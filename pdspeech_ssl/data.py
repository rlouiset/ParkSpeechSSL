from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import lightning.pytorch as pl
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from pdspeech_ssl.augment import augment_waveform
from pdspeech_ssl.config import AugmentHParams, DataHParams

HC_PD_LABELS = ("HC", "PD")
IndividualKey = Tuple[str, str]  # (dataset_name, patient_hash) -- globally unique


@dataclass(frozen=True)
class Individual:
    key: IndividualKey
    dataset: str
    label: str
    paths: List[Path]


def scan_derivatives(root: Path) -> List[Individual]:
    """Every derivatives/{Dataset}/*.wav is named {LABEL}_{patient_hash}_{idx}_{Dataset}.wav.
    (dataset folder name, patient_hash) is the globally-unique individual key --
    patient_hash alone isn't safe to use across datasets since each preprocessing
    script salts its own hash independently."""
    by_key: Dict[IndividualKey, Individual] = {}
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for wav_path in sorted(dataset_dir.glob("*.wav")):
            parts = wav_path.stem.split("_")
            label, phash = parts[0], parts[1]
            key = (dataset_dir.name, phash)
            if key not in by_key:
                by_key[key] = Individual(key=key, dataset=dataset_dir.name, label=label, paths=[])
            by_key[key].paths.append(wav_path)
    return list(by_key.values())


def split_individuals(
    individuals: List[Individual], val_fraction: float, seed: int
) -> Tuple[List[Individual], List[Individual]]:
    """HC/PD individuals are split val_fraction/rest, stratified per (dataset,
    label) so small datasets aren't dropped entirely into one split. Every
    other-diagnosis individual (MSA/PSP/DYS) goes to train only, per instruction."""
    rng = random.Random(seed)
    train, val = [], []

    groups: Dict[Tuple[str, str], List[Individual]] = {}
    for ind in individuals:
        if ind.label in HC_PD_LABELS:
            groups.setdefault((ind.dataset, ind.label), []).append(ind)
        else:
            train.append(ind)

    for group in groups.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_fraction)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    return train, val


def normalize_loudness(wav: torch.Tensor, sample_rate: int, target_lufs: float) -> torch.Tensor:
    """LUFS integrated-loudness normalization (single global gain per clip --
    within-clip dynamics/dynamic-range are gain-invariant and pass through
    untouched; only absolute level, the part most confounded by this
    project's wildly different per-corpus recording setups, is removed)."""
    import pyloudnorm as pyln

    audio = wav.numpy()
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError:
        return wav  # shorter than the meter's gating block -- can't measure, nothing to normalize
    if loudness == float("-inf"):
        return wav  # silence / near-silence -- meter can't measure, nothing to normalize
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

    # pyln.normalize.loudness applies one flat gain with no headroom/peak awareness, so a
    # quiet clip needing a big boost can land past full scale. Back off with a single extra
    # proportional rescale of the whole clip (preserves waveform shape) instead of clamping
    # samples (which would hard-clip/distort exactly the loudest peaks).
    peak = float(abs(normalized).max())
    if peak > 1.0:
        normalized = normalized * (0.99 / peak)

    return torch.from_numpy(normalized).float()


def load_waveform(
    path: Path, sample_rate: int, max_seconds: float | None = None, target_lufs: float | None = None
) -> torch.Tensor:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    wav = torch.from_numpy(audio)
    if sr != sample_rate:
        import torchaudio.functional as AF

        wav = AF.resample(wav, sr, sample_rate)
    if target_lufs is not None:
        # normalize on the full clip before truncating -- integrated loudness measured
        # over a fixed 10s crop would depend on where that crop happens to land.
        wav = normalize_loudness(wav, sample_rate, target_lufs)
    if max_seconds is not None:
        wav = wav[: int(max_seconds * sample_rate)]
    return wav


class PairBatch(TypedDict):
    view1: torch.Tensor
    len1: torch.Tensor
    view2: torch.Tensor
    len2: torch.Tensor
    labels: List[str]
    keys: List[IndividualKey]


class IndividualPairDataset(Dataset):
    """One item = one individual's contrastive pair. If the individual has
    >=2 segments, the pair is two distinct (augmented) segments; if only 1
    (e.g. every FredPrior patient), the pair is that one segment augmented
    twice -- standard SimCLR-style fallback, so no individual is dropped."""

    def __init__(self, individuals: List[Individual], data_cfg: DataHParams, augment_cfg: AugmentHParams):
        self.individuals = individuals
        self.data_cfg = data_cfg
        self.augment_cfg = augment_cfg

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, idx: int):
        ind = self.individuals[idx]
        if len(ind.paths) >= 2:
            p1, p2 = random.sample(ind.paths, 2)
        else:
            p1 = p2 = ind.paths[0]

        wav1 = augment_waveform(
            load_waveform(p1, self.data_cfg.sample_rate, self.data_cfg.max_audio_seconds, self.data_cfg.target_lufs),
            self.augment_cfg, self.data_cfg.sample_rate,
        )
        wav2 = augment_waveform(
            load_waveform(p2, self.data_cfg.sample_rate, self.data_cfg.max_audio_seconds, self.data_cfg.target_lufs),
            self.augment_cfg, self.data_cfg.sample_rate,
        )
        return {"view1": wav1, "view2": wav2, "label": ind.label, "key": ind.key}


def collate_pairs(batch: list) -> PairBatch:
    view1 = [b["view1"] for b in batch]
    view2 = [b["view2"] for b in batch]
    len1 = torch.tensor([v.shape[-1] for v in view1], dtype=torch.long)
    len2 = torch.tensor([v.shape[-1] for v in view2], dtype=torch.long)
    return {
        "view1": pad_sequence(view1, batch_first=True),
        "len1": len1,
        "view2": pad_sequence(view2, batch_first=True),
        "len2": len2,
        "labels": [b["label"] for b in batch],
        "keys": [b["key"] for b in batch],
    }


class SegmentBatch(TypedDict):
    wav: torch.Tensor
    lengths: torch.Tensor
    labels: List[str]


class SegmentDataset(Dataset):
    """One item per *segment* (not per individual) -- used only for the HC/PD
    linear probe, which wants many samples for a stable metric, not one
    embedding per individual."""

    def __init__(self, individuals: List[Individual], data_cfg: DataHParams, max_samples: int | None, seed: int):
        segments = []
        for ind in individuals:
            if ind.label not in HC_PD_LABELS:
                continue
            for path in ind.paths:
                segments.append((path, ind.label))
        if max_samples is not None and len(segments) > max_samples:
            rng = random.Random(seed)
            segments = rng.sample(segments, max_samples)
        self.segments = segments
        self.data_cfg = data_cfg

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int):
        path, label = self.segments[idx]
        wav = load_waveform(path, self.data_cfg.sample_rate, self.data_cfg.max_audio_seconds, self.data_cfg.target_lufs)
        return {"wav": wav, "label": label}


def collate_segments(batch: list) -> SegmentBatch:
    wavs = [b["wav"] for b in batch]
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.long)
    return {
        "wav": pad_sequence(wavs, batch_first=True),
        "lengths": lengths,
        "labels": [b["label"] for b in batch],
    }


class PDSpeechDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: DataHParams, augment_cfg: AugmentHParams, training_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.augment_cfg = augment_cfg
        self.training_cfg = training_cfg
        self.train_individuals: List[Individual] = []
        self.val_individuals: List[Individual] = []

    def setup(self, stage: str | None = None):
        individuals = scan_derivatives(Path(self.data_cfg.derivatives_root))
        self.train_individuals, self.val_individuals = split_individuals(
            individuals, self.data_cfg.val_fraction, self.data_cfg.split_seed
        )
        print(
            f"[data] {len(self.train_individuals)} train individuals "
            f"({sum(1 for i in self.train_individuals if i.label in HC_PD_LABELS)} HC/PD), "
            f"{len(self.val_individuals)} val individuals (all HC/PD)"
        )

    def train_dataloader(self) -> DataLoader:
        ds = IndividualPairDataset(self.train_individuals, self.data_cfg, self.augment_cfg)
        return DataLoader(
            ds,
            batch_size=self.training_cfg.batch_size_per_gpu,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_pairs,
            num_workers=self.data_cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        # No-op contrastive val loader (same pairing, no shuffle needed for correctness,
        # kept mainly to log a comparable contrastive val loss); the meaningful
        # HC/PD metrics come from the rank-0-only linear probe dataloaders below.
        ds = IndividualPairDataset(self.val_individuals, self.data_cfg, self.augment_cfg)
        return DataLoader(
            ds,
            batch_size=self.training_cfg.batch_size_per_gpu,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_pairs,
            num_workers=self.data_cfg.num_workers,
        )

    def probe_train_dataloader(self) -> DataLoader:
        ds = SegmentDataset(
            self.train_individuals, self.data_cfg, self.training_cfg.probe_max_train_samples, self.data_cfg.split_seed
        )
        return DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_segments, num_workers=2)

    def probe_val_dataloader(self) -> DataLoader:
        ds = SegmentDataset(
            self.val_individuals, self.data_cfg, self.training_cfg.probe_max_val_samples, self.data_cfg.split_seed
        )
        return DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_segments, num_workers=2)

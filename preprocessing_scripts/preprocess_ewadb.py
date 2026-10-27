"""
Preprocess the EWA-DB (Slovak) raw dataset into the derivatives/EWADB folder.

Source layout: {group}/{speaker_code}/{task}/{item}.wav (+ a .json of
precomputed features next to every recording). Groups are Healthy, Parkinson,
Alzheimer-Parkinson, Alzheimer and MCI; only the first three are used here.
Speakers who did not sign the publish agreement only have the .json files, so
speakers are taken from the .wav files actually on disk, not from
SPEAKERS.TSV. All audio is natively 16000Hz mono.

Labels: Parkinson -> PD, Healthy -> HC. Alzheimer-Parkinson (2 speakers, only
1 with audio) is also written as PD: those speakers do have PD, and the
dataset is too small for a separate label to be useful.

Healthy subsampling: EWA-DB has ~9x more healthy speakers with audio than PD
ones, so to avoid swamping the HC/PD balance only as many HC speakers as PD
speakers are kept. Every speaker follows the same recording protocol, so
matching the speaker count also roughly matches the file count. Rather than a
plain random draw, HC speakers are age- and sex-matched to the PD speakers
(SPEAKERS.TSV): for each PD speaker, oldest first, the unused HC speaker of
the same sex with the closest age is taken, and ties are broken by a seeded
shuffle. Healthy speakers in EWA-DB skew younger than the PD group, so a
random draw would make age a shortcut for the label.

Per-task routing (same rules as NeuroVoz):
- phonation (sustained vowel, ~15s): kept whole, dropped if longer than
  MAX_PHONATION_DURATION (20s; never triggers, the app caps at 15s).
- pataka (diadochokinesis, ~8s): kept whole, cropped to PATAKA_MAX_DURATION.
- naming (single words, median ~3s): under MIN_KEEP_DURATION, so kept whole.
  The ~2% under MIN_OUTPUT_DURATION (2s) are dropped.
- picture (picture description, up to 90s): VAD-split like NeuroVoz FREE,
  chunked into <=10s pieces with pieces under 6s dropped.

Output files are named:
    {HC|PD}_{patient_hash}_{segment_index}_EWADB.wav
patient_hash = md5("{label}::{speaker_code}")[:8].
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

RAW_ROOT = Path("/Users/robinlouiset/Documents/EWA-DB")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/EWADB")

GROUP_LABELS = {"Parkinson": "PD", "Alzheimer-Parkinson": "PD", "Healthy": "HC"}
HC_MATCH_SEED = 0

SR = 16000
MIN_OUTPUT_DURATION = 2.0  # drop any output segment shorter than this, regardless of which path produced it
MIN_KEEP_DURATION = 10.0  # below this, keep the file whole
MIN_SEGMENT_DURATION = 6.0  # drop VAD segments (or chunks) shorter than this
MAX_SEGMENT_DURATION = 10.0  # VAD segments longer than this are chunked into pieces of this size
MAX_PHONATION_DURATION = 20.0  # drop vowel-phonation files longer than this entirely
PATAKA_MAX_DURATION = 20.0  # crop pataka files longer than this to this length
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 2500  # same as NeuroVoz FREE (spontaneous speech with long pauses)

VAD_MODEL = load_silero_vad()


@dataclass(frozen=True)
class Segment:
    start: float
    end: float


class AudioData:
    def __init__(self, array: np.ndarray, sr: int):
        self.array = array
        self.sr = sr

    @classmethod
    def from_file(cls, path: Path, segment: Segment) -> "AudioData":
        audio, sr = librosa.load(
            path,
            sr=SR,
            mono=True,
            res_type="kaiser_fast",
            offset=segment.start,
            duration=segment.end - segment.start,
        )
        return cls(array=audio, sr=sr)


@dataclass(frozen=True)
class PatientFile:
    path: Path
    label: str  # "HC" or "PD"
    task: str
    patient_hash: str


@dataclass(frozen=True)
class Speaker:
    code: str
    group: str
    age: int | None
    sex: str | None


def patient_hash(label: str, patient_id: str) -> str:
    digest = hashlib.md5(f"{label}::{patient_id}".encode()).hexdigest()
    return digest[:8]


def load_speaker_info() -> dict[str, tuple[int | None, str | None]]:
    info = {}
    with open(RAW_ROOT / "SPEAKERS.TSV", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            age = int(row["AGE"]) if row["AGE"].isdigit() else None
            if age is not None and not 15 <= age <= 110:  # a few rows hold a year (e.g. 2022) instead of an age
                age = None
            info[row["SPEAKER_CODE"]] = (age, row["SEX"] or None)
    return info


def speakers_with_audio(group: str, info: dict) -> list[Speaker]:
    codes = sorted({p.parts[-3] for p in (RAW_ROOT / group).glob("*/*/*.wav")})
    return [Speaker(code, group, *info.get(code, (None, None))) for code in codes]


def match_healthy(pd_speakers: list[Speaker], hc_pool: list[Speaker]) -> list[Speaker]:
    """One age/sex-matched HC speaker per PD speaker (nearest age, same sex,
    without replacement). PD speakers with unknown age/sex get a random HC."""
    rng = random.Random(HC_MATCH_SEED)
    pool = list(hc_pool)
    rng.shuffle(pool)
    chosen = []
    for pd in sorted(pd_speakers, key=lambda s: -(s.age or 0)):
        candidates = [s for s in pool if s.age is not None and s.sex == pd.sex]
        if pd.age is None or pd.sex is None or not candidates:
            pick = pool[0]
        else:
            pick = min(candidates, key=lambda s: abs(s.age - pd.age))
        pool.remove(pick)
        chosen.append(pick)
    return chosen


def select_speakers() -> list[Speaker]:
    info = load_speaker_info()
    pd_speakers = [s for g, l in GROUP_LABELS.items() if l == "PD" for s in speakers_with_audio(g, info)]
    hc_pool = speakers_with_audio("Healthy", info)
    hc_speakers = match_healthy(pd_speakers, hc_pool)

    def summary(speakers: list[Speaker]) -> str:
        ages = [s.age for s in speakers if s.age is not None]
        n_female = sum(s.sex == "female" for s in speakers)
        return f"{len(speakers)} speakers, age {np.mean(ages):.1f}±{np.std(ages):.1f}, {n_female} female"

    print(f"PD: {summary(pd_speakers)}")
    print(f"HC: {summary(hc_speakers)} (matched from a pool of {summary(hc_pool)})")
    return pd_speakers + hc_speakers


def collect_files(speakers: list[Speaker]) -> list[PatientFile]:
    files = []
    for spk in speakers:
        label = GROUP_LABELS[spk.group]
        phash = patient_hash(label, spk.code)
        for wav_path in sorted((RAW_ROOT / spk.group / spk.code).glob("*/*.wav")):
            files.append(PatientFile(wav_path, label, wav_path.parent.name, phash))
    return files


def chunk_segment(segment: Segment) -> list[Segment]:
    """Split a segment into pieces of at most MAX_SEGMENT_DURATION, dropping
    any leftover piece shorter than MIN_SEGMENT_DURATION."""
    length = segment.end - segment.start
    if length < MIN_SEGMENT_DURATION:
        return []
    if length <= MAX_SEGMENT_DURATION:
        return [segment]

    chunks = []
    pos = segment.start
    while segment.end - pos > MAX_SEGMENT_DURATION:
        chunks.append(Segment(pos, pos + MAX_SEGMENT_DURATION))
        pos += MAX_SEGMENT_DURATION
    if segment.end - pos >= MIN_SEGMENT_DURATION:
        chunks.append(Segment(pos, segment.end))
    return chunks


def speech_segments(path: Path, duration: float) -> list[Segment]:
    audio = AudioData.from_file(path, Segment(0.0, duration))
    tensor = torch.from_numpy(audio.array)
    timestamps = get_speech_timestamps(
        tensor,
        VAD_MODEL,
        sampling_rate=SR,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        return_seconds=True,
    )
    segments = []
    for ts in timestamps:
        segments.extend(chunk_segment(Segment(ts["start"], ts["end"])))
    return segments


def segments_for_file(path: Path, task: str, duration: float) -> list[Segment]:
    if task == "phonation":
        if duration > MAX_PHONATION_DURATION:
            return []
        return [Segment(0.0, duration)]
    if task == "pataka":
        return [Segment(0.0, min(duration, PATAKA_MAX_DURATION))]
    if duration < MIN_KEEP_DURATION:
        return [Segment(0.0, duration)]
    return speech_segments(path, duration)


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_EWADB.wav"


def process_file(pf: PatientFile, next_index: dict[str, int]) -> int:
    duration = sf.info(str(pf.path)).duration
    segments = [
        seg for seg in segments_for_file(pf.path, pf.task, duration)
        if seg.end - seg.start >= MIN_OUTPUT_DURATION
    ]

    written = 0
    for segment in segments:
        idx = next_index.get(pf.patient_hash, 1)
        audio = AudioData.from_file(pf.path, segment)
        out_path = OUT_ROOT / out_name(pf.label, pf.patient_hash, idx)
        sf.write(str(out_path), audio.array, audio.sr)
        next_index[pf.patient_hash] = idx + 1
        written += 1
    return written


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_files = collect_files(select_speakers())
    print(f"found {len(all_files)} source .wav files")

    next_index: dict[str, int] = {}
    total_segments = 0
    for i, pf in enumerate(all_files, 1):
        written = process_file(pf, next_index)
        total_segments += written
        if i % 500 == 0 or i == len(all_files):
            print(f"[{i}/{len(all_files)}] processed, {total_segments} segments written so far")

    print(f"done: {total_segments} segments written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

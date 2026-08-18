"""
Preprocess the Czech-PD (czech-pd-JanHlavnicka) raw dataset into the
derivatives/CzechPD folder.

Source layout: a single flat directory (records/) containing, per recording,
several related files (raw .wav, _LF.wav, _impulses.csv, _clean.wav); only
the _clean.wav files are used here. Filenames look like "HC10a1_clean.wav" =
label "HC", patient id "10", vowel task "a", take "1" (dataset.csv confirms
this same {record, subject, group, vowel} structure). Four diagnosis groups
are present, not just HC/PD: HC (88 recordings/22 patients), PD (93/22), MSA
-- Multiple System Atrophy (84/21), PSP -- Progressive Supranuclear Palsy
(72/18). All four are kept as their own output label (HC_/PD_/MSA_/PSP_)
rather than merging MSA/PSP into PD.

Patient hash is derived from the label+id pair embedded in the filename
(e.g. "HC1" vs "MSA1" vs "PD1" are three different patients, even though the
numeric id repeats across groups), matching the same (label, patient_id)
hashing scheme used for the other datasets.

These are all sustained-vowel-phonation recordings (tasks "a" and "i"), and
the _clean files are already pre-trimmed by the dataset's own curators -- so
no VAD is applied at all. Same phonation treatment as the vowel tasks in
IPVS/NeuroVoz: each file is kept whole (resampled to 16000Hz), but dropped
entirely if longer than MAX_PHONATION_DURATION (20s, same value as the other
datasets). Note this cutoff is asymmetric here -- HC recordings run longer on
average than the disease groups, so HC loses proportionally more files to
the cap (46.6%) than PD/MSA/PSP (~13-16%) -- but 20s was kept for consistency
with IPVS/NeuroVoz per instruction.

Output files are named:
    {HC|PD|MSA|PSP}_{patient_hash}_{segment_index}_CzechPD.wav
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/czech-pd-JanHlavnicka/records")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/CzechPD")

SR = 16000
MAX_PHONATION_DURATION = 20.0  # drop vowel-phonation files longer than this entirely

FILENAME_RE = re.compile(r"^([A-Za-z]+)(\d+)([a-zA-Z]+)(\d+)_clean$")


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
    label: str  # "HC", "PD", "MSA", or "PSP"
    vowel: str
    patient_hash: str


def patient_hash(label: str, patient_id: str) -> str:
    digest = hashlib.md5(f"{label}::{patient_id}".encode()).hexdigest()
    return digest[:8]


def collect_files() -> list[PatientFile]:
    files = []
    for wav_path in sorted(RAW_ROOT.glob("*_clean.wav")):
        match = FILENAME_RE.match(wav_path.stem)
        label, pid, vowel, _take = match.groups()
        files.append(PatientFile(wav_path, label, vowel, patient_hash(label, pid)))
    return files


def segments_for_file(duration: float) -> list[Segment]:
    if duration > MAX_PHONATION_DURATION:
        return []
    return [Segment(0.0, duration)]


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_CzechPD.wav"


def process_file(pf: PatientFile, next_index: dict[str, int]) -> int:
    duration = sf.info(str(pf.path)).duration
    segments = segments_for_file(duration)

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

    all_files = collect_files()
    print(f"found {len(all_files)} source _clean.wav files")

    next_index: dict[str, int] = {}
    total_segments = 0
    for i, pf in enumerate(all_files, 1):
        written = process_file(pf, next_index)
        total_segments += written
        if i % 50 == 0 or i == len(all_files):
            print(f"[{i}/{len(all_files)}] processed, {total_segments} segments written so far")

    print(f"done: {total_segments} segments written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

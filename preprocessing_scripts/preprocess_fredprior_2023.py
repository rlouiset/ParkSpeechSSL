"""
Preprocess the FredPrior_AnuHIER_2023_SciRep raw dataset into the
derivatives/FredPrior folder.

Source layout: two flat directories, HC_AH (41 files) and PD_AH (40 files).
Filenames encode a patient id but with two different formats between the two
folders:
    HC_AH: AH_<id>_<UUID>.wav        e.g. "AH_121A_BD5BA248-E807-....wav"
           (id is alphanumeric, e.g. "121A"; underscore before the UUID)
    PD_AH: AH_<id>-<UUID>.wav        e.g. "AH_545616858-3A749CBC-....wav"
           (id is numeric, e.g. "545616858"; dash directly before the UUID,
           no separating underscore)
All ids are confirmed unique within their folder (no repeats), so each file
is one patient. Every file is a single sustained-vowel ("AH") recording,
natively 8000Hz (telephone-quality) and short (1.5-7.2s, well under any
phonation cap) -- so each file maps to exactly one output segment: kept
whole and resampled to 16000Hz (upsampled from 8kHz; this doesn't add real
information above the original 4kHz Nyquist limit, but keeps output
consistent with the other datasets).

Output files are named:
    {HC|PD}_{patient_hash}_{segment_index}_FredPrior.wav
patient_hash is a short, stable hash of (label, patient_id).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/FredPrior_AnuHIER_2023_SciRep")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/FredPrior")

SR = 16000
MAX_PHONATION_DURATION = 20.0  # drop vowel-phonation files longer than this entirely (safety net)

SUBDIRS = {
    "HC_AH": "HC",
    "PD_AH": "PD",
}
HC_FILENAME_RE = re.compile(r"^AH_([A-Za-z0-9]+)_[0-9A-F-]+$", re.IGNORECASE)
PD_FILENAME_RE = re.compile(r"^AH_(\d+)-[0-9A-F-]+$", re.IGNORECASE)


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
    patient_hash: str


def patient_hash(label: str, patient_id: str) -> str:
    digest = hashlib.md5(f"{label}::{patient_id}".encode()).hexdigest()
    return digest[:8]


def collect_files() -> list[PatientFile]:
    files = []
    for subdir, label in SUBDIRS.items():
        pattern = HC_FILENAME_RE if label == "HC" else PD_FILENAME_RE
        for wav_path in sorted((RAW_ROOT / subdir).glob("*.wav")):
            match = pattern.match(wav_path.stem)
            pid = match.group(1)
            files.append(PatientFile(wav_path, label, patient_hash(label, pid)))
    return files


def segments_for_file(duration: float) -> list[Segment]:
    if duration > MAX_PHONATION_DURATION:
        return []
    return [Segment(0.0, duration)]


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_FredPrior.wav"


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
    print(f"found {len(all_files)} source .wav files")

    next_index: dict[str, int] = {}
    total_segments = 0
    for i, pf in enumerate(all_files, 1):
        written = process_file(pf, next_index)
        total_segments += written

    print(f"done: {total_segments} segments written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

"""
Preprocess the SJTU_YongFu raw dataset into the derivatives/SJTU folder.

Source layout is denoised-speech-dataset/, which despite looking like "5
folders, one per individual" is actually: Tessi, LW, DL as single
individuals; emma split into two task-type subfolders (WP1111, IC1111) for
the same person -- only WP1111 is used, IC1111 is skipped per instruction;
and Faces split into six separate individuals by initials (BG_au, MJ_au,
TP_au, TS_au, SK_au, JC_au). So there are really 9 individuals processed
here (Tessi, LW, DL, emma, BG, MJ, TP, TS, SK, JC), not 5. All are PD (no
HC in this dataset).

Files are already natively 16000Hz and short, pre-segmented single
utterances (3-15s typically) -- there's no VAD/splitting to do. Instead, per
individual: recordings >=MAX_DURATION (10s) are dropped entirely, and of the
rest, only the TOP_N (10) longest are kept (or fewer if an individual
doesn't have that many recordings under 10s to begin with, e.g. Faces/TP_au
only has 6 total).

Output files are named:
    PD_{patient_hash}_{segment_index}_SJTU.wav
where segment_index 1 is that individual's longest recording, 2 the second
longest, etc. patient_hash is a short, stable hash of the individual's id
(Tessi/LW/DL/emma/BG/MJ/TP/TS/SK/JC).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/SJTU_YongFu/denoised-speech-dataset")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/SJTU")

SR = 16000
MAX_DURATION = 10.0  # recordings at or above this are dropped entirely
TOP_N = 10  # of the recordings under MAX_DURATION, keep at most this many (longest first)

INDIVIDUALS = {
    "Tessi": RAW_ROOT / "Tessi",
    "LW": RAW_ROOT / "LW",
    "DL": RAW_ROOT / "DL",
    "emma": RAW_ROOT / "emma" / "WP1111",
    "BG": RAW_ROOT / "Faces" / "BG_au",
    "MJ": RAW_ROOT / "Faces" / "MJ_au",
    "TP": RAW_ROOT / "Faces" / "TP_au",
    "TS": RAW_ROOT / "Faces" / "TS_au",
    "SK": RAW_ROOT / "Faces" / "SK_au",
    "JC": RAW_ROOT / "Faces" / "JC_au",
}


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


def patient_hash(person_id: str) -> str:
    digest = hashlib.md5(f"PD::{person_id}".encode()).hexdigest()
    return digest[:8]


def selected_files(person_dir: Path) -> list[tuple[Path, float]]:
    wavs = sorted(person_dir.glob("*.wav"))
    durations = [(w, sf.info(str(w)).duration) for w in wavs]
    durations = [x for x in durations if x[1] < MAX_DURATION]
    durations.sort(key=lambda x: x[1], reverse=True)
    return durations[:TOP_N]


def out_name(phash: str, segment_index: int) -> str:
    return f"PD_{phash}_{segment_index}_SJTU.wav"


def process_individual(person_id: str, person_dir: Path) -> int:
    phash = patient_hash(person_id)
    ranked = selected_files(person_dir)

    for idx, (wav_path, duration) in enumerate(ranked, 1):
        audio = AudioData.from_file(wav_path, Segment(0.0, duration))
        out_path = OUT_ROOT / out_name(phash, idx)
        sf.write(str(out_path), audio.array, audio.sr)
    return len(ranked)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_segments = 0
    for person_id, person_dir in INDIVIDUALS.items():
        n = process_individual(person_id, person_dir)
        total_segments += n
        print(f"{person_id}: {n} segments kept")

    print(f"done: {total_segments} segments written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

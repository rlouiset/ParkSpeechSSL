"""
Preprocess the YeTalkBank raw dataset into the derivatives/YeTalkBank
folder.

Source layout: a single flat directory of 43 files, one per patient. Despite
being described as .wav, they are actually .mp3 (confirmed soundfile 0.13.1
decodes them directly, no ffmpeg needed). Sample rates are mixed (44100,
48000, 22050, 16000Hz across different files) and are all resampled to
16000Hz via librosa as usual. All patients are PD (per instruction) --
filenames don't consistently start with "PD" (e.g. "BC002", "BJ_MZ_015",
"DF009" alongside "PD015" etc.), but the diagnosis is PD regardless of the
filename prefix; the filename stem (with extension stripped) is used
directly as the patient id.

Recordings are long-form (~60-71s, mean 62.5s) -- these read like
spontaneous-speech/reading recordings, not short phonation clips -- so they
go through the same VAD-split pipeline used for the equivalent task type in
the other datasets (IPVS's B/FB/PR, NeuroVoz's FREE, KCL): Silero VAD with
min_silence_duration_ms=2500 (the value settled on for spontaneous speech
elsewhere in this project), chunked into pieces of at most
MAX_SEGMENT_DURATION (10s), dropping any piece under MIN_SEGMENT_DURATION
(6s).

Output files are named:
    PD_{patient_hash}_{segment_index}_YeTalkBank.wav
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/YeTalkBank")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/YeTalkBank")

SR = 16000
MIN_SEGMENT_DURATION = 6.0  # drop VAD segments (or chunks) shorter than this
MAX_SEGMENT_DURATION = 10.0  # VAD segments longer than this are chunked into pieces of this size
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 2500  # matches the value used for spontaneous speech in the other datasets

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
    patient_hash: str


def patient_hash(patient_id: str) -> str:
    digest = hashlib.md5(f"PD::{patient_id}".encode()).hexdigest()
    return digest[:8]


def collect_files() -> list[PatientFile]:
    files = []
    for path in sorted(RAW_ROOT.glob("*.mp3")):
        files.append(PatientFile(path, patient_hash(path.stem)))
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


def out_name(phash: str, segment_index: int) -> str:
    return f"PD_{phash}_{segment_index}_YeTalkBank.wav"


def process_file(pf: PatientFile) -> int:
    duration = sf.info(str(pf.path)).duration
    segments = speech_segments(pf.path, duration)

    for idx, segment in enumerate(segments, 1):
        audio = AudioData.from_file(pf.path, segment)
        out_path = OUT_ROOT / out_name(pf.patient_hash, idx)
        sf.write(str(out_path), audio.array, audio.sr)
    return len(segments)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_files = collect_files()
    print(f"found {len(all_files)} source files")

    total_segments = 0
    zero_count = 0
    for i, pf in enumerate(all_files, 1):
        n = process_file(pf)
        total_segments += n
        if n == 0:
            zero_count += 1
        print(f"[{i}/{len(all_files)}] {pf.path.name}: {n} segments")

    print(f"done: {total_segments} segments written to {OUT_ROOT}")
    print(f"files with zero output segments: {zero_count}/{len(all_files)}")


if __name__ == "__main__":
    main()

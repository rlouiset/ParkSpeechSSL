"""
Preprocess the KCL (26-29_09_2017_KCL) raw dataset into the derivatives/KCL
folder.

Source layout:
    {ReadText|SpontaneousDialogue}/{HC|PD}/{patient_id}_{hc|pd}_{n}_{n}_{n}.wav
e.g. "ReadText/HC/ID00_hc_0_0_0.wav". One file
(SpontaneousDialogue/HC/ID22hc_0_0_0.wav) is missing the underscore before
the label, so the filename is parsed with a regex rather than a plain split.

All recordings are long-form (73-220s, natively 44100Hz) and contain the
clinician talking in the background at lower volume in addition to the
patient. Checked whether raising Silero VAD's confidence threshold could
isolate the (louder, closer) patient from the (quieter, farther) clinician:
it can't -- per-frame speech probabilities are strongly bimodal (most frames
are either near-0 or near-1 confidence) and raising the threshold from 0.5 to
0.99 only trims detected speech from 52.4% to 42.4% of a sample file, with no
clean separation point tied to loudness (mean RMS 0.0085 above 0.9 confidence
vs 0.0043 in the 0.5-0.9 band -- a weak, continuous correlation, not a
boundary). So this is accepted as a known limitation: some clinician speech
will leak into output segments, especially in SpontaneousDialogue where
speaker turns interleave. Standard VAD (threshold=0.5) is used as-is.

Per the "don't keep silences bigger than 1s" requirement, VAD's silence-merge
window is set to 1000ms: any pause >=1s becomes a segment boundary, while
shorter pauses are bridged within a kept segment. Segments are then bounded
to [MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION] = [6s, 10s], same as IPVS and
NeuroVoz, by chunking anything longer and dropping any leftover piece
shorter than 6s.

Output files are named:
    {HC|PD}_{patient_hash}_{segment_index}_KCL.wav
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
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/26-29_09_2017_KCL")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/KCL")

TASK_DIRS = ["ReadText", "SpontaneousDialogue"]
LABEL_DIRS = ["HC", "PD"]

SR = 16000
MIN_KEEP_DURATION = 10.0  # below this, keep the file whole (never triggers here in practice)
MIN_SEGMENT_DURATION = 6.0  # drop VAD segments (or chunks) shorter than this
MAX_SEGMENT_DURATION = 10.0  # VAD segments longer than this are chunked into pieces of this size
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 1000  # don't keep silences bigger than 1s

FILENAME_RE = re.compile(r"^(ID\d+)_?(hc|pd)_(\d+)_(\d+)_(\d+)$", re.IGNORECASE)

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
    task: str  # "ReadText" or "SpontaneousDialogue"
    patient_hash: str


def patient_hash(label: str, patient_id: str) -> str:
    digest = hashlib.md5(f"{label}::{patient_id}".encode()).hexdigest()
    return digest[:8]


def collect_files() -> list[PatientFile]:
    files = []
    for task in TASK_DIRS:
        for label in LABEL_DIRS:
            for wav_path in sorted((RAW_ROOT / task / label).glob("*.wav")):
                match = FILENAME_RE.match(wav_path.stem)
                pid = match.group(1).upper()
                files.append(PatientFile(wav_path, label, task, patient_hash(label, pid)))
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


def segments_for_file(path: Path, duration: float) -> list[Segment]:
    if duration < MIN_KEEP_DURATION:
        return [Segment(0.0, duration)]
    return speech_segments(path, duration)


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_KCL.wav"


def process_file(pf: PatientFile, next_index: dict[str, int]) -> int:
    duration = sf.info(str(pf.path)).duration
    segments = segments_for_file(pf.path, duration)

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
        print(f"[{i}/{len(all_files)}] {pf.path.name}: {written} segments")

    print(f"done: {total_segments} segments written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

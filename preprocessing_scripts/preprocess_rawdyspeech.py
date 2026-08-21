"""
Preprocess the RAWDysPeech raw dataset into the derivatives/RAWDysPeech
folder.

Source layout: two folders, "0" (healthy, label HC) and "1" (dysarthric,
label DYS -- etiology isn't confirmed by any metadata in the raw dataset,
so it is NOT assumed to be PD specifically). Each folder mixes two
completely different filename schemes:

1. Standard: "{id}_B{b}_{word}_M{m}.wav", e.g. "CF02_B1_C10_M2.wav". Only
   B1 (per instruction, B2/B3 discarded) and M2 (per instruction -- there is
   no M1 anywhere in the raw data; repetitions run M2-M8, so M2 stands in
   for "the first take"). word ids come from several vocabularies (C##,
   CW##, UW##, D#, and single-letter codes LA-LZ), all valid "id of the
   word being pronounced". Within the B1/M2 subset, each individual has at
   most one file per word id (verified, no duplicates), so picking distinct
   word ids is automatic.
2. wav_*Mic_*: "wav_{arrayMic|headMic}_{token}_{word}.wav", e.g.
   "wav_headMic_M05S02_0369.wav". Per instruction, {token} (e.g. "M05S02")
   IS the individual id here (including its session suffix -- treated as
   its own identity, not merged with the standard scheme's ids, since nothing
   confirms the two naming schemes refer to the same underlying people).
   Each word typically has both an arrayMic and a headMic recording of the
   same utterance (not independent repetitions); the longer (post-VAD-trim)
   of the two is used as that word's single candidate, so word ids stay
   unique per individual here too, consistent with pattern 1.

For both patterns: every candidate recording is passed through Silero VAD to
find its speech span (bounding box from the first detected segment's start
to the last one's end -- these are short single-word/utterance clips, so
this is a trim, not a multi-segment split) and trimmed to that span. If VAD
finds no speech at all, the original file is kept untrimmed as a fallback
(tracked and reported, rather than silently dropping the recording).

Selection, per individual, per pattern:
- pattern 1: rank by trimmed duration, keep the TOP_N (15) longest.
- pattern 2: drop anything with trimmed duration >= WAVMIC_MAX_DURATION
  (10s) entirely, then rank the rest by trimmed duration and keep the
  TOP_N (15) longest.

Output files are named:
    {HC|DYS}_{patient_hash}_{segment_index}_RAWDysPeech.wav
patient_hash is salted by which naming pattern produced it (in addition to
label and id), since pattern-1 and pattern-2 ids are different, unmerged
identity spaces even when their id strings happen to look related.
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

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/RAWDysPeech")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/RAWDysPeech")

GROUP_LABELS = {"0": "HC", "1": "DYS"}

SR = 16000
TOP_N = 15
MIN_DURATION = 2.0  # candidates (either pattern) shorter than this are dropped entirely
MAX_DURATION = 10.0  # candidates (either pattern) at or above this are dropped entirely
VAD_THRESHOLD = 0.5

STANDARD_RE = re.compile(r"^([A-Za-z]+\d+)_B(\d+)_([A-Za-z]+\d*)_M(\d+)\.wav$")
WAVMIC_RE = re.compile(r"^wav_(arrayMic|headMic)_([A-Za-z]+\d+S\d+)_(\d+)\.wav$")

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


def patient_hash(label: str, pattern: str, individual_id: str) -> str:
    digest = hashlib.md5(f"{label}::{pattern}::{individual_id}".encode()).hexdigest()
    return digest[:8]


def vad_trim(path: Path) -> tuple[Segment, bool]:
    """Return (trimmed span, used_fallback). used_fallback=True means VAD
    found no speech and the original full-file span was used instead."""
    duration = sf.info(str(path)).duration
    audio = AudioData.from_file(path, Segment(0.0, duration))
    tensor = torch.from_numpy(audio.array)
    timestamps = get_speech_timestamps(
        tensor, VAD_MODEL, sampling_rate=SR, threshold=VAD_THRESHOLD, return_seconds=True
    )
    if not timestamps:
        return Segment(0.0, duration), True
    return Segment(timestamps[0]["start"], timestamps[-1]["end"]), False


def collect_standard(label_dir: Path) -> dict[str, list[tuple[Path, str]]]:
    """individual_id -> list of (path, word_id) for B1/M2 files."""
    by_individual: dict[str, list[tuple[Path, str]]] = {}
    for wav_path in sorted(label_dir.glob("*.wav")):
        match = STANDARD_RE.match(wav_path.name)
        if not match:
            continue
        pid, b, word, mrep = match.groups()
        if b != "1" or mrep != "2":
            continue
        by_individual.setdefault(pid, []).append((wav_path, word))
    return by_individual


def collect_wavmic(label_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """individual_token -> word_id -> list of candidate paths (arrayMic/headMic)."""
    by_token: dict[str, dict[str, list[Path]]] = {}
    for wav_path in sorted(label_dir.glob("wav_*.wav")):
        match = WAVMIC_RE.match(wav_path.name)
        if not match:
            continue
        _mic, token, word = match.groups()
        by_token.setdefault(token, {}).setdefault(word, []).append(wav_path)
    return by_token


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_RAWDysPeech.wav"


def write_segment(label: str, phash: str, idx: int, path: Path, span: Segment) -> None:
    audio = AudioData.from_file(path, span)
    out_path = OUT_ROOT / out_name(label, phash, idx)
    sf.write(str(out_path), audio.array, audio.sr)


def process_standard(label: str, label_dir: Path, fallback_counter: list[int]) -> int:
    total = 0
    by_individual = collect_standard(label_dir)
    for pid, files in by_individual.items():
        ranked = []
        for path, word in files:
            span, fell_back = vad_trim(path)
            if fell_back:
                fallback_counter[0] += 1
            ranked.append((path, span, span.end - span.start))
        ranked = [r for r in ranked if MIN_DURATION <= r[2] < MAX_DURATION]
        ranked.sort(key=lambda x: x[2], reverse=True)

        phash = patient_hash(label, "standard", pid)
        for idx, (path, span, _dur) in enumerate(ranked[:TOP_N], 1):
            write_segment(label, phash, idx, path, span)
            total += 1
    return total


def process_wavmic(label: str, label_dir: Path, fallback_counter: list[int]) -> int:
    total = 0
    by_token = collect_wavmic(label_dir)
    for token, words in by_token.items():
        candidates = []
        for word, paths in words.items():
            best = None
            for path in paths:
                span, fell_back = vad_trim(path)
                if fell_back:
                    fallback_counter[0] += 1
                dur = span.end - span.start
                if best is None or dur > best[2]:
                    best = (path, span, dur)
            candidates.append(best)

        candidates = [c for c in candidates if MIN_DURATION <= c[2] < MAX_DURATION]
        candidates.sort(key=lambda x: x[2], reverse=True)

        phash = patient_hash(label, "wavmic", token)
        for idx, (path, span, _dur) in enumerate(candidates[:TOP_N], 1):
            write_segment(label, phash, idx, path, span)
            total += 1
    return total


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    fallback_counter = [0]
    grand_total = 0
    for folder, label in GROUP_LABELS.items():
        label_dir = RAW_ROOT / folder
        n1 = process_standard(label, label_dir, fallback_counter)
        print(f"{label} standard pattern: {n1} segments")
        n2 = process_wavmic(label, label_dir, fallback_counter)
        print(f"{label} wavmic pattern: {n2} segments")
        grand_total += n1 + n2

    print(f"done: {grand_total} segments written to {OUT_ROOT}")
    print(f"VAD fallback (no speech detected, kept untrimmed) count: {fallback_counter[0]}")


if __name__ == "__main__":
    main()

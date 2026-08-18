"""
Preprocess the NeuroVoz raw dataset into the derivatives/NeuroVoz folder.

Source layout: a single flat directory of .wav files named
    {HC|PD}_{TASK}_{patient_id}.wav
e.g. "HC_A1_0034.wav", "PD_PATATA_BLANDA_0077.wav" (a few tasks are
two-word, e.g. PAN_VINO / PATATA_BLANDA / PETACA_BLANCA -- the label is
always the first field and the patient id the last field, so the task is
everything in between).

Per-task routing (all files are natively 44100Hz; resampled to 16000Hz via
librosa's from_file below):

- Vowel-phonation tasks (A1/A2/A3, E1/E2/E3, I1/I2/I3, O1/O2/O3, U1/U2/U3):
  same rationale as IPVS -- the patient sustains a vowel and it naturally
  fades out, so these are always kept whole (never VAD-split), but dropped
  entirely if longer than MAX_PHONATION_DURATION (20s; rarely triggers here,
  since these recordings top out around 12s in practice).
- PATAKA (rapid diadochokinetic syllable repetition): mean ~12s, max ~27s,
  63% of recordings already exceed 10s -- long enough that it needs explicit
  handling, but it's a continuous repeated-articulation task without
  linguistic pauses (like the vowels), so it is kept whole and cropped to
  the first PATAKA_MAX_DURATION (20s) when longer, rather than VAD-split or
  dropped.
- FREE (spontaneous monologue): VAD-split same as everything else below.
- All other (word/phrase) tasks: in practice every single one is under 10s,
  so they are always kept whole via the general MIN_KEEP_DURATION rule. If
  one ever isn't, it falls through to the same VAD-split path as FREE.

For files >= MIN_KEEP_DURATION that aren't vowel/PATAKA: split into speech
segments with Silero VAD, chunk anything longer than MAX_SEGMENT_DURATION
(10s) into consecutive pieces, and drop any resulting piece shorter than
MIN_SEGMENT_DURATION (6s).

Output files are named:
    {HC|PD}_{patient_hash}_{segment_index}_NeuroVoz.wav
patient_hash is a short, stable hash of (label, patient_id), so the same
patient always gets the same hash across runs. Patient ids are per-label
(a couple of numeric ids, e.g. 0068/0069, are reused between HC and PD for
different individuals), so the hash is salted with the label to avoid any
collision.
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

RAW_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/raw/NeuroVoz/data/audios")
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/NeuroVoz")

SR = 16000
MIN_KEEP_DURATION = 10.0  # below this, keep the file whole
MIN_SEGMENT_DURATION = 6.0  # drop VAD segments (or chunks) shorter than this
MAX_SEGMENT_DURATION = 10.0  # VAD segments longer than this are chunked into pieces of this size
MAX_PHONATION_DURATION = 20.0  # drop vowel-phonation files longer than this entirely
PATAKA_MAX_DURATION = 20.0  # crop PATAKA files longer than this to this length
VAD_THRESHOLD = 0.5
# FREE (spontaneous speech) has natural thinking/breathing pauses well over 500ms.
# Measured per-group (of patients who actually have a FREE recording -- only
# 53/58 HC and 23/56 PD do):
#   1500ms -> HC 0/53 zero-output, 82.7% retained; PD 1/23 zero-output, 57.8% retained
#   2000ms -> HC 0/53 zero-output, 89.9% retained; PD 1/23 zero-output, 69.6% retained
#   2500ms -> HC 0/53 zero-output, 93.7% retained; PD 0/23 zero-output, 78.2% retained
# 2500ms is strictly best on every metric above, so that's what's used.
VAD_MIN_SILENCE_MS = 2500

VOWEL_TASK_RE = re.compile(r"^[AEIOU][123]$")

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


def parse_filename(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("_")
    label, pid = parts[0], parts[-1]
    task = "_".join(parts[1:-1])
    return label, task, pid


def patient_hash(label: str, patient_id: str) -> str:
    digest = hashlib.md5(f"{label}::{patient_id}".encode()).hexdigest()
    return digest[:8]


def collect_files() -> list[PatientFile]:
    files = []
    for wav_path in sorted(RAW_ROOT.glob("*.wav")):
        label, task, pid = parse_filename(wav_path)
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


def segments_for_file(path: Path, task: str, duration: float) -> list[Segment]:
    if duration < MIN_KEEP_DURATION:
        return [Segment(0.0, duration)]
    if VOWEL_TASK_RE.match(task):
        if duration > MAX_PHONATION_DURATION:
            return []
        return [Segment(0.0, duration)]
    if task == "PATAKA":
        return [Segment(0.0, min(duration, PATAKA_MAX_DURATION))]
    return speech_segments(path, duration)


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_NeuroVoz.wav"


def process_file(pf: PatientFile, next_index: dict[str, int]) -> int:
    duration = sf.info(str(pf.path)).duration
    segments = segments_for_file(pf.path, pf.task, duration)

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
        if i % 200 == 0 or i == len(all_files):
            print(f"[{i}/{len(all_files)}] processed, {total_segments} segments written so far")

    print(f"done: {total_segments} segments written to {OUT_ROOT}")


if __name__ == "__main__":
    main()

"""
Preprocess the "Italian Parkinson's Voice and Speech" (IPVS) raw dataset into
the derivatives/IPVS folder.

For every .wav found under the raw dataset:
  - if its duration is < 10s, it is copied over unchanged (just resampled to
    16kHz mono and renamed).
  - otherwise, it is split into speech segments with Silero VAD, and any
    resulting segment shorter than 6s is dropped.

Output files are named:
    {HC|PD}_{patient_hash}_{segment_index}_IPVS.wav

- HC is used for the two healthy-control folders, PD for the Parkinson's folder.
- patient_hash is a short, stable hash of the patient's identity (group +
  patient folder name), so the same patient always gets the same hash across
  runs.
- Patients appearing in more than one "x-y" range folder under the PD group
  (e.g. "Vito S" is under both "6-10" and "11-16") are merged into a single
  patient identity, with segment numbering continuing across their files.

Sample rate note: not all raw files are natively 16kHz (checked with
`soundfile.info`, no decoding needed) -- 655/815 are 16kHz but 160 files
(all inside the PD group) are natively 44100Hz. librosa.load(..., sr=16000)
resamples explicitly, so this is handled transparently as long as sr=16000
is always passed, which it is below.

VAD note: the dataset's filenames encode a task type as a prefix (B, D, FB,
PR, VA, VE, VI, VO, VU). VA/VE/VI/VO/VU are true sustained-vowel/phonation
tasks (duration ~4-35s, mean ~10-11s): the patient holds a vowel and their
voice naturally fades out as breath support runs out, often decaying to very
low energy well before the recording ends (e.g. peak RMS ~0.1 down to
~0.002-0.005 for the last ~9s of a 12s file, observed on a real sample).
Silero VAD -- tuned for normal conversational loudness -- does not reliably
detect that quiet tail as speech, and running VAD on these truncates or
entirely drops exactly the fade-out that is often the clinically relevant
signal (voice breaks, reduced phonation time). So these are always kept
whole rather than VAD-split -- except files longer than
MAX_PHONATION_DURATION (20s) are dropped entirely, since a single sustained
vowel running that long is an outlier for this task (only ~9% of vowel
recordings exceed 20s; PD patients contribute proportionally more of these
than HC).

B was initially bucketed with the vowel tasks too, but its duration profile
(min 38.3s, mean 68.6s, max 250.3s) doesn't fit a single sustained vowel --
it's almost certainly a monologue/spontaneous-speech task, so it is routed
through the same VAD-splitting path as D/FB/PR instead (structural pauses
between words/sentences, e.g. read-passage or spontaneous-speech
recordings).
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

RAW_ROOT = Path(
    "/Users/robinlouiset/Documents/ParkSpeechData/raw/"
    "Italian Parkinson's Voice and speech"
)
OUT_ROOT = Path("/Users/robinlouiset/Documents/ParkSpeechData/derivatives/IPVS")

HC_GROUP_DIRS = [
    RAW_ROOT / "15 Young Healthy Control",
    RAW_ROOT / "22 Elderly Healthy Control",
]
PD_GROUP_DIR = RAW_ROOT / "28 People with Parkinson's disease"

SR = 16000
MIN_KEEP_DURATION = 10.0  # below this, keep the file whole
MIN_SEGMENT_DURATION = 6.0  # drop VAD segments (or chunks) shorter than this
MAX_SEGMENT_DURATION = 10.0  # VAD segments longer than this are chunked into pieces of this size
MAX_PHONATION_DURATION = 20.0  # drop vowel-phonation files longer than this entirely
VAD_THRESHOLD = 0.5
# B/FB/PR have natural pauses well over 500ms. Measured on the VAD-eligible
# B/FB/PR files (232 total):
#   500ms  -> e.g. PD FB 10.7% retained, 16/26 zero-output
#   1500ms -> PD FB 78.4% retained, 1/26 zero-output
#   2500ms -> PD FB 89.5% retained, 0/26 zero-output; diminishing returns past this
# 2500ms matches the value used for NeuroVoz's FREE task.
VAD_MIN_SILENCE_MS = 2500

# Task-prefixes that are true sustained-vowel/phonation recordings: always
# kept whole (never VAD-split), but dropped entirely if longer than
# MAX_PHONATION_DURATION (see VAD note above). B, D, FB, PR are eligible for
# VAD splitting once they clear MIN_KEEP_DURATION.
PHONATION_PREFIXES = {"VA", "VE", "VI", "VO", "VU"}
TASK_PREFIX_RE = re.compile(r"^([A-Za-z]+)\d")

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
    patient_hash: str


def patient_hash(group_key: str, patient_name: str) -> str:
    normalized = patient_name.strip().upper()
    digest = hashlib.md5(f"{group_key}::{normalized}".encode()).hexdigest()
    return digest[:8]


def iter_patient_dirs(group_dir: Path):
    for entry in sorted(group_dir.iterdir()):
        if entry.is_dir():
            yield entry


def collect_hc_files() -> list[PatientFile]:
    files = []
    for group_dir in HC_GROUP_DIRS:
        group_key = group_dir.name
        for patient_dir in iter_patient_dirs(group_dir):
            phash = patient_hash(group_key, patient_dir.name)
            for wav_path in sorted(patient_dir.glob("*.wav")):
                files.append(PatientFile(wav_path, "HC", phash))
    return files


def collect_pd_files() -> list[PatientFile]:
    group_key = PD_GROUP_DIR.name
    files = []
    for range_dir in iter_patient_dirs(PD_GROUP_DIR):
        for patient_dir in iter_patient_dirs(range_dir):
            phash = patient_hash(group_key, patient_dir.name)
            for wav_path in sorted(patient_dir.glob("*.wav")):
                files.append(PatientFile(wav_path, "PD", phash))
    return files


def task_prefix(path: Path) -> str:
    match = TASK_PREFIX_RE.match(path.name)
    return match.group(1) if match else ""


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
    if task_prefix(path) in PHONATION_PREFIXES:
        if duration > MAX_PHONATION_DURATION:
            return []
        return [Segment(0.0, duration)]
    return speech_segments(path, duration)


def out_name(label: str, phash: str, segment_index: int) -> str:
    return f"{label}_{phash}_{segment_index}_IPVS.wav"


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

    all_files = collect_hc_files() + collect_pd_files()
    print(f"found {len(all_files)} source .wav files")

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
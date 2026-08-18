"""
Waveform augmentations for the SSL contrastive objective.

Deliberately excludes anything that would perturb voice *production*
characteristics -- pitch-shift, time-stretch, formant warping -- since those
are exactly the acoustic dimensions (F0, jitter/shimmer, speech rate/rhythm)
that carry PD-relevant signal. Instead these augmentations perturb *recording
condition* (noise, gain, reverb, bandwidth), which both leaves biomarkers
intact and directly helps the model generalize across the very heterogeneous
recording setups in this project's combined dataset (8kHz phone-quality
FredPrior clips next to 44.1kHz studio recordings, near vs. far-mic KCL,
etc.) instead of learning to fingerprint which source dataset a clip is from.
"""

from __future__ import annotations

import random

import torch

from pdspeech_ssl.config import AugmentHParams


def _rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x**2).clamp_min(1e-12))


def add_noise(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_rms = _rms(x)
    noise = torch.randn_like(x)
    noise_rms = _rms(noise)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)
    return x + noise


def apply_gain(x: torch.Tensor, gain_db: float) -> torch.Tensor:
    return x * (10 ** (gain_db / 20))


def random_crop(x: torch.Tensor, ratio: float) -> torch.Tensor:
    n = x.shape[-1]
    keep = max(1, int(n * ratio))
    if keep >= n:
        return x
    start = random.randint(0, n - keep)
    return x[..., start : start + keep]


def synthetic_reverb(x: torch.Tensor, decay: float, sr: int = 16000) -> torch.Tensor:
    """Convolve with a short synthetic exponential-decay impulse response,
    a lightweight stand-in for a real RIR corpus (kept dependency-free)."""
    ir_len = int(sr * 0.15)
    t = torch.arange(ir_len, dtype=x.dtype, device=x.device)
    ir = torch.exp(-t / (decay * sr)) * (torch.rand(ir_len, device=x.device) * 2 - 1)
    ir = ir / (ir.abs().max().clamp_min(1e-8))
    ir[0] = 1.0  # keep the direct path dominant
    wet = torch.nn.functional.conv1d(
        x.view(1, 1, -1), ir.view(1, 1, -1), padding=ir_len - 1
    ).view(-1)[: x.shape[-1]]
    peak = wet.abs().max().clamp_min(1e-8)
    return wet * (x.abs().max().clamp_min(1e-8) / peak)


def bandlimit(x: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Simulate lower-bandwidth recording equipment (e.g. FredPrior's native
    8kHz) by downsampling and upsampling back."""
    import torchaudio.functional as AF

    down = AF.resample(x, orig_sr, target_sr)
    return AF.resample(down, target_sr, orig_sr)


def augment_waveform(x: torch.Tensor, cfg: AugmentHParams, sr: int = 16000) -> torch.Tensor:
    if random.random() < cfg.crop_prob:
        ratio = random.uniform(*cfg.crop_ratio_range)
        x = random_crop(x, ratio)
    if random.random() < cfg.reverb_prob:
        decay = random.uniform(*cfg.reverb_decay_range)
        x = synthetic_reverb(x, decay, sr)
    if random.random() < cfg.bandlimit_prob:
        target_sr = random.choice(cfg.bandlimit_sr_choices)
        x = bandlimit(x, sr, target_sr)
    if random.random() < cfg.gain_prob:
        gain_db = random.uniform(*cfg.gain_db_range)
        x = apply_gain(x, gain_db)
    if random.random() < cfg.noise_prob:
        snr_db = random.uniform(*cfg.noise_snr_db_range)
        x = add_noise(x, snr_db)
    return x


def feature_time_mask(x: torch.Tensor, lengths: torch.Tensor, mask_fraction: float) -> torch.Tensor:
    """Randomly zero out a contiguous fraction of timesteps per-sample, applied
    to the wav2vec2 output sequence (B, T, F). Mirrors wav2vec2's own SSL
    pretraining masking recipe; applied independently per contrastive view."""
    x = x.clone()
    B, T, _ = x.shape
    for i in range(B):
        length = int(lengths[i].item())
        mask_len = max(1, int(length * mask_fraction))
        if length <= mask_len:
            continue
        start = random.randint(0, length - mask_len)
        x[i, start : start + mask_len, :] = 0.0
    return x

from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard SimCLR NT-Xent over two views. z1[i] and z2[i] must be the two
    views of the same individual (same position i in both tensors is the
    positive pair); every other position (including the same individual's
    slot from a *different* batch entry -- which shouldn't exist, since the
    sampler guarantees each individual appears at most once per global step)
    is treated as a negative.

    z1, z2: (N, D), not necessarily pre-normalized.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    N = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)  # (2N, D)
    sim = z @ z.T / temperature  # (2N, 2N)

    self_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    positive_idx = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, positive_idx)


def alignment_loss(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Wang & Isola (2020) alignment: expected distance between positive pairs.
    z1, z2 must already be unit-norm (on the hypersphere) -- this and
    uniformity_loss below are only meaningful for normalized inputs."""
    return (z1 - z2).norm(p=2, dim=-1).pow(alpha).mean()


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola (2020) uniformity: log E[exp(-t ||x-y||^2)] over all pairs
    in z, minimized to push embeddings toward a uniform spread on the
    hypersphere (the anti-collapse counterpart to alignment_loss). z must
    already be unit-norm."""
    sq_dist = torch.pdist(z, p=2).pow(2)
    return sq_dist.mul(-t).exp().mean().log()

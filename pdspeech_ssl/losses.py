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


def within_class_variance_loss(
    embd: torch.Tensor, groups: torch.Tensor, gamma: float, eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg-style variance (+ covariance) terms, computed within each class instead of
    over the whole batch, to keep a classifier loss from collapsing each class to a point.

    Scale-invariant on purpose: every dimension is first divided by its std over the whole
    batch (both classes), so the hinge is on the *fraction* of each dimension's spread that
    is within-class. With VICReg's raw std, BCE could meet any threshold by just inflating
    ||embd|| while the classes still collapse relative to their separation.
      var = mean_{class, dim} relu(gamma - std_within / std_total)
      cov = mean_class sum_{i != j} corr_within[i, j]^2 / dim   (within-class decorrelation)
    Classes with fewer than 2 samples are skipped. Also returns the mean std ratio (no grad)
    for logging. embd: (N, D), groups: (N,) integer class ids.
    """
    embd = embd.float()
    total_std = torch.sqrt(embd.var(0) + eps)
    var_terms, cov_terms, ratios = [], [], []
    for g in groups.unique():
        z = embd[groups == g]
        if len(z) < 2:
            continue
        within_std = torch.sqrt(z.var(0) + eps)
        ratio = within_std / total_std
        var_terms.append(F.relu(gamma - ratio).mean())
        zn = (z - z.mean(0)) / within_std
        corr = zn.T @ zn / (len(z) - 1)
        off_diag = corr - torch.diag(torch.diag(corr))
        cov_terms.append(off_diag.pow(2).sum() / z.shape[1])
        ratios.append(ratio.detach().mean())
    if not var_terms:
        zero = embd.sum() * 0.0
        return zero, zero, zero.detach()
    return torch.stack(var_terms).mean(), torch.stack(cov_terms).mean(), torch.stack(ratios).mean()

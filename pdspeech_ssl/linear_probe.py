from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def train_and_eval_linear_probe(
    train_embd: torch.Tensor,
    train_labels: torch.Tensor,
    val_embd: torch.Tensor,
    val_labels: torch.Tensor,
    lr: float,
    epochs: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Trains a single nn.Linear probe on frozen (already-detached) embeddings
    to discriminate HC (0) vs PD (1), returns (balanced_accuracy, auc, val_probs)
    on val, where val_probs is the per-segment P(PD) (aligned with val_embd's rows).

    This is called from inside Lightning's on_validation_epoch_end, which
    Lightning runs under torch.inference_mode() -- tensors created there are
    permanently non-differentiable, and inference_mode() can't be undone by
    torch.enable_grad() alone. So this whole routine explicitly exits
    inference_mode and clones the (detached) input embeddings into ordinary
    tensors before training the probe.
    """
    with torch.inference_mode(False), torch.enable_grad():
        train_embd = train_embd.clone().to(device)
        val_embd = val_embd.clone().to(device)
        train_labels = train_labels.to(device).float()

        in_dim = train_embd.shape[1]
        probe = nn.Linear(in_dim, 1).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

        probe.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = probe(train_embd).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_embd).squeeze(-1)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
    val_preds = (val_probs >= 0.5).astype(int)
    val_labels_np = val_labels.cpu().numpy()

    balanced_acc = balanced_accuracy_score(val_labels_np, val_preds)
    try:
        auc = roc_auc_score(val_labels_np, val_probs)
    except ValueError:
        # only one class present in val -- can happen with small probe_max_val_samples
        auc = float("nan")
    return balanced_acc, auc, val_probs


def individual_level_metrics(
    keys: list,
    labels: torch.Tensor,
    probs: np.ndarray,
) -> tuple[float, float]:
    """Aggregates per-segment P(PD) into one score per individual (mean over that
    individual's segments), then returns individual-level (balanced_accuracy at a
    0.5 threshold, auc). An individual's label is the same for all its segments."""
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    by_individual: dict = {}
    for key, label, prob in zip(keys, labels_np, probs):
        by_individual.setdefault(key, (label, []))[1].append(prob)

    ind_labels = np.array([label for label, _ in by_individual.values()])
    ind_probs = np.array([np.mean(segment_probs) for _, segment_probs in by_individual.values()])

    balanced_acc = balanced_accuracy_score(ind_labels, (ind_probs >= 0.5).astype(int))
    try:
        auc = roc_auc_score(ind_labels, ind_probs)
    except ValueError:
        # only one class present among val individuals
        auc = float("nan")
    return balanced_acc, auc


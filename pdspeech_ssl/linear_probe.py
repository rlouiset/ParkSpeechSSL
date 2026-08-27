from __future__ import annotations

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
) -> tuple[float, float]:
    """Trains a single nn.Linear probe on frozen (already-detached) embeddings
    to discriminate HC (0) vs PD (1), returns (balanced_accuracy, auc) on val.

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
    return balanced_acc, auc


def auc_of_scores(labels: torch.Tensor, scores) -> float:
    """ROC-AUC of a single scalar score column against binary labels (HC=0,
    disease=1) -- e.g. the norm of a disease-specific embedding subspace.
    No probe fitting needed since it's already a 1D signal."""
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    try:
        return roc_auc_score(labels_np, scores)
    except ValueError:
        # only one class present -- can happen with small probe_max_val_samples
        return float("nan")

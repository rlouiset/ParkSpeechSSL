from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from pdspeech_ssl.config import HParams
from pdspeech_ssl.data import PairBatch, SegmentBatch
from pdspeech_ssl.linear_probe import auc_of_scores, train_and_eval_linear_probe
from pdspeech_ssl.losses import alignment_loss, nt_xent_loss, uniformity_loss
from pdspeech_ssl.model import SSLEncoder

LABEL_TO_BINARY = {"HC": 0, "PD": 1}
OBJECTIVES = ("simclr", "hc_vs_rest_bce", "disease_uniformity")


def _hc_vs_rest_targets(labels: list, device: torch.device) -> torch.Tensor:
    """0.0 for HC, 1.0 for everything else (PD/MSA/PSP/DYS) -- a reachability
    sanity check for whether the encoder can learn anything at all, kept
    available alongside the primary SimCLR objective (see TrainingHParams.objective)."""
    return torch.tensor([0.0 if label == "HC" else 1.0 for label in labels], device=device)


def _hc_mask(labels: list, device: torch.device) -> torch.Tensor:
    return torch.tensor([label == "HC" for label in labels], device=device)


def _zero_disease_for_hc(z: torch.Tensor, hc_mask: torch.Tensor, d_disease: int) -> torch.Tensor:
    """Zero the Z_D block (first d_disease dims) for HC rows, then renormalize the
    whole vector back onto the unit hypersphere -- HC individuals end up living
    exactly on the Z_C-only equatorial subsphere; non-HC rows pass through
    unchanged (already unit-norm, so renormalizing them again is a no-op). The
    zeroed entries carry no gradient for HC rows, which is what forces the
    *other* view's Z_D toward 0 via the alignment loss."""
    z = z.clone()
    z[hc_mask, :d_disease] = 0.0
    return F.normalize(z, dim=-1)


class SSLLightningModule(pl.LightningModule):
    def __init__(self, cfg: HParams):
        super().__init__()
        if cfg.training.objective not in OBJECTIVES:
            raise ValueError(f"Unknown training.objective: {cfg.training.objective!r}, expected one of {OBJECTIVES}")
        self.cfg = cfg
        self.model = SSLEncoder(cfg.encoder, cfg.model)
        # cls_head only exists for the hc_vs_rest_bce objective -- keeping it out of the
        # graph entirely under simclr (rather than just unused) avoids padding DDP's
        # unused-parameter bookkeeping and the checkpoint with dead weights every step.
        self.cls_head = nn.Linear(cfg.model.d_emb, 1) if cfg.training.objective == "hc_vs_rest_bce" else None
        self.bce = nn.BCEWithLogitsLoss() if cfg.training.objective == "hc_vs_rest_bce" else None

    def _embed_pair(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
        out1 = self.model(batch["view1"], batch["len1"], augment_cfg=self.cfg.augment)
        out2 = self.model(batch["view2"], batch["len2"], augment_cfg=self.cfg.augment)
        return out1, out2

    def _contrastive_loss(self, batch: PairBatch) -> torch.Tensor:
        """SimCLR NT-Xent: aligns view1[i]/view2[i] -- the two augmented views of the
        *same individual* from IndividualPairDataset -- as the positive pair, against
        every other individual in the batch (gathered across GPUs) as negatives."""
        out1, out2 = self._embed_pair(batch)
        z1, z2 = out1["proj"], out2["proj"]
        if self.cfg.loss.gather_across_gpus and self.trainer.world_size > 1:
            # sync_grads=True: gradients flow back through the gather to every
            # rank's own local forward pass, so this is a true global negative pool,
            # not a detached copy of the other ranks' embeddings.
            z1 = self.all_gather(z1, sync_grads=True).flatten(0, 1)
            z2 = self.all_gather(z2, sync_grads=True).flatten(0, 1)
        return nt_xent_loss(z1, z2, self.cfg.loss.temperature)

    def _classification_loss(self, batch: PairBatch) -> torch.Tensor:
        out1, out2 = self._embed_pair(batch)
        targets = _hc_vs_rest_targets(batch["labels"], self.device)
        logit1 = self.cls_head(out1["embd"]).squeeze(-1)
        logit2 = self.cls_head(out2["embd"]).squeeze(-1)
        return 0.5 * (self.bce(logit1, targets) + self.bce(logit2, targets))

    def _project_disease_space(self, proj: torch.Tensor) -> torch.Tensor:
        """Normalize onto the hypersphere, apply leaky-ReLU to the Z_D block only
        (keeps disease deviation mostly one-sided -- see DiseaseHParams.leaky_slope),
        then renormalize since the nonlinearity moves the vector off the sphere."""
        z = F.normalize(proj, dim=-1)
        d = self.cfg.disease.d_disease
        z_d = F.leaky_relu(z[:, :d], negative_slope=self.cfg.disease.leaky_slope)
        z = torch.cat([z_d, z[:, d:]], dim=-1)
        return F.normalize(z, dim=-1)

    def _disease_uniformity_loss(self, batch: PairBatch, stage: str) -> torch.Tensor:
        """Alignment+uniformity (Wang & Isola) on the Z_D/Z_C-split proj_head output.
        HC individuals' Z_D is pinned to 0 (see _zero_disease_for_hc); non-HC
        individuals' Z_D is left free, so its norm should emerge as an unsupervised
        disease-severity signal under the anti-collapse pressure of uniformity_loss."""
        out1, out2 = self._embed_pair(batch)
        hc_mask = _hc_mask(batch["labels"], self.device)

        z1 = self._project_disease_space(out1["proj"])
        z2 = self._project_disease_space(out2["proj"])

        if self.cfg.loss.gather_across_gpus and self.trainer.world_size > 1:
            z1 = self.all_gather(z1, sync_grads=True).flatten(0, 1)
            z2 = self.all_gather(z2, sync_grads=True).flatten(0, 1)
            hc_mask = self.all_gather(hc_mask.long(), sync_grads=False).flatten(0, 1).bool()

        d = self.cfg.disease.d_disease
        z1_hc0 = _zero_disease_for_hc(z1, hc_mask, d)
        z2_hc0 = _zero_disease_for_hc(z2, hc_mask, d)

        align = 0.5 * (alignment_loss(z1, z2_hc0) + alignment_loss(z2, z1_hc0))
        uniform = 0.5 * (
            uniformity_loss(z1_hc0, self.cfg.disease.uniformity_t)
            + uniformity_loss(z2_hc0, self.cfg.disease.uniformity_t)
        )
        self.log(f"{stage}/align_loss", align, on_epoch=True, sync_dist=True, batch_size=z1.shape[0])
        self.log(f"{stage}/uniformity_loss", uniform, on_epoch=True, sync_dist=True, batch_size=z1.shape[0])

        # Diagnostic only: is Z_D norm actually separating HC from non-HC? No sync_dist --
        # unconditional self.log calls are still made every step either way, but the mean
        # itself is intentionally left per-rank (a debug signal, not something ModelCheckpoint
        # depends on) so this never needs a cross-rank collective to agree on class presence.
        with torch.no_grad():
            disease_norm = z1_hc0[:, :d].norm(dim=-1)
            hc_norm = disease_norm[hc_mask].mean() if hc_mask.any() else disease_norm.new_zeros(())
            non_hc_norm = disease_norm[~hc_mask].mean() if (~hc_mask).any() else disease_norm.new_zeros(())
        self.log(f"{stage}/z_disease_norm_hc", hc_norm, on_epoch=True, batch_size=z1.shape[0])
        self.log(f"{stage}/z_disease_norm_non_hc", non_hc_norm, on_epoch=True, batch_size=z1.shape[0])

        return self.cfg.disease.align_weight * align + self.cfg.disease.uniform_weight * uniform

    def _step_loss(self, batch: PairBatch, stage: str) -> torch.Tensor:
        if self.cfg.training.objective == "simclr":
            return self._contrastive_loss(batch)
        elif self.cfg.training.objective == "hc_vs_rest_bce":
            return self._classification_loss(batch)
        elif self.cfg.training.objective == "disease_uniformity":
            return self._disease_uniformity_loss(batch, stage)
        raise AssertionError(f"unreachable: {self.cfg.training.objective!r} not in {OBJECTIVES}")

    @property
    def _loss_metric_name(self) -> str:
        if self.cfg.training.objective == "simclr":
            return "contrastive_loss"
        elif self.cfg.training.objective == "hc_vs_rest_bce":
            return "hc_vs_rest_bce"
        return "align_uniform_loss"

    def training_step(self, batch: PairBatch, batch_idx: int) -> torch.Tensor:
        loss = self._step_loss(batch, stage="Train")
        self.log(
            f"Train/{self._loss_metric_name}", loss, on_step=True, on_epoch=True,
            sync_dist=True, batch_size=batch["view1"].shape[0], prog_bar=True,
        )
        return loss

    def validation_step(self, batch: PairBatch, batch_idx: int) -> None:
        loss = self._step_loss(batch, stage="Val")
        self.log(
            f"Val/{self._loss_metric_name}", loss, on_step=False, on_epoch=True,
            sync_dist=True, batch_size=batch["view1"].shape[0],
        )

    @torch.no_grad()
    def _embed_segments(self, dataloader) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embd, proj, labels) for every segment in dataloader."""
        self.model.eval()
        all_embd, all_proj, all_labels = [], [], []
        for batch in dataloader:
            batch: SegmentBatch
            wav = batch["wav"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            out = self.model(wav, lengths, augment_cfg=None)
            all_embd.append(out["embd"].detach().cpu())
            all_proj.append(out["proj"].detach().cpu())
            all_labels.extend(LABEL_TO_BINARY[label] for label in batch["labels"])
        return torch.cat(all_embd, dim=0), torch.cat(all_proj, dim=0), torch.tensor(all_labels, dtype=torch.long)

    def on_validation_epoch_end(self) -> None:
        # -1.0 sentinel (outside balanced-accuracy/AUC's [0, 1] range) for epochs where
        # the probe doesn't run or can't be evaluated (missing a class) -- ModelCheckpoint's
        # mode="max" monitor will simply never pick these as the best epoch.
        balanced_acc, auc, z_disease_auc = -1.0, -1.0, -1.0
        should_probe = (self.current_epoch + 1) % self.cfg.training.probe_every_n_epochs == 0

        if should_probe and self.trainer.is_global_zero:
            datamodule = self.trainer.datamodule
            train_embd, train_proj, train_labels = self._embed_segments(datamodule.probe_train_dataloader())
            val_embd, val_proj, val_labels = self._embed_segments(datamodule.probe_val_dataloader())
            self.model.train()

            if train_labels.unique().numel() >= 2 and val_labels.unique().numel() >= 2:
                balanced_acc, auc = train_and_eval_linear_probe(
                    train_embd, train_labels, val_embd, val_labels,
                    lr=self.cfg.training.probe_lr,
                    epochs=self.cfg.training.probe_epochs,
                    weight_decay=self.cfg.training.probe_weight_decay,
                    device=self.device,
                )

            # Direct AUC of ||Z_D|| against the HC/PD label, on the same held-out val
            # segments the probe above uses -- no probe fitting needed, it's already a
            # 1D signal. Only meaningful under disease_uniformity (Z_D isn't a trained
            # subspace under the other objectives).
            if self.cfg.training.objective == "disease_uniformity" and val_labels.unique().numel() >= 2:
                z_val = self._project_disease_space(val_proj.to(self.device))
                disease_norm = z_val[:, : self.cfg.disease.d_disease].norm(dim=-1).cpu().numpy()
                z_disease_auc = auc_of_scores(val_labels, disease_norm)

        # The probe only runs on rank 0 (it's expensive); broadcast the result so every
        # rank logs the same value -- otherwise ModelCheckpoint's monitor check, which runs
        # independently per rank, crashes on ranks that never called self.log for this key.
        if self.trainer.world_size > 1:
            balanced_acc, auc, z_disease_auc = self.trainer.strategy.broadcast(
                (balanced_acc, auc, z_disease_auc), src=0
            )

        self.log("Val/hc_pd_balanced_accuracy", balanced_acc, rank_zero_only=True, prog_bar=True)
        self.log("Val/hc_pd_auc", auc, rank_zero_only=True, prog_bar=True)
        self.log("Val/z_disease_norm_auc", z_disease_auc, rank_zero_only=True, prog_bar=True)
        # slash-free alias so ModelCheckpoint's filename template can reference it
        # (see NOTE in main_ssl.py -- "/" in a template key is read as a subdirectory)
        self.log("bal_acc", balanced_acc, rank_zero_only=True, prog_bar=False)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # wav2vec2's frozen base weights (~1.2GB) never change from the pretrained
        # checkpoint and don't need saving every time -- only the LoRA adapters and
        # the small custom heads are actually trainable. Cuts checkpoint size ~100x.
        # NOTE: reloading one of these later needs load_state_dict(..., strict=False),
        # since the frozen backbone is intentionally absent from the saved state_dict.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in trainable}

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cls_head is not None:
            params += list(self.cls_head.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay
        )
        return {
            "optimizer": optimizer
        }

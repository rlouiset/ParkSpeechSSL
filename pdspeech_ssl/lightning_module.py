from __future__ import annotations

import torch
import torch.nn as nn
import lightning.pytorch as pl

from pdspeech_ssl.config import HParams
from pdspeech_ssl.data import PairBatch, SegmentBatch
from pdspeech_ssl.linear_probe import train_and_eval_linear_probe
from pdspeech_ssl.losses import nt_xent_loss
from pdspeech_ssl.model import SSLEncoder

LABEL_TO_BINARY = {"HC": 0, "PD": 1}
OBJECTIVES = ("simclr", "hc_vs_rest_bce")


def _hc_vs_rest_targets(labels: list, device: torch.device) -> torch.Tensor:
    """0.0 for HC, 1.0 for everything else (PD/MSA/PSP/DYS) -- a reachability
    sanity check for whether the encoder can learn anything at all, kept
    available alongside the primary SimCLR objective (see TrainingHParams.objective)."""
    return torch.tensor([0.0 if label == "HC" else 1.0 for label in labels], device=device)


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

    def _step_loss(self, batch: PairBatch) -> torch.Tensor:
        if self.cfg.training.objective == "simclr":
            return self._contrastive_loss(batch)
        return self._classification_loss(batch)

    @property
    def _loss_metric_name(self) -> str:
        return "contrastive_loss" if self.cfg.training.objective == "simclr" else "hc_vs_rest_bce"

    def training_step(self, batch: PairBatch, batch_idx: int) -> torch.Tensor:
        loss = self._step_loss(batch)
        self.log(
            f"Train/{self._loss_metric_name}", loss, on_step=True, on_epoch=True,
            sync_dist=True, batch_size=batch["view1"].shape[0], prog_bar=True,
        )
        return loss

    def validation_step(self, batch: PairBatch, batch_idx: int) -> None:
        loss = self._step_loss(batch)
        self.log(
            f"Val/{self._loss_metric_name}", loss, on_step=False, on_epoch=True,
            sync_dist=True, batch_size=batch["view1"].shape[0],
        )

    @torch.no_grad()
    def _embed_segments(self, dataloader) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        all_embd, all_labels = [], []
        for batch in dataloader:
            batch: SegmentBatch
            wav = batch["wav"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            embd = self.model(wav, lengths, augment_cfg=None)["embd"].detach()
            all_embd.append(embd.cpu())
            all_labels.extend(LABEL_TO_BINARY[label] for label in batch["labels"])
        return torch.cat(all_embd, dim=0), torch.tensor(all_labels, dtype=torch.long)

    def on_validation_epoch_end(self) -> None:
        # -1.0 sentinel (outside balanced-accuracy/AUC's [0, 1] range) for epochs where
        # the probe doesn't run or can't be evaluated (missing a class) -- ModelCheckpoint's
        # mode="max" monitor will simply never pick these as the best epoch.
        balanced_acc, auc = -1.0, -1.0
        should_probe = (self.current_epoch + 1) % self.cfg.training.probe_every_n_epochs == 0

        if should_probe and self.trainer.is_global_zero:
            datamodule = self.trainer.datamodule
            train_embd, train_labels = self._embed_segments(datamodule.probe_train_dataloader())
            val_embd, val_labels = self._embed_segments(datamodule.probe_val_dataloader())
            self.model.train()

            if train_labels.unique().numel() >= 2 and val_labels.unique().numel() >= 2:
                balanced_acc, auc = train_and_eval_linear_probe(
                    train_embd, train_labels, val_embd, val_labels,
                    lr=self.cfg.training.probe_lr,
                    epochs=self.cfg.training.probe_epochs,
                    weight_decay=self.cfg.training.probe_weight_decay,
                    device=self.device,
                )

        # The probe only runs on rank 0 (it's expensive); broadcast the result so every
        # rank logs the same value -- otherwise ModelCheckpoint's monitor check, which runs
        # independently per rank, crashes on ranks that never called self.log for this key.
        if self.trainer.world_size > 1:
            balanced_acc, auc = self.trainer.strategy.broadcast((balanced_acc, auc), src=0)

        self.log("Val/hc_pd_balanced_accuracy", balanced_acc, rank_zero_only=True, prog_bar=True)
        self.log("Val/hc_pd_auc", auc, rank_zero_only=True, prog_bar=True)
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

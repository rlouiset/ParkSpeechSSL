from __future__ import annotations

import torch
import lightning.pytorch as pl

from pdspeech_ssl.config import HParams
from pdspeech_ssl.data import PairBatch, SegmentBatch
from pdspeech_ssl.linear_probe import train_and_eval_linear_probe
from pdspeech_ssl.losses import nt_xent_loss
from pdspeech_ssl.model import SSLEncoder

LABEL_TO_BINARY = {"HC": 0, "PD": 1}


class SSLLightningModule(pl.LightningModule):
    def __init__(self, cfg: HParams):
        super().__init__()
        self.cfg = cfg
        self.model = SSLEncoder(cfg.encoder, cfg.model)

    def _step_embeddings(self, batch: PairBatch) -> tuple[torch.Tensor, torch.Tensor]:
        out1 = self.model(batch["view1"], batch["len1"], augment_cfg=self.cfg.augment)
        out2 = self.model(batch["view2"], batch["len2"], augment_cfg=self.cfg.augment)
        return out1["proj"], out2["proj"]

    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        if self.cfg.loss.gather_across_gpus and self.trainer.world_size > 1:
            # sync_grads=True: gradients flow back through the gather to every
            # rank's own local forward pass, so this is a true global negative pool,
            # not a detached copy of the other ranks' embeddings.
            z1 = self.all_gather(z1, sync_grads=True).flatten(0, 1)
            z2 = self.all_gather(z2, sync_grads=True).flatten(0, 1)
        return nt_xent_loss(z1, z2, self.cfg.loss.temperature)

    def training_step(self, batch: PairBatch, batch_idx: int) -> torch.Tensor:
        z1, z2 = self._step_embeddings(batch)
        loss = self._contrastive_loss(z1, z2)
        self.log(
            "Train/contrastive_loss", loss, on_step=True, on_epoch=True,
            sync_dist=True, batch_size=batch["view1"].shape[0], prog_bar=True,
        )
        return loss

    def validation_step(self, batch: PairBatch, batch_idx: int) -> None:
        z1, z2 = self._step_embeddings(batch)
        loss = self._contrastive_loss(z1, z2)
        self.log(
            "Val/contrastive_loss", loss, on_step=False, on_epoch=True,
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
        if not self.trainer.is_global_zero:
            return
        if (self.current_epoch + 1) % self.cfg.training.probe_every_n_epochs != 0:
            return

        datamodule = self.trainer.datamodule
        train_embd, train_labels = self._embed_segments(datamodule.probe_train_dataloader())
        val_embd, val_labels = self._embed_segments(datamodule.probe_val_dataloader())
        self.model.train()

        if train_labels.unique().numel() < 2 or val_labels.unique().numel() < 2:
            return  # can't fit/evaluate a binary probe without both classes present

        balanced_acc, auc = train_and_eval_linear_probe(
            train_embd, train_labels, val_embd, val_labels,
            lr=self.cfg.training.probe_lr,
            epochs=self.cfg.training.probe_epochs,
            weight_decay=self.cfg.training.probe_weight_decay,
            device=self.device,
        )
        self.log("Val/hc_pd_balanced_accuracy", balanced_acc, rank_zero_only=True, prog_bar=True)
        self.log("Val/hc_pd_auc", auc, rank_zero_only=True, prog_bar=True)
        # slash-free alias so ModelCheckpoint's filename template can reference it
        # (see NOTE in main_ssl.py -- "/" in a template key is read as a subdirectory)
        self.log("bal_acc", balanced_acc, rank_zero_only=True, prog_bar=False)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.cfg.training.lr, weight_decay=self.cfg.training.weight_decay
        )

        warmup_steps = self.cfg.training.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

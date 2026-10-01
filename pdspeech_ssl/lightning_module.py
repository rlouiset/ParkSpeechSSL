from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import lightning.pytorch as pl

from pdspeech_ssl.config import HParams
from pdspeech_ssl.data import PairBatch, SegmentBatch
from pdspeech_ssl.linear_probe import individual_level_metrics, train_and_eval_linear_probe
from pdspeech_ssl.losses import nt_xent_loss
from pdspeech_ssl.model import SSLEncoder, SSLOutput

LABEL_TO_BINARY = {"HC": 0, "PD": 1}
OBJECTIVES = ("simclr", "hc_vs_rest_bce")
SIMCLR_VIEWS = ("masked", "unmasked")
W2V2_KEYS = ("w2v2_loss", "w2v2_contrastive", "w2v2_diversity", "codevector_perplexity")
_QUANTIZER_PREFIXES = ("model._w2v2_base.quantizer.", "model._w2v2_base.project_q.")
_BACKBONE_PREFIX = "model._w2v2_base."


def _hc_vs_rest_targets(labels: list, device: torch.device) -> torch.Tensor:
    """0.0 for HC, 1.0 for everything else (PD/MSA/PSP/DYS) -- a reachability
    sanity check for whether the encoder can learn anything at all, kept
    available alongside the primary SimCLR objective (see TrainingHParams.objective)."""
    return torch.tensor([0.0 if label == "HC" else 1.0 for label in labels], device=device)


def _warmup_cosine(step: int, warmup_steps: int, total_steps: int, offset: int = 0) -> float:
    """LR multiplier: 0 before `offset`, then linear warmup over warmup_steps, then
    cosine decay to 0 at total_steps (all counted in optimizer steps)."""
    step -= offset
    total_steps = max(1, total_steps - offset)
    if step < 0:
        return 0.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class SSLLightningModule(pl.LightningModule):
    def __init__(self, cfg: HParams):
        super().__init__()
        if cfg.training.objective not in OBJECTIVES:
            raise ValueError(f"Unknown training.objective: {cfg.training.objective!r}, expected one of {OBJECTIVES}")
        if cfg.w2v2.simclr_view not in SIMCLR_VIEWS:
            raise ValueError(f"Unknown w2v2.simclr_view: {cfg.w2v2.simclr_view!r}, expected one of {SIMCLR_VIEWS}")
        if cfg.encoder.trainable_mode == "frozen" and cfg.w2v2.weight > 0:
            raise ValueError(
                "encoder.trainable_mode='frozen' with w2v2.weight > 0: the whole wav2vec2 backbone is "
                "frozen, so the w2v2 loss has nothing to train. Set w2v2.weight=0 or use "
                "trainable_mode='full'."
            )
        self.cfg = cfg
        self.model = SSLEncoder(cfg.encoder, cfg.model, cfg.w2v2)
        # cls_head only exists for the hc_vs_rest_bce objective -- keeping it out of the
        # graph entirely under simclr (rather than just unused) avoids padding DDP's
        # unused-parameter bookkeeping and the checkpoint with dead weights every step.
        self.cls_head = nn.Linear(cfg.model.d_emb, 1) if cfg.training.objective == "hc_vs_rest_bce" else None
        self.bce = nn.BCEWithLogitsLoss() if cfg.training.objective == "hc_vs_rest_bce" else None

    @property
    def _use_w2v2(self) -> bool:
        return self.cfg.w2v2.weight > 0

    def _embed_view(self, wav: torch.Tensor, lengths: torch.Tensor) -> SSLOutput:
        if not self._use_w2v2:
            return self.model(wav, lengths, augment_cfg=self.cfg.augment)
        if self.cfg.w2v2.simclr_view == "masked":
            # one masked forward feeds both the objective head and the w2v2 loss
            return self.model(wav, lengths, augment_cfg=self.cfg.augment, w2v2_mask=True)
        # "unmasked" ablation: the head sees an unmasked forward, and a second masked
        # forward is run for the w2v2 loss only (2x backbone cost)
        out = self.model(wav, lengths, augment_cfg=self.cfg.augment)
        masked_out = self.model(wav, lengths, augment_cfg=None, w2v2_mask=True)
        out.update({k: masked_out[k] for k in W2V2_KEYS})
        return out

    def _contrastive_loss(self, out1: SSLOutput, out2: SSLOutput) -> torch.Tensor:
        """SimCLR NT-Xent: aligns view1[i]/view2[i] -- the two augmented views of the
        *same individual* from IndividualPairDataset -- as the positive pair, against
        every other individual in the batch (gathered across GPUs) as negatives."""
        z1, z2 = out1["proj"], out2["proj"]
        if self.cfg.loss.gather_across_gpus and self.trainer.world_size > 1:
            # sync_grads=True: gradients flow back through the gather to every
            # rank's own local forward pass, so this is a true global negative pool,
            # not a detached copy of the other ranks' embeddings.
            z1 = self.all_gather(z1, sync_grads=True).flatten(0, 1)
            z2 = self.all_gather(z2, sync_grads=True).flatten(0, 1)
        return nt_xent_loss(z1, z2, self.cfg.loss.temperature)

    def _classification_loss(self, out1: SSLOutput, out2: SSLOutput, batch: PairBatch) -> torch.Tensor:
        targets = _hc_vs_rest_targets(batch["labels"], self.device)
        logit1 = self.cls_head(out1["embd"]).squeeze(-1)
        logit2 = self.cls_head(out2["embd"]).squeeze(-1)
        return 0.5 * (self.bce(logit1, targets) + self.bce(logit2, targets))

    def _objective_loss(self, out1: SSLOutput, out2: SSLOutput, batch: PairBatch) -> torch.Tensor:
        if self.cfg.training.objective == "simclr":
            return self._contrastive_loss(out1, out2)
        elif self.cfg.training.objective == "hc_vs_rest_bce":
            return self._classification_loss(out1, out2, batch)
        raise AssertionError(f"unreachable: {self.cfg.training.objective!r} not in {OBJECTIVES}")

    @property
    def _loss_metric_name(self) -> str:
        if self.cfg.training.objective == "simclr":
            return "contrastive_loss"
        return "hc_vs_rest_bce"

    def _step(self, batch: PairBatch, stage: str) -> torch.Tensor:
        """total = loss.simclr_weight * objective_loss + w2v2.weight * mean_over_views(w2v2_loss).
        Validation runs the same masked procedure (explicit masks still apply in eval
        mode since apply_spec_augment=True), just without dropout."""
        out1 = self._embed_view(batch["view1"], batch["len1"])
        out2 = self._embed_view(batch["view2"], batch["len2"])

        objective_loss = self._objective_loss(out1, out2, batch)
        total = self.cfg.loss.simclr_weight * objective_loss

        is_train = stage == "Train"
        log_kwargs = dict(on_step=is_train, on_epoch=True, sync_dist=True, batch_size=batch["view1"].shape[0])
        self.log(f"{stage}/{self._loss_metric_name}", objective_loss, **log_kwargs)

        if self._use_w2v2:
            w2v2_terms = {k: 0.5 * (out1[k] + out2[k]) for k in W2V2_KEYS}
            total = total + self.cfg.w2v2.weight * w2v2_terms["w2v2_loss"]
            for k, v in w2v2_terms.items():
                self.log(f"{stage}/{k}", v, **log_kwargs)

        self.log(f"{stage}/total_loss", total, prog_bar=True, **log_kwargs)
        return total

    def training_step(self, batch: PairBatch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="Train")

    def validation_step(self, batch: PairBatch, batch_idx: int) -> None:
        self._step(batch, stage="Val")

    @torch.no_grad()
    def _embed_segments(self, dataloader) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Returns (embd, labels, individual keys) for every segment in dataloader."""
        self.model.eval()
        all_embd, all_labels, all_keys = [], [], []
        for batch in dataloader:
            batch: SegmentBatch
            wav = batch["wav"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            out = self.model(wav, lengths, augment_cfg=None)
            all_embd.append(out["embd"].detach().cpu())
            all_labels.extend(LABEL_TO_BINARY[label] for label in batch["labels"])
            all_keys.extend(batch["keys"])
        return torch.cat(all_embd, dim=0), torch.tensor(all_labels, dtype=torch.long), all_keys

    def on_validation_epoch_end(self) -> None:
        layer_weights = self.model.layer_weight_probs()
        if layer_weights is not None:
            # identical on every rank under DDP, so no sync needed
            for i, w in enumerate(layer_weights.tolist()):
                self.log(f"Val/layer_weight_{i:02d}", w)

        # -1.0 sentinel (outside balanced-accuracy/AUC's [0, 1] range) for epochs where
        # the probe doesn't run or can't be evaluated (missing a class) -- ModelCheckpoint's
        # mode="max" monitor will simply never pick these as the best epoch.
        balanced_acc, auc, balanced_acc_ind, auc_ind = -1.0, -1.0, -1.0, -1.0
        should_probe = (self.current_epoch + 1) % self.cfg.training.probe_every_n_epochs == 0

        if should_probe and self.trainer.is_global_zero:
            datamodule = self.trainer.datamodule
            was_training = self.model.training
            train_embd, train_labels, _ = self._embed_segments(datamodule.probe_train_dataloader())
            val_embd, val_labels, val_keys = self._embed_segments(datamodule.probe_val_dataloader())
            self.model.train(was_training)

            if train_labels.unique().numel() >= 2 and val_labels.unique().numel() >= 2:
                balanced_acc, auc, val_probs = train_and_eval_linear_probe(
                    train_embd, train_labels, val_embd, val_labels,
                    lr=self.cfg.training.probe_lr,
                    epochs=self.cfg.training.probe_epochs,
                    weight_decay=self.cfg.training.probe_weight_decay,
                    device=self.device,
                )
                # one score per individual: mean of its segments' P(PD)
                balanced_acc_ind, auc_ind = individual_level_metrics(val_keys, val_labels, val_probs)

        # The probe only runs on rank 0 (it's expensive); broadcast the result so every
        # rank logs the same value -- otherwise ModelCheckpoint's monitor check, which runs
        # independently per rank, crashes on ranks that never called self.log for this key.
        if self.trainer.world_size > 1:
            balanced_acc, auc, balanced_acc_ind, auc_ind = self.trainer.strategy.broadcast(
                (balanced_acc, auc, balanced_acc_ind, auc_ind), src=0
            )

        self.log("Val/hc_pd_balanced_accuracy", balanced_acc, rank_zero_only=True, prog_bar=True)
        self.log("Val/hc_pd_auc", auc, rank_zero_only=True, prog_bar=True)
        self.log("Val/hc_pd_balanced_accuracy_individual", balanced_acc_ind, rank_zero_only=True, prog_bar=True)
        self.log("Val/hc_pd_auc_individual", auc_ind, rank_zero_only=True)
        # slash-free aliases so ModelCheckpoint's filename template can reference them
        # (see NOTE in main_ssl.py -- "/" in a template key is read as a subdirectory)
        self.log("bal_acc", balanced_acc, rank_zero_only=True, prog_bar=False)
        self.log("bal_acc_ind", balanced_acc_ind, rank_zero_only=True, prog_bar=False)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Frozen weights never change from the pretrained checkpoint and don't need
        # saving every time -- only trainable params are kept. Under trainable_mode="full"
        # that drops the CNN feature encoder (+ quantizer/project_q unless
        # w2v2.train_quantizer); under "frozen", everything but the small custom heads.
        # NOTE: to reload one of these, first build the module (which loads the base
        # checkpoint from encoder.model_name_or_path), then
        # load_state_dict(checkpoint["state_dict"], strict=False) on top of it.
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in trainable}

    def _param_groups(self) -> list[dict]:
        """Backbone / head / quantizer groups, each split into decay / no-decay (biases,
        LayerNorm params and the layer-weight vector get no weight decay)."""
        optim_cfg = self.cfg.optim
        no_decay_ids = {id(p) for m in self.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}
        if self.model.layer_weights is not None:
            no_decay_ids.add(id(self.model.layer_weights))

        grouped: dict[tuple[str, bool], list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(_QUANTIZER_PREFIXES):
                role = "quantizer"
            elif name.startswith(_BACKBONE_PREFIX):
                role = "backbone"
            else:
                role = "head"  # layer_weights, projector, BiLSTM, pooling, embedding_projector, proj_head, cls_head
            decay = id(p) not in no_decay_ids and "bias" not in name.rsplit(".", 1)[-1]
            grouped.setdefault((role, decay), []).append(p)

        lrs = {"backbone": optim_cfg.lr_backbone, "head": optim_cfg.lr_head, "quantizer": optim_cfg.lr_quantizer}
        return [
            {
                "params": params,
                "lr": lrs[role],
                "weight_decay": optim_cfg.weight_decay if decay else 0.0,
                "name": role if decay else f"{role}_no_decay",  # shown by LearningRateMonitor
                "role": role,
            }
            for (role, decay), params in grouped.items()
        ]

    def configure_optimizers(self):
        optim_cfg = self.cfg.optim
        param_groups = self._param_groups()
        optimizer = torch.optim.AdamW(param_groups)

        total_steps = int(self.trainer.estimated_stepping_batches)
        # freeze_backbone_steps: the backbone groups' LR stays at 0 for the first N steps
        # (gradients are still computed, just not applied), then their own warmup+cosine
        # schedule starts from step N; the head/quantizer groups start at step 0.
        lr_lambdas = [
            functools.partial(
                _warmup_cosine,
                warmup_steps=optim_cfg.warmup_steps,
                total_steps=total_steps,
                offset=optim_cfg.freeze_backbone_steps if group["role"] == "backbone" else 0,
            )
            for group in param_groups
        ]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdas)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

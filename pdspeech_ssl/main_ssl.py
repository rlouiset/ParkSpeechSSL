import os

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from pdspeech_ssl.config import HParams
from pdspeech_ssl.data import PDSpeechDataModule
from pdspeech_ssl.lightning_module import SSLLightningModule

cs = ConfigStore.instance()
cs.store(name="base_config", node=HParams)

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: HParams) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = PDSpeechDataModule(cfg.data, cfg.augment, cfg.training)
    module = SSLLightningModule(cfg)

    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name)

    # relative "checkpoints" would land under $SLURM_SUBMIT_DIR ($HOME on Jean Zay --
    # small quota); default to $WORK there, fall back to a local relative dir otherwise.
    checkpoint_dir = os.path.join(os.environ["WORK"], "pdspeech_ssl_checkpoints") if "WORK" in os.environ else "checkpoints"
    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        # NOTE: metric names containing "/" can't be used inside the filename
        # template itself (ModelCheckpoint doesn't escape it, so "Val/foo" is
        # read as a subdirectory) -- "bal_acc" here is a plain manually-logged
        # alias of Val/hc_pd_balanced_accuracy, see lightning_module.py.
        filename="epoch{epoch}-bal_acc{bal_acc:.3f}",
        auto_insert_metric_name=False,
        monitor="Val/hc_pd_balanced_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
    )

    # find_unused_parameters=True: training_step calls self.model(...) twice per step
    # (once per contrastive view), and with LoRA freezing most of wav2vec2, the two
    # calls' autograd graphs don't always touch an identical set of trainable params --
    # plain strategy="ddp" (find_unused_parameters=False) crashes on that mismatch.
    strategy = DDPStrategy(find_unused_parameters=True) if cfg.training.strategy == "ddp" else cfg.training.strategy

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
        check_val_every_n_epoch=1,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

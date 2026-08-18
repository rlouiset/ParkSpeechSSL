import hydra
import lightning.pytorch as pl
import torch
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

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

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
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
    lr_monitor_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        gradient_clip_val=cfg.training.gradient_clip_val,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        logger=wandb_logger,
        callbacks=[checkpoint_cb, lr_monitor_cb],
        check_val_every_n_epoch=1,
    )

    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()

"""Top-level composition of a D4RT training experiment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from d4rt_lab.data import D4RTDataModule
from d4rt_lab.loss import D4RTLoss
from d4rt_lab.model import D4RTModel
from d4rt_lab.model.weights import initialize_model

if TYPE_CHECKING:
    from d4rt_lab.config.schema import D4RTConfig

from .callbacks import (
    ExperimentMetadataCallback,
    ModelArtifactCallback,
)
from .module import D4RTSystem


class D4RTExperiment:
    """Compose data, model, loss, callbacks, and Lightning execution.

    Parameters
    ----------
    cfg : D4RTConfig
        Fully composed Hydra configuration for one experiment.
    """

    def __init__(self, cfg: D4RTConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.experiment.output_dir)

    def _total_steps(self) -> int:
        override = self.cfg.schedule.local_repro_override
        return int(
            override.total_steps if override.enabled else self.cfg.schedule.total_steps
        )

    def _callbacks(self) -> list[L.Callback]:
        checkpoint_dir = self.output_dir / "checkpoints"
        return [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="step_{step:07d}",
                every_n_train_steps=int(self.cfg.checkpoint.step_save_every_steps),
                save_last=True,
                save_top_k=int(self.cfg.checkpoint.keep_last_k),
                save_on_train_epoch_end=False,
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                monitor=str(self.cfg.checkpoint.keep_best_by),
                mode="min",
                save_top_k=1,
                auto_insert_metric_name=False,
            ),
            LearningRateMonitor(logging_interval="step"),
            ExperimentMetadataCallback(self.output_dir, self.cfg),
            ModelArtifactCallback(self.output_dir, self.cfg.model),
        ]

    def _logger(self) -> L.loggers.Logger | bool:
        return (
            TensorBoardLogger(self.output_dir, name="tensorboard", version="")
            if self.cfg.launch.tensorboard
            else False
        )

    def fit(self) -> None:
        """Build and execute the configured training experiment."""
        L.seed_everything(int(self.cfg.experiment.seed), workers=True)

        # build the system
        model = D4RTModel(self.cfg.model)
        initialize_model(model, self.cfg.initialization)
        total_steps = self._total_steps()
        system = D4RTSystem(
            model,
            D4RTLoss(self.cfg.loss),
            self.cfg.optimizer,
            total_steps,
        )

        # build the dataModule
        data = D4RTDataModule(
            data_cfg=self.cfg.data,
            sampling_cfg=self.cfg.train_sampling,
            augmentation_cfg=self.cfg.augmentation,
            dataloader_cfg=self.cfg.dataloader,
            seed=self.cfg.experiment.seed,
            train_manifest=self.cfg.launch.train_manifest,
            val_manifest=self.cfg.launch.val_manifest,
        )

        # build the trainer and run
        trainer = L.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.cfg.runtime.devices,
            num_nodes=int(self.cfg.runtime.num_nodes),
            strategy=str(self.cfg.runtime.strategy),
            max_steps=total_steps,
            precision="16-mixed" if self.cfg.runtime.mixed_precision else "32-true",
            gradient_clip_val=float(self.cfg.optimizer.gradient_clip_l2_norm),
            log_every_n_steps=int(self.cfg.logging.log_every_steps),
            val_check_interval=int(self.cfg.logging.validate_every_steps),
            limit_val_batches=int(self.cfg.logging.validate_max_batches_global),
            callbacks=self._callbacks(),
            logger=self._logger(),
            default_root_dir=self.output_dir,
        )
        trainer.fit(
            system,
            datamodule=data,
            ckpt_path=self.cfg.launch.resume_checkpoint_path,
        )

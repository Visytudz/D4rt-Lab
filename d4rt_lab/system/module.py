"""Lightning system for D4RT training and validation."""

from __future__ import annotations

from typing import TypedDict

import lightning as L
import torch

from d4rt_lab.core.types import ModelOutput, TrainingBatch
from d4rt_lab.loss import D4RTLoss
from d4rt_lab.model import D4RTModel

from .optim import configure_optimization
from .config import OptimizerConfig


class StepOutput(TypedDict):
    """Values returned from one Lightning training or validation step."""

    loss: torch.Tensor
    predictions: ModelOutput


class D4RTSystem(L.LightningModule):
    """Combine the pure model, loss, and optimization policy.

    Parameters
    ----------
    model : D4RTModel
        Network mapping :class:`TrainingBatch` to predictions with leading
        shape ``[B, Q]``.
    loss : D4RTLoss
        Multi-task loss consuming typed predictions and supervision.
    optimizer_cfg : OptimizerConfig
        AdamW and learning-rate schedule configuration.
    total_steps : int
        Number of optimizer steps executed by Lightning.
    """

    def __init__(
        self,
        model: D4RTModel,
        loss: D4RTLoss,
        optimizer_cfg: OptimizerConfig,
        total_steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer_cfg = optimizer_cfg
        self.total_steps = int(total_steps)

    def forward(self, batch: TrainingBatch) -> ModelOutput:
        """Run the pure D4RT model."""
        return self.model(batch)

    def _shared_step(self, batch: TrainingBatch, stage: str) -> StepOutput:
        predictions = self.model(batch)
        total, metrics = self.loss(predictions, batch)
        self.log_dict(
            {f"{stage}_{name}": value for name, value in metrics.items()},
            on_step=stage == "train",
            on_epoch=stage == "val",
            prog_bar=True,
            sync_dist=True,
            batch_size=int(batch.video.shape[0]),
        )
        return {"loss": total, "predictions": predictions}

    def training_step(self, batch: TrainingBatch, batch_idx: int) -> StepOutput:
        del batch_idx
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> StepOutput:
        del batch_idx
        stage = "val" if dataloader_idx == 0 else f"val_dataset_{dataloader_idx}"
        return self._shared_step(batch, stage)

    def configure_optimizers(self) -> dict[str, object]:
        return configure_optimization(
            self.model.parameters(), self.optimizer_cfg, self.total_steps
        )

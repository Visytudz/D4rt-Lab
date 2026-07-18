"""Lightning data module for D4RT datasets."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
from torch.utils.data import DataLoader

from d4rt_lab.core.types import TrainingBatch
from d4rt_lab.data.builder import build_dataloader
from d4rt_lab.data.config import (
    AugmentationConfig,
    DataBuildConfig,
    DataConfig,
    DataLoaderConfig,
    TrainSamplingConfig,
)


class D4RTDataModule(L.LightningDataModule):
    """Own construction and transfer of D4RT training data.

    Parameters
    ----------
    data_cfg : DataConfig
        Dataset locations and clip geometry.
    sampling_cfg : TrainSamplingConfig
        Query and timestep sampling policy.
    augmentation_cfg : AugmentationConfig
        Training augmentation policy.
    dataloader_cfg : DataLoaderConfig
        Batch size and dataloader worker settings.
    seed : int
        Base seed used by datasets and workers.
    train_manifest, val_manifest : str or None
        Optional launch-time manifest paths.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        sampling_cfg: TrainSamplingConfig,
        augmentation_cfg: AugmentationConfig,
        dataloader_cfg: DataLoaderConfig,
        seed: int,
        train_manifest: str | None = None,
        val_manifest: str | None = None,
    ) -> None:
        super().__init__()
        self.cfg = DataBuildConfig(
            data=data_cfg,
            train_sampling=sampling_cfg,
            augmentation=augmentation_cfg,
            dataloader=dataloader_cfg,
            seed=seed,
        )
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and self._train_loader is None:
            self._train_loader = build_dataloader(
                "train", self.cfg, self.train_manifest
            )
        if stage in (None, "fit", "validate") and self._val_loader is None:
            self._val_loader = build_dataloader("val", self.cfg, self.val_manifest)

    def train_dataloader(self) -> DataLoader:
        if self._train_loader is None:
            self.setup("fit")
        assert self._train_loader is not None
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        if self._val_loader is None:
            self.setup("validate")
        assert self._val_loader is not None
        return self._val_loader

    def on_before_batch_transfer(
        self, batch: Mapping[str, object] | TrainingBatch, dataloader_idx: int
    ) -> TrainingBatch:
        """Establish the typed training boundary after collation."""
        del dataloader_idx
        return batch if isinstance(batch, TrainingBatch) else TrainingBatch.from_mapping(batch)

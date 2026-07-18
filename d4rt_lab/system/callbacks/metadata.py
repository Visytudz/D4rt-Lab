"""Experiment metadata logging."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
from typing import TYPE_CHECKING

import lightning as L
import yaml

if TYPE_CHECKING:
    from d4rt_lab.config import D4RTConfig


class ExperimentMetadataCallback(L.Callback):
    """Persist the resolved Hydra configuration at fit start."""

    def __init__(self, output_dir: Path, cfg: D4RTConfig) -> None:
        self.output_dir = Path(output_dir)
        self.cfg = cfg

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        path = self.output_dir / "config" / "composed.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                asdict(self.cfg),
                handle,
                allow_unicode=True,
                sort_keys=False,
            )

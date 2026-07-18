"""Export framework-independent model artifacts."""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import lightning as L

from d4rt_lab.model import D4RTModelConfig
from d4rt_lab.model.weights import save_model_artifact


class ModelArtifactCallback(L.Callback):
    """Export the final pure model independently of Lightning trainer state."""

    def __init__(self, output_dir: Path, model_cfg: D4RTModelConfig) -> None:
        self.path = Path(output_dir) / "artifacts" / "last.model.pt"
        self.model_config = asdict(model_cfg)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        save_model_artifact(
            self.path,
            model_config=self.model_config,
            state_dict=pl_module.model.state_dict(),
            metadata={"global_step": int(trainer.global_step)},
        )

"""Hydra command-line entrypoint for D4RT training."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from d4rt_lab.config import D4RTConfig, register_configs
from d4rt_lab.system import D4RTExperiment

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

register_configs()


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Run one fully composed D4RT training experiment."""
    app_cfg = OmegaConf.to_object(cfg)
    if not isinstance(app_cfg, D4RTConfig):
        raise TypeError(f"Expected D4RTConfig, got {type(app_cfg).__name__}")
    D4RTExperiment(app_cfg).fit()


if __name__ == "__main__":
    main()

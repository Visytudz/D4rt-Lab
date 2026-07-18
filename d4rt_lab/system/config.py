"""Structured configuration for training execution."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str = "d4rt"
    seed: int = 42
    output_dir: str = "output/d4rt"


@dataclass
class RuntimeConfig:
    devices: int | str = "auto"
    num_nodes: int = 1
    strategy: str = "auto"
    mixed_precision: bool = True


@dataclass
class LearningRateConfig:
    warmup_steps: int = 500
    peak_lr: float = 4e-6
    final_lr: float = 4e-7


@dataclass
class OptimizerConfig:
    weight_decay: float = 0.03
    gradient_clip_norm: float = 10.0
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)


@dataclass
class ScheduleConfig:
    total_steps: int = 30000


@dataclass
class CheckpointConfig:
    save_every_steps: int = 1000
    keep_last_k: int = 5
    keep_best_by: str = "val_loss_total"
    resume_checkpoint_path: str | None = None


@dataclass
class LoggingConfig:
    log_every_steps: int = 100
    validate_every_steps: int = 2000
    validate_max_batches: int = 64

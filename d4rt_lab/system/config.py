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
    framework: str = "lightning"
    devices: int | str = "auto"
    num_nodes: int = 1
    strategy: str = "auto"
    mixed_precision: bool = True
    fail_on_non_finite: bool = True


@dataclass
class LearningRateConfig:
    schedule: str = "cosine_decay"
    warmup_steps: int = 500
    peak_lr: float = 4e-6
    final_lr: float = 4e-7


@dataclass
class OptimizerConfig:
    type: str = "adamw"
    weight_decay: float = 0.03
    gradient_clip_l2_norm: float = 10.0
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)


@dataclass
class LocalReproductionConfig:
    enabled: bool = False
    total_steps: int = 30000


@dataclass
class ScheduleConfig:
    total_steps: int = 30000
    paper_reference_runtime: str = ""
    local_repro_override: LocalReproductionConfig = field(
        default_factory=LocalReproductionConfig
    )


@dataclass
class CheckpointConfig:
    step_save_every_steps: int = 1000
    keep_last_k: int = 20
    keep_best_by: str = "val_loss_total"


@dataclass
class LoggingConfig:
    log_every_steps: int = 100
    validate_every_steps: int = 2000
    validate_max_batches_global: int = 64


@dataclass
class LaunchConfig:
    train_manifest: str | None = None
    val_manifest: str | None = None
    resume_checkpoint_path: str | None = None
    tensorboard: bool = False

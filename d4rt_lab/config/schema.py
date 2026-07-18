"""Root Hydra schema for D4RT-Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from d4rt_lab.data.config import AugmentationConfig, DataConfig, DataLoaderConfig, TrainSamplingConfig
from d4rt_lab.loss.config import D4RTLossConfig
from d4rt_lab.model.config import D4RTModelConfig
from d4rt_lab.model.weights.config import InitializationConfig
from d4rt_lab.system.config import (
    CheckpointConfig,
    ExperimentConfig,
    LaunchConfig,
    LoggingConfig,
    OptimizerConfig,
    RuntimeConfig,
    ScheduleConfig,
)


@dataclass
class D4RTConfig:
    """Fully composed and structured D4RT-Lab configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: D4RTModelConfig = field(default_factory=D4RTModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train_sampling: TrainSamplingConfig = field(default_factory=TrainSamplingConfig)
    loss: D4RTLossConfig = field(default_factory=D4RTLossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)

    def __post_init__(self) -> None:
        """Validate configuration contracts shared across modules."""
        if self.model.input.clip_frames != self.data.clip_frames:
            raise ValueError(
                "model.input.clip_frames must equal data.clip_frames "
                f"({self.model.input.clip_frames} != {self.data.clip_frames})"
            )

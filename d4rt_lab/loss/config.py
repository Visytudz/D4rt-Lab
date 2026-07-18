"""Structured configuration for D4RT losses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LossTermConfig:
    enabled: bool = True
    weight: float = 1.0


@dataclass
class XYZLossConfig(LossTermConfig):
    normalize_by_mean_depth: bool = True
    use_signed_log: bool = True


@dataclass
class UVLossConfig(LossTermConfig):
    weight: float = 0.1


@dataclass
class VisibilityLossConfig(LossTermConfig):
    weight: float = 0.1


@dataclass
class DisplacementLossConfig(LossTermConfig):
    weight: float = 0.1


@dataclass
class NormalLossConfig(LossTermConfig):
    weight: float = 0.4


@dataclass
class ConfidenceTargetConfig(LossTermConfig):
    enabled: bool = False
    weight: float = 0.2


@dataclass
class ConfidenceLossConfig(LossTermConfig):
    weight: float = 0.2
    confidence_weights_xyz_error: bool = True
    confidence_target: ConfidenceTargetConfig = field(
        default_factory=ConfidenceTargetConfig
    )


@dataclass
class ReprojectionLossConfig(LossTermConfig):
    enabled: bool = False
    weight: float = 0.05
    detach_xyz: bool = False
    robust_loss: str = "huber"
    huber_delta_px: float = 8.0


@dataclass
class D4RTLossConfig:
    xyz_3d: XYZLossConfig = field(default_factory=XYZLossConfig)
    uv_2d: UVLossConfig = field(default_factory=UVLossConfig)
    visibility: VisibilityLossConfig = field(default_factory=VisibilityLossConfig)
    displacement: DisplacementLossConfig = field(default_factory=DisplacementLossConfig)
    normal: NormalLossConfig = field(default_factory=NormalLossConfig)
    confidence: ConfidenceLossConfig = field(default_factory=ConfidenceLossConfig)
    reprojection_uv_from_xyz: ReprojectionLossConfig = field(
        default_factory=ReprojectionLossConfig
    )

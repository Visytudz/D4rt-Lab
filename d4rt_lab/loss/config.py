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
    value_transform: str = "sign_x_log1p_abs_x"


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
class ConfidenceAblationConfig(LossTermConfig):
    enabled: bool = False
    weight: float = 0.2


@dataclass
class ConfidenceLossConfig(LossTermConfig):
    weight: float = 0.2
    mode: str = "main_text"
    confidence_penalty: str = "-log(c)"
    confidence_weights_xyz_error: bool = True
    lconf_ablation: ConfidenceAblationConfig = field(
        default_factory=ConfidenceAblationConfig
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
    total: str = "weighted_sum"
    xyz_3d: XYZLossConfig = field(default_factory=XYZLossConfig)
    uv_2d: UVLossConfig = field(default_factory=UVLossConfig)
    visibility: VisibilityLossConfig = field(default_factory=VisibilityLossConfig)
    displacement: DisplacementLossConfig = field(default_factory=DisplacementLossConfig)
    normal: NormalLossConfig = field(default_factory=NormalLossConfig)
    confidence: ConfidenceLossConfig = field(default_factory=ConfidenceLossConfig)
    apply_only_where_gt_available: bool = True
    reprojection_uv_from_xyz: ReprojectionLossConfig = field(
        default_factory=ReprojectionLossConfig
    )

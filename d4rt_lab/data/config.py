"""Structured data and augmentation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


# -----------------------------------------------------------------------------
# Data loading infrastructure
# -----------------------------------------------------------------------------


@dataclass
class BadSampleConfig:
    path: str = "data/meta/bad_sample.json"
    max_retries: int = 64


@dataclass
class DataLoaderConfig:
    batch_size: int = 1
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 1
    train_num_workers: int = 1
    val_num_workers: int = 1
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 1
    train_prefetch_factor: int = 1
    timeout: int = 300
    train_drop_last: bool = True
    val_drop_last: bool = False


@dataclass
class BenchmarkTrackingConfig:
    enabled: bool = False
    max_queries: int = 0


@dataclass
class ReprojectionCheckConfig:
    enabled: bool = False
    max_frames: int = 0
    max_points_per_frame: int = 0
    max_scenes: int = 0
    median_threshold_px: float = 0.0
    mode: str = "off"


# -----------------------------------------------------------------------------
# Dataset-specific configuration
# -----------------------------------------------------------------------------


@dataclass
class PointOdysseyConfig:
    root: str = "data/pointodyssey/v2"
    split_map: dict[str, str] = field(default_factory=dict)
    max_cached_scenes: int = 2
    max_scenes: int | None = None
    val_clips_per_scene: int = 1


@dataclass
class DynamicReplicaConfig:
    root: str = "data/dynamic-replica/v2"
    split_map: dict[str, str] = field(default_factory=dict)
    camera_convention: str = "dynamic_replica_v2"
    depth_decode_mode: str = "auto"
    depth_divisor: float = 10000.0
    max_scenes: int | None = None
    benchmark_tracking: BenchmarkTrackingConfig = field(
        default_factory=BenchmarkTrackingConfig
    )
    reprojection_self_check: ReprojectionCheckConfig = field(
        default_factory=ReprojectionCheckConfig
    )


@dataclass
class KubricConfig:
    root: str = "data/kubric_full/movi-f_full/512x512"
    backend: str = "preprocess"
    processed_root: str = "data/kubric_full/kubric_full_process_v1"
    mmap_mode: str = "r"
    eval_cache_max_items: int = 2
    tfds_split_map: dict[str, str] = field(default_factory=dict)
    processed_split_map: dict[str, str] = field(default_factory=dict)
    shuffle_buffer_size: int = 64
    max_scenes: int | None = None
    benchmark_tracking: BenchmarkTrackingConfig = field(
        default_factory=BenchmarkTrackingConfig
    )


@dataclass
class TartanAirConfig:
    root: str = "data/tartanair/v2"
    camera_name: str = "lcam_front"
    difficulties: list[str] = field(default_factory=list)
    split_modulo: int = 20
    split_map: dict[str, str] = field(default_factory=dict)
    max_depth_m: float = 1000.0
    max_scenes: int | None = None
    intrinsics: list[float] = field(default_factory=list)


@dataclass
class VirtualKitti2Config:
    root: str = "data/virtual-kitti-2/v2"
    variants: list[str] = field(default_factory=list)
    camera_ids: list[int] = field(default_factory=list)
    split_scenes: dict[str, list[str]] = field(default_factory=dict)
    max_scenes: int | None = None


@dataclass
class ScanNetConfig:
    root: str = "data/scannet/plus-v2/data"
    split_files: dict[str, str] = field(default_factory=dict)
    source: str = "iphone_rgbd"
    max_scenes: int | None = None


@dataclass
class BlendedMVSConfig:
    roots: list[str] = field(default_factory=list)
    split_map: dict[str, str] = field(default_factory=dict)
    split_modulo: int = 20
    max_depth_m: float = 1000.0
    depth_clip_percentile: float = 98.0
    min_depth_valid_ratio: float = 0.01
    min_valid_frames_ratio: float = 0.5
    require_complete_frames: bool = False
    use_masked_images: bool = False
    max_scenes: int | None = None


@dataclass
class CO3DConfig:
    root: str = "data/co3d/v2"
    use_depth_masks: bool = True
    categories: list[str] = field(default_factory=list)
    split_map: dict[str, str] = field(default_factory=dict)
    min_viewpoint_quality: float = 0.0
    max_scenes: int | None = None


@dataclass
class MVSSynthConfig:
    root: str = "data/mvs-synth/v1"
    sequence_dir: str = ""
    split_map: dict[str, str] = field(default_factory=dict)
    split_modulo: int = 20
    depth_scale: float = 1.0
    max_depth_m: float = 1000.0
    depth_clip_percentile: float = 98.0
    min_depth_valid_ratio: float = 0.01
    min_valid_frames_ratio: float = 0.5
    require_complete_frames: bool = False
    max_scenes: int | None = None


# -----------------------------------------------------------------------------
# Dataset collection and mixture configuration
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    clip_frames: int = 48
    image_size: tuple[int, int] = (256, 256)
    bad_sample_registry: BadSampleConfig = field(default_factory=BadSampleConfig)
    train_sources: list[str] = field(default_factory=list)
    val_sources: list[str] = field(default_factory=list)
    sampling_weights: dict[str, float] = field(default_factory=dict)
    pointodyssey: PointOdysseyConfig = field(default_factory=PointOdysseyConfig)
    dynamic_replica: DynamicReplicaConfig = field(default_factory=DynamicReplicaConfig)
    kubric_full: KubricConfig = field(default_factory=KubricConfig)
    tartanair: TartanAirConfig = field(default_factory=TartanAirConfig)
    virtual_kitti2: VirtualKitti2Config = field(default_factory=VirtualKitti2Config)
    scannet: ScanNetConfig = field(default_factory=ScanNetConfig)
    blendermvs: BlendedMVSConfig = field(default_factory=BlendedMVSConfig)
    co3d: CO3DConfig = field(default_factory=CO3DConfig)
    mvs_synth: MVSSynthConfig = field(default_factory=MVSSynthConfig)


# -----------------------------------------------------------------------------
# Query and timestep sampling
# -----------------------------------------------------------------------------


@dataclass
class TimestepSamplingConfig:
    t_src: str = "uniform"
    t_tgt: str = "uniform"
    t_cam: str = "uniform"
    prob_t_tgt_equals_t_cam: float = 0.4
    t_src_tgt_delta_mode: str = "static_local_global"
    t_src_tgt_delta_choices: list[int | None] | None = None
    t_src_tgt_delta_probs: list[float] | None = None


@dataclass
class TrainSamplingConfig:
    queries_per_clip: int = 4096
    hard_query_ratio: float = 0.2
    hard_query_source: str = "depth_and_motion_boundaries_sobel"
    timestep_sampling: TimestepSamplingConfig = field(
        default_factory=TimestepSamplingConfig
    )


# -----------------------------------------------------------------------------
# Data augmentation
# -----------------------------------------------------------------------------


@dataclass
class ToggleAugmentationConfig:
    enabled: bool = False
    probability: float = 0.0


@dataclass
class ColorJitterConfig(ToggleAugmentationConfig):
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0


@dataclass
class RandomCropConfig(ToggleAugmentationConfig):
    scale_min_max: tuple[float, float] = (1.0, 1.0)
    aspect_ratio_min_max: tuple[float, float] = (1.0, 1.0)
    random_zoom_in_probability: float = 0.0
    aspect_ratio_sampling: str = "log_uniform"
    sampling_domain: str = "original"
    boundary_mode: str = "rejection"


@dataclass
class TemporalSubsampleConfig:
    enabled: bool = True
    stride_min: int = 1
    stride_max: int = 2
    stride_sampling: str = "random"


@dataclass
class AugmentationConfig:
    color_jitter: ColorJitterConfig = field(default_factory=ColorJitterConfig)
    color_drop: ToggleAugmentationConfig = field(default_factory=ToggleAugmentationConfig)
    gaussian_blur: ToggleAugmentationConfig = field(default_factory=ToggleAugmentationConfig)
    random_crop: RandomCropConfig = field(default_factory=RandomCropConfig)
    temporal_subsample: TemporalSubsampleConfig = field(default_factory=TemporalSubsampleConfig)


# -----------------------------------------------------------------------------
# Builder input boundary
# -----------------------------------------------------------------------------


@dataclass
class DataBuildConfig:
    """Configuration boundary consumed by dataset and dataloader builders.

    Parameters
    ----------
    data : DataConfig
        Dataset locations, mixtures, and clip geometry.
    train_sampling : TrainSamplingConfig
        Query and timestep sampling policy.
    augmentation : AugmentationConfig
        Training augmentation policy.
    dataloader : DataLoaderConfig
        Batching and worker settings.
    seed : int
        Base random seed for datasets and dataloader workers.
    """

    data: DataConfig
    train_sampling: TrainSamplingConfig
    augmentation: AugmentationConfig
    dataloader: DataLoaderConfig
    seed: int = 42

"""Shared configuration and lifecycle for dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset

from .bad_samples import BadSampleRegistry
from ..sampling.augmentation import RawAugmentConfig
from .seeding import SeededDatasetMixin


@dataclass(kw_only=True)
class DatasetConfig:
    """Resolved configuration shared by single-source datasets.

    Parameters
    ----------
    clip_frames : int
        Number of video frames in each sample.
    image_size : tuple[int, int]
        Output image size as ``(height, width)``.
    queries_per_clip : int
        Number of spatiotemporal queries sampled from each clip.
    hard_query_ratio : float
        Fraction of queries sampled near geometric or motion boundaries.
    prob_t_tgt_equals_t_cam : float
        Probability that target and camera timesteps are identical.
    training : bool
        Whether stochastic training augmentation is enabled.
    """

    clip_frames: int
    image_size: tuple[int, int]
    queries_per_clip: int
    hard_query_ratio: float
    prob_t_tgt_equals_t_cam: float
    training: bool
    t_src_tgt_delta_choices: tuple[int | None, ...] | None = None
    t_src_tgt_delta_probs: tuple[float, ...] | None = None
    max_scenes: int | None = None
    augment: RawAugmentConfig | None = None
    bad_sample_registry_path: Path = Path("datasets/meta/bad_sample.json")
    max_sample_retries: int = 64


class BaseDataset(SeededDatasetMixin, Dataset):
    """Base class for common dataset state without constraining sampling logic."""

    def __init__(
        self,
        config: DatasetConfig,
        *,
        namespace: str,
        default_seed: int,
    ) -> None:
        """Initialize geometry, augmentation, random state, and retry state.

        Parameters
        ----------
        config : DatasetConfig
            Resolved dataset configuration. ``image_size`` has shape ``(H, W)``.
        namespace : str
            Stable dataset name used to derive an independent random stream.
        default_seed : int
            Seed used before the experiment seed is assigned by the builder.
        """
        self.cfg = config
        self.h, self.w = config.image_size
        self._init_dataset_seeding(namespace=namespace, default_seed=default_seed)
        self.augment = (
            config.augment
            if config.training and config.augment is not None
            else RawAugmentConfig()
        )
        self.bad_registry = BadSampleRegistry(config.bad_sample_registry_path)
        self.max_sample_retries = max(1, int(config.max_sample_retries))

"""Dataset and dataloader builders for the D4RT 9Mix training recipe."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from .config import DataBuildConfig
from .datasets.blendermvs import BlendermvsRawConfig, BlendermvsRawDataset
from .datasets.co3d import Co3dRawConfig, Co3dRawDataset
from .datasets.dynamic_replica import DynamicReplicaRawConfig, DynamicReplicaRawDataset
from .datasets.kubric import KubricFullRobustConfig, KubricFullRobustDataset
from .datasets.kubric_preprocessed import (
    KubricFullRobustPreprocessConfig,
    KubricFullRobustPreprocessDataset,
)
from .datasets.mixture import MixtureDataset, MixtureDatasetConfig
from .datasets.mvs_synth import MvsSynthRawConfig, MvsSynthRawDataset
from .datasets.pointodyssey import PointOdysseyRawConfig, PointOdysseyRawDataset
from .sampling.augmentation import (
    STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES,
    STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS,
    augment_cfg_from_train_config,
)
from .datasets.scannet import ScannetRawConfig, ScannetRawDataset
from .datasets.seeding import configure_dataset_seeding, seed_dataloader_worker
from .datasets.tartanair import TartanairRawConfig, TartanairRawDataset
from .datasets.virtual_kitti2 import VirtualKitti2RawConfig, VirtualKitti2RawDataset



def _image_size(cfg: DataBuildConfig) -> tuple[int, int]:
    image_size_cfg = cfg.data.image_size
    if not isinstance(image_size_cfg, (list, tuple)) or len(image_size_cfg) != 2:
        raise ValueError(f"data.image_size must be [H, W], got: {image_size_cfg}")
    return int(image_size_cfg[0]), int(image_size_cfg[1])


def _queries_per_clip(cfg: DataBuildConfig) -> int:
    return int(cfg.train_sampling.queries_per_clip)


def _hard_query_ratio(cfg: DataBuildConfig) -> float:
    return float(cfg.train_sampling.hard_query_ratio)


def _prob_t_tgt_equals_t_cam(cfg: DataBuildConfig) -> float:
    return float(cfg.train_sampling.timestep_sampling.prob_t_tgt_equals_t_cam)


def _query_timestep_delta_kwargs(
    cfg: DataBuildConfig,
) -> dict[str, tuple[int | None, ...] | tuple[float, ...] | None]:
    path = "train_sampling.timestep_sampling"
    timestep = cfg.train_sampling.timestep_sampling
    mode_raw = timestep.t_src_tgt_delta_mode
    choices_raw = timestep.t_src_tgt_delta_choices
    probs_raw = timestep.t_src_tgt_delta_probs

    if choices_raw is None:
        mode = str(mode_raw).strip().lower().replace("-", "_") if mode_raw is not None else ""
        if mode in {"", "none", "off", "disabled", "uniform", "independent_uniform"}:
            choices: tuple[int | None, ...] | None = None
            probs: tuple[float, ...] | None = None
        elif mode in {"static_local_global", "local_global", "local_global_static"}:
            choices = STATIC_LOCAL_GLOBAL_TGT_DELTA_CHOICES
            probs = STATIC_LOCAL_GLOBAL_TGT_DELTA_PROBS
        else:
            raise ValueError(
                f"Unsupported {path}.t_src_tgt_delta_mode={mode_raw!r}; "
                "supported: off, independent_uniform, static_local_global"
            )
    else:
        if not isinstance(choices_raw, (list, tuple)):
            raise ValueError(f"{path}.t_src_tgt_delta_choices must be a list, got {choices_raw!r}")
        parsed_choices: list[int | None] = []
        for item in choices_raw:
            if item is None:
                parsed_choices.append(None)
                continue
            if isinstance(item, str):
                token = item.strip().lower().replace("-", "_")
                if token in {"none", "null", "full", "full_range", "global", "all"}:
                    parsed_choices.append(None)
                    continue
            parsed_choices.append(int(item))
        if not parsed_choices:
            raise ValueError(f"{path}.t_src_tgt_delta_choices must be non-empty")
        choices = tuple(parsed_choices)
        if probs_raw is None:
            probs = tuple([1.0 / float(len(choices))] * len(choices))
        else:
            if not isinstance(probs_raw, (list, tuple)):
                raise ValueError(f"{path}.t_src_tgt_delta_probs must be a list, got {probs_raw!r}")
            probs = tuple(float(v) for v in probs_raw)
            if len(probs) != len(choices):
                raise ValueError(f"{path}.t_src_tgt_delta_probs length does not match choices length")

    return {"t_src_tgt_delta_choices": choices, "t_src_tgt_delta_probs": probs}


def _clip_frames(cfg: DataBuildConfig) -> int:
    return int(cfg.data.clip_frames)


def _bad_sample_registry_path(cfg: DataBuildConfig) -> Path:
    return Path(cfg.data.bad_sample_registry.path)


def _bad_sample_max_retries(cfg: DataBuildConfig) -> int:
    return int(cfg.data.bad_sample_registry.max_retries)


def _dataset_kwargs(split: str, cfg: DataBuildConfig) -> dict[str, object]:
    """Resolve parameters shared by all single-source dataset configs."""
    return {
        "clip_frames": _clip_frames(cfg),
        "image_size": _image_size(cfg),
        "queries_per_clip": _queries_per_clip(cfg),
        "hard_query_ratio": _hard_query_ratio(cfg),
        "prob_t_tgt_equals_t_cam": _prob_t_tgt_equals_t_cam(cfg),
        **_query_timestep_delta_kwargs(cfg),
        "training": split == "train",
        "augment": augment_cfg_from_train_config(cfg.augmentation),
        "bad_sample_registry_path": _bad_sample_registry_path(cfg),
        "max_sample_retries": _bad_sample_max_retries(cfg),
    }


def _normalize_dataset_type(dataset_type: str) -> str:
    key = str(dataset_type).strip().lower()
    aliases = {
        "blendermvs": "blendedmvs_raw",
        "blendermvs_raw": "blendedmvs_raw",
        "blendedmvs": "blendedmvs_raw",
        "blended_mvs": "blendedmvs_raw",
        "co3d": "co3d_raw",
        "co3dv2": "co3d_raw",
        "co3d_v2": "co3d_raw",
        "dynamic_replica": "dynamic_replica_raw",
        "dynamic-replica": "dynamic_replica_raw",
        "kubric_full": "kubric_full_robust",
        "kubric-full": "kubric_full_robust",
        "kubric_full_preprocess": "kubric_full_robust_preprocess",
        "kubric_full_processed": "kubric_full_robust_preprocess",
        "mvs_synth": "mvs_synth_raw",
        "mvs-synth": "mvs_synth_raw",
        "pointodyssey": "pointodyssey_raw",
        "point_odyssey": "pointodyssey_raw",
        "scannet": "scannet_raw",
        "tartanair": "tartanair_raw",
        "virtual_kitti2": "virtual_kitti2_raw",
        "virtual-kitti-2": "virtual_kitti2_raw",
        "virtualkitti2": "virtual_kitti2_raw",
        "vkitti2": "virtual_kitti2_raw",
        "vitual-kitti-2": "virtual_kitti2_raw",
        "vitual_kitti_2": "virtual_kitti2_raw",
        "mixture": "mixture_raw",
    }
    return aliases.get(key, key)


def _blendermvs_roots(cfg: DataBuildConfig) -> list[Path]:
    roots = [Path(item) for item in cfg.data.blendermvs.roots]
    if roots:
        return roots
    return [Path("data/blendermvs/base-low-res/BlendedMVS")]


def _build_blendedmvs_raw(split: str, cfg: DataBuildConfig):
    roots = _blendermvs_roots(cfg)
    data_cfg = cfg.data.blendermvs
    split_map = data_cfg.split_map or None
    return BlendermvsRawDataset(
        BlendermvsRawConfig(
            root=roots[0],
            roots=tuple(roots),
            split=str(split),
            **_dataset_kwargs(split, cfg),
            split_map=split_map,
            max_scenes=data_cfg.max_scenes,
            split_modulo=data_cfg.split_modulo,
            use_masked_images=data_cfg.use_masked_images,
            max_depth_m=data_cfg.max_depth_m,
            depth_clip_percentile=data_cfg.depth_clip_percentile,
            min_depth_valid_ratio=data_cfg.min_depth_valid_ratio,
            min_valid_frames_ratio=data_cfg.min_valid_frames_ratio,
            require_complete_frames=data_cfg.require_complete_frames,
        )
    )


def _build_co3d_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.co3d
    root = Path(data_cfg.root)
    categories = data_cfg.categories
    split_map = data_cfg.split_map or None
    return Co3dRawDataset(
        Co3dRawConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            split_map=split_map,
            max_scenes=data_cfg.max_scenes,
            categories=categories or None,
            min_viewpoint_quality=data_cfg.min_viewpoint_quality,
            use_depth_masks=data_cfg.use_depth_masks,
        )
    )


def _build_kubric_full_robust(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.kubric_full
    root = Path(data_cfg.root)
    split_map_raw = data_cfg.tfds_split_map or {"train": "train", "val": "validation", "test": "validation"}
    split_map = split_map_raw if isinstance(split_map_raw, dict) else {"train": "train", "val": "validation", "test": "validation"}
    return KubricFullRobustDataset(
        KubricFullRobustConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            max_scenes=cfg.data.kubric_full.max_scenes,
            tfds_split_map=split_map,
            shuffle_buffer_size=cfg.data.kubric_full.shuffle_buffer_size,
            eval_cache_max_items=cfg.data.kubric_full.eval_cache_max_items,
            benchmark_tracking_enabled=cfg.data.kubric_full.benchmark_tracking.enabled,
            benchmark_max_queries=cfg.data.kubric_full.benchmark_tracking.max_queries,
        )
    )


def _build_kubric_full_robust_preprocess(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.kubric_full
    root = Path(data_cfg.processed_root)
    split_map = data_cfg.processed_split_map or {"train": "train", "val": "validation", "test": "validation"}
    mmap_mode_raw = data_cfg.mmap_mode
    mmap_mode = None if str(mmap_mode_raw).lower() in {"", "none", "false"} else str(mmap_mode_raw)
    return KubricFullRobustPreprocessDataset(
        KubricFullRobustPreprocessConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            max_scenes=data_cfg.max_scenes,
            split_map=split_map,
            mmap_mode=mmap_mode,
            eval_cache_max_items=data_cfg.eval_cache_max_items,
            benchmark_tracking_enabled=data_cfg.benchmark_tracking.enabled,
            benchmark_max_queries=data_cfg.benchmark_tracking.max_queries,
        )
    )


def _build_pointodyssey_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.pointodyssey
    root = Path(data_cfg.root)
    split_map = data_cfg.split_map or None
    return PointOdysseyRawDataset(
        PointOdysseyRawConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            split_map=split_map,
            max_scenes=data_cfg.max_scenes,
            max_cached_scenes=data_cfg.max_cached_scenes,
            val_clips_per_scene=data_cfg.val_clips_per_scene,
        )
    )


def _build_virtual_kitti2_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.virtual_kitti2
    root = Path(data_cfg.root)
    split_scenes = data_cfg.split_scenes or None
    variants = data_cfg.variants
    camera_ids = data_cfg.camera_ids
    return VirtualKitti2RawDataset(
        VirtualKitti2RawConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            split_scenes=split_scenes,
            variants=variants or None,
            camera_ids=camera_ids or [0],
            max_scenes=data_cfg.max_scenes,
        )
    )


def _dynamic_replica_split(split: str, cfg: DataBuildConfig) -> str:
    mapped = cfg.data.dynamic_replica.split_map.get(split)
    if mapped:
        return mapped
    return {"train": "train", "val": "valid", "test": "test"}.get(split, split)


def _build_dynamic_replica_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.dynamic_replica
    root = Path(data_cfg.root)
    reprojection = data_cfg.reprojection_self_check
    benchmark = data_cfg.benchmark_tracking
    return DynamicReplicaRawDataset(
        DynamicReplicaRawConfig(
            root=root,
            split=_dynamic_replica_split(split, cfg),
            **_dataset_kwargs(split, cfg),
            max_scenes=data_cfg.max_scenes,
            camera_convention=data_cfg.camera_convention,
            depth_decode_mode=data_cfg.depth_decode_mode,
            depth_divisor=data_cfg.depth_divisor,
            reprojection_self_check_enabled=reprojection.enabled,
            reprojection_self_check_mode=reprojection.mode,
            reprojection_self_check_median_threshold_px=reprojection.median_threshold_px,
            reprojection_self_check_max_scenes=reprojection.max_scenes,
            reprojection_self_check_max_frames=reprojection.max_frames,
            reprojection_self_check_max_points=reprojection.max_points_per_frame,
            benchmark_tracking_enabled=benchmark.enabled,
            benchmark_max_queries=benchmark.max_queries,
        )
    )


def _build_mvs_synth_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.mvs_synth
    root = Path(data_cfg.root)
    split_map = data_cfg.split_map or None
    return MvsSynthRawDataset(
        MvsSynthRawConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            split_map=split_map,
            sequence_dir=data_cfg.sequence_dir,
            max_scenes=data_cfg.max_scenes,
            split_modulo=data_cfg.split_modulo,
            depth_scale=data_cfg.depth_scale,
            max_depth_m=data_cfg.max_depth_m,
            depth_clip_percentile=data_cfg.depth_clip_percentile,
            min_depth_valid_ratio=data_cfg.min_depth_valid_ratio,
            min_valid_frames_ratio=data_cfg.min_valid_frames_ratio,
            require_complete_frames=data_cfg.require_complete_frames,
        )
    )


def _scannet_split_file(split: str, cfg: DataBuildConfig) -> Path:
    picked = cfg.data.scannet.split_files.get(split)
    if picked:
        return Path(picked)
    defaults = {
        "train": "data/scannet/plus-v2/splits/nvs_sem_train.txt",
        "val": "data/scannet/plus-v2/splits/nvs_sem_val.txt",
        "test": "data/scannet/plus-v2/splits/nvs_test.txt",
    }
    return Path(defaults.get(split, defaults["val"]))


def _build_scannet_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.scannet
    root = Path(data_cfg.root)
    return ScannetRawDataset(
        ScannetRawConfig(
            root=root,
            split_file=_scannet_split_file(split, cfg),
            **_dataset_kwargs(split, cfg),
            max_scenes=data_cfg.max_scenes,
            source=data_cfg.source,
        )
    )


def _build_tartanair_raw(split: str, cfg: DataBuildConfig):
    data_cfg = cfg.data.tartanair
    root = Path(data_cfg.root)
    split_map = data_cfg.split_map or None
    difficulties = data_cfg.difficulties
    intrinsics = data_cfg.intrinsics or None
    return TartanairRawDataset(
        TartanairRawConfig(
            root=root,
            split=str(split),
            **_dataset_kwargs(split, cfg),
            split_map=split_map,
            camera_name=data_cfg.camera_name,
            difficulties=difficulties or ["Data_easy", "Data_hard"],
            max_scenes=data_cfg.max_scenes,
            split_modulo=data_cfg.split_modulo,
            max_depth_m=data_cfg.max_depth_m,
            intrinsics=intrinsics,
        )
    )


def _normalize_mixture_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "blendermvs_raw": "blendermvs",
        "blendedmvs": "blendermvs",
        "blendedmvs_raw": "blendermvs",
        "co3d_raw": "co3d",
        "co3dv2": "co3d",
        "dynamic_replica_raw": "dynamic_replica",
        "kubric_full_robust": "kubric_full",
        "kubric_full_robust_preprocess": "kubric_full",
        "mvs_synth_raw": "mvs_synth",
        "pointodyssey_raw": "pointodyssey",
        "point_odyssey": "pointodyssey",
        "scannet_raw": "scannet",
        "tartanair_raw": "tartanair",
        "virtual_kitti2_raw": "virtual_kitti2",
        "virtualkitti2": "virtual_kitti2",
        "vkitti2": "virtual_kitti2",
        "vitual_kitti_2": "virtual_kitti2",
    }
    return aliases.get(key, key)


def _resolve_mixture_sources(split: str, cfg: DataBuildConfig) -> list[tuple[str, str]]:
    names = (
        cfg.data.train_sources
        if split == "train"
        else cfg.data.val_sources
    )

    kubric_full_backend = cfg.data.kubric_full.backend.strip().lower()
    kubric_full_builder = (
        "kubric_full_robust_preprocess"
        if kubric_full_backend in {"preprocess", "processed", "npy", "preprocessed"}
        else "kubric_full_robust"
    )
    source_to_builder = {
        "blendermvs": "blendedmvs_raw",
        "co3d": "co3d_raw",
        "dynamic_replica": "dynamic_replica_raw",
        "kubric_full": kubric_full_builder,
        "mvs_synth": "mvs_synth_raw",
        "pointodyssey": "pointodyssey_raw",
        "scannet": "scannet_raw",
        "tartanair": "tartanair_raw",
        "virtual_kitti2": "virtual_kitti2_raw",
    }

    resolved: list[tuple[str, str]] = []
    unknown_sources: list[str] = []
    for name in names:
        normalized = _normalize_mixture_name(name)
        builder_name = source_to_builder.get(normalized)
        if builder_name is None:
            unknown_sources.append(str(name))
        else:
            resolved.append((name, builder_name))
    if unknown_sources:
        supported = ", ".join(sorted(source_to_builder))
        unknown = ", ".join(unknown_sources)
        raise ValueError(f"Unsupported mixture source(s): {unknown}. Supported sources: {supported}")
    return resolved


def _resolve_mixture_weights(cfg: DataBuildConfig, selected_sources: list[str]) -> list[float] | None:
    raw = cfg.data.sampling_weights
    if not raw or not selected_sources:
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) < len(selected_sources):
            return None
        return [float(raw[idx]) for idx in range(len(selected_sources))]
    if isinstance(raw, dict):
        out: list[float] = []
        for name in selected_sources:
            normalized = _normalize_mixture_name(name)
            candidate = raw.get(name, raw.get(normalized))
            if candidate is None:
                return None
            out.append(float(candidate))
        return out
    return None


def _build_mixture_raw(split: str, cfg: DataBuildConfig):
    resolved_sources = _resolve_mixture_sources(split, cfg)
    if not resolved_sources:
        raise ValueError("mixture_raw requires at least one source in data.dataset_mixture")

    datasets = []
    source_names: list[str] = []
    for source_name, dataset_type in resolved_sources:
        builder = _dataset_builder(dataset_type)
        try:
            datasets.append(builder(split=split, cfg=cfg))
            source_names.append(source_name)
        except Exception as exc:
            warnings.warn(f"Skip mixture source '{source_name}' ({dataset_type}): {exc}", stacklevel=2)

    if not datasets:
        raise ValueError("No valid datasets could be built for mixture_raw")

    return MixtureDataset(
        MixtureDatasetConfig(
            datasets=datasets,
            weights=_resolve_mixture_weights(cfg, source_names),
        )
    )


DATASET_BUILDERS = {
    "blendermvs_raw": _build_blendedmvs_raw,
    "blendedmvs_raw": _build_blendedmvs_raw,
    "co3d_raw": _build_co3d_raw,
    "kubric_full_robust": _build_kubric_full_robust,
    "kubric_full_robust_preprocess": _build_kubric_full_robust_preprocess,
    "pointodyssey_raw": _build_pointodyssey_raw,
    "virtual_kitti2_raw": _build_virtual_kitti2_raw,
    "dynamic_replica_raw": _build_dynamic_replica_raw,
    "mvs_synth_raw": _build_mvs_synth_raw,
    "scannet_raw": _build_scannet_raw,
    "tartanair_raw": _build_tartanair_raw,
    "mixture_raw": _build_mixture_raw,
}


def _dataset_builder(dataset_type: str):
    """Resolve an internal dataset factory by normalized type name."""
    try:
        return DATASET_BUILDERS[dataset_type]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(f"Unsupported dataset type '{dataset_type}'. Known: [{known}]") from exc


def build_dataset(split: str, cfg: DataBuildConfig):
    dataset = _build_mixture_raw(split=split, cfg=cfg)
    if isinstance(dataset, list):
        dataset = ConcatDataset(dataset)
    configure_dataset_seeding(dataset, base_seed=cfg.seed)
    return dataset


def _worker_count(split: str, cfg: DataBuildConfig) -> int:
    return int(
        cfg.dataloader.train_num_workers
        if split == "train"
        else cfg.dataloader.val_num_workers
    )


def build_dataloader(
    split: str,
    cfg: DataBuildConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = build_dataset(split=split, cfg=cfg)
    batch_size = int(
        cfg.dataloader.train_batch_size
        if split == "train"
        else cfg.dataloader.val_batch_size
    )
    drop_last = bool(
        cfg.dataloader.train_drop_last
        if split == "train"
        else cfg.dataloader.val_drop_last
    )
    num_workers = _worker_count(split, cfg)

    sampler = None
    shuffle = split == "train"
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        shuffle = False

    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": cfg.dataloader.pin_memory,
        "drop_last": drop_last if sampler is None else False,
        "worker_init_fn": seed_dataloader_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.dataloader.persistent_workers
        prefetch_factor = (
            cfg.dataloader.train_prefetch_factor
            if split == "train"
            else cfg.dataloader.prefetch_factor
        )
        loader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        loader_kwargs["timeout"] = max(0, cfg.dataloader.timeout)

    loader = DataLoader(**loader_kwargs)
    loader.dist_sampler = sampler  # type: ignore[attr-defined]
    return loader

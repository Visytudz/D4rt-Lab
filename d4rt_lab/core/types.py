"""Shared tensor data types used across D4RT-Lab components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from typing import cast
import torch


class TensorRecord(Mapping[str, object]):
    """Mapping interface shared by immutable batch records."""

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return len(fields(self))

    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


@dataclass
class QueryBatch(TensorRecord):
    """Batched spatio-temporal queries consumed by D4RT.

    Parameters
    ----------
    u, v : torch.Tensor, shape [B, Q], dtype floating
        Normalized source-pixel coordinates in ``[0, 1]``.
    t_src : torch.Tensor, shape [B, Q], dtype torch.long
        Source-frame indices.
    t_tgt : torch.Tensor, shape [B, Q], dtype torch.long
        Target-frame indices.
    t_cam : torch.Tensor, shape [B, Q], dtype torch.long
        Camera-frame indices used for 3D output coordinates, with shape
        ``[B, Q]``.
    """

    u: torch.Tensor
    v: torch.Tensor
    t_src: torch.Tensor
    t_tgt: torch.Tensor
    t_cam: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "QueryBatch":
        """Create a query batch from dictionary format."""
        return cls(
            value["u"], value["v"], value["t_src"], value["t_tgt"], value["t_cam"]
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "QueryBatch":
        """Move every query tensor to a device.

        Parameters
        ----------
        device : torch.device
            Destination PyTorch device.
        non_blocking : bool
            Whether eligible tensor copies may occur asynchronously.

        Returns
        -------
        QueryBatch
            Query batch backed by tensors on ``device``.
        """
        return QueryBatch(
            u=self.u.to(device=device, non_blocking=non_blocking),
            v=self.v.to(device=device, non_blocking=non_blocking),
            t_src=self.t_src.to(device=device, non_blocking=non_blocking),
            t_tgt=self.t_tgt.to(device=device, non_blocking=non_blocking),
            t_cam=self.t_cam.to(device=device, non_blocking=non_blocking),
        )


@dataclass(frozen=True)
class ModelOutput(TensorRecord):
    """Predictions produced for every D4RT query.

    Parameters
    ----------
    xyz_3d : torch.Tensor, shape [B, Q, 3], dtype floating
        Predicted 3D positions in the coordinate system selected by ``t_cam``.
    uv_2d : torch.Tensor, shape [B, Q, 2], dtype floating
        Predicted normalized image coordinates.
    visibility : torch.Tensor, shape [B, Q], dtype floating
        Visibility logits.
    displacement : torch.Tensor, shape [B, Q, 3], dtype floating
        Predicted 3D displacements.
    normal : torch.Tensor, shape [B, Q, 3], dtype floating
        Predicted surface normals.
    confidence : torch.Tensor, shape [B, Q], dtype floating
        Confidence logits.
    """

    xyz_3d: torch.Tensor
    uv_2d: torch.Tensor
    visibility: torch.Tensor
    displacement: torch.Tensor
    normal: torch.Tensor
    confidence: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "ModelOutput":
        """Create typed outputs without copying prediction tensors."""
        return cls(
            value["xyz_3d"],
            value["uv_2d"],
            value["visibility"],
            value["displacement"],
            value["normal"],
            value["confidence"],
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return a shallow dictionary view of the prediction tensors."""
        return {
            "xyz_3d": self.xyz_3d,
            "uv_2d": self.uv_2d,
            "visibility": self.visibility,
            "displacement": self.displacement,
            "normal": self.normal,
            "confidence": self.confidence,
        }


@dataclass
class TargetBatch(TensorRecord):
    """Ground-truth query targets with leading shape ``[B, Q]``."""

    xyz_3d: torch.Tensor
    uv_2d: torch.Tensor
    visibility: torch.Tensor
    displacement: torch.Tensor
    normal: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "TargetBatch":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass
class TargetMask(TensorRecord):
    """Boolean validity masks with leading shape ``[B, Q]``."""

    xyz_3d: torch.Tensor
    uv_2d: torch.Tensor
    visibility: torch.Tensor
    displacement: torch.Tensor
    normal: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> "TargetMask":
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass
class CameraBatch(TensorRecord):
    """Per-frame camera calibration tensors."""

    K: torch.Tensor | None = None
    T_wc: torch.Tensor | None = None
    camera_valid: torch.Tensor | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor] | None) -> "CameraBatch":
        value = value or {}
        return cls(
            K=value.get("K"),
            T_wc=value.get("T_wc"),
            camera_valid=value.get("camera_valid"),
        )


@dataclass
class TrainingBatch(TensorRecord):
    """Typed boundary between datasets, the model, and the training loss.

    Parameters
    ----------
    video : torch.Tensor, shape [B, T, C, H, W]
        Batched video clips.
    query : QueryBatch
        Batched spatio-temporal queries.
    target : TargetBatch
        Batched ground-truth query targets.
    mask : TargetMask
        Batched validity masks for each target.
    aspect_ratio : torch.Tensor, shape [B] or None
        Batched video aspect ratios, or ``None`` if not provided by the dataset.
    camera : CameraBatch | None
        Batched camera calibration tensors, or ``None`` if not provided by the dataset.
    meta : object
        Optional metadata provided by the dataset, e.g. video path or frame indices.
    """

    video: torch.Tensor
    query: QueryBatch
    target: TargetBatch
    mask: TargetMask
    aspect_ratio: torch.Tensor | None = None
    camera: CameraBatch | None = None
    meta: object = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrainingBatch":
        """Convert the dataloader's collated mapping into a typed batch."""
        return cls(
            video=cast(torch.Tensor, value["video"]),
            query=QueryBatch.from_mapping(
                cast(Mapping[str, torch.Tensor], value["query"])
            ),
            target=TargetBatch.from_mapping(
                cast(Mapping[str, torch.Tensor], value["target"])
            ),
            mask=TargetMask.from_mapping(
                cast(Mapping[str, torch.Tensor], value["mask"])
            ),
            aspect_ratio=cast(torch.Tensor | None, value.get("aspect_ratio")),
            camera=CameraBatch.from_mapping(
                cast(Mapping[str, torch.Tensor] | None, value.get("camera"))
            ),
            meta=value.get("meta"),
        )

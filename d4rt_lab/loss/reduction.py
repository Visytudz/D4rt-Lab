"""Masked reductions used by D4RT losses."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Average values over valid mask entries."""
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(eps)
    return (values * mask_f).sum() / denom


def masked_mean_per_sample(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a mask-aware mean independently for each batch item."""
    if values.ndim != mask.ndim:
        raise ValueError(f"values/mask ndim mismatch: {values.shape} vs {mask.shape}")
    if values.ndim == 1:
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)
    if values.ndim < 2:
        raise ValueError(f"Expected at least [B, N], got {values.shape}")
    mask_f = mask.to(dtype=values.dtype)
    reduce_dims = tuple(range(1, values.ndim))
    denom = mask_f.sum(dim=reduce_dims, keepdim=True).clamp_min(eps)
    return (values * mask_f).sum(dim=reduce_dims, keepdim=True) / denom

"""Optimizer and scheduler construction."""

from __future__ import annotations

from collections.abc import Iterable
import torch

from .config import OptimizerConfig


def configure_optimization(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_cfg: OptimizerConfig,
    total_steps: int,
) -> dict[str, object]:
    """Create AdamW with linear warmup followed by cosine decay."""
    learning_rate = optimizer_cfg.learning_rate
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate.peak_lr),
        weight_decay=float(optimizer_cfg.weight_decay),
    )
    warmup_steps = min(int(learning_rate.warmup_steps), total_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=float(learning_rate.final_lr),
    )
    if warmup_steps:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            total_iters=warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_steps]
        )
    else:
        scheduler = cosine
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }

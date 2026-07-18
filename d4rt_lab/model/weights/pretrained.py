"""VideoMAEv2 encoder checkpoint adaptation."""

from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .config import EncoderPretrainedConfig


_LOGGER = logging.getLogger(__name__)
_WRAPPER_PREFIXES = ("", "module.", "model.", "backbone.", "trunk.", "visual.", "network.", "net.")


def _unpack_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "module", "network", "net"):
            value = payload.get(key)
            if isinstance(value, Mapping) and all(
                isinstance(name, str) and torch.is_tensor(tensor)
                for name, tensor in value.items()
            ):
                return dict(value)
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return {
                str(key): value
                for key, value in payload.items()
                if torch.is_tensor(value)
            }
    return {}


def _candidate_pretrained_keys(target_key: str) -> list[str]:
    base = [target_key]
    if target_key.startswith("patch_embed."):
        base.append(target_key.replace("patch_embed.", "patch_embed.proj.", 1))
    if target_key.startswith("final_norm."):
        base.append(target_key.replace("final_norm.", "norm.", 1))
    prefixes = ("", "module.", "model.", "encoder.", "backbone.", "trunk.", "visual.")
    return [prefix + item for item in base for prefix in prefixes]


def _find_tensor(
    source: dict[str, torch.Tensor],
    candidate_keys: list[str] | tuple[str, ...],
    expected_shape: torch.Size | None = None,
) -> torch.Tensor | None:
    for key in candidate_keys:
        for prefix in _WRAPPER_PREFIXES:
            value = source.get(prefix + key)
            if not torch.is_tensor(value):
                continue
            if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
                continue
            return value
    return None


def _resize_patch_embed_weight(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if source.ndim != 5 or len(target_shape) != 5:
        return None
    if source.shape[:2] != target_shape[:2]:
        return None
    if tuple(source.shape) == tuple(target_shape):
        return source
    out_channels, in_channels = source.shape[:2]
    resized = F.interpolate(
        source.reshape(out_channels * in_channels, 1, *source.shape[2:]).float(),
        size=tuple(int(v) for v in target_shape[2:]),
        mode="trilinear",
        align_corners=False,
    )
    return resized.reshape(out_channels, in_channels, *target_shape[2:]).to(source.dtype)


def _structured_tensor(
    target_key: str,
    target_value: torch.Tensor,
    source: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    if target_key == "patch_embed.weight":
        for key in ("encoder.patch_embed.proj.weight", "patch_embed.proj.weight", "patch_embed.weight"):
            value = _find_tensor(source, [key])
            if value is not None:
                resized = _resize_patch_embed_weight(value, target_value.shape)
                if resized is not None and tuple(resized.shape) == tuple(target_value.shape):
                    return resized
        return None
    if target_key == "patch_embed.bias":
        return _find_tensor(
            source,
            ["encoder.patch_embed.proj.bias", "patch_embed.proj.bias", "patch_embed.bias"],
            target_value.shape,
        )
    if target_key.startswith("final_norm."):
        suffix = target_key.split(".", 1)[1]
        return _find_tensor(
            source, [f"encoder.norm.{suffix}", f"norm.{suffix}", target_key], target_value.shape
        )
    if not target_key.startswith("blocks."):
        return None

    parts = target_key.split(".")
    if len(parts) < 4:
        return None
    try:
        block_id = int(parts[1])
    except ValueError:
        return None
    suffix = ".".join(parts[2:])
    bases = (f"encoder.blocks.{block_id}", f"blocks.{block_id}")
    suffix_map = {
        "norm_attn.weight": "norm1.weight",
        "norm_attn.bias": "norm1.bias",
        "attn.in_proj_weight": "attn.qkv.weight",
        "attn.out_proj.weight": "attn.proj.weight",
        "attn.out_proj.bias": "attn.proj.bias",
        "norm_ff.weight": "norm2.weight",
        "norm_ff.bias": "norm2.bias",
        "ff.0.weight": "mlp.fc1.weight",
        "ff.0.bias": "mlp.fc1.bias",
        "ff.3.weight": "mlp.fc2.weight",
        "ff.3.bias": "mlp.fc2.bias",
    }
    mapped_suffix = suffix_map.get(suffix)
    if mapped_suffix is not None:
        value = _find_tensor(source, [f"{base}.{mapped_suffix}" for base in bases], target_value.shape)
        if value is not None:
            return value
    if suffix == "attn.in_proj_bias":
        q_bias = _find_tensor(source, [f"{base}.attn.q_bias" for base in bases])
        v_bias = _find_tensor(source, [f"{base}.attn.v_bias" for base in bases])
        if q_bias is not None and v_bias is not None and tuple(q_bias.shape) == tuple(v_bias.shape):
            value = torch.cat([q_bias, torch.zeros_like(q_bias), v_bias], dim=0)
            if tuple(value.shape) == tuple(target_value.shape):
                return value
    return None


def load_pretrained_encoder(
    encoder: torch.nn.Module,
    config: EncoderPretrainedConfig,
) -> None:
    """Load a VideoMAEv2-style checkpoint into the D4RT encoder.

    Parameters
    ----------
    encoder : torch.nn.Module
        Target video encoder.
    config : EncoderPretrainedConfig
        Pretrained encoder checkpoint policy.
    """
    checkpoint_type = config.type.strip().lower()

    def fail(message: str) -> None:
        if config.must_succeed:
            raise RuntimeError(message)
        _LOGGER.warning(message)

    if checkpoint_type not in {"videomae_v2", "videomaev2"}:
        fail(f"Unsupported encoder pretrained type: {config.type!r}")
        return
    if not config.path:
        fail("Encoder pretrained path is empty.")
        return
    path = Path(config.path)
    if not path.exists():
        fail(f"Encoder pretrained path does not exist: {path}")
        return
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        fail(f"Failed to load pretrained checkpoint {path}: {exc}")
        return
    source = _unpack_state_dict(payload)
    if not source:
        fail(f"No usable state_dict found in pretrained checkpoint: {path}")
        return

    target = encoder.state_dict()
    matched: dict[str, torch.Tensor] = {}
    for key, value in target.items():
        converted = _structured_tensor(key, value, source)
        if converted is not None:
            matched[key] = converted
            continue
        for source_key in _candidate_pretrained_keys(key):
            source_value = source.get(source_key)
            if torch.is_tensor(source_value) and tuple(source_value.shape) == tuple(value.shape):
                matched[key] = source_value
                break
    if not matched:
        fail(f"No encoder tensors matched pretrained checkpoint: {path}")
        return
    core_ok = (
        "patch_embed.weight" in matched
        and "final_norm.weight" in matched
        and any(key.startswith("blocks.") for key in matched)
    )
    if not core_ok:
        fail("Pretrained checkpoint did not match the core VideoMAE encoder tensors.")
        return
    encoder.load_state_dict(matched, strict=False)
    loaded_elements = sum(value.numel() for value in matched.values())
    total_elements = sum(value.numel() for value in target.values())
    _LOGGER.info(
        "Loaded encoder pretrained tensors: %d/%d params (%d/%d elems) from %s",
        len(matched), len(target), loaded_elements, total_elements, path,
    )

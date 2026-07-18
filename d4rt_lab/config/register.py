"""Hydra ConfigStore registration."""

from hydra.core.config_store import ConfigStore

from .schema import D4RTConfig


def register_configs() -> None:
    """Register the root structured schema with Hydra."""
    ConfigStore.instance().store(name="d4rt_lab_schema", node=D4RTConfig)

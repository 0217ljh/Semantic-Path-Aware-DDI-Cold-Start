"""Unified registry: models (name -> class). Features live in data/registry (per-dataset). See docs/2-Repo_Layout."""
from __future__ import annotations

from typing import Any, Dict

# --- Models ---
_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Register a model class. Used from models/__init__.py."""
    def _reg(cls):
        _REGISTRY[name] = cls
        return cls
    return _reg


def _ensure_models_registered() -> None:
    if _REGISTRY:
        return
    import my_code.models  # noqa: F401


def load_method(name: str, cfg: dict) -> Any:
    _ensure_models_registered()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method: {name}. Registered: {list(_REGISTRY.keys())}")
    cls = _REGISTRY[name]
    return cls(cfg)

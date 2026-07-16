"""Checkpoint bundle I/O - the FROZEN-mode load side.

The runner writes a self-describing bundle {state, model, hp, task_kind,
n_classes, best_epoch, cold_best_val} (see runner._save_ckpt). The FROZEN adapter
mode reads it back to reconstruct + freeze a pre-trained backbone:

    b = load_checkpoint(hp["backbone_ckpt"])      # -> the bundle
    # adapter wrapper builds the backbone with b["hp"] / b["task_kind"] to match,
    # then: self.backbone.load_state_dict(b["state"]); freeze its parameters.
"""
from __future__ import annotations

from pathlib import Path


def load_checkpoint(path) -> dict:
    """Load a checkpoint bundle written by the runner. Returns the full dict
    (keys: state, model, hp, task_kind, n_classes, ...)."""
    import torch
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"backbone checkpoint not found: {p}")
    bundle = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or "state" not in bundle:
        raise ValueError(f"{p}: not a code-adapter checkpoint bundle (missing 'state')")
    return bundle


__all__ = ["load_checkpoint"]

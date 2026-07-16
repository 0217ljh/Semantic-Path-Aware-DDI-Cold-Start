"""Checkpoint rotation helper per CLAUDE.md '训练中间 eval / save 规范'.

Provides `rotate_checkpoints(dir, max_keep)` that deletes the oldest
intermediate checkpoints (by numeric tag in dir name) until at most
`max_keep` remain.

Naming convention: `<run_dir>/checkpoints/checkpoint-<step_or_epoch>/`
Each baseline's `.save(path)` writes the checkpoint into the given dir.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List


def list_checkpoints(ckpt_root: Path) -> List[Path]:
    """List existing `checkpoint-<N>` subdirs under `ckpt_root`, sorted by N."""
    if not ckpt_root.is_dir():
        return []
    out = []
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            try:
                n = int(p.name.split("-", 1)[1])
                out.append((n, p))
            except ValueError:
                continue
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def rotate_checkpoints(ckpt_root: Path, max_keep: int) -> int:
    """Keep only the `max_keep` latest checkpoints. Delete the rest.

    Returns the number of checkpoints deleted.
    """
    if max_keep <= 0:
        return 0
    ckpts = list_checkpoints(ckpt_root)
    if len(ckpts) <= max_keep:
        return 0
    to_delete = ckpts[:-max_keep]
    for p in to_delete:
        try:
            shutil.rmtree(p)
        except Exception:
            pass
    return len(to_delete)

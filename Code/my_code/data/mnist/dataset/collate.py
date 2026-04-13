"""MNIST collate: stack x,y and add train/val/test masks (all True)."""
from __future__ import annotations

from typing import List

import torch


def collate_torch_batch(batch: List[dict]) -> dict:
    if not batch:
        return {}
    x = torch.stack([b["x"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    B = len(batch)
    device = x.device if x.is_cuda else None
    ones = torch.ones(B, dtype=torch.bool, device=device)
    return {
        "x": x,
        "y": y,
        "train_mask": ones,
        "val_mask": ones,
        "test_mask": ones,
    }

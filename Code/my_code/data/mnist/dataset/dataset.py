"""MNIST Dataset: TorchSplitDataset; build_datasets(processed, features) -> {train, val, test}."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class TorchSplitDataset(Dataset):
    """Wrap x,y tensors per split; __getitem__ returns dict for collate."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {"x": self.x[i], "y": self.y[i]}


def build_datasets(processed: dict, features: dict = None) -> Dict[str, Dataset]:
    features = features or {}
    train = processed["train"]
    val = processed["val"]
    test = processed["test"]
    return {
        "train": TorchSplitDataset(train["x"], train["y"]),
        "val": TorchSplitDataset(val["x"], val["y"]),
        "test": TorchSplitDataset(test["x"], test["y"]),
    }

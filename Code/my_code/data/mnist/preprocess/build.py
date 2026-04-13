"""MNIST builder: raw -> processed with dataset_type=torch, train/val/test x,y tensors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from my_code.data.base import BaseDatasetBuilder

try:
    from torchvision import datasets
    from torchvision import transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False


class MNISTBuilder(BaseDatasetBuilder):
    dataset_name = "mnist"

    def build_base_processed(self, cfg_data: dict) -> Dict[str, Any]:
        if not HAS_TV:
            raise ImportError("torchvision is required for MNIST. pip install torchvision")
        root = Path("data/mnist")
        root.mkdir(parents=True, exist_ok=True)
        seed = cfg_data.get("global_seed", 42)
        torch.manual_seed(seed)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        full_train = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)

        n_train = len(full_train)
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed))
        val_size = 5000
        train_idx = perm[val_size:]
        val_idx = perm[:val_size]

        train_x = torch.stack([full_train[i][0] for i in train_idx])
        train_y = torch.tensor([full_train[i][1] for i in train_idx], dtype=torch.long)
        val_x = torch.stack([full_train[i][0] for i in val_idx])
        val_y = torch.tensor([full_train[i][1] for i in val_idx], dtype=torch.long)
        test_x = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
        test_y = torch.tensor([test_ds[i][1] for i in range(len(test_ds))], dtype=torch.long)

        return {
            "dataset_type": "torch",
            "train": {"x": train_x, "y": train_y},
            "val": {"x": val_x, "y": val_y},
            "test": {"x": test_x, "y": test_y},
            "num_classes": 10,
            "extra": {"num_classes": 10},
            "meta": {
                "dataset": "mnist",
                "global_seed": seed,
                "preprocess_version": cfg_data.get("preprocess_version", "v1"),
            },
        }

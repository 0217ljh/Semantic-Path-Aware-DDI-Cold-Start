"""Cora builder: raw -> processed/<data_key>/base.pkl."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from my_code.data.base import BaseDatasetBuilder

try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class CoraBuilder(BaseDatasetBuilder):
    dataset_name = "cora"

    def build_base_processed(self, cfg_data: dict) -> Dict[str, Any]:
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for Cora. pip install torch_geometric")
        root = Path("data/cora")
        root.mkdir(parents=True, exist_ok=True)
        dataset = Planetoid(root=str(root), name="Cora", transform=NormalizeFeatures())
        data = dataset[0]
        seed = cfg_data.get("global_seed", 0)
        if hasattr(data, "train_mask") and data.train_mask is not None:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
        else:
            import torch
            torch.manual_seed(seed)
            n = data.num_nodes
            perm = torch.randperm(n)
            train_mask = torch.zeros(n, dtype=torch.bool)
            val_mask = torch.zeros(n, dtype=torch.bool)
            test_mask = torch.zeros(n, dtype=torch.bool)
            train_mask[perm[:140]] = True
            val_mask[perm[140:640]] = True
            test_mask[perm[640:1640]] = True

        return {
            "data": {
                "x": data.x,
                "edge_index": data.edge_index,
                "y": data.y,
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
                "num_classes": dataset.num_classes,
                "num_nodes": data.num_nodes,
            },
            "splits": {
                "train": train_mask.nonzero(as_tuple=True)[0].tolist(),
                "val": val_mask.nonzero(as_tuple=True)[0].tolist(),
                "test": test_mask.nonzero(as_tuple=True)[0].tolist(),
            },
            "extra": {"num_classes": dataset.num_classes},
            "meta": {
                "dataset": cfg_data.get("dataset", "cora"),
                "global_seed": seed,
                "preprocess_version": cfg_data.get("preprocess_version", "v1"),
                "subset": cfg_data.get("subset", cfg_data.get("subset_tag", "none")),
            },
        }

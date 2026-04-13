"""Cora Dataset: PyG InMemoryDataset; build_datasets(processed, features) -> {train, val, test}."""
from __future__ import annotations

from typing import Any, Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data

def _concat_node_features(x: torch.Tensor, by_index: dict, num_nodes: int) -> torch.Tensor:
    extra_list = []
    for k, v in by_index.items():
        if k.endswith("_offsets"):
            continue
        off_key = f"{k}_offsets"
        if off_key in by_index:
            continue
        v = torch.as_tensor(v, dtype=torch.float)
        if v.dim() == 1 and v.shape[0] == num_nodes:
            extra_list.append(v.unsqueeze(1))
        elif v.dim() == 2 and v.shape[0] == num_nodes:
            extra_list.append(v)
    if not extra_list:
        return x
    return torch.cat([x] + extra_list, dim=-1)


class MyPyGDataset(InMemoryDataset):
    """PyG InMemoryDataset for single/multi graph; by_index merged into x."""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, processed=None, split="train", features=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self._processed = processed or {}
        self._split = split
        self.data_base = self._processed.get("data")
        self.split_idx = self._processed.get("splits", {}).get(split, [])
        self.by_index = (features or {}).get("by_index", {})
        self._single_graph = (
            isinstance(self.data_base, dict)
            and "x" in self.data_base
            and "edge_index" in self.data_base
        )

    def len(self):
        if self._single_graph:
            return 1
        return len(self.split_idx)

    def get(self, i):
        if self._single_graph:
            return self._get_single_graph(i)
        return self._get_multi_graph(i)

    def _download(self):
        pass

    def _process(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def _get_single_graph(self, i):
        base = self.data_base
        x = torch.as_tensor(base["x"], dtype=torch.float)
        edge_index = torch.as_tensor(base["edge_index"], dtype=torch.long)
        num_nodes = x.shape[0]
        x = _concat_node_features(x, self.by_index, num_nodes)
        data = Data(x=x, edge_index=edge_index)
        if "y" in base:
            data.y = torch.as_tensor(base["y"], dtype=torch.long)
        for key in ("train_mask", "val_mask", "test_mask", "num_classes"):
            if key in base:
                data[key] = base[key]
        return data

    def _get_multi_graph(self, i):
        gidx = int(self.split_idx[i])
        base = self.data_base[gidx]
        x = torch.as_tensor(base["x"], dtype=torch.float)
        edge_index = torch.as_tensor(base["edge_index"], dtype=torch.long)
        num_nodes = x.shape[0]
        x = _concat_node_features(x, self.by_index, num_nodes)
        data = Data(x=x, edge_index=edge_index)
        if "y" in base:
            data.y = torch.as_tensor(base["y"], dtype=torch.long)
        for k, v in self.by_index.items():
            if k.endswith("_offsets"):
                continue
            off_key = f"{k}_offsets"
            if off_key in self.by_index:
                offsets = self.by_index[off_key]
                s, e = int(offsets[gidx]), int(offsets[gidx + 1])
                data[k] = v[s:e]
            else:
                try:
                    data[k] = v[gidx]
                except (IndexError, TypeError):
                    pass
        data.gidx = gidx
        return data


def build_datasets(processed: dict, features: dict = None) -> Dict[str, Dataset]:
    features = features or {}
    root = "."
    return {
        "train": MyPyGDataset(root=root, processed=processed, split="train", features=features),
        "val": MyPyGDataset(root=root, processed=processed, split="val", features=features),
        "test": MyPyGDataset(root=root, processed=processed, split="test", features=features),
    }

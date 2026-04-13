"""Base dataset builder interface. Registration is done per-dataset in data/<name>/__init__.py via registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDatasetBuilder(ABC):
    """Interface for dataset-specific preprocessing: compute_data_key, build_base_processed."""

    dataset_name: str = ""

    def compute_data_key(self, cfg_data: dict) -> str:
        """data_key = f(dataset, split_spec, split_seed, subset, preprocess_version)."""
        dataset = cfg_data.get("dataset", self.dataset_name or "unknown")
        split_spec = cfg_data.get("split_spec", "random")
        seed = cfg_data.get("split_seed", cfg_data.get("global_seed", 42))
        subset = cfg_data.get("subset", cfg_data.get("subset_tag", "none"))
        version = cfg_data.get("preprocess_version", "v1")
        base = f"{dataset}__sp-{split_spec}__sd{seed}"
        if subset and str(subset).lower() != "none":
            base += f"__sub-{subset}"
        return f"{base}__{version}"

    @abstractmethod
    def build_base_processed(self, cfg_data: dict) -> Dict[str, Any]:
        """Build base processed dict (raw -> processed)."""
        pass

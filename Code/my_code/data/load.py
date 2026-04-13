"""Thin dispatcher: get processed, features, datasets, collate, loaders by dataset name from registry."""
from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

from my_code.data.registry import (
    get_preprocess_builder,
    get_feature_fns,
    get_dataset_builder,
    get_collate_fn,
)

logger = logging.getLogger(__name__)


def compute_data_key(cfg_data: dict) -> str:
    """data_key from registered preprocess builder."""
    name = (cfg_data.get("dataset") or "cora").strip().lower()
    builder = get_preprocess_builder(name)
    return builder.compute_data_key(cfg_data)


def load_or_build_base_processed(cfg_data: dict, rank: int = 0) -> Dict[str, Any]:
    """Load or build processed; cache at data/<dataset>/processed/<data_key>/base.pkl."""
    dataset = (cfg_data.get("dataset") or "cora").strip().lower()
    builder = get_preprocess_builder(dataset)
    data_key = builder.compute_data_key(cfg_data)
    cache_dir = Path("data") / dataset / "processed" / data_key
    done_file = cache_dir / "DONE"
    base_file = cache_dir / "base.pkl"

    if done_file.exists():
        try:
            with open(base_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            if rank == 0 and cache_dir.exists():
                shutil.rmtree(cache_dir)

    processed = builder.build_base_processed(cfg_data)
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(base_file, "wb") as f:
            pickle.dump(processed, f)
        done_file.write_text("")
    return processed


def get_or_load_features(processed: dict, cfg: dict, rank: int = 0) -> Dict[str, Any]:
    """Build or load feature cache for (dataset, feature_method). Return {} if disabled."""
    data_cfg = cfg.get("data") or {}
    feature_method = data_cfg.get("feature_method")
    if feature_method is None or (
        isinstance(feature_method, str) and feature_method.strip().lower() in ("none", "")
    ):
        return {}
    method_key = (
        (feature_method if isinstance(feature_method, str) and feature_method.strip() else None)
        or cfg.get("model", {}).get("name")
    )
    if not method_key:
        return {}
    dataset = data_cfg.get("dataset", "cora")
    build_fn, load_fn = get_feature_fns(dataset, method_key)
    if not build_fn or not load_fn:
        return {}

    data_key = compute_data_key(data_cfg)
    resource_dir = Path("data") / dataset / "processed" / data_key / "features" / method_key
    done_file = resource_dir / "DONE"

    if done_file.exists():
        try:
            out = load_fn(str(resource_dir))
            logger.info("feature: loaded from cache (dir=%s)", resource_dir)
            return out
        except Exception:
            if rank == 0 and resource_dir.exists():
                shutil.rmtree(resource_dir)

    features = build_fn(processed, cfg, str(resource_dir))
    logger.info("feature: built (dir=%s)", resource_dir)
    return features

def get_or_load_data(cfg: dict, rank: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (processed, features)."""
    processed = load_or_build_base_processed(cfg.get("data", {}), rank=rank)
    features = get_or_load_features(processed, cfg, rank=rank)
    return processed, features

def build_dataset(cfg: dict, processed: dict, features: dict = None) -> Dict[str, Dataset]:
    """Return {train, val, test} from registered dataset builder."""
    name = (cfg.get("data") or {}).get("dataset", "cora")
    build_fn = get_dataset_builder(name)
    return build_fn(processed, features or {})


def build_collate_fn(cfg: dict, features: dict = None) -> Callable:
    """Return registered collate for dataset."""
    name = (cfg.get("data") or {}).get("dataset", "cora")
    return get_collate_fn(name)


def build_loaders(
    cfg: dict,
    datasets: Dict[str, Any],
    collate_fn: Callable = None,
) -> Dict[str, DataLoader]:
    """Build DataLoaders; PyG vs torch by sample type; skip empty splits."""
    batch_size = cfg.get("training", {}).get("batch_size", 1)
    if collate_fn is None:
        collate_fn = build_collate_fn(cfg, {})
    loaders = {}
    for split, ds in datasets.items():
        n = len(ds)
        if n == 0:
            logger.warning("Dataset for split '%s' is empty (len=0); skipping DataLoader.", split)
            continue
        first_sample = ds[0]
        is_pyg_data = isinstance(first_sample, (Data, HeteroData))

        if is_pyg_data:
            loaders[split] = PyGDataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=None,
                num_workers=0,
            )
        elif isinstance(ds, Dataset):
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=collate_fn,
                num_workers=0,
            )
        else:
            raise ValueError(f"Unsupported dataset type: {type(ds)}")
    return loaders

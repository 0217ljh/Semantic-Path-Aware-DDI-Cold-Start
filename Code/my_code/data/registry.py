"""Central registry: preprocess builders, feature fns, dataset builders, collate fns. Keyed by dataset name."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

# name -> builder instance (has compute_data_key, build_base_processed)
_PREPROCESS: Dict[str, Any] = {}

# (dataset_name, method_key) -> (build_fn, load_fn)
_FEATURES: Dict[Tuple[str, str], Tuple[Callable[..., Dict[str, Any]], Callable[..., Dict[str, Any]]]] = {}

# dataset_name -> (processed, features) -> {train, val, test} Dataset
_DATASET_BUILDERS: Dict[str, Callable[..., Dict[str, Any]]] = {}

# dataset_name -> collate_fn(batch_list) -> batch_dict
_COLLATE: Dict[str, Callable] = {}


def register_preprocess(name: str, builder: Any) -> None:
    _PREPROCESS[(name or "").strip().lower()] = builder


def get_preprocess_builder(name: str) -> Any:
    key = (name or "").strip().lower()
    if key not in _PREPROCESS:
        raise KeyError(f"Unknown dataset: {name}. Registered: {list(_PREPROCESS.keys())}")
    return _PREPROCESS[key]


def register_feature(dataset_name: str, method_key: str, build_fn: Callable, load_fn: Callable) -> None:
    _FEATURES[((dataset_name or "").strip().lower(), (method_key or "").strip().lower())] = (build_fn, load_fn)


def get_feature_fns(dataset_name: str, method_key: str) -> Tuple[Any, Any]:
    key = ((dataset_name or "").strip().lower(), (method_key or "").strip().lower())
    return _FEATURES.get(key, (None, None))


def register_dataset_builder(dataset_name: str, build_fn: Callable[..., Dict[str, Any]]) -> None:
    _DATASET_BUILDERS[(dataset_name or "").strip().lower()] = build_fn


def get_dataset_builder(name: str) -> Callable[..., Dict[str, Any]]:
    key = (name or "").strip().lower()
    if key not in _DATASET_BUILDERS:
        raise KeyError(f"No dataset builder for: {name}. Registered: {list(_DATASET_BUILDERS.keys())}")
    return _DATASET_BUILDERS[key]


def register_collate(dataset_name: str, collate_fn: Callable) -> None:
    _COLLATE[(dataset_name or "").strip().lower()] = collate_fn


def get_collate_fn(name: str) -> Callable:
    key = (name or "").strip().lower()
    if key not in _COLLATE:
        raise KeyError(f"No collate for dataset: {name}. Registered: {list(_COLLATE.keys())}")
    return _COLLATE[key]

# Trigger registration of all datasets (cora, mnist)
import my_code.data.cora  # noqa: F401
import my_code.data.mnist  # noqa: F401

from my_code.data.load import (
    compute_data_key,
    load_or_build_base_processed,
    get_or_load_features,
    get_or_load_data,
    build_dataset,
    build_collate_fn,
    build_loaders,
)

__all__ = [
    "compute_data_key",
    "load_or_build_base_processed",
    "get_or_load_features",
    "get_or_load_data",
    "build_dataset",
    "build_collate_fn",
    "build_loaders",
]

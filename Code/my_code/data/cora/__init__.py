"""Cora dataset: register preprocess, feature, dataset, collate."""
from __future__ import annotations

from my_code.data.registry import (
    register_preprocess,
    register_dataset_builder,
    register_collate,
)
from my_code.data.cora.preprocess.build import CoraBuilder
from my_code.data.cora.dataset import build_datasets, collate_cora

# Trigger feature registration
import my_code.data.cora.feature  # noqa: F401

register_preprocess("cora", CoraBuilder())
register_dataset_builder("cora", build_datasets)
register_collate("cora", collate_cora)

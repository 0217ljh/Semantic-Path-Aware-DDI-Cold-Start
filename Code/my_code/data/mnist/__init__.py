"""MNIST dataset: register preprocess, dataset, collate."""
from __future__ import annotations

from my_code.data.registry import (
    register_preprocess,
    register_dataset_builder,
    register_collate,
)
from my_code.data.mnist.preprocess.build import MNISTBuilder
from my_code.data.mnist.dataset import build_datasets, collate_torch_batch

register_preprocess("mnist", MNISTBuilder())
register_dataset_builder("mnist", build_datasets)
register_collate("mnist", collate_torch_batch)

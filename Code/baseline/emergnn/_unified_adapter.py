"""EmerGNN leaf->PairDataset adapter — now a thin re-export of the SHARED adapter
`data_utils.leaf_adapter` (the legacy cores all read the same surface, so the adapter is
shared; see that module's docstring). Kept as a stable import path for the EmerGNN wrapper.
"""
from __future__ import annotations

from data_utils.leaf_adapter import _LeafDataset, _LeafSplits, make_dataset

__all__ = ["make_dataset", "_LeafDataset", "_LeafSplits"]

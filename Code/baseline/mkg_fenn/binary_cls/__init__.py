"""MKG-FENN binary task variant (DDI-existence, regime-aware cold-start; Case-B).

Re-exports the binary core + unified wrapper. Importing this package registers
``MKGFENNUnifiedBinary`` under ``("mkg_fenn", "binary")`` in the unified registry
(via ``baseline_unified``). Reuses the multiclass regime-aware cold-start machinery
(MKGFENN / MKGFENNCold + drug_sim/test_adj helpers), swapping only the task surface
(2-way head, deterministic negatives, AUPRC).
"""

from __future__ import annotations

from baseline.mkg_fenn.binary_cls.baseline import (
    BINARY_EVENT_NUM,
    MKGFENNBinaryBaseline,
    PAPER_HYPERPARAMS,
)
from baseline.mkg_fenn.binary_cls.baseline_unified import MKGFENNUnifiedBinary

__all__ = [
    "MKGFENNBinaryBaseline",
    "MKGFENNUnifiedBinary",
    "PAPER_HYPERPARAMS",
    "BINARY_EVENT_NUM",
]

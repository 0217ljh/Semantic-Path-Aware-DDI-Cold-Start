"""MKG-FENN multilabel task variant (TWOSIDES 200-label, regime-aware cold-start).

Re-exports the multilabel core + unified wrapper. Importing this package registers
``MKGFENNUnifiedMultilabel`` under ``("mkg_fenn", "multilabel")`` in the unified
registry (via ``baseline_unified``). The paper's COLD-START ALGORITHM CORE is reused
UNCHANGED from ``multi_cls`` (regime switch, cold plumbing, models, KGs); only the
task surface (fixed 200-label sigmoid + masked BCE + macro-AUPRC) is swapped.
"""

from __future__ import annotations

from baseline.mkg_fenn.multi_label_cls.baseline import (
    MKGFENNMultilabelBaseline,
    PAPER_HYPERPARAMS,
    DEFAULT_N_LABELS,
)
from baseline.mkg_fenn.multi_label_cls.baseline_unified import MKGFENNUnifiedMultilabel

__all__ = [
    "MKGFENNMultilabelBaseline",
    "MKGFENNUnifiedMultilabel",
    "PAPER_HYPERPARAMS",
    "DEFAULT_N_LABELS",
]

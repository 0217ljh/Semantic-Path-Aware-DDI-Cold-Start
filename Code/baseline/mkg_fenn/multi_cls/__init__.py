"""MKG-FENN multiclass task variant (65-event-style DDI, regime-aware cold-start).

Re-exports the multiclass core + cold model + unified wrapper. Importing this
package registers ``MKGFENNUnifiedMulticlass`` under ``("mkg_fenn", "multiclass")``
in the unified registry (via ``baseline_unified``).
"""

from __future__ import annotations

from baseline.mkg_fenn.multi_cls.baseline import (
    MKGFENNMulticlassBaseline,
    PAPER_HYPERPARAMS,
    find_dif,
    jaccard,
)
from baseline.mkg_fenn.multi_cls.baseline_unified import MKGFENNUnifiedMulticlass
from baseline.mkg_fenn.multi_cls.model_cold import MKGFENNCold

__all__ = [
    "MKGFENNMulticlassBaseline",
    "MKGFENNUnifiedMulticlass",
    "MKGFENNCold",
    "PAPER_HYPERPARAMS",
    "find_dif",
    "jaccard",
]

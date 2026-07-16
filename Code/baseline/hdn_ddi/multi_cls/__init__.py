"""HDN-DDI multi-class classification task (per-pair: predict which of K types).

NOT in the original paper (paper uses triplet-binary natively); this is a
multi-class adaptation we added so HDN-DDI can compete on K-way DDI
prediction benchmarks.

Contains:

* :class:`HDNDDIMulticlassBaseline` (registered as ``"hdn_ddi_mc"``):
    same encoder as binary variant (BRICS-aware, y==1 bipartite, LambdaLR
    scheduler all inherited), head swapped to K-dim softmax output.
* :class:`HDN_DDI_MC`: the K-way model subclass.
"""

from __future__ import annotations

from baseline.hdn_ddi.multi_cls.baseline import HDNDDIMulticlassBaseline
from baseline.hdn_ddi.multi_cls.model import HDN_DDI_MC

__all__ = ["HDNDDIMulticlassBaseline", "HDN_DDI_MC"]

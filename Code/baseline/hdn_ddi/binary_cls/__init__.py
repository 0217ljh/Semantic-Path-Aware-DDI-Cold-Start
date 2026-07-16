"""HDN-DDI binary classification task (per-pair: predict existence).

Matches the original paper formulation (Sun & Zheng 2025, BMC
Bioinformatics): triplet-binary `(drug_a, rel, drug_b) -> {0, 1}` with
sigmoid + BCE.

Single paper-faithful BRICS-aware variant:

* :class:`HDNDDIBaseline` (registered as ``"hdn_ddi"``):
    refined BRICS 3-level graph + y==1 substructure bipartite +
    66-dim Table-S1 features + LambdaLR(0.96^epoch) scheduler +
    per-epoch dynamic negatives.  Old flat (non-paper-faithful)
    variant was removed.
"""

from __future__ import annotations

from baseline.hdn_ddi.binary_cls.baseline import HDNDDIBaseline

__all__ = ["HDNDDIBaseline"]

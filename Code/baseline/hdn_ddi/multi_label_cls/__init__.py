"""HDN-DDI MULTILABEL classification task (TWOSIDES, fixed L side-effect labels).

Case-B adaptation (CLAUDE.md §"从 reproduction 派生 baseline" case B): the
paper never did multilabel, but the paper's ALGORITHM CORE (BRICS 3-level
hierarchical molecular-graph encoder + y==1 substructure bipartite + co-
attention + RESCAL all-relation head) is reused UNCHANGED from the multiclass
core. Only the head activation (sigmoid), loss (masked BCE), targets (fixed
n_labels multihot), and val metric (macro-AUPRC) are swapped.

Contains:

* :class:`HDNDDIMultilabelBaseline` (registered as ``"hdn_ddi_ml"``):
    subclasses the multiclass core; all BRICS-aware paper contributions
    (3-level graph, y==1 bipartite, co-attention, RESCAL, LambdaLR scheduler)
    inherited, head scored with sigmoid over a fixed n_labels column space.
* :class:`HDNDDIUnifiedMultilabel` (registered as ``"hdn_ddi"``,
    task="multilabel"): the unified-benchmark wrapper.
"""

from __future__ import annotations

from baseline.hdn_ddi.multi_label_cls.baseline import HDNDDIMultilabelBaseline
from baseline.hdn_ddi.multi_label_cls.baseline_unified import HDNDDIUnifiedMultilabel

__all__ = ["HDNDDIMultilabelBaseline", "HDNDDIUnifiedMultilabel"]

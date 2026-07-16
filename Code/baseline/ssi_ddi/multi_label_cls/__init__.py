"""SSI-DDI MULTILABEL classification task (TWOSIDES, fixed L side-effect labels).

Case-B adaptation (CLAUDE.md §"从 reproduction 派生 baseline" case B): the
paper never did multilabel, but the paper's ALGORITHM CORE (substructure-aware
GAT blocks + co-attention + RESCAL all-relation head) is reused UNCHANGED from
the multiclass core. Only the head activation (sigmoid), loss (masked BCE),
targets (fixed n_labels multihot), and val metric (macro-AUPRC) are swapped.

Contains:

* :class:`SSIDDIMultilabelBaseline` (registered as ``"ssi_ddi_ml"``):
    subclasses the multiclass core; all SSI-DDI paper contributions (per-drug
    PyG mol-graphs, GAT blocks, co-attention, RESCAL) inherited, head scored
    with sigmoid over a fixed n_labels column space.
* :class:`SSIDDIUnifiedMultilabel` (registered as ``"ssi_ddi"``,
    task="multilabel"): the unified-benchmark wrapper.
"""

from __future__ import annotations

from baseline.ssi_ddi.multi_label_cls.baseline import SSIDDIMultilabelBaseline
from baseline.ssi_ddi.multi_label_cls.baseline_unified import SSIDDIUnifiedMultilabel

__all__ = ["SSIDDIMultilabelBaseline", "SSIDDIUnifiedMultilabel"]

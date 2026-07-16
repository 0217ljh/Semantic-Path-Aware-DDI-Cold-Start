"""KnowDDI baseline (KG-subgraph graph-structure-learning DDI) for the unified benchmark.

Per-task unified wrappers live under ``multi_cls/`` (multiclass DrugBank), ``multi_label_cls/``
(TWOSIDES / BioSNAP branch), and ``binary_cls/`` (Case-B GraIL-style binary), imported by the
runner by module path. The ported KnowDDI core is under ``model/`` (GraphSAGE-once + GSL +
head), ``data_processor/`` (enclosing-subgraph extraction + directional ``extract_r_digraph``
pruning), ``manager/`` (Trainer + Evaluator_multiclass + Evaluator_multilabel), and
``utils/`` — all independent DGL-2.x copies of the reproduction. On-disk data builders are
``_data/necessary/build_knowddi_data.py`` (multiclass), ``build_knowddi_twoside.py``
(multilabel), and ``build_knowddi_binary_data.py`` (binary).

Top-level re-export of all task wrappers (mirrors the SumGNN package shape).
"""
from __future__ import annotations

from .binary_cls.baseline_unified import KnowDDIUnifiedBinary
from .multi_cls.baseline_unified import KnowDDIUnifiedMulticlass
from .multi_label_cls.baseline_unified import KnowDDIUnifiedMultilabel

__all__ = ["KnowDDIUnifiedMulticlass", "KnowDDIUnifiedMultilabel", "KnowDDIUnifiedBinary"]

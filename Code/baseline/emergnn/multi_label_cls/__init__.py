"""EmerGNN multilabel task (TWOSIDES, 200 side-effect labels).

Ported from the TWOSIDES reproduction (LARS-research/EmerGNN, TWOSIDES/) as an
independent core — see ``model.py`` (fixed 200-output head) and ``_core_twoside.py``
(KG build + per-epoch shuffle_train + paired-pos/neg BCE loop). The unified wrapper
is ``baseline_unified.EmerGNNUnifiedMultilabel`` (registered as ``"emergnn"``,
task="multilabel").

Re-exports only the unified multilabel wrapper (this task subfolder), per the
baseline task-subfolder convention (CLAUDE.md §Baseline 规范 §3).
"""
from __future__ import annotations

from baseline.emergnn.multi_label_cls.baseline_unified import EmerGNNUnifiedMultilabel

__all__ = ["EmerGNNUnifiedMultilabel"]

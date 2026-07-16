"""EmerGNN binary baseline (official, multimode 3-sub-model dispatch).

Promoted 2026-05-20 from the candidate ``binary_cls_multimode`` variant
after winning the seed-42 comparison against ``binary_cls_kgonly``. See
``baseline.py`` docstring for design rationale and ``_results/`` for the
head-to-head numbers.
"""
from __future__ import annotations

from baseline.emergnn.binary_cls.baseline import EmerGNNBaseline

__all__ = ["EmerGNNBaseline"]

"""DEPRECATED 2026-05-20 — EmerGNN binary baseline Method B (kgonly).

Method B candidate that dropped ``shuffle_train`` and used base_kg-only for
both training and eval. Compared head-to-head against Method A (multimode)
on seed 42; Method A won. Class kept here as a reference experiment.

See ``baseline.py`` docstring for design rationale and
``baseline/emergnn/_results/2026-05-20__binary_cls__seed42__final.md`` for
the comparison numbers.
"""
from __future__ import annotations

from baseline.emergnn.deprecate.binary_cls_kgonly.baseline import (
    EmerGNNKGOnlyBaseline,
)

__all__ = ["EmerGNNKGOnlyBaseline"]

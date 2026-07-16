"""EmerGNN baseline (Zhang et al., Nat Comp Sci 2023; pure-PyTorch reimpl).

Layout — task-separated subfolders:

  emergnn/
  ├── __init__.py            ← this file (re-exports for external code)
  ├── _per_mode.py           ← internal per-mode training helper
  │                             (_PerModeEmerGNN, NOT a registered baseline)
  ├── model.py               ← shared base EmerGNN model class
  ├── kg_builder.py          ← shared KG construction utils (paper-style + merged)
  ├── kg_builder_merged.py
  ├── morgan_features.py     ← shared Morgan FP featurizer
  ├── _shared.py             ← shared setup helpers
  ├── _data/                 ← baseline-specific aux data
  ├── _results/              ← experiment result reports
  ├── _reviews/              ← code review reports
  ├── binary_cls/            ← OFFICIAL binary baseline (multimode 3-sub-model)
  │   ├── __init__.py
  │   └── baseline.py        ← EmerGNNBaseline (registered as "emergnn")
  ├── multi_cls/             ← per-pair multi-class task (matches paper DrugBank)
  │   ├── __init__.py
  │   ├── baseline.py        ← EmerGNNMulticlassBaseline (registered as "emergnn_mc")
  │   └── model.py           ← EmerGNN_MC subclass (replaces output head)
  └── deprecate/             ← old / experimental variants kept for reference
      ├── binary_cls_legacy_buggy/   ← original single-S2-mode binary, broken
      ├── binary_cls_kgonly/         ← Method B candidate (no shuffle_train)
      └── _smoke_save_load.py        ← one-off save/load test

History — binary baseline 2026-05-20 promotion:
    The earlier binary baseline at ``binary_cls/`` (now in deprecate/)
    had a hardcoded ``shuffle_train_mode="S2"`` that caused degenerate
    AUC=0.5 on test_s0/s1 (see ``_results/2026-05-18__binary_cls__seed42.md``).
    Two candidate fixes were evaluated on seed 42:
      - Method A (multimode): 3 sub-models per S0/S1/S2 (paper-faithful)
      - Method B (kgonly):    drop shuffle_train + base_kg-only KG
    Method A won: s0=0.99 > 0.91, s1=0.83 > 0.81, s2 tied. Promoted to
    the official binary baseline. Method B kept in deprecate/ as reference.

Importing this submodule registers the public task variants:
  - :class:`EmerGNNBaseline`            under ``"emergnn"``     (binary, multimode)
  - :class:`EmerGNNMulticlassBaseline`  under ``"emergnn_mc"``  (multi-class, K-way)
"""
from __future__ import annotations

from baseline.emergnn.binary_cls.baseline import EmerGNNBaseline
from baseline.emergnn.multi_cls.baseline import EmerGNNMulticlassBaseline

# Side-effect import: trigger the @register("emergnn-kgonly") decorator in
# the deprecated kgonly variant so the registry is populated consistently
# whenever the user imports `baseline.emergnn`. Without this, the deprecated
# baseline would only register itself if/when someone happens to import
# its module directly (e.g., run_baseline.py's deprecated branch).
# The deprecated class is NOT re-exported in __all__ — registry slot only.
from baseline.emergnn.deprecate.binary_cls_kgonly.baseline import (  # noqa: F401
    EmerGNNKGOnlyBaseline as _EmerGNNKGOnlyBaseline_for_registry,
)

__all__ = ["EmerGNNBaseline", "EmerGNNMulticlassBaseline"]

"""MRCGNN multiclass baseline for the UNIFIED benchmark.

One leaf = one regime -> a SINGLE MRCGNN multiclass model
(:class:`baseline.mrcgnn.multi_cls.baseline.MRCGNNMulticlassBaseline`: multi-relational
RGCN over the leaf's train DDI-event graph + DGI-style contrastive learning + TrimNet
molecular skip, positives-only CE over the GLOBAL K-class axis, best checkpoint by
macro-F1). KG-FREE (molecular + DDI graph), so applies to every dataset.

`predict` returns (n, n_labels_GLOBAL): the core already emits (n, K_global) columns in
GLOBAL class-id order (it trains a K_global-way head; `_idx_to_ddi_type` is the identity
`['0',...,'K-1']`), so the scatter loop below is a pass-through — kept verbatim from the
SSI-DDI multiclass wrapper for a uniform contract. Test rows whose gold class was unseen
in train still map to a valid global column (the head spans all K); the uniform-1/K
fallback from the core handles pairs whose drug lacks a molecular graph.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.mrcgnn.multi_cls.baseline import MRCGNNMulticlassBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]


@register_unified("mrcgnn")
class MRCGNNUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, n_epochs: int = 100, batch_size: int = 256, device: str = "auto",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.run_dir = run_dir
        self._core: MRCGNNMulticlassBaseline | None = None
        self._n_global: int | None = None

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._n_global = int(resources.meta["labels"]["n_labels"])
        self._core = MRCGNNMulticlassBaseline(
            n_classes=self._n_global, n_epochs=self.n_epochs,
            batch_size=self.batch_size, device=self.device,
            fold=resources.meta.get("split_code", 0), run_dir=self.run_dir)
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="multiclass")
        self._core.fit(ds, ds)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        p = self._core.predict_proba(test_df[_PAIR])          # (n, K_train == K_global)
        out = np.zeros((len(test_df), self._n_global), dtype=np.float32)
        for i, t in enumerate(self._core._idx_to_ddi_type):   # dense idx -> global id (identity)
            out[:, int(t)] = p[:, i]
        return out


__all__ = ["MRCGNNUnifiedMulticlass"]

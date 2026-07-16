"""SSI-DDI multiclass baseline for the UNIFIED benchmark.

One leaf = one regime -> a SINGLE SSI-DDI multiclass model
(:class:`baseline.ssi_ddi.multi_cls.baseline.SSIDDIMulticlassBaseline`: all-relation
RESCAL scoring, positives-only CE over the train-observed ddi_type vocab, best
checkpoint by macro-F1). KG-FREE (molecular), so applies to every dataset.

`predict` returns (n, n_labels_GLOBAL): the core's train-vocab class probabilities
scattered back to their global class indices (via `_idx_to_ddi_type`), so the runner's
argmax compares directly to the leaf's global `y_cls`. Global classes UNSEEN in train
get 0 mass (the output matrix is zero-initialized and only train-vocab columns are
filled), so a test row whose gold class was unseen-in-train can never be argmax-selected
-> counted wrong (surfaced via the runner's oov_target_rate). Mirrors the EmerGNN
multiclass unified wrapper.
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
from baseline.ssi_ddi.multi_cls.baseline import SSIDDIMulticlassBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]


@register_unified("ssi_ddi")
class SSIDDIUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 256, device: str = "auto",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.run_dir = run_dir
        self._core: SSIDDIMulticlassBaseline | None = None
        self._n_global: int | None = None

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._n_global = int(resources.meta["labels"]["n_labels"])
        self._core = SSIDDIMulticlassBaseline(
            n_classes=self._n_global, n_epochs=self.n_epochs,
            batch_size=self.batch_size, device=self.device)
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="multiclass")
        self._core.fit(ds, ds)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        p = self._core.predict_proba(test_df[_PAIR])          # (n, K_train)
        out = np.zeros((len(test_df), self._n_global), dtype=np.float32)
        for i, t in enumerate(self._core._idx_to_ddi_type):   # train dense idx -> global id
            out[:, int(t)] = p[:, i]
        return out


__all__ = ["SSIDDIUnifiedMulticlass"]

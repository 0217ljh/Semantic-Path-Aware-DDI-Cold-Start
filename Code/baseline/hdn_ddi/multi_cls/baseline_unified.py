"""HDN-DDI multiclass baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

One leaf = one regime -> a SINGLE HDN-DDI multiclass model (`HDNDDIMulticlassBaseline`,
positives-only CE over the train-observed ddi_type vocab, BRICS 3-level graph + y==1
bipartite + LambdaLR scheduler inherited from the binary core). NO cross-regime routing.
Core reused UNCHANGED via the shared adapter (task="multiclass" feeds a `ddi_type` column
= y_cls; the core builds its train vocab from train uniques). HDN-DDI is KG-FREE, so no
merged-KG path is required — it auto-builds its mol-graph pkl from drugs[[drugbank_id,
smiles]].

`predict` returns (n, n_labels_GLOBAL): the core's train-vocab class probabilities scattered
back to their global class indices via `core._idx_to_ddi_type` (whose entries are the string
form of the GLOBAL y_cls ids, since the adapter feeds ddi_type = y_cls). Test rows whose gold
class was unseen-in-train get ~0 mass at their global index -> counted wrong (codex),
surfaced via the runner's oov_target_rate.
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
from baseline.hdn_ddi.multi_cls.baseline import HDNDDIMulticlassBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]


@register_unified("hdn_ddi")
class HDNDDIUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 512, device: str = "auto",
                 log_step_every: int = 50, run_dir: str | Path | None = None,
                 **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: HDNDDIMulticlassBaseline | None = None
        self._n_global: int | None = None

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._n_global = int(resources.meta["labels"]["n_labels"])
        # n_classes is a hint; the core re-vocabs from train uniques of ddi_type (= y_cls
        # global ids) and adjusts n_classes to the observed count.
        self._core = HDNDDIMulticlassBaseline(
            n_classes=self._n_global, n_epochs=self.n_epochs, batch_size=self.batch_size,
            device=self.device, log_step_every=self.log_step_every,
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
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


__all__ = ["HDNDDIUnifiedMulticlass"]

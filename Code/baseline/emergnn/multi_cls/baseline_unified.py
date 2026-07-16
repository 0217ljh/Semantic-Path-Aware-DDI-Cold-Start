"""EmerGNN multiclass baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

One leaf = one regime -> a SINGLE EmerGNN multiclass model (`EmerGNNMulticlassBaseline`,
positives-only CE over the train-observed ddi_type vocab; per-epoch shuffle_train),
`shuffle_train_mode` = leaf split_code. NO cross-regime routing. Core reused UNCHANGED via
the shared adapter (task="multiclass" feeds a `ddi_type` column = y_cls; codex: feed the
GLOBAL class id and let the core build its train vocab from train uniques).

`predict` returns (n, n_labels_GLOBAL): the core's train-vocab class probabilities scattered
back to their global class indices, so the runner's argmax compares directly to the leaf's
global `y_cls`. Test rows whose gold class was unseen-in-train get ~0 mass at their global
index -> counted wrong (codex), surfaced via the runner's oov_target_rate.

EmerGNN is KG-based -> applies to KG datasets (drugbank_latest_full/partial via merged KG).
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
from baseline.emergnn._backend import get_backend  # noqa: E402
from baseline.emergnn.multi_cls.baseline import EmerGNNMulticlassBaseline  # noqa: E402
from baseline.emergnn.binary_cls.baseline_unified import _MODE_CFG, MERGED_EDGES  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]


@register_unified("emergnn")
class EmerGNNUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, n_dim: int = 64, length: int = 3, n_epochs: int = 100,
                 learning_rate: float = 1e-3, device: str = "auto",
                 log_step_every: int = 50, run_dir: str | Path | None = None,
                 **_ignored) -> None:
        self.n_dim = n_dim
        self.length = length
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: EmerGNNMulticlassBaseline | None = None
        self._n_global: int | None = None

    @staticmethod
    def _merged_kg_path(resources: "LeafResources") -> Path:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("EmerGNN requires a KG; this leaf declares none.")
        edges = Path(kg.source) / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        split_code = resources.meta["split_code"]
        cfg = _MODE_CFG[split_code]
        self._n_global = int(resources.meta["labels"]["n_labels"])
        backend = get_backend()
        if backend == "rspmm":
            from baseline.emergnn.multi_cls.baseline_rspmm import EmerGNNMulticlassBaseline_RSPMM
            core_cls = EmerGNNMulticlassBaseline_RSPMM
        else:
            core_cls = EmerGNNMulticlassBaseline
        print(f"[emergnn_mc] backend={backend} core={core_cls.__name__}", flush=True)
        self._core = core_cls(
            n_classes=self._n_global, n_dim=self.n_dim, length=self.length,
            n_epochs=self.n_epochs, learning_rate=self.learning_rate, device=self.device,
            log_step_every=self.log_step_every, run_dir=self.run_dir,
            backbone_kg_source="merged", merged_kg_path=self._merged_kg_path(resources),
            shuffle_train_mode=split_code, feat=cfg["feat"],
            batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"])
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


__all__ = ["EmerGNNUnifiedMulticlass"]

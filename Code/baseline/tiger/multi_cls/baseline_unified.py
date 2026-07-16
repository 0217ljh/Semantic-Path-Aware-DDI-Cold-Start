"""TIGER multiclass baseline for the UNIFIED benchmark (decision B).

One leaf = one regime -> a SINGLE `TIGERMulticlassBaseline` (dual-channel molecular
GraphTransformer + BKG subgraph channel, relation-aware attention, MI loss; positives-only
CE over the train-observed ddi_type vocab, `fc2` head sized to K_train). Core reused
UNCHANGED via the shared adapter (task="multiclass" feeds a `ddi_type` column = y_cls; the
core re-vocabs to observed classes -> n_classes=K_train). cold_start_patch OFF, extractor
randomWalk (paper-faithful). BKG defaults to the FULL merged KG via TIGER_KG_SCOPE (default
"full", standing decision 2026-07-01).

`predict` returns (n, n_labels_GLOBAL): the core outputs (n, K_train) softmax over dense
train-vocab classes; we scatter each dense column back to its GLOBAL y_cls index via
`core._idx_to_ddi_type` (entries are the string form of the global y_cls, since the adapter
feeds ddi_type = y_cls). Test rows whose gold class was unseen-in-train get ~0 mass ->
counted wrong (codex convention; surfaced via the runner's oov_target_rate). Mirrors the
EmerGNN / HDN-DDI / SumGNN multiclass unified wrappers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.tiger.multi_cls.baseline import TIGERMulticlassBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"


@register_unified("tiger")
class TIGERUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 64, device: str = "auto",
                 extractor: str = "randomWalk", cold_start_patch: bool = False,
                 log_step_every: int = 50, run_dir: str | Path | None = None,
                 **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.extractor = extractor
        self.cold_start_patch = cold_start_patch
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: TIGERMulticlassBaseline | None = None
        self._n_global: int | None = None

    @staticmethod
    def _merged_kg_path(resources: "LeafResources") -> Path:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("TIGER requires a KG; this leaf declares none.")
        edges = Path(kg.source) / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

    @staticmethod
    def _shim_test_s2(ds) -> None:
        """Dead-code guard: core reads splits.test_s2 in an unused all_drugs union; empty
        frame is faithful (g1∪g2 already cover every leaf drug). See binary wrapper."""
        if not hasattr(ds.splits, "test_s2"):
            ds.splits.test_s2 = pd.DataFrame(columns=_PAIR)

    def fit(self, train_df, val_df=None, *, resources: "LeafResources") -> None:
        self._n_global = int(resources.meta["labels"]["n_labels"])
        # n_classes is a hint; the core re-vocabs to the train-observed ddi_type count.
        self._core = TIGERMulticlassBaseline(
            n_classes=self._n_global, kg_source="merged",
            merged_kg_path=self._merged_kg_path(resources),
            extractor=self.extractor, cold_start_patch=self.cold_start_patch,
            n_epochs=self.n_epochs, batch_size=self.batch_size, device=self.device,
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="multiclass")
        self._shim_test_s2(ds)
        self._core.fit(ds, ds)

    def predict(self, test_df) -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        p = self._core.predict_proba(test_df[_PAIR])          # (n, K_train)
        out = np.zeros((len(test_df), self._n_global), dtype=np.float32)
        for i, t in enumerate(self._core._idx_to_ddi_type):   # dense idx -> global y_cls id
            out[:, int(t)] = p[:, i]
        return out


__all__ = ["TIGERUnifiedMulticlass"]

"""EmerGNN binary baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

One leaf = one regime, so this trains a SINGLE `_PerModeEmerGNN` whose `shuffle_train_mode`
= the leaf's split_code (S0/S1/S2) — NO cross-regime routing (that was only needed when
the legacy bundle mixed all three settings into one prediction surface). The paper-faithful
core (`_PerModeEmerGNN`: KG message passing, per-epoch shuffle_train, Morgan/learned feats,
ReduceLROnPlateau, sum-reduction BCE) is reused UNCHANGED via `_unified_adapter`.

EmerGNN is KG-based, so it is applicable only to datasets whose leaf declares a KG
(drugbank_latest_full/partial via the merged KG; twoside via its KG). KG-free datasets
(deng/ryu) are out of scope for this baseline.

Per-regime hyperparameter bundles are the paper's dispatch (evaluate.py:run_model):
  S0 -> feat='E' (learned), batch 128, weight_decay 1e-6
  S1 -> feat='M' (Morgan),  batch 32,  weight_decay 1e-8
  S2 -> feat='M' (Morgan),  batch 32,  weight_decay 1e-8
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
from baseline.emergnn._per_mode import _PerModeEmerGNN  # noqa: E402
from baseline.emergnn import _unified_adapter as _adapt  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

    from data_utils.unified_loader import LeafResources

MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
#: per-split-code (mode, feat, batch_size, weight_decay) — paper dispatch bundles.
_MODE_CFG = {
    "S0": dict(feat="E", batch_size=128, weight_decay=1e-6),
    "S1": dict(feat="M", batch_size=32, weight_decay=1e-8),
    "S2": dict(feat="M", batch_size=32, weight_decay=1e-8),
}


@register_unified("emergnn")
class EmerGNNUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, n_dim: int = 64, length: int = 3, n_epochs: int = 100,
                 learning_rate: float = 1e-3, device: str = "auto",
                 log_step_every: int = 50, run_dir: str | Path | None = None) -> None:
        self.n_dim = n_dim
        self.length = length
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: _PerModeEmerGNN | None = None

    @staticmethod
    def _merged_kg_path(resources: "LeafResources") -> Path:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("EmerGNN requires a KG; this leaf declares none "
                             "(KG-free dataset is out of scope for emergnn).")
        edges = Path(kg.source) / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        split_code = resources.meta["split_code"]            # S0 | S1 | S2
        cfg = _MODE_CFG[split_code]
        # Backend dispatch (EMERGNN_BACKEND=chunk|rspmm). Same constructor for
        # both cores (rspmm core subclasses the chunk core); only the message-
        # passing kernel + KG representation differ. Recorded for provenance.
        backend = get_backend()
        if backend == "rspmm":
            from baseline.emergnn._per_mode_rspmm import _PerModeEmerGNN_RSPMM
            core_cls = _PerModeEmerGNN_RSPMM
        else:
            core_cls = _PerModeEmerGNN
        print(f"[emergnn] backend={backend} core={core_cls.__name__}", flush=True)
        self._core = core_cls(
            n_dim=self.n_dim, length=self.length, n_epochs=self.n_epochs,
            learning_rate=self.learning_rate, device=self.device,
            log_step_every=self.log_step_every, run_dir=self.run_dir,
            backbone_kg_source="merged", merged_kg_path=self._merged_kg_path(resources),
            shuffle_train_mode=split_code, feat=cfg["feat"],
            batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"],
        )
        ds = _adapt.make_dataset(train_df, val_df if val_df is not None else train_df,
                                 resources)
        self._core.fit(ds, ds)        # ds serves as both train and val provider

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[["drug_a_id", "drug_b_id"]])


__all__ = ["EmerGNNUnifiedBinary"]

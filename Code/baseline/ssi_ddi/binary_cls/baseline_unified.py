"""SSI-DDI binary baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

SSI-DDI is a molecular substructure model (substructure-aware GAT + co-attention RESCAL);
KG-FREE, so it applies to EVERY dataset (drugbank_latest_full/partial, deng, ryu). One
leaf = one regime; the paper-faithful core `SSIDDIBaseline` is reused UNCHANGED via the
shared `data_utils.leaf_adapter` (SSI-DDI reads the same legacy PairDataset surface as
EmerGNN minus the KG). Negatives are the leaf's materialized y_bin==0 rows.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.ssi_ddi.baseline import SSIDDIBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from data_utils.unified_loader import LeafResources


@register_unified("ssi_ddi")
class SSIDDIUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 256, device: str = "auto",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self._core: SSIDDIBaseline | None = None

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._core = SSIDDIBaseline(n_epochs=self.n_epochs, batch_size=self.batch_size,
                                    device=self.device)
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="binary")
        self._core.fit(ds, ds)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[["drug_a_id", "drug_b_id"]])


__all__ = ["SSIDDIUnifiedBinary"]

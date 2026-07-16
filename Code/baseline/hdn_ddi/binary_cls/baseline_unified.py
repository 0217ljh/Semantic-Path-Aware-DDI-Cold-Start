"""HDN-DDI binary baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

HDN-DDI is a molecular hierarchical-decomposition model (refined BRICS 3-level graph +
y==1 substructure bipartite + 66-dim node features + RESCAL readout); KG-FREE, so it
applies to EVERY dataset regardless of whether the leaf declares a KG. One leaf = one
regime; the paper-faithful core `HDNDDIBaseline` is reused UNCHANGED via the shared
`data_utils.leaf_adapter` (HDN-DDI reads `splits.train` POS, `get_train_negatives`,
`splits.val_s2`, `get_negatives("val_s2")`, and `drugs[[drugbank_id, smiles]]`). The core
auto-builds its 3-level hierarchical mol-graph pkl from `drugs[[drugbank_id, smiles]]` via
`_shared.ensure_mol_graphs_pkl` (canonical `_data/necessary/hdn_ddi_mol_graphs__mine.pkl`),
so no KG / merged-kg path is required. Negatives are the leaf's materialized y_bin==0 rows.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.hdn_ddi.binary_cls.baseline import HDNDDIBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]


@register_unified("hdn_ddi")
class HDNDDIUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 512, device: str = "auto",
                 log_step_every: int = 50, run_dir: str | Path | None = None,
                 **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: HDNDDIBaseline | None = None

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._core = HDNDDIBaseline(
            n_epochs=self.n_epochs, batch_size=self.batch_size, device=self.device,
            log_step_every=self.log_step_every,
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="binary")
        self._core.fit(ds, ds)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[_PAIR])


__all__ = ["HDNDDIUnifiedBinary"]

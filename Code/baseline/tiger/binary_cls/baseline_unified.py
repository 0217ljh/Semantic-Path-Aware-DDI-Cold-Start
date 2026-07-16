"""TIGER binary baseline for the UNIFIED benchmark (decision B).

TIGER (Su et al. AAAI 2024): DUAL-CHANNEL — drug MOLECULAR-graph GraphTransformer +
biomedical-KG (BKG) subgraph channel — with a relation-aware self-attention heterogeneous
graph transformer, one of 3 subgraph extractors (default `randomWalk`), and a
mutual-information (MI) loss. KG-based (BKG from the merged KG), so it applies to datasets
whose leaf declares a KG. One leaf = one regime; the paper-faithful core `TIGERBaseline`
(codex PASS 2026-05-18) is reused UNCHANGED via the shared `data_utils.leaf_adapter`.

Paper-faithful defaults: `cold_start_patch=False` (the cold-start center-node patch is a
project extension, NOT in the paper), `extractor="randomWalk"`. BKG defaults to the FULL
merged KG (standing decision 2026-07-01) via the `TIGER_KG_SCOPE` env var (default "full"),
read inside `_shared`/`kg_builder`; set `TIGER_KG_SCOPE=drug_incident` for the legacy 1-hop.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.tiger.binary_cls.baseline import TIGERBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402

if TYPE_CHECKING:
    import numpy as np

    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"


@register_unified("tiger")
class TIGERUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 64, device: str = "auto",
                 extractor: str = "randomWalk", cold_start_patch: bool = False,
                 log_step_every: int = 50, run_dir: str | Path | None = None,
                 **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.extractor = extractor              # paper: randomWalk (TIGER-DW) default
        self.cold_start_patch = cold_start_patch  # paper-faithful default = off
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: TIGERBaseline | None = None

    @staticmethod
    def _merged_kg_path(resources: "LeafResources") -> Path:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("TIGER requires a KG; this leaf declares none "
                             "(KG-free dataset is out of scope for tiger).")
        edges = Path(kg.source) / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

    @staticmethod
    def _shim_test_s2(ds) -> None:
        """The core's `_build_bkg_and_subgraphs` reads `splits.test_s2` in a DEAD all_drugs
        union (binary_cls/baseline.py:308); the unified adapter has no test_s2. The real BKG
        drug universe comes from `splits.items()` (incl. the all_drugs frame) + g1/g2, and
        g1∪g2 already covers every leaf drug, so an EMPTY test_s2 is faithful (no universe
        change). This shim only prevents the dead-code AttributeError; it touches nothing else."""
        if not hasattr(ds.splits, "test_s2"):
            ds.splits.test_s2 = pd.DataFrame(columns=_PAIR)

    def fit(self, train_df, val_df=None, *, resources: "LeafResources") -> None:
        self._core = TIGERBaseline(
            kg_source="merged", merged_kg_path=self._merged_kg_path(resources),
            extractor=self.extractor, cold_start_patch=self.cold_start_patch,
            n_epochs=self.n_epochs, batch_size=self.batch_size, device=self.device,
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="binary")
        self._shim_test_s2(ds)
        self._core.fit(ds, ds)

    def predict(self, test_df) -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[_PAIR])


__all__ = ["TIGERUnifiedBinary"]

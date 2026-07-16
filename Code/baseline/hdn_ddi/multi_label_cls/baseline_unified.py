"""HDN-DDI MULTILABEL baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

TWOSIDES-only: a FIXED n_labels-side-effect head + sigmoid + masked BCE (no
per-fold re-vocab, unlike multiclass). One leaf = one regime -> a SINGLE
HDN-DDI multilabel model (:class:`baseline.hdn_ddi.multi_label_cls.baseline.
HDNDDIMultilabelBaseline`) reusing the paper's ALGORITHM CORE UNCHANGED (BRICS
3-level hierarchical molecular-graph encoder + y==1 substructure bipartite +
co-attention + RESCAL all-relation head). Case-B adaptation: only the head
activation / loss / targets / metric are swapped. NO cross-regime routing.

HDN-DDI is KG-FREE / molecular — no merged-KG path is required; it auto-builds
its BRICS mol-graph pkl from ``drugs[[drugbank_id, smiles]]`` (TWOSIDES ships
SMILES for all 604 drugs). The only bridge to the unified benchmark is the DDI
pairs, delivered as a paired pos/neg 200-multihot bundle via
:func:`data_utils.leaf_adapter.make_multilabel_bundle` (pos_pair_id ->
neg_pair_id links loaded from the leaf's per-fold ``*_pair_links.parquet``).

Independence: imports ONLY ``baseline.hdn_ddi.*`` + ``data_utils`` (per
CLAUDE.md §"文件级独立性"); the EmerGNN multilabel wrapper was READ as a
reference pattern but is not imported.
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
from baseline.hdn_ddi.multi_label_cls.baseline import HDNDDIMultilabelBaseline  # noqa: E402
from data_utils.leaf_adapter import make_dataset, make_multilabel_bundle  # noqa: E402
from data_utils import unified  # noqa: E402

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]


def _project_root(start) -> Path:
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / "Code" / "data" / "ddi_unified").is_dir():
            return c
    return p


@register_unified("hdn_ddi")
class HDNDDIUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(self, *, n_epochs: int = 5, batch_size: int = 512,
                 device: str = "auto", log_step_every: int = 50,
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: HDNDDIMultilabelBaseline | None = None
        self._n_labels: int | None = None
        self._fold: str = "fold0"

    def _locate_pair_links(self, resources: "LeafResources", fold: str):
        """Load {train,val}_pair_links.parquet from the leaf's fold dir, resolved
        from the standard ddi_unified layout (loader keeps no path, so rebuild it
        — mirrors the EmerGNN multilabel wrapper)."""
        meta = resources.meta
        root = _project_root(_ROOT) / "Code" / "data" / "ddi_unified"
        parent = unified.layout_dir(root, meta["dataset_group"], meta["task"],
                                    meta["split_type"])
        fdir = parent / fold
        tr = pd.read_parquet(fdir / "train_pair_links.parquet")
        vp = fdir / "val_pair_links.parquet"
        va = pd.read_parquet(vp) if vp.is_file() else None
        return tr, va

    def fit_leaf(self, leaf: "Leaf") -> None:
        # capture fold (base fit_leaf drops it); multilabel needs the pair_links
        # sidecar which is per-fold and not carried by LeafResources.
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources)

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._n_labels = int(resources.meta["labels"]["n_labels"])

        tr_links, va_links = self._locate_pair_links(resources, self._fold)
        bundle = make_multilabel_bundle(
            train_df, val_df if val_df is not None else train_df,
            tr_links, va_links, n_labels=self._n_labels,
        )
        # ``ds`` only feeds the inherited ``_build_graphs`` (BRICS mol-graph
        # cache from drugs[[drugbank_id, smiles]]); the actual training targets
        # come from ``bundle``. task="multilabel" so the adapter attaches the
        # y_label_ids column for the splits surface (unused by the ml core but
        # required by make_dataset's multilabel path).
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="multilabel")

        self._core = HDNDDIMultilabelBaseline(
            n_labels=self._n_labels, n_epochs=self.n_epochs,
            batch_size=self.batch_size, device=self.device,
            log_step_every=self.log_step_every,
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
        self._core.fit(ds, bundle)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[_PAIR])       # (n, n_labels)


__all__ = ["HDNDDIUnifiedMultilabel"]

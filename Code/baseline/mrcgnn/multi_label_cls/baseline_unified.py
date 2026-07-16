"""MRCGNN MULTILABEL baseline for the UNIFIED benchmark (TWOSIDES-only, Case-B).

A FIXED ``n_labels``-side-effect head + sigmoid + masked BCE_cls (no per-fold
re-vocab, unlike multiclass). One leaf = one regime -> a SINGLE MRCGNN multilabel
model (:class:`baseline.mrcgnn.multi_label_cls.baseline.MRCGNNMultilabelBaseline`)
reusing the paper's ALGORITHM CORE UNCHANGED (2x multi-relational RGCN over the
label-exploded DDI graph + two DGI contrastive views + TrimNet molecular skip +
MLP pair head + 3-loss). Only the head width / activation / loss / targets /
metric are swapped. NO cross-regime routing.

MRCGNN is KG-FREE / molecular (DDI graph + TrimNet features), so no merged-KG
path is required; TWOSIDES ships SMILES for all 604 drugs, so TrimNet can
featurize every drug. The only bridge to the unified benchmark is the DDI pairs,
delivered as a paired pos/neg ``n_labels``-multihot bundle via
:func:`data_utils.leaf_adapter.make_multilabel_bundle` (pos_pair_id ->
neg_pair_id links loaded from the leaf's per-fold ``*_pair_links.parquet``).

Mirrors ``baseline/hdn_ddi/multi_label_cls/baseline_unified.py`` (the sibling
molecular ml wrapper): locate ``*_pair_links.parquet``, ``make_multilabel_bundle``,
``make_dataset(task="multilabel")`` for the drug universe, ``core.fit(ds, bundle)``,
``predict -> (n, n_labels)``.

Independence (CLAUDE.md §文件级独立性): imports ONLY ``baseline.mrcgnn.*`` +
``data_utils``; the EmerGNN / HDN-DDI ml wrappers were READ as reference patterns
but are not imported.
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
from baseline.mrcgnn.multi_label_cls.baseline import MRCGNNMultilabelBaseline  # noqa: E402
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


@register_unified("mrcgnn")
class MRCGNNUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(self, *, n_epochs: int = 100, batch_size: int = 256,
                 device: str = "auto", log_step_every: int = 50,
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: MRCGNNMultilabelBaseline | None = None
        self._n_labels: int | None = None
        self._fold: str = "fold0"

    def _locate_pair_links(self, resources: "LeafResources", fold: str):
        """Load {train,val}_pair_links.parquet from the leaf's fold dir, resolved
        from the standard ddi_unified layout (loader keeps no path, so rebuild it
        — mirrors the EmerGNN / HDN-DDI multilabel wrappers)."""
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
        # ``ds`` feeds the inherited ``_build_graphs`` (drug universe + molecular
        # graphs from drugs[[drugbank_id, smiles]]) + ``_drug_smiles_dict`` (TrimNet
        # input); the actual training targets come from ``bundle``.
        # task="multilabel" so the adapter attaches y_label_ids for the splits
        # surface (unused by the ml core but required by make_dataset's ml path).
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="multilabel")

        self._core = MRCGNNMultilabelBaseline(
            n_labels=self._n_labels, n_epochs=self.n_epochs,
            batch_size=self.batch_size, device=self.device,
            log_step_every=self.log_step_every,
            fold=resources.meta.get("split_code", 0),
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
        self._core.fit(ds, bundle)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        return self._core.predict_proba(test_df[_PAIR])       # (n, n_labels)


__all__ = ["MRCGNNUnifiedMultilabel"]

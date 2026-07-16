"""EmerGNN multilabel baseline for the UNIFIED benchmark (decision B, codex 019f19fb).

TWOSIDES-only: a FIXED 200-side-effect head + BCE (no per-fold re-vocab, unlike
multiclass). One leaf = one regime -> a SINGLE ported TWOSIDES EmerGNN
(:class:`baseline.emergnn.multi_label_cls._core_twoside.BaseModelTwoside`) trained
with the paper's per-epoch shuffle_train, KG from ``resources.kg.source`` (the
reproduction's TWOSIDES data dir), Morgan feats. NO cross-regime routing.

Unlike binary/multiclass (which reuse the merged DrugBank KG via ``_per_mode``),
multilabel ports the TWOSIDES core independently because TWOSIDES ships its own
per-fold KG (``train_KG.txt``) + entity/relation vocab + Morgan pkl, and the head
is a fixed 200-output sigmoid (not 1-logit binary or K-way softmax).

Per-regime hyperparameter bundles are the paper's dispatch (evaluate.py:run_model
S1/S2 override):
  S1 -> feat='M', n_dim=32, lr=1e-3,  batch 32, weight_decay 1e-6
  S2 -> feat='M', n_dim=64, lr=3e-3,  batch 64, weight_decay 1e-6
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
from baseline.emergnn._backend import get_backend  # noqa: E402
from baseline.emergnn.multi_label_cls._core_twoside import BaseModelTwoside  # noqa: E402
from data_utils.leaf_adapter import make_multilabel_bundle  # noqa: E402
from data_utils import unified  # noqa: E402

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]

#: per-split-code (n_dim, lr, batch_size, weight_decay) — paper TWOSIDES dispatch
#: (evaluate.py:57-71). feat='M' for S1/S2; length=3.
_MODE_CFG = {
    "S1": dict(n_dim=32, learning_rate=1e-3, batch_size=32, weight_decay=1e-6),
    "S2": dict(n_dim=64, learning_rate=3e-3, batch_size=64, weight_decay=1e-6),
    # S0 fallback (paper uses feat='E'); TWOSIDES leaves here are S1/S2 cold only.
    "S0": dict(n_dim=32, learning_rate=1e-2, batch_size=32, weight_decay=1e-6),
}


def _project_root(start) -> Path:
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / "Code" / "data" / "ddi_unified").is_dir():
            return c
    return p


@register_unified("emergnn")
class EmerGNNUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(self, *, length: int = 3, n_epochs: int = 100,
                 device: str = "auto", log_step_every: int = 50,
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.length = length
        self.n_epochs = n_epochs
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: BaseModelTwoside | None = None
        self._n_labels: int | None = None
        self._fold: str = "fold0"

    @staticmethod
    def _kg_source(resources: "LeafResources") -> str:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("EmerGNN multilabel requires a KG; this leaf declares none.")
        src = Path(kg.source)
        if not src.is_dir():
            raise FileNotFoundError(f"TWOSIDES KG source dir not found: {src}")
        return str(src)

    def _locate_pair_links(self, resources: "LeafResources", fold: str):
        """Load {train,val}_pair_links.parquet from the leaf's fold dir, resolved
        from the standard ddi_unified layout (loader keeps no path, so rebuild it)."""
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
        split_code = resources.meta["split_code"]
        cfg = _MODE_CFG[split_code]
        self._n_labels = int(resources.meta["labels"]["n_labels"])

        tr_links, va_links = self._locate_pair_links(resources, self._fold)
        bundle = make_multilabel_bundle(
            train_df, val_df if val_df is not None else train_df,
            tr_links, va_links, n_labels=self._n_labels,
        )

        backend = get_backend()
        if backend == "rspmm":
            from baseline.emergnn.multi_label_cls._core_twoside_rspmm import BaseModelTwoside_RSPMM
            core_cls = BaseModelTwoside_RSPMM
        else:
            core_cls = BaseModelTwoside
        print(f"[emergnn_ml] backend={backend} core={core_cls.__name__}", flush=True)
        self._core = core_cls(
            kg_source=self._kg_source(resources), split_code=split_code,
            fold=self._fold, n_labels=self._n_labels, n_dim=cfg["n_dim"],
            length=self.length, feat="M", learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
            n_epochs=self.n_epochs, device=self.device,
            log_step_every=self.log_step_every, run_dir=self.run_dir,
        )
        self._core.fit(bundle)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None:
            raise RuntimeError("fit() before predict().")
        ht = test_df[_PAIR].to_numpy().astype(np.int64)
        return self._core.predict_proba(ht)          # (n, n_labels)


__all__ = ["EmerGNNUnifiedMultilabel"]

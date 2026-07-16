"""TIGER MULTILABEL baseline for the UNIFIED benchmark (TWOSIDES, decision B).

TWOSIDES-only: a FIXED 200-side-effect head + sigmoid + masked BCE. One leaf = one
regime -> a SINGLE dual-channel :class:`baseline.tiger.multi_label_cls.baseline.
TIGERMultilabelBaseline` (mol GraphTransformer + BKG-subgraph GraphTransformer +
relation-aware attention + MI loss). Paper-faithful defaults: cold_start_patch
OFF, extractor randomWalk (TIGER-DW). BKG defaults to the FULL merged KG
(standing decision 2026-07-01) via TIGER_KG_SCOPE.

CRITICAL id-remap (why NOT ``data_utils.leaf_adapter.make_dataset``):
  * The generic ``make_dataset`` renames the unified ``drug_id`` -> ``drugbank_id``.
    That is fine for SMILES-only baselines, but WRONG for TIGER's merged-KG branch:
    TIGER connects drug nodes to KG entity nodes by their DrugBank id, and the
    merged-KG node names are DrugBank ids (``DB00813``, ...), not the TWOSIDES
    integer ``drug_id`` (``0``..``603``).
  * TWOSIDES ships 604 drugs; only 335 carry a real DrugBank id (all UNIQUE, no
    collision — verified). 269 drugs have ``drugbank_id == '<NA>'``.
  * We therefore build a TIGER-specific remapped leaf surface where each drug's
    TIGER id = its DrugBank id if present, else a synthetic stable
    ``twoside:<int_drug_id>``. The remap is applied to the drug table, the
    train/val pair endpoints, drug_split (g1/g2), and the paired pos/neg bundle
    endpoints so every id lives in the SAME TIGER-id space.
  * merged_kg_path = the canonical ``Code/data/KG/_merged_kg/...`` merged edges,
    NOT ``resources.kg.source`` (which points at TWOSIDES's own KG dir).

Data limitation (documented, not hidden): the 269 unmapped drugs (~45%) join the
merged KG only as ISOLATED self-loop nodes (built by
:mod:`baseline.tiger.kg_builder`), so their KG channel has no neighbourhood
signal; the molecular channel still covers ALL 604. Unmapped drugs are KEPT in
``drug_ids`` (NOT dropped) — dropping them would make ``_make_pair_batch`` drop
every pair that touches them, discarding most of the data.

Independence (CLAUDE.md §"文件级独立性"): imports ONLY ``baseline.tiger.*`` +
``data_utils`` + framework.
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
from baseline.tiger.multi_label_cls.baseline import TIGERMultilabelBaseline  # noqa: E402
from data_utils.leaf_adapter import _LeafDataset, _LeafSplits, _dense_multihot  # noqa: E402
from data_utils import unified  # noqa: E402

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]
#: canonical merged KG edges (NOT resources.kg.source, which is TWOSIDES's own KG).
MERGED_KG_REL = Path("Code") / "data" / "KG" / "_merged_kg"
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"


def _project_root(start) -> Path:
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / MERGED_KG_REL).is_dir():
            return c
    return p


@register_unified("tiger")
class TIGERUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(self, *, n_epochs: int = 100, batch_size: int = 64, device: str = "auto",
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
        self._core: TIGERMultilabelBaseline | None = None
        self._n_labels: int | None = None
        self._fold: str = "fold0"
        self._tiger_id: dict[str, str] | None = None   # unified drug_id -> TIGER id

    # ------------------------------------------------------------------
    # KG path + pair-links location
    # ------------------------------------------------------------------
    @staticmethod
    def _merged_kg_path(resources: "LeafResources") -> Path:
        """Canonical merged KG edges parquet (NOT resources.kg.source)."""
        root = _project_root(_ROOT)
        edges = root / MERGED_KG_REL / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

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

    # ------------------------------------------------------------------
    # id-remap
    # ------------------------------------------------------------------
    def _build_tiger_id_map(self, resources: "LeafResources") -> dict[str, str]:
        """unified drug_id (str) -> TIGER id.

        TIGER id = DrugBank id if present, else synthetic stable ``twoside:<id>``.
        FIRST verify drug_id->drugbank_id is 1-to-1 (no two twoside drugs share a
        DrugBank id); on any collision, fall back to the synthetic id for the
        colliding drugs (and log it) so no two drugs share a TIGER id.
        """
        drugs = resources.drugs
        raw = {str(r.drug_id): (None if r.drugbank_id is None else str(r.drugbank_id))
               for r in drugs[["drug_id", "drugbank_id"]].itertuples()}
        # pandas string-NA renders as the literal '<NA>'; also guard None/'nan'.
        _MISSING = {"", "<NA>", "None", "nan", "NaN"}

        def _real(bid: str | None) -> str | None:
            if bid is None:
                return None
            b = str(bid).strip()
            return None if b in _MISSING else b

        # collision check on the REAL DrugBank ids
        bid_to_drugs: dict[str, list[str]] = {}
        for did, bid in raw.items():
            rb = _real(bid)
            if rb is not None:
                bid_to_drugs.setdefault(rb, []).append(did)
        colliding = {b for b, ds in bid_to_drugs.items() if len(ds) > 1}
        if colliding:
            n_col = sum(len(bid_to_drugs[b]) for b in colliding)
            print(f"[tiger_ml] WARNING: {len(colliding)} DrugBank id(s) shared by "
                  f"{n_col} twoside drugs; falling back to synthetic id for those.",
                  file=sys.stderr)

        tiger_id: dict[str, str] = {}
        n_mapped = 0
        for did, bid in raw.items():
            rb = _real(bid)
            if rb is not None and rb not in colliding:
                tiger_id[did] = rb
                n_mapped += 1
            else:
                tiger_id[did] = f"twoside:{did}"
        n_synth = len(tiger_id) - n_mapped
        print(f"[tiger_ml] id-remap: {n_mapped}/{len(tiger_id)} drugs -> DrugBank id, "
              f"{n_synth} -> synthetic 'twoside:<id>' (isolated self-loop BKG). "
              f"~{100*n_synth/max(1,len(tiger_id)):.0f}% of twoside drugs get an "
              f"isolated BKG self-loop (data limitation).", flush=True)
        return tiger_id

    def _remap_endpoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with drug_a_id / drug_b_id in TIGER-id space."""
        out = df.copy()
        m = self._tiger_id
        out["drug_a_id"] = out["drug_a_id"].astype(str).map(m)
        out["drug_b_id"] = out["drug_b_id"].astype(str).map(m)
        return out

    def _build_remapped_ds(self, train_df, val_df, resources) -> "_LeafDataset":
        """Build a duck-typed leaf dataset whose drug ids all live in the TIGER-id
        space (DrugBank id or synthetic). ``drugs`` keyed by TIGER id with SMILES
        for ALL 604 drugs (molecular channel covers all); g1/g2 + all_drugs
        remapped so kg_builder gives unmapped drugs isolated self-loop BKG (NOT
        dropped)."""
        m = self._tiger_id
        roles = resources.drug_split
        g1 = [m[str(d)] for d in
              roles[roles["role"].isin(["train", "train_seen"])]["drug_id"].astype(str)]
        g2 = [m[str(d)] for d in
              roles[roles["role"].isin(["test", "val", "eval_unseen"])]["drug_id"].astype(str)]
        all_drugs = [m[str(d)] for d in resources.drugs["drug_id"].astype(str)]

        tr = self._remap_endpoints(train_df)
        va = self._remap_endpoints(val_df if val_df is not None else train_df)
        # multilabel task: _LeafSplits.train reads is_positive==1 + y_label_ids;
        # the endpoints are already remapped above.
        splits = _LeafSplits(tr, va, all_drugs, g1, g2, task="multilabel")

        drugs = resources.drugs[["drug_id", "smiles"]].copy()
        drugs["drug_id"] = drugs["drug_id"].astype(str).map(m)   # -> TIGER id
        drugs.columns = ["drugbank_id", "smiles"]

        ds = _LeafDataset(splits=splits, drugs=drugs, kg=None,
                          _train_neg=tr.iloc[0:0][_PAIR], _val_neg=va.iloc[0:0][_PAIR])
        # Dead-code guard: core's _build_bkg_and_subgraphs reads splits.test_s2 in
        # an unused all_drugs union; g1∪g2 already cover every leaf drug, so an
        # empty frame is faithful (mirrors the binary / multiclass wrappers).
        if not hasattr(ds.splits, "test_s2"):
            ds.splits.test_s2 = pd.DataFrame(columns=_PAIR)
        return ds

    # ------------------------------------------------------------------
    # paired pos/neg bundle in the TIGER-id (string) space
    # ------------------------------------------------------------------
    def _make_bundle(self, train_df, val_df, tr_links, va_links) -> dict:
        """Mirror of ``data_utils.leaf_adapter.make_multilabel_bundle`` but keeps
        endpoints as STRING TIGER ids (the generic builder casts to int64, which
        would break on 'DB00813' / 'twoside:12'). Uses the shared ``_dense_multihot``
        for the label vectors (neg carries the POS multihot as a MASK)."""
        m = self._tiger_id
        n_labels = self._n_labels

        def _pairs(df: pd.DataFrame, links: pd.DataFrame):
            by_id = df.set_index("pair_id")
            pos = by_id.loc[links["pos_pair_id"].to_numpy()]
            neg = by_id.loc[links["neg_pair_id"].to_numpy()]
            pos_ht = np.array(
                [[m[str(a)], m[str(b)]] for a, b in
                 zip(pos["drug_a_id"], pos["drug_b_id"])], dtype=object
            )
            neg_ht = np.array(
                [[m[str(a)], m[str(b)]] for a, b in
                 zip(neg["drug_a_id"], neg["drug_b_id"])], dtype=object
            )
            y = _dense_multihot(list(pos["y_label_ids"]), n_labels)
            return pos_ht, neg_ht, y

        tr_pos_ht, tr_neg_ht, tr_y = _pairs(train_df, tr_links)
        if val_df is not None and va_links is not None and len(va_links):
            va_pos_ht, va_neg_ht, va_y = _pairs(val_df, va_links)
        else:
            va_pos_ht = np.zeros((0, 2), object)
            va_neg_ht = np.zeros((0, 2), object)
            va_y = np.zeros((0, n_labels), np.float32)

        return {
            "train_pos_ht": tr_pos_ht, "train_pos_y": tr_y,
            "train_neg_ht": tr_neg_ht, "train_neg_y": tr_y,
            "val_pos_ht": va_pos_ht, "val_pos_y": va_y,
            "val_neg_ht": va_neg_ht, "val_neg_y": va_y,
        }

    # ------------------------------------------------------------------
    # UnifiedBaseline surface
    # ------------------------------------------------------------------
    def fit_leaf(self, leaf: "Leaf") -> None:
        # capture fold (base fit_leaf drops it); multilabel needs the pair_links
        # sidecar which is per-fold and not carried by LeafResources.
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources)

    def fit(self, train_df, val_df=None, *, resources: "LeafResources") -> None:
        self._n_labels = int(resources.meta["labels"]["n_labels"])
        self._tiger_id = self._build_tiger_id_map(resources)

        self._core = TIGERMultilabelBaseline(
            n_labels=self._n_labels, kg_source="merged",
            merged_kg_path=self._merged_kg_path(resources),
            extractor=self.extractor, cold_start_patch=self.cold_start_patch,
            n_epochs=self.n_epochs, batch_size=self.batch_size, device=self.device,
            log_step_every=self.log_step_every,
            run_dir=str(self.run_dir) if self.run_dir is not None else None,
        )
        ds = self._build_remapped_ds(train_df, val_df, resources)
        tr_links, va_links = self._locate_pair_links(resources, self._fold)
        bundle = self._make_bundle(
            train_df, val_df if val_df is not None else train_df, tr_links, va_links
        )
        self._core.fit(ds, bundle)

    def predict(self, test_df) -> "np.ndarray":
        if self._core is None or self._tiger_id is None:
            raise RuntimeError("fit() before predict().")
        m = self._tiger_id
        ht = np.array(
            [[m[str(a)], m[str(b)]] for a, b in
             zip(test_df["drug_a_id"], test_df["drug_b_id"])], dtype=object
        )
        return self._core.predict_proba(ht)          # (n, n_labels), aligned to test rows


__all__ = ["TIGERUnifiedMultilabel"]

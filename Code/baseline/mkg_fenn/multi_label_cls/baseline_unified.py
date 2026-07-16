"""MKG-FENN MULTILABEL baseline for the UNIFIED benchmark (TWOSIDES, decision B).

TWOSIDES-only: a FIXED 200-side-effect head + sigmoid + masked BCE (Case-B). One
leaf = one regime -> a SINGLE regime-aware ``MKGFENNMultilabelBaseline``:
  * S0 (transductive) -> the warm 4-channel model (``MKGFENN``, event_num=200,
    via the core's ``cold=False`` path).
  * S1/S2 (inductive) -> the cold 3-channel model + nearest-seen imputation
    (``MKGFENNCold``, event_num=200), with drug_sim1..4 + test_adj built from the
    leaf's seen/unseen drug split.

CRITICAL KG1-ON-TWOSIDES RE-KEY (the one genuinely tricky part; codex thread
019f2485-1815-73b1-b9fc-a555a161886a confirmed the approach):
  * TWOSIDES pairs use an INTEGER pool ``drug_id`` (0..603). KG2 (Morgan/SMILES),
    KG3 (train DDI), KG4 (property/SMILES) all build NATIVELY in that pool-id
    space (all 604 twoside drugs have SMILES).
  * KG1 (enzyme/target/transporter/carrier/pathway) tables are keyed by DrugBank
    id. ``build_kg1`` only keeps a drug if its ``drugbank_id`` is a key of
    ``dict1``. With ``dict1`` in the pool-id space, drugbank-keyed rows would NEVER
    match -> KG1 empty for all drugs.
  * FIX: keep ``dict1`` in the pool-id space (so KG2/3/4 stay native) and RE-KEY
    the KG1 tables' ``drugbank_id`` column drugbank_id -> pool-id BEFORE building.
    Only ~335/604 twoside drugs carry a real drugbank_id (verified, 0 collisions);
    of those, ~312 hit the filtered KG1 tables. Twoside drugs with no drugbank_id
    (or whose drugbank_id isn't in the filtered tables) get NO KG1 edges -> the
    existing ``_ghost_pad_kg1`` cold-path gives them a single ghost KG1 slot. ALL
    604 drugs stay in ``dict1`` (unmapped drugs are NOT dropped). This keeps all 4
    KGs in ONE pool-id space.
  * The crosswalk is one-to-many-safe (drugbank_id -> list of pool ids), per codex:
    if two pool drugs ever shared a drugbank_id they would both inherit the same
    KG1 rows (biologically correct; flagged). TWOSIDES has 0 collisions.

Data limitation (documented, not hidden): ~48% of twoside drugs get an isolated
ghost KG1 slot (no enzyme/target signal), analogous to TIGER's ~45% isolated-BKG
note. The molecular channels (KG2/KG4) and KG3 still cover ALL 604 drugs.

KG1 source = the native filtered tables at ``Code/data/KG/drugbank/filtered/`` via
``KnowledgeGraph.from_filtered_dir`` (drugbank-intrinsic drug side-info, cold-start
safe, shared across leaves). ``Code/data/KG/twosides`` holds only the raw decagon
DDI source, no entity tables — so the drugbank filtered dir is the right source.

Independence (CLAUDE.md §"文件级独立性"): imports ONLY ``baseline.mkg_fenn.*`` +
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
from baseline.mkg_fenn.multi_label_cls.baseline import (  # noqa: E402
    MKGFENNMultilabelBaseline, PAPER_HYPERPARAMS,
)
from data_utils.kg import KnowledgeGraph  # noqa: E402
from data_utils.leaf_adapter import make_dataset, _dense_multihot  # noqa: E402
from data_utils import unified  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]

#: Native KG1 source (drug->{enzyme,target,transporter,carrier,pathway}) — intrinsic
#: drug side-info, NOT the merged triple KG. Relative to the repo root.
_KG1_FILTERED_DIR = "Code/data/KG/drugbank/filtered"

#: filtered KG1 tables carrying a drugbank_id column (re-keyed to pool id).
_KG1_TABLE_ATTRS = ("enzymes", "targets", "transporters", "carriers", "pathways")

#: pandas string-NA / sentinel forms that count as "no drugbank id".
_MISSING = frozenset({"", "<NA>", "None", "nan", "NaN"})


def _project_root(start: Path) -> Path:
    """Walk up to the repo root (dir holding Code/data/KG)."""
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    return p


@register_unified("mkg_fenn")
class MKGFENNUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(
        self, *, n_epochs: int = 50, embedding_num: int = 128,
        neighbor_sample_size: int = 6, dropout: float = 0.3,
        learning_rate: float = 1e-2, weight_decay: float = 1e-8,
        batch_size: int = 256, seed: int = 1, device: str = "auto",
        log_step_every: int = 50, run_dir: "str | Path | None" = None,
        **_ignored,
    ) -> None:
        self.n_epochs = n_epochs
        self.embedding_num = embedding_num
        self.neighbor_sample_size = neighbor_sample_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._core: MKGFENNMultilabelBaseline | None = None
        self._n_labels: int | None = None
        self._fold: str = "fold0"
        self._dict1: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # KG1 re-key (drugbank_id -> twoside pool id)
    # ------------------------------------------------------------------
    @staticmethod
    def _real_bid(bid) -> str | None:
        """DrugBank id if real, else None (handles pandas NA / sentinel strings)."""
        if bid is None:
            return None
        b = str(bid).strip()
        return None if b in _MISSING else b

    def _build_crosswalk(self, resources: "LeafResources") -> dict[str, list[str]]:
        """drugbank_id -> [pool_drug_id, ...] (one-to-many safe, codex caveat).

        Built from ``resources.drugs`` (has both the integer pool ``drug_id`` and a
        ``drugbank_id`` present for ~335/604 twoside drugs). Collisions (one
        drugbank_id shared by >1 pool drug) are kept as a list and flagged.
        """
        drugs = resources.drugs
        cross: dict[str, list[str]] = {}
        n_real = 0
        for did, bid in zip(drugs["drug_id"].astype(str), drugs["drugbank_id"]):
            rb = self._real_bid(bid)
            if rb is None:
                continue
            cross.setdefault(rb, []).append(did)
            n_real += 1
        collisions = {b: ds for b, ds in cross.items() if len(ds) > 1}
        if collisions:
            n_col = sum(len(ds) for ds in collisions.values())
            print(f"[mkg_fenn_ml] WARNING: {len(collisions)} drugbank id(s) shared by "
                  f"{n_col} twoside pool drugs; both inherit the same KG1 rows.",
                  file=sys.stderr)
        print(f"[mkg_fenn_ml] crosswalk: {n_real}/{len(drugs)} twoside drugs carry a "
              f"real drugbank_id ({len(cross)} distinct).", flush=True)
        return cross

    def _rekey_kg1(self, kg1: KnowledgeGraph,
                   crosswalk: dict[str, list[str]]) -> tuple[KnowledgeGraph, int]:
        """Return a NEW KnowledgeGraph whose 5 tables' ``drugbank_id`` column holds
        the twoside pool-id string (drugbank_id -> pool id via ``crosswalk``).

        Rows whose drugbank_id isn't in the crosswalk are dropped. One-to-many
        collisions duplicate the row for each mapped pool id (codex caveat). The
        on-disk files are NOT mutated (shallow-copied tables only). Returns the
        re-keyed KG + the count of DISTINCT pool ids that received >=1 KG1 row.
        """
        pool_hits: set[str] = set()
        new_tables: dict[str, pd.DataFrame] = {}
        for attr in _KG1_TABLE_ATTRS:
            tbl = getattr(kg1, attr)
            if tbl is None or not isinstance(tbl, pd.DataFrame) or tbl.empty \
                    or "drugbank_id" not in tbl.columns:
                new_tables[attr] = tbl
                continue
            rows = []
            for _i, row in tbl.iterrows():
                bid = str(row["drugbank_id"])
                pool_ids = crosswalk.get(bid)
                if not pool_ids:
                    continue
                for pid in pool_ids:
                    r = row.copy()
                    r["drugbank_id"] = pid            # NOW a pool-id string
                    rows.append(r)
                    pool_hits.add(pid)
            new_tables[attr] = (pd.DataFrame(rows).reset_index(drop=True)
                                if rows else tbl.iloc[0:0].copy())
        rekeyed = KnowledgeGraph(
            enzymes=new_tables["enzymes"], targets=new_tables["targets"],
            transporters=new_tables["transporters"], carriers=new_tables["carriers"],
            pathways=new_tables["pathways"],
        )
        return rekeyed, len(pool_hits)

    def _load_rekeyed_kg1(self, resources: "LeafResources") -> KnowledgeGraph:
        """Load native KG1 tables and re-key drugbank_id -> twoside pool id."""
        root = _project_root(Path(__file__))
        kg_dir = root / _KG1_FILTERED_DIR
        if not kg_dir.is_dir():
            raise FileNotFoundError(f"native KG1 filtered dir not found: {kg_dir}")
        kg1 = KnowledgeGraph.from_filtered_dir(kg_dir)
        crosswalk = self._build_crosswalk(resources)
        rekeyed, n_cov = self._rekey_kg1(kg1, crosswalk)
        n_drug = len(resources.drugs)
        print(f"[mkg_fenn_ml] KG1 coverage after re-key: {n_cov}/{n_drug} twoside "
              f"pool drugs get >=1 KG1 edge (~{100*n_cov/max(1,n_drug):.0f}%); the "
              f"rest get a ghost KG1 slot (data limitation). KG2/KG3/KG4 cover all "
              f"{n_drug}.", flush=True)
        return rekeyed

    # ------------------------------------------------------------------
    # regime + vocab
    # ------------------------------------------------------------------
    @staticmethod
    def _is_cold(resources: "LeafResources") -> bool:
        """Cold iff inductive regime (S1/S2). Transductive S0 -> warm."""
        return str(resources.meta.get("regime", "")).lower() == "inductive"

    def _build_dict1(self, resources: "LeafResources") -> dict[str, int]:
        """All pool drugs -> contiguous idx (sorted). Covers seen + unseen (keeps
        cold drugs reachable). Keyed on the pool ``drug_id`` (str)."""
        drug_ids = set(resources.drugs["drug_id"].astype(str))
        return {did: i for i, did in enumerate(sorted(drug_ids))}

    @staticmethod
    def _unseen_ids(resources: "LeafResources", dict1: dict[str, int]) -> list[int]:
        """Unseen (test/eval) pool drug ids in dict1 index space (cold regimes)."""
        roles = resources.drug_split
        g2 = roles[roles["role"].isin(["test", "val", "eval_unseen"])]["drug_id"].astype(str)
        return sorted({dict1[d] for d in g2 if d in dict1})

    # ------------------------------------------------------------------
    # pair-links + paired bundle (endpoints -> dict1 pool-id index space)
    # ------------------------------------------------------------------
    def _locate_pair_links(self, resources: "LeafResources", fold: str):
        """Load {train,val}_pair_links.parquet from the leaf's fold dir."""
        meta = resources.meta
        root = _project_root(Path(__file__)) / "Code" / "data" / "ddi_unified"
        parent = unified.layout_dir(root, meta["dataset_group"], meta["task"],
                                    meta["split_type"])
        fdir = parent / fold
        tr = pd.read_parquet(fdir / "train_pair_links.parquet")
        vp = fdir / "val_pair_links.parquet"
        va = pd.read_parquet(vp) if vp.is_file() else None
        return tr, va

    def _make_bundle(self, train_df, val_df, tr_links, va_links) -> dict:
        """Mirror ``make_multilabel_bundle`` but map endpoints -> dict1 int index
        (MKG-FENN scores by pool-id index, not raw id). neg_y == pos_y (the POS
        multihot reused as an ACTIVE-LABEL MASK)."""
        assert self._dict1 is not None
        d1 = self._dict1
        n_labels = self._n_labels

        def _pairs(df: pd.DataFrame, links: pd.DataFrame):
            by_id = df.set_index("pair_id")
            pos = by_id.loc[links["pos_pair_id"].to_numpy()]
            neg = by_id.loc[links["neg_pair_id"].to_numpy()]
            pos_ht = np.array(
                [[d1[str(a)], d1[str(b)]] for a, b in
                 zip(pos["drug_a_id"], pos["drug_b_id"])], dtype=np.int64
            )
            neg_ht = np.array(
                [[d1[str(a)], d1[str(b)]] for a, b in
                 zip(neg["drug_a_id"], neg["drug_b_id"])], dtype=np.int64
            )
            y = _dense_multihot(list(pos["y_label_ids"]), n_labels)
            return pos_ht, neg_ht, y

        tr_pos_ht, tr_neg_ht, tr_y = _pairs(train_df, tr_links)
        if val_df is not None and va_links is not None and len(va_links):
            va_pos_ht, va_neg_ht, va_y = _pairs(val_df, va_links)
        else:
            va_pos_ht = np.zeros((0, 2), np.int64)
            va_neg_ht = np.zeros((0, 2), np.int64)
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
        # capture fold (base fit_leaf drops it); multilabel needs the per-fold
        # pair_links sidecar which is not carried by LeafResources.
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources)

    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        self._n_labels = int(resources.meta["labels"]["n_labels"])
        cold = self._is_cold(resources)
        self._dict1 = self._build_dict1(resources)
        unseen_ids = self._unseen_ids(resources, self._dict1) if cold else []
        kg1 = self._load_rekeyed_kg1(resources)
        print(f"[mkg_fenn_ml] regime={resources.meta.get('regime')} "
              f"split={resources.meta.get('split_code')} cold={cold} "
              f"n_drug={len(self._dict1)} n_unseen={len(unseen_ids)} "
              f"n_labels={self._n_labels}", flush=True)

        self._core = MKGFENNMultilabelBaseline(
            n_labels=self._n_labels, cold=cold,
            embedding_num=self.embedding_num,
            neighbor_sample_size=self.neighbor_sample_size,
            dropout=self.dropout, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, batch_size=self.batch_size,
            n_epochs=self.n_epochs, seed=self.seed, device=self.device,
            log_step_every=self.log_step_every, run_dir=self.run_dir,
        )
        # Duck-typed leaf dataset (drug vocab / SMILES / KG3 train-DDI pairs).
        ds = make_dataset(train_df, val_df if val_df is not None else train_df,
                          resources, task="multilabel")
        tr_links, va_links = self._locate_pair_links(resources, self._fold)
        bundle = self._make_bundle(
            train_df, val_df if val_df is not None else train_df, tr_links, va_links
        )
        self._core.fit(ds, bundle, kg=kg1, dict1=self._dict1, unseen_ids=unseen_ids)

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._core is None or self._dict1 is None:
            raise RuntimeError("fit() before predict().")
        d1 = self._dict1
        # Map test drug ids -> dict1 pool index; OOV -> -1 (core returns a 0.5 row).
        ht = np.array(
            [[d1.get(str(a), -1), d1.get(str(b), -1)] for a, b in
             zip(test_df["drug_a_id"], test_df["drug_b_id"])], dtype=np.int64
        )
        return self._core.predict_proba(ht)          # (n, n_labels), aligned to test rows


__all__ = ["MKGFENNUnifiedMultilabel", "PAPER_HYPERPARAMS"]

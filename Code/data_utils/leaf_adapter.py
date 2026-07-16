"""Shared adapter: unified `Leaf` -> the minimal legacy-PairDataset surface that the
ColdDDI-derived baselines read. Lets all legacy cores (EmerGNN, SSI-DDI, HDN-DDI, TIGER,
MKG-FENN, ...) run on the new benchmark UNCHANGED — only the data-plumbing layer is
replaced (decision B, codex 019f19fb).

The legacy cores read a stable surface: `splits.train` (POS), `splits.val_s2` (val POS),
`splits.items()` (drug-pool union — we widen it with an all-drugs frame so unseen test
drugs join the entity/feature vocab, since `fit` never sees test_df), `drugs[[drugbank_id,
smiles]]` keyed on `drug_id`, `get_train_negatives()/get_negatives("val_s2")`, and
`g1_drugs/g2_drugs`. Negatives are the leaf's MATERIALIZED y_bin==0 rows (benchmark-fixed;
cores may sub-sample per epoch, but no fresh negatives are generated).

Task support:
  binary     -> splits carry POS pairs; negatives from y_bin==0 rows.
  multiclass -> splits.train also carries a `ddi_type` column (= y_cls); positives only.
  multilabel -> splits.train carries `y_label_ids`; positives only (is_positive==1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_PAIR = ["drug_a_id", "drug_b_id"]


def _pos(df: pd.DataFrame) -> pd.DataFrame:
    if "y_bin" in df.columns:
        return df[df["y_bin"] == 1][_PAIR].reset_index(drop=True)
    if "is_positive" in df.columns:
        return df[df["is_positive"] == 1].reset_index(drop=True)
    return df.reset_index(drop=True)          # multiclass: positives-only already


def _neg(df: pd.DataFrame) -> pd.DataFrame:
    if "y_bin" not in df.columns:
        return df.iloc[0:0][_PAIR]
    return df[df["y_bin"] == 0][_PAIR].reset_index(drop=True)


def _train_pos_frame(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Positives frame for `splits.train`, with the task's label column attached."""
    if task == "binary":
        return _pos(df)
    if task == "multiclass":
        out = df[_PAIR].copy()
        out["ddi_type"] = df["y_cls"].to_numpy()       # legacy cores read `ddi_type`
        return out.reset_index(drop=True)
    if task == "multilabel":
        out = df[df["is_positive"] == 1][_PAIR + ["y_label_ids"]]
        return out.reset_index(drop=True)
    raise ValueError(f"unknown task {task!r}")


class _LeafSplits:
    """Duck-typed SplitFolds. `train`/`val_s2` are POS (with label cols for mc/ml);
    `items()` yields train/val pair frames + an all-drugs frame so the entity/feature
    vocab covers every leaf drug (incl. unseen test drugs)."""

    def __init__(self, train_df, val_df, all_drugs, g1, g2, task):
        self.train = _train_pos_frame(train_df, task)
        self.val_s2 = _train_pos_frame(val_df, task)
        ad = [str(d) for d in all_drugs]
        all_frame = pd.DataFrame({"drug_a_id": ad, "drug_b_id": ad})
        self._frames = {"train": train_df[_PAIR], "val_s2": val_df[_PAIR],
                        "all_drugs": all_frame}
        self.g1_drugs = list(g1)
        self.g2_drugs = list(g2)

    def items(self):
        return self._frames.items()


@dataclass
class _LeafDataset:
    """Duck-typed PairDataset exposing only what the legacy cores read."""
    splits: _LeafSplits
    drugs: pd.DataFrame
    kg: object | None
    _train_neg: pd.DataFrame
    _val_neg: pd.DataFrame
    #: per-epoch deterministic train-negative provider (binary only; design B,
    #: leaf_negatives.LeafTrainNegatives). None -> fall back to fixed negatives.
    _epoch_neg: object | None = None

    def get_train_negatives(self, epoch: int = 0, *, regenerate: bool = True) -> pd.DataFrame:
        """``regenerate=True`` (paper default) -> deterministic per-epoch draw
        (design B); ``regenerate=False`` -> the leaf's FIXED materialized
        negatives (audit anchor). Falls back to fixed when no provider (e.g.
        resources lacked fold_dir, or non-binary)."""
        if regenerate and self._epoch_neg is not None:
            return self._epoch_neg.for_epoch(self.splits, epoch)
        return self._train_neg

    def get_negatives(self, split: str) -> pd.DataFrame:
        if split in ("val_s2", "val"):
            return self._val_neg
        raise KeyError(f"_LeafDataset only serves val negatives; got {split!r}")


def make_dataset(train_df, val_df, resources, *, task: str = "binary") -> _LeafDataset:
    """Build a duck-typed dataset from a unified leaf's (train_df, val_df, resources).
    Usable as BOTH the train and val arg of a legacy core's `fit`."""
    roles = resources.drug_split
    g1 = roles[roles["role"].isin(["train", "train_seen"])]["drug_id"].astype(str).tolist()
    g2 = roles[roles["role"].isin(["test", "val", "eval_unseen"])]["drug_id"].astype(str).tolist()
    all_drugs = resources.drugs["drug_id"].astype(str).tolist()
    splits = _LeafSplits(train_df, val_df, all_drugs, g1, g2, task)
    drugs = resources.drugs[["drug_id", "smiles"]].copy()
    drugs.columns = ["drugbank_id", "smiles"]
    # BINARY: attach the deterministic per-epoch train-negative provider (design B)
    # so get_train_negatives(epoch, regenerate=True) returns a fresh, reconstructable
    # draw per epoch (shared identically across baselines). Needs the leaf's fold_dir
    # + identity (added to LeafResources 2026-07-01). Non-binary / missing fold_dir ->
    # None (keeps the fixed-negatives fallback).
    epoch_neg = None
    if task == "binary" and getattr(resources, "fold_dir", None) is not None:
        from data_utils.leaf_negatives import LeafTrainNegatives  # local import (avoid cycle)
        m = resources.meta
        epoch_neg = LeafTrainNegatives(
            fold_dir=resources.fold_dir, dataset_id=m["dataset_id"],
            regime=m["regime"], split_code=m["split_code"], fold=resources.fold,
        )
    return _LeafDataset(splits=splits, drugs=drugs, kg=None,
                        _train_neg=_neg(train_df), _val_neg=_neg(val_df),
                        _epoch_neg=epoch_neg)


def _dense_multihot(y_label_ids, n_labels: int) -> np.ndarray:
    """Build a dense (n, n_labels) 0/1 multihot from per-row lists of label ids."""
    mat = np.zeros((len(y_label_ids), n_labels), dtype=np.float32)
    for i, ids in enumerate(y_label_ids):
        if ids is None:
            continue
        idx = np.asarray(list(ids), dtype=np.int64)
        if idx.size:
            mat[i, idx] = 1.0
    return mat


def make_multilabel_bundle(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_pair_links: pd.DataFrame,
    val_pair_links: pd.DataFrame,
    *,
    n_labels: int,
) -> dict:
    """Multilabel adapter (multilabel path only; binary/multiclass unchanged).

    Produces the paired pos/neg bundle the ported TWOSIDES core consumes: ONE ROW
    PER PAIR with a DENSE ``n_labels``-multihot (from ``y_label_ids``), preserving
    the PAIRED pos/neg structure via ``*_pair_links`` (pos_pair_id -> neg_pair_id).
    Faithful to upstream base_model.py where each positive and its endpoint-
    corrupting negative both carry the SAME multihot and scores are masked by
    label>0 for BCE.

    Returns arrays keyed ``{train,val}_{pos,neg}_{ht,y}``:
      *_ht: (n,2) int endpoints; *_y: (n, n_labels) float32 multihot.
    Each row j pairs pos[j] with its linked neg[j] (same multihot).
    """
    def _pairs(df: pd.DataFrame, links: pd.DataFrame):
        by_id = df.set_index("pair_id")
        pos_ids = links["pos_pair_id"].to_numpy()
        neg_ids = links["neg_pair_id"].to_numpy()
        pos = by_id.loc[pos_ids]
        neg = by_id.loc[neg_ids]
        pos_ht = pos[_PAIR].to_numpy().astype(np.int64)
        neg_ht = neg[_PAIR].to_numpy().astype(np.int64)
        # upstream: negative carries the positive's multihot (endpoint corruption
        # keeps the same side-effect vector). Build from the POS labels so pos/neg
        # share the vector exactly, matching base_model.py's train_pos/train_neg.
        y = _dense_multihot(list(pos["y_label_ids"]), n_labels)
        return pos_ht, neg_ht, y

    tr_pos_ht, tr_neg_ht, tr_y = _pairs(train_df, train_pair_links)
    if val_df is not None and val_pair_links is not None and len(val_pair_links):
        va_pos_ht, va_neg_ht, va_y = _pairs(val_df, val_pair_links)
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


__all__ = ["make_dataset", "_LeafDataset", "_LeafSplits", "make_multilabel_bundle",
           "_dense_multihot"]

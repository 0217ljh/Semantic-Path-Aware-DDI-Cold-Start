"""Dataset loading for the rank-analysis pipeline, in DRUG-ID space (model-agnostic).

Produces drug-id pairs + labels + the seen-DDI edge set (with types for multiclass)
+ a SHARED reference bio-degree (merged KG) used by every model's hub baseline, so
degree_only is comparable across models regardless of each model's own KG.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import TaskSpec

_ROOT = Path(__file__).resolve().parents[3]
_UNIFIED = _ROOT / "Code" / "data" / "ddi_unified"


@dataclass
class RankData:
    task: TaskSpec
    dataset: str
    fold: str
    train_pairs: np.ndarray          # (n, 2) drug-id strings
    train_labels: np.ndarray         # (n,) int  (binary: 0/1 ; multiclass: class id)
    cold_val_pairs: np.ndarray
    cold_val_labels: np.ndarray
    cold_test_pairs: np.ndarray
    cold_test_labels: np.ndarray
    seen_ddi: np.ndarray             # (D, 3) [drug_a, drug_b, ddi_type] positive train DDIs
    train_neg: np.ndarray            # (N, 2) dataset negatives (binary only; empty for multiclass)
    train_drugs: np.ndarray          # (n_drugs,) unique drug ids in train positives
    ref_logdeg: dict                 # drug_id -> shared merged-KG log bio-degree (hub baseline)


def _pairs(df):
    return df[["drug_a_id", "drug_b_id"]].astype(str).to_numpy()


def load_rank_data(dataset: str, task: TaskSpec, fold: str = "fold0") -> RankData:
    sub = "binary_cls" if task.is_binary else "multi_cls"
    base = _UNIFIED / sub / dataset / "inductive" / "S2" / fold
    tr = pd.read_parquet(base / "train.parquet")
    va = pd.read_parquet(base / "val.parquet")
    te = pd.read_parquet(base / "test.parquet")

    if task.is_binary:
        lab = "y_bin"
        tr_ok = tr; va_ok = va; te_ok = te
        pos = tr[tr[lab] == 1]
        seen = np.column_stack([_pairs(pos), np.zeros(len(pos), dtype=np.int64)])  # single DDI type
        neg = _pairs(tr[tr[lab] == 0])
        tl = tr[lab].to_numpy().astype(np.int64)
    else:
        lab = "y_cls_train"
        tr_ok = tr[tr[lab] >= 0]; va_ok = va[va[lab] >= 0]; te_ok = te[te[lab] >= 0]
        seen = np.column_stack([_pairs(tr_ok), tr_ok[lab].to_numpy().astype(np.int64)])  # typed
        neg = np.empty((0, 2), dtype=object)
        tl = tr_ok[lab].to_numpy().astype(np.int64)

    train_pairs = _pairs(tr_ok)
    # drug universe = ALL train drugs (incl. those only in binary negatives), so
    # emerging/kept/source sampling is over the true S2 drug pool (codex fix).
    train_drugs = np.unique(train_pairs.reshape(-1)) if len(train_pairs) else np.array([])
    ref_logdeg = _ref_logdeg()

    return RankData(
        task=task, dataset=dataset, fold=fold,
        train_pairs=train_pairs, train_labels=tl,
        cold_val_pairs=_pairs(va_ok), cold_val_labels=va_ok[lab].to_numpy().astype(np.int64),
        cold_test_pairs=_pairs(te_ok), cold_test_labels=te_ok[lab].to_numpy().astype(np.int64),
        seen_ddi=seen, train_neg=neg, train_drugs=train_drugs, ref_logdeg=ref_logdeg)


_REF_CACHE = {}


def _ref_logdeg() -> dict:
    """Shared merged-KG log bio-degree per drug id (hub baseline reference).
    Cached across calls."""
    if _REF_CACHE:
        return _REF_CACHE["d"]
    import sys
    sys.path.insert(0, str(_ROOT / "Code"))
    from my_code.models.spmn_v1.retrieval import (
        MergedKG, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
    kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
    logdeg = np.log1p(kg.degree.astype(np.float64))
    inv = {v: k for k, v in kg.id_to_idx.items()}
    d = {inv[i]: float(logdeg[i]) for i in range(len(logdeg)) if i in inv}
    _REF_CACHE["d"] = d
    return d


def ref_degree_matrix(pairs: np.ndarray, ref_logdeg: dict) -> np.ndarray:
    """(n, 2) shared log-degree features for a set of drug-id pairs (unknown -> 0)."""
    return np.array([[ref_logdeg.get(str(a), 0.0), ref_logdeg.get(str(b), 0.0)]
                     for a, b in pairs], dtype=np.float32)

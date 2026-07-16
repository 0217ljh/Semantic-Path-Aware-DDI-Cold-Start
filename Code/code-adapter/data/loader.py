"""Stage 1 - dataset load into ``RankData`` (self-contained copy).

Copied from ``my_code/rank_analysis/data.py`` and scoped to the code-adapter
pipeline (codex-reviewed 2026-07-09):

* Loads ONLY the two locked datasets (``drugbank_ryu``,
  ``drugbank_latest_partial``) and folds ``fold0..fold4``, cold-start S2.
* NO KG loading here. ``ref_logdeg`` is left ``None`` and populated by the
  stage-2 (KG prep) module - a self-contained split reader must not drag in
  the merged-KG hub-baseline dependency.
* NO pair canonicalization / dedup - orientation is preserved as stored;
  normalization belongs to the protocol stage downstream.
* Multiclass uses the closed-set column ``y_cls_train`` and drops ``< 0`` rows
  on ALL three splits; label range is validated against ``task.n_classes``.

Import convention: the folder ``Code/code-adapter`` (hyphen) is not an
importable package name. Put ``Code/code-adapter`` on ``sys.path`` and import
``data.loader`` / ``specs`` directly (module names have no hyphen).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from specs import TaskSpec

# Code/code-adapter/data/loader.py -> parents[3] = project root
_ROOT = Path(__file__).resolve().parents[3]
_UNIFIED = _ROOT / "Code" / "data" / "ddi_unified"

#: datasets locked for this project (General Settings). Others are rejected so
#: an unsupported id fails loudly instead of hitting a missing path.
SUPPORTED_DATASETS = ("drugbank_ryu", "drugbank_latest_partial")
SUPPORTED_FOLDS = tuple(f"fold{i}" for i in range(5))


@dataclass
class RankData:
    """In-memory payload for one (dataset, task, fold) cold-start S2 leaf, in
    DRUG-ID (string) space. Consumed by the downstream protocol / model stages."""
    task: TaskSpec
    dataset: str
    fold: str
    train_pairs: np.ndarray          # (n, 2) drug-id strings
    train_labels: np.ndarray         # (n,) int (binary 0/1 ; multiclass class id)
    cold_val_pairs: np.ndarray
    cold_val_labels: np.ndarray
    cold_test_pairs: np.ndarray
    cold_test_labels: np.ndarray
    seen_ddi: np.ndarray             # (D, 3) [drug_a, drug_b, ddi_type] train positives
    train_neg: np.ndarray            # (N, 2) dataset negatives (binary only; empty for multiclass)
    train_drugs: np.ndarray          # (n_drugs,) unique drug ids in train positives
    ref_logdeg: Optional[dict] = None  # populated by stage-2 KG prep (hub baseline), not here


def _pairs(df: pd.DataFrame) -> np.ndarray:
    return df[["drug_a_id", "drug_b_id"]].astype(str).to_numpy()


def load_rank_data(dataset: str, task: TaskSpec, fold: str = "fold0") -> RankData:
    """Load one cold-start S2 leaf into ``RankData``. No KG, no canonicalization."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    if fold not in SUPPORTED_FOLDS:
        raise ValueError(f"unsupported fold {fold!r}; expected one of {SUPPORTED_FOLDS}")

    sub = "binary_cls" if task.is_binary else "multi_cls"
    base = _UNIFIED / sub / dataset / "inductive" / "S2" / fold
    for name in ("train", "val", "test"):
        p = base / f"{name}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"missing split parquet: {p}")
    tr = pd.read_parquet(base / "train.parquet")
    va = pd.read_parquet(base / "val.parquet")
    te = pd.read_parquet(base / "test.parquet")

    if task.is_binary:
        lab = "y_bin"
        tr_ok, va_ok, te_ok = tr, va, te
        pos = tr[tr[lab] == 1]
        seen = np.column_stack([_pairs(pos), np.zeros(len(pos), dtype=np.int64)])  # single DDI type 0
        neg = _pairs(tr[tr[lab] == 0])
        tl = tr[lab].to_numpy().astype(np.int64)
    else:
        lab = "y_cls_train"
        tr_ok = tr[tr[lab] >= 0]; va_ok = va[va[lab] >= 0]; te_ok = te[te[lab] >= 0]
        seen = np.column_stack([_pairs(tr_ok), tr_ok[lab].to_numpy().astype(np.int64)])  # typed
        neg = np.empty((0, 2), dtype=object)
        tl = tr_ok[lab].to_numpy().astype(np.int64)

    train_pairs = _pairs(tr_ok)
    train_drugs = np.unique(train_pairs.reshape(-1)) if len(train_pairs) else np.array([], dtype=object)

    va_labels = va_ok[lab].to_numpy().astype(np.int64)
    te_labels = te_ok[lab].to_numpy().astype(np.int64)

    # Validate label range against the declared task (catches a wrong K passed
    # into TaskSpec.multiclass(K) - a common silent bug).
    for split_name, y in (("train", tl), ("val", va_labels), ("test", te_labels)):
        if len(y) and (int(y.min()) < 0 or int(y.max()) >= task.n_classes):
            raise ValueError(
                f"{dataset}/{fold}/{split_name}: label out of range "
                f"[0,{task.n_classes}) (min={int(y.min())}, max={int(y.max())})")

    return RankData(
        task=task, dataset=dataset, fold=fold,
        train_pairs=train_pairs, train_labels=tl,
        cold_val_pairs=_pairs(va_ok), cold_val_labels=va_labels,
        cold_test_pairs=_pairs(te_ok), cold_test_labels=te_labels,
        seen_ddi=seen, train_neg=neg, train_drugs=train_drugs, ref_logdeg=None)


__all__ = ["RankData", "load_rank_data", "SUPPORTED_DATASETS", "SUPPORTED_FOLDS"]

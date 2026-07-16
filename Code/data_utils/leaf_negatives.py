"""Deterministic per-epoch training negatives for a unified BINARY leaf.

Design B (user directive 2026-07-01, codex thread 019f1f62): restore the
per-epoch negative resampling that the native ``PairDataset`` already has
(``data_utils.negatives.build_train_negatives``) but that the unified leaf
adapter had frozen to a single fixed set. This is a THIN wrapper around the
existing, tested :func:`data_utils.negatives.build_train_negatives`; it adds
only the leaf-specific pieces:

  * a STABLE per-leaf ``base_seed`` (blake2b of the leaf identity, so any
    epoch's negatives are reconstructable at any time — NOT Python ``hash()``
    which is process-randomized),
  * the ``exclude_extra`` set required to keep the benchmark's train/val/test
    canonical-pair disjointness invariant (audit ``analyze_unified_audit``):
    dataset-global positives UNION this leaf's FIXED val negatives UNION its
    FIXED test negatives. Without the fixed eval negatives in the exclude,
    a fresh per-epoch train draw could collide with them.

Seed scheme (matches the native PairDataset, negatives.py:264):
    epoch_seed = base_seed + TRAIN_NEGATIVES_SEED_BASE(=1000) + epoch
so per-epoch draws differ deterministically and never collide with the
val/test PHASE_OFFSETS (100..600).

Scope: BINARY train negatives only. val/test negatives stay fixed (eval must
be deterministic); multiclass (positives-only) and multilabel (endpoint
corruption) are unaffected.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from data_utils.negatives import build_train_negatives

_PAIR = ["drug_a_id", "drug_b_id"]

#: dataset-global positive-pair cache, keyed by the dataset dir (avoids
#: re-reading the S0 sibling on every leaf load). In-memory only.
_GLOBAL_POS_CACHE: dict[str, set[tuple[str, str]]] = {}


def leaf_base_seed(dataset_id: str, regime: str, split_code: str, fold: str) -> int:
    """Stable 63-bit base seed for a leaf's train-negative sequence.

    Deterministic across processes / machines (blake2b, NOT ``hash()``), so
    ``negatives_for_epoch(e)`` is reconstructable at any time.
    """
    key = f"ddi_unified.binary_train_neg.v1|{dataset_id}|{regime}|{split_code}|{fold}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) >> 1  # keep positive


def _canon(pairs) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for a, b in pairs:
        a_s, b_s = str(a), str(b)
        if a_s > b_s:
            a_s, b_s = b_s, a_s
        out.add((a_s, b_s))
    return out


def _pairs_where(df: pd.DataFrame, col: str, val: int) -> list[tuple[str, str]]:
    if col not in df.columns:
        return []
    sub = df[df[col] == val]
    return list(zip(sub["drug_a_id"].astype(str), sub["drug_b_id"].astype(str)))


def _dataset_global_positives(dataset_dir: Path) -> set[tuple[str, str]]:
    """All binary positive pairs of the dataset (canonical), unioned across EVERY
    regime/split/fold leaf (matches the audit's global_positives,
    analyze_unified_audit.py:53-66). Using only transductive/S0/fold0 is NOT
    safe for MRCGNN-style datasets (drugbank_deng/ryu) whose S0 is written per
    official source fold and does not cover the full positive universe (codex
    019f1f75). We therefore scan all leaves and take y_bin==1."""
    ck = str(dataset_dir.resolve())
    if ck in _GLOBAL_POS_CACHE:
        return _GLOBAL_POS_CACHE[ck]
    pos: set[tuple[str, str]] = set()
    for st in ("transductive/S0", "inductive/S1", "inductive/S2"):
        d = dataset_dir / st
        if not d.is_dir():
            continue
        for fold in sorted(d.glob("fold*")):
            for sp in ("train", "val", "test"):
                fp = fold / f"{sp}.parquet"
                if fp.is_file():
                    pos |= _canon(_pairs_where(pd.read_parquet(fp), "y_bin", 1))
    _GLOBAL_POS_CACHE[ck] = pos
    return pos


class LeafTrainNegatives:
    """Per-epoch deterministic train-negative provider for one binary leaf.

    Build once per leaf (reads the fixed val/test negatives + dataset-global
    positives), then call :meth:`for_epoch` per training epoch.
    """

    def __init__(self, fold_dir: str | Path, dataset_id: str, regime: str,
                 split_code: str, fold: str) -> None:
        fold_dir = Path(fold_dir)
        self.base_seed = leaf_base_seed(dataset_id, regime, split_code, fold)
        # Fixed eval negatives of THIS leaf (must not be re-sampled as train negs).
        fixed_neg: set[tuple[str, str]] = set()
        for name in ("val.parquet", "test.parquet"):
            fp = fold_dir / name
            if fp.is_file():
                df = pd.read_parquet(fp)
                fixed_neg |= _canon(_pairs_where(df, "y_bin", 0))
        dataset_dir = fold_dir.parents[2]      # <root>/binary_cls/<dataset>
        global_pos = _dataset_global_positives(dataset_dir)
        # exclude_extra passed to build_train_negatives. Note build_train_negatives
        # ALSO excludes everything in splits.items(), which for _LeafSplits is the
        # RAW train/val frames (both pos AND neg rows) — so at runtime the effective
        # exclusion is: all dataset-global positives, this leaf's fixed val+test
        # negatives, AND this leaf's fixed train+val rows. That is a safe superset:
        # it guarantees no epoch draw is a real positive or a fixed eval negative
        # (preserves train/val/test canonical disjointness), at the mild cost of also
        # avoiding the fixed train negatives (harmless — they are valid negs either way).
        self._exclude_extra: set[tuple[str, str]] = global_pos | fixed_neg

    def for_epoch(self, splits, epoch: int) -> pd.DataFrame:
        """Deterministic 1:1 training negatives for ``epoch`` (canonical, sorted).

        ``splits`` is the leaf's duck-typed SplitFolds (train POS + g1 pool)."""
        return build_train_negatives(
            splits,
            base_seed=self.base_seed,
            epoch=int(epoch),
            n_per_pos=1,
            exclude_extra=self._exclude_extra,
        )


__all__ = ["LeafTrainNegatives", "leaf_base_seed"]

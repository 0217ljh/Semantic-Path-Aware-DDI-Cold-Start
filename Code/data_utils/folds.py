"""5-fold CV split generation for the unified DDI benchmark.

Builds both regimes from a dataset's full positive DDI set + drug list, using ONE
uniform protocol (codex-reviewed 2026-06-30, thread 019f19b6):

* transductive/S0 (warm): pair-level k-fold. All drugs appear in every fold's train.
* inductive (cold-start): drug-disjoint k-fold. The drug pool is partitioned into k
  groups; for fold j the test group is G[j], the val group is G[(j+1)%k], train is the
  remaining groups. From the SAME partition we carve both:
    - S2 = pairs with BOTH endpoints in the held-out (test/val) group,
    - S1 = pairs with EXACTLY ONE endpoint held-out, the other a TRAIN ("seen") drug.
  train is shared by S1 and S2 (pairs with both endpoints in train_drugs). Pairs with
  one endpoint in the test group and the other in the val group (both-unseen, different
  groups) are neither S1 nor S2 and are DROPPED (counted as `unused`).

Negatives (binary) are sampled with :class:`data_utils.negatives.UniformNegativeSampler`
using POOLS MATCHED to each split's positive geometry (train: train×train; S2: held×held;
S1: held×train) and excluding the GLOBAL positive set, so a cold-start classifier cannot
win by detecting endpoint novelty instead of interaction.

This module only builds pair frames + drug-role tables + validators; each prepare_*
script assembles the task-specific columns (y_bin / y_cls / ...) and calls
``unified.write_dataset``.
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd

PairSet = pd.DataFrame  # columns at least: drug_a_id, drug_b_id (+ caller extras)


def _canon(a: str, b: str) -> tuple[str, str]:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def canonical_pairs(df: pd.DataFrame) -> set[tuple[str, str]]:
    return {_canon(a, b) for a, b in zip(df["drug_a_id"], df["drug_b_id"])}


# -- partition + role assignment -------------------------------------------------

def drug_cv_partition(drugs, k: int, seed: int) -> list[list[str]]:
    """Deterministic disjoint k-partition of the drug pool (sorted then seeded shuffle)."""
    order = np.random.default_rng(seed).permutation(sorted(set(map(str, drugs))))
    return [sorted(g.tolist()) for g in np.array_split(order, k)]


def cold_fold_drug_roles(groups: list[list[str]], j: int) -> tuple[set, set, set]:
    """fold j: test = G[j], val = G[(j+1)%k], train = the rest. Returns (train, val, test)
    as string sets."""
    k = len(groups)
    test = set(groups[j])
    val = set(groups[(j + 1) % k])
    train = set().union(*[set(groups[m]) for m in range(k) if m not in (j, (j + 1) % k)])
    return train, val, test


# -- pair carving ----------------------------------------------------------------

def _mask(df, a_in: set, b_in: set):
    a = df["drug_a_id"].astype(str); b = df["drug_b_id"].astype(str)
    return a.isin(a_in), b.isin(b_in)


def carve_cold(pos: pd.DataFrame, train_d: set, val_d: set, test_d: set
               ) -> tuple[dict[str, pd.DataFrame], int, set]:
    """Carve a positive frame into train / S1 / S2 (test & val) by drug roles.

    The S1 "seen" endpoint must be a drug that ACTUALLY appears in a train positive
    (not merely a member of the nominal train group) — otherwise an isolated train-group
    drug would make an S1 pair effectively both-unseen and fail check_cold_s1. So S1 is
    carved against `seen` = the train-positive drug universe.

    Returns ({"train","s1_val","s1_test","s2_val","s2_test"} -> frame, unused_count, seen).
    Every returned frame is a row-subset of `pos` (preserves caller columns like y_cls).
    """
    a = pos["drug_a_id"].astype(str); b = pos["drug_b_id"].astype(str)
    in_tr = a.isin(train_d) & b.isin(train_d)
    train_part = pos[in_tr]
    seen = set(train_part["drug_a_id"].astype(str)) | set(train_part["drug_b_id"].astype(str))
    s2_test = a.isin(test_d) & b.isin(test_d)
    s2_val = a.isin(val_d) & b.isin(val_d)
    s1_test = ((a.isin(test_d) & b.isin(seen)) | (b.isin(test_d) & a.isin(seen)))
    s1_val = ((a.isin(val_d) & b.isin(seen)) | (b.isin(val_d) & a.isin(seen)))
    out = {"train": train_part, "s1_val": pos[s1_val], "s1_test": pos[s1_test],
           "s2_val": pos[s2_val], "s2_test": pos[s2_test]}
    used = in_tr | s2_test | s2_val | s1_test | s1_val
    return ({k: v.reset_index(drop=True) for k, v in out.items()},
            int((~used).sum()), seen)


def warm_cv_pairs(pos: pd.DataFrame, k: int, seed: int, val_frac: float = 0.1
                  ) -> list[dict[str, pd.DataFrame]]:
    """Pair-level k-fold (transductive/S0): each fold test = one partition of the
    canonical positive pairs, val = a seeded slice of the remaining, train = the rest.
    Operates on canonical-deduped pairs so (a,b)/(b,a) never split across train/test."""
    cps = sorted(canonical_pairs(pos))
    rng = np.random.default_rng(seed)
    folds_idx = np.array_split(rng.permutation(len(cps)), k)
    # map each canonical pair -> the row-subset of pos carrying it (all type rows)
    pos = pos.copy()
    pos["_cp"] = [(_canon(a, b)) for a, b in zip(pos["drug_a_id"], pos["drug_b_id"])]
    by_cp = {cp: g for cp, g in pos.groupby("_cp")}
    out = []
    for j in range(k):
        test_cp = {cps[i] for i in folds_idx[j]}
        rest = [cps[i] for i in range(len(cps)) if cps[i] not in test_cp]
        rest = [rest[i] for i in rng.permutation(len(rest))]   # keep tuples intact
        n_val = max(1, int(round(val_frac * len(rest))))
        val_cp = set(rest[:n_val]); train_cp = set(rest[n_val:])
        def cat(cset):
            if not cset:
                return pos.iloc[0:0].drop(columns="_cp")
            return pd.concat([by_cp[cp] for cp in cset], ignore_index=True).drop(columns="_cp")
        out.append({"train": cat(train_cp), "val": cat(val_cp), "test": cat(test_cp)})
    return out


# -- drug-role table (extends unified.build_drug_split to {train,val,test,unused}) -

def build_drug_split_cv(train_d: set, val_d: set, test_d: set, all_drugs: set
                        ) -> pd.DataFrame:
    """Per-drug role table for a cold CV fold: train / val / test / unused (a drug that
    is in none of the three active groups for this fold). codex: record all four."""
    rows = []
    for d in sorted(all_drugs, key=lambda x: (len(x), x)):
        if d in train_d:
            r = "train"
        elif d in val_d:
            r = "val"
        elif d in test_d:
            r = "test"
        else:
            r = "unused"
        rows.append((d, r))
    return pd.DataFrame(rows, columns=["drug_id", "role"])


# -- validators (codex-required, raise on violation) -----------------------------

def assert_pair_disjoint(frames: dict[str, pd.DataFrame], tag: str = "") -> None:
    """No canonical pair shared across train/val/test of one fold."""
    cps = {name: canonical_pairs(df) for name, df in frames.items()}
    names = list(cps)
    for i in range(len(names)):
        for jx in range(i + 1, len(names)):
            ov = cps[names[i]] & cps[names[jx]]
            if ov:
                raise AssertionError(f"[{tag}] {len(ov)} canonical pairs shared between "
                                     f"{names[i]} and {names[jx]} (e.g. {sorted(ov)[:3]})")


def assert_no_dup_pairs(df: pd.DataFrame, tag: str = "") -> None:
    """Binary: no duplicate canonical pair within a split."""
    cps = [_canon(a, b) for a, b in zip(df["drug_a_id"], df["drug_b_id"])]
    if len(cps) != len(set(cps)):
        raise AssertionError(f"[{tag}] {len(cps) - len(set(cps))} duplicate canonical pairs")


def assert_neg_excluded(neg: pd.DataFrame, global_pos: set[tuple[str, str]],
                        tag: str = "") -> None:
    bad = canonical_pairs(neg) & global_pos
    if bad:
        raise AssertionError(f"[{tag}] {len(bad)} negatives are real positives "
                             f"(e.g. {sorted(bad)[:3]})")


def assert_drugs_disjoint(train_d: set, val_d: set, test_d: set, tag: str = "") -> None:
    if train_d & val_d or train_d & test_d or val_d & test_d:
        raise AssertionError(f"[{tag}] train/val/test drug groups overlap")


def assert_cv_coverage(groups: list[list[str]], tag: str = "") -> None:
    """Each drug is the test group exactly once and the val group exactly once across
    the k folds (test=G[j], val=G[(j+1)%k]). Verifies the partition is disjoint AND
    simulates the rotation to confirm the exactly-once property explicitly."""
    k = len(groups)
    sizes = [set(g) for g in groups]
    alld = set().union(*sizes)
    if sum(len(s) for s in sizes) != len(alld):
        raise AssertionError(f"[{tag}] partition groups overlap or miss drugs")
    test_cnt: collections.Counter = collections.Counter()
    val_cnt: collections.Counter = collections.Counter()
    for j in range(k):
        for d in sizes[j]:
            test_cnt[d] += 1
        for d in sizes[(j + 1) % k]:
            val_cnt[d] += 1
    if set(test_cnt.values()) != {1} or set(val_cnt.values()) != {1}:
        raise AssertionError(f"[{tag}] cv rotation not exactly-once "
                             f"(test={set(test_cnt.values())}, val={set(val_cnt.values())})")


def assert_train_covers(train_df: pd.DataFrame, all_drugs: set, tag: str = "") -> None:
    """transductive guarantee: every drug appears in this fold's train (as an endpoint)."""
    td = set(train_df["drug_a_id"].astype(str)) | set(train_df["drug_b_id"].astype(str))
    miss = set(map(str, all_drugs)) - td
    if miss:
        raise AssertionError(f"[{tag}] {len(miss)} drugs absent from train "
                             f"(e.g. {sorted(miss)[:5]})")


__all__ = ["canonical_pairs", "drug_cv_partition", "cold_fold_drug_roles", "carve_cold",
           "warm_cv_pairs", "build_drug_split_cv", "assert_pair_disjoint",
           "assert_no_dup_pairs", "assert_neg_excluded", "assert_drugs_disjoint",
           "assert_cv_coverage", "assert_train_covers"]

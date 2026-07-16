"""Build the unified all-regime 5-fold layout for an MRCGNN-format dataset (deng / ryu).

Source ships 5 warm CV folds of `d1,type,d2` rows. We produce, per the uniform protocol
(codex-reviewed 2026-06-30, see data_utils.folds):
  - transductive/S0 : the source's 5 warm CV folds (pair-level, all drugs seen), as
    binary (canonical positives + 1:1 structure-matched negatives) and multiclass
    (source rows, closed-set y_cls_train).
  - inductive/S1, S2 : drug-disjoint 5-fold generated from the FULL positive set.

A canonical pair may carry multiple event types (ryu) and/or appear in both orientations;
binary canonicalizes+dedups, multiclass keeps every (d1,type,d2) row and assigns its fold
by the pair's drug groups (so all rows of a pair stay in one fold — no cross-fold leak).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import unified, folds
from data_utils.negatives import UniformNegativeSampler

K = 5
PART_SEED = 42
NEG_BASE = 80000


def _cp_frame(cps) -> pd.DataFrame:
    """Minimal positive frame (drug_a_id, drug_b_id) from a set of canonical pairs."""
    cps = sorted(cps)
    return pd.DataFrame({"drug_a_id": [p[0] for p in cps],
                         "drug_b_id": [p[1] for p in cps]})


def _bin_split(canon_pairs, neg_df) -> pd.DataFrame:
    pdf = pd.DataFrame({"drug_a_id": [p[0] for p in canon_pairs],
                        "drug_b_id": [p[1] for p in canon_pairs], "y_bin": 1})
    ndf = pd.DataFrame({"drug_a_id": neg_df.drug_a_id.astype(str),
                        "drug_b_id": neg_df.drug_b_id.astype(str), "y_bin": 0})
    df = pd.concat([pdf, ndf], ignore_index=True)
    df.insert(0, "pair_id", df.index.astype(np.int64))
    return df[["pair_id", "drug_a_id", "drug_b_id", "y_bin"]]


def _mc_split(pos_rows, train_types) -> pd.DataFrame:
    df = pos_rows.reset_index(drop=True).copy()
    df.insert(0, "pair_id", df.index.astype(np.int64))
    df["y_cls_train"] = np.array([train_types.get(int(t), -1) for t in df.y_cls],
                                 dtype=np.int64)
    return df[["pair_id", "drug_a_id", "drug_b_id", "y_cls", "y_cls_train"]]


def _bin_leaf(parts, pools, sampler, global_pos, seed0):
    """Build a binary {train,val,test} leaf with running-exclude negatives.
    parts: {"train","val","test"} positive frames; pools: {split: (pool_a, pool_b)}."""
    sf, used = {}, set()
    for i, sp in enumerate(("train", "val", "test")):
        cps = sorted(folds.canonical_pairs(parts[sp]))
        pa, pb = pools[sp]
        neg = sampler.sample(drug_pool_a=list(pa), drug_pool_b=list(pb),
                             n_pairs=len(cps), exclude=global_pos | used, seed=seed0 + i)
        used |= folds.canonical_pairs(neg)
        sf[sp] = _bin_split(cps, neg)
        folds.assert_no_dup_pairs(sf[sp], f"bin/{sp}")
        folds.assert_neg_excluded(neg, global_pos, f"bin/{sp}")
    folds.assert_pair_disjoint(sf, "bin-leaf")
    return sf


def build(*, group: str, src_dir: Path, fold_files: dict[str, str], n_source_folds: int,
          smiles: pd.DataFrame, n_classes: int, label_vocab: pd.DataFrame,
          out_root: Path) -> dict:
    """Materialize all 6 leaves (binary+mc × S0/S1/S2) for an MRCGNN dataset.

    smiles: DataFrame[drug_id, smiles] covering every used drug. Returns a stats dict.
    """
    # ---- read source warm folds + full positive set ----
    def read_fold(f, sp):
        d = pd.read_csv(src_dir / str(f) / fold_files[sp])
        assert list(d.columns) == ["d1", "type", "d2"], f"cols {list(d.columns)}"
        return pd.DataFrame({"drug_a_id": d.d1.astype(str), "drug_b_id": d.d2.astype(str),
                             "y_cls": d.type.astype(np.int64)})

    warm_folds = {f: {sp: read_fold(f, sp) for sp in ("train", "val", "test")}
                  for f in range(K)}
    full = pd.concat([read_fold(f, sp) for f in range(n_source_folds)
                      for sp in ("train", "val", "test")], ignore_index=True)
    full = full.drop_duplicates(["drug_a_id", "drug_b_id", "y_cls"]).reset_index(drop=True)
    drugs_all = sorted(set(full.drug_a_id) | set(full.drug_b_id))
    global_pos = folds.canonical_pairs(full)

    smiles = smiles.copy(); smiles["drug_id"] = smiles.drug_id.astype(str)
    smiles = smiles.drop_duplicates("drug_id")
    miss = set(drugs_all) - set(smiles.drug_id)
    if miss:
        raise ValueError(f"[{group}] {len(miss)} drugs lack SMILES: {sorted(miss)[:5]}")
    smap = smiles.set_index("drug_id").smiles
    drugs = unified.make_drugs_frame(drugs_all, smiles=[str(smap[d]) for d in drugs_all],
                                     smiles_source="mrcgnn_source")
    sampler = UniformNegativeSampler(drug_pool_a=drugs_all, drug_pool_b=drugs_all)

    def write(task, split_type, frames, dsplits=None):
        lv = label_vocab if task == "multiclass" else None
        nlab = n_classes if task == "multiclass" else None
        pf = "canonical" if task == "binary" else "ordered_source"
        unified.write_dataset(
            unified.layout_dir(out_root, group, task, split_type),
            unified.base_meta(unified.DATASET_DIRS[group], group, task, split_type,
                              [f"fold{j}" for j in range(K)],
                              smiles_source="mrcgnn_source", n_labels=nlab, pair_format=pf),
            drugs, lv, frames, dsplits)

    # ---- transductive/S0 from source folds ----
    bin_f, mc_f = {}, {}
    for j in range(K):
        fr = warm_folds[j]
        pools = {sp: (drugs_all, drugs_all) for sp in ("train", "val", "test")}
        # binary is symmetric: a canonical pair may appear in >1 source split (ryu has
        # directional/multi-type rows). Assign each canonical pair to ONE split with
        # priority test>val>train so eval splits stay clean (no cross-split leak).
        te_cp = folds.canonical_pairs(fr["test"])
        va_cp = folds.canonical_pairs(fr["val"]) - te_cp
        tr_cp = folds.canonical_pairs(fr["train"]) - te_cp - va_cp
        bfr = {"train": _cp_frame(tr_cp), "val": _cp_frame(va_cp), "test": _cp_frame(te_cp)}
        bin_f[f"fold{j}"] = _bin_leaf(bfr, pools, sampler, global_pos, NEG_BASE + 10 * j)
        tt = {int(t): i for i, t in enumerate(sorted(fr["train"].y_cls.unique()))}
        msf = {sp: _mc_split(fr[sp], tt) for sp in ("train", "val", "test")}
        unified.check_multiclass_closed_set(msf["train"], msf["val"], msf["test"])
        # NOTE: no all-drugs-in-train assertion here — these are the OFFICIAL source warm
        # CV folds; a drug may be eval-only in a given fold (it appears in other folds'
        # train). Imposing full coverage would corrupt the official split.
        mc_f[f"fold{j}"] = msf
    write("binary", "transductive", bin_f)
    write("multiclass", "transductive", mc_f)

    # ---- inductive/S1, S2 (drug-disjoint 5-fold) ----
    groups = folds.drug_cv_partition(drugs_all, K, PART_SEED)
    folds.assert_cv_coverage(groups, group)
    s1_bin, s2_bin, s1_mc, s2_mc, s1_ds, s2_ds = {}, {}, {}, {}, {}, {}
    unused_total = 0
    for j in range(K):
        train_d, val_d, test_d = folds.cold_fold_drug_roles(groups, j)
        folds.assert_drugs_disjoint(train_d, val_d, test_d, f"{group} f{j}")
        parts, unused, seen = folds.carve_cold(full, train_d, val_d, test_d)
        unused_total += unused
        s2_bin[f"fold{j}"] = _bin_leaf(
            {"train": parts["train"], "val": parts["s2_val"], "test": parts["s2_test"]},
            {"train": (train_d, train_d), "val": (val_d, val_d), "test": (test_d, test_d)},
            sampler, global_pos, NEG_BASE + 5000 + j * 3)
        # S1 negatives' seen endpoint drawn from `seen` (actual train-positive drugs) so
        # they satisfy exactly-one-seen, matching the S1 positives.
        s1_bin[f"fold{j}"] = _bin_leaf(
            {"train": parts["train"], "val": parts["s1_val"], "test": parts["s1_test"]},
            {"train": (train_d, train_d), "val": (val_d, seen), "test": (test_d, seen)},
            sampler, global_pos, NEG_BASE + 1000 + j * 3)
        ttj = {int(t): i for i, t in enumerate(sorted(parts["train"].y_cls.unique()))}
        s2_mc[f"fold{j}"] = {"train": _mc_split(parts["train"], ttj),
                             "val": _mc_split(parts["s2_val"], ttj),
                             "test": _mc_split(parts["s2_test"], ttj)}
        s1_mc[f"fold{j}"] = {"train": _mc_split(parts["train"], ttj),
                             "val": _mc_split(parts["s1_val"], ttj),
                             "test": _mc_split(parts["s1_test"], ttj)}
        for sf in (s2_mc[f"fold{j}"], s1_mc[f"fold{j}"]):
            unified.check_multiclass_closed_set(sf["train"], sf["val"], sf["test"])
        for leaf in (s2_bin[f"fold{j}"], s2_mc[f"fold{j}"]):
            unified.check_cold_s2(leaf["train"], [("val", leaf["val"]), ("test", leaf["test"])])
        for leaf in (s1_bin[f"fold{j}"], s1_mc[f"fold{j}"]):
            unified.check_cold_s1(leaf["train"], [("val", leaf["val"]), ("test", leaf["test"])])
        s2_ds[f"fold{j}"] = folds.build_drug_split_cv(train_d, val_d, test_d, set(drugs_all))
        s1_ds[f"fold{j}"] = folds.build_drug_split_cv(train_d, val_d, test_d, set(drugs_all))
    write("binary", "cold_s1", s1_bin, s1_ds)
    write("binary", "cold_s2", s2_bin, s2_ds)
    write("multiclass", "cold_s1", s1_mc, s1_ds)
    write("multiclass", "cold_s2", s2_mc, s2_ds)
    return {"drugs": len(drugs_all), "n_classes": n_classes, "dropped": unused_total}


__all__ = ["build"]

"""Materialize OUR DrugBank corpus (drugbank_latest) into the unified layout, 5-fold.

Two versions of the SAME corpus (positives + types from ddi_edges.csv):
  - drugbank_latest_partial : the 800-drug fast-test subset (group "ddi800"); drug set =
    the legacy 800drug_3seed universe. ~165 event types present.
  - drugbank_latest_full    : the FULL DrugBank (group "ddi_full"); drug set = all 1900
    drugs in ddi_edges.csv. 215 event types.

Both built with the uniform 5-fold CV protocol (codex-reviewed 2026-06-30, data_utils.folds):
  - transductive/S0 : pair-level 5-fold (all drugs seen in every fold's train).
  - inductive/S1, S2 : drug-disjoint 5-fold (20% drugs held out per fold). S1 = one
    endpoint held-out + one SEEN, S2 = both held-out; S1/S2 share partition + train.
Tasks: binary (1:1 structure-matched negatives, global-positive exclusion) and multiclass
(positives only, closed-set y_cls_train). KG = dataset-scoped merged-KG ref.
Read-only on source; writes Code/data/ddi_unified/<task>/<dataset>/<regime>/<split>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
from data_utils import unified, folds  # noqa: E402
from data_utils.negatives import UniformNegativeSampler  # noqa: E402

LEGACY = ROOT / "Code/data/coldddi_legacy/800drug_3seed/seed42"
SMI = ROOT / "Code/data/coldddi_legacy/800drug/drug_smiles__seed42.csv"
DDI_EDGES = ROOT / "Code/data/KG/drugbank/filtered/ddi_edges.csv"
MERGED_KG = "Code/data/KG/_merged_kg/"
OUT = ROOT / "Code/data/ddi_unified"
K = 5
PART_SEED = 42          # drug-partition seed (inductive)
WARM_SEED = 4200        # pair-fold seed (transductive)
NEG_BASE = 70000        # negative-sampler seed base


def _canon(a, b):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def _bin_split(pos_pairs, neg_df) -> pd.DataFrame:
    pdf = pd.DataFrame({"drug_a_id": [p[0] for p in pos_pairs],
                        "drug_b_id": [p[1] for p in pos_pairs], "y_bin": 1})
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


def build_latest(group: str, drugs_set: set, edges: pd.DataFrame, smap: pd.Series) -> dict:
    """Build all 6 leaves (binary+mc × S0/S1/S2) for one drug universe."""
    du = set(map(str, drugs_set))
    drugs_sorted = sorted(du)
    e = edges[edges.a.isin(du) & edges.b.isin(du)]
    types = sorted(e.ddi_type.astype(str).unique())
    type2idx = {t: i for i, t in enumerate(types)}
    emap_sets: dict[tuple[str, str], set[str]] = {}
    for a, b, t in zip(e.a, e.b, e.ddi_type):
        emap_sets.setdefault(_canon(a, b), set()).add(str(t))
    conflict = {k: v for k, v in emap_sets.items() if len(v) > 1}
    if conflict:
        raise ValueError(f"[{group}] {len(conflict)} canonical pairs conflict on type")
    cps = sorted(emap_sets)
    mc_pos = pd.DataFrame({"drug_a_id": [p[0] for p in cps],
                           "drug_b_id": [p[1] for p in cps],
                           "y_cls": [type2idx[next(iter(emap_sets[p]))] for p in cps]})
    global_pos = set(cps)
    miss = du - set(smap.index)
    if miss:
        raise ValueError(f"[{group}] {len(miss)} drugs lack SMILES: {sorted(miss)[:5]}")
    drugs = unified.make_drugs_frame(drugs_sorted, smiles=[str(smap[d]) for d in drugs_sorted],
                                     smiles_source="drug_smiles__seed42.csv")
    label_vocab = pd.DataFrame({"label_idx_global": np.arange(len(types), dtype=np.int64),
                                "label_name": types})
    kg = {"scope": "dataset", "source": MERGED_KG, "drug_node_key": "drug_id"}
    sampler = UniformNegativeSampler(drug_pool_a=drugs_sorted, drug_pool_b=drugs_sorted)

    def write(task, split_type, frames, dsplits=None):
        lv = label_vocab if task == "multiclass" else None
        nlab = len(types) if task == "multiclass" else None
        unified.write_dataset(
            unified.layout_dir(OUT, group, task, split_type),
            unified.base_meta(unified.DATASET_DIRS[group], group, task, split_type,
                              [f"fold{j}" for j in range(K)], kg=kg,
                              smiles_source="drug_smiles__seed42.csv",
                              n_labels=nlab, pair_format="canonical"),
            drugs, lv, frames, dsplits)

    # ---- transductive/S0 ----
    warm = folds.warm_cv_pairs(mc_pos, K, WARM_SEED)
    bin_f, mc_f = {}, {}
    for j, fr in enumerate(warm):
        bsf, used_neg = {}, set()
        for sp in ("train", "val", "test"):
            cps_sp = sorted(folds.canonical_pairs(fr[sp]))
            neg = sampler.sample(n_pairs=len(cps_sp), exclude=global_pos | used_neg,
                                 seed=NEG_BASE + 10 * j + {"train": 0, "val": 1, "test": 2}[sp])
            used_neg |= folds.canonical_pairs(neg)
            bsf[sp] = _bin_split(cps_sp, neg)
            folds.assert_no_dup_pairs(bsf[sp], f"{group} S0 bin f{j}/{sp}")
            folds.assert_neg_excluded(neg, global_pos, f"{group} S0 bin f{j}/{sp}")
        folds.assert_pair_disjoint(bsf, f"{group} S0 bin f{j}")
        folds.assert_train_covers(bsf["train"], du, f"{group} S0 bin f{j}")
        bin_f[f"fold{j}"] = bsf
        tt = {int(t): i for i, t in enumerate(sorted(fr["train"].y_cls.unique()))}
        msf = {sp: _mc_split(fr[sp], tt) for sp in ("train", "val", "test")}
        unified.check_multiclass_closed_set(msf["train"], msf["val"], msf["test"])
        folds.assert_train_covers(msf["train"], du, f"{group} S0 mc f{j}")
        mc_f[f"fold{j}"] = msf
    write("binary", "transductive", bin_f)
    write("multiclass", "transductive", mc_f)

    # ---- inductive/S1, S2 ----
    groups = folds.drug_cv_partition(drugs_sorted, K, PART_SEED)
    folds.assert_cv_coverage(groups, group)
    s1_bin, s2_bin, s1_mc, s2_mc, s1_ds, s2_ds = {}, {}, {}, {}, {}, {}
    unused_total = 0
    for j in range(K):
        train_d, val_d, test_d = folds.cold_fold_drug_roles(groups, j)
        folds.assert_drugs_disjoint(train_d, val_d, test_d, f"{group} f{j}")
        parts, unused, seen = folds.carve_cold(mc_pos, train_d, val_d, test_d)
        unused_total += unused

        def bin_leaf(train_part, val_part, test_part, val_pool, test_pool, tag):
            sf, used = {}, set()
            toff = {"S1": 1000, "S2": 5000}[tag]
            specs = [("train", train_part, (list(train_d), list(train_d))),
                     ("val", val_part, val_pool), ("test", test_part, test_pool)]
            for sp, part, (pa, pb) in specs:
                cps_sp = sorted(folds.canonical_pairs(part))
                neg = sampler.sample(drug_pool_a=list(pa), drug_pool_b=list(pb),
                                     n_pairs=len(cps_sp), exclude=global_pos | used,
                                     seed=NEG_BASE + toff + j * 3
                                     + {"train": 0, "val": 1, "test": 2}[sp])
                used |= folds.canonical_pairs(neg)
                sf[sp] = _bin_split(cps_sp, neg)
                folds.assert_no_dup_pairs(sf[sp], f"{group} {tag} bin f{j}/{sp}")
                folds.assert_neg_excluded(neg, global_pos, f"{group} {tag} bin f{j}/{sp}")
            folds.assert_pair_disjoint(sf, f"{group} {tag} bin f{j}")
            return sf

        s2_bin[f"fold{j}"] = bin_leaf(parts["train"], parts["s2_val"], parts["s2_test"],
                                      (list(val_d), list(val_d)), (list(test_d), list(test_d)), "S2")
        s1_bin[f"fold{j}"] = bin_leaf(parts["train"], parts["s1_val"], parts["s1_test"],
                                      (list(val_d), list(seen)), (list(test_d), list(seen)), "S1")
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
        s2_ds[f"fold{j}"] = folds.build_drug_split_cv(train_d, val_d, test_d, du)
        s1_ds[f"fold{j}"] = folds.build_drug_split_cv(train_d, val_d, test_d, du)
    write("binary", "cold_s1", s1_bin, s1_ds)
    write("binary", "cold_s2", s2_bin, s2_ds)
    write("multiclass", "cold_s1", s1_mc, s1_ds)
    write("multiclass", "cold_s2", s2_mc, s2_ds)
    return {"group": group, "drugs": len(du), "types": len(types), "dropped": unused_total}


def main() -> None:
    edges = pd.read_csv(DDI_EDGES)
    edges["a"] = edges.drug_a_id.astype(str); edges["b"] = edges.drug_b_id.astype(str)
    s = pd.read_csv(SMI)
    cols = {c.lower(): c for c in s.columns}
    idc = cols.get("drug_id") or s.columns[0]; smc = cols.get("smiles") or s.columns[1]
    s = s.rename(columns={idc: "drug_id", smc: "smiles"}).drop_duplicates("drug_id")
    smap = s.set_index(s.drug_id.astype(str)).smiles

    # partial = legacy 800-drug universe
    partial = set()
    for f in ("train", "val_s2", "test_s2"):
        d = pd.read_parquet(LEGACY / f"{f}.parquet")
        partial |= set(d.drug_a_id.astype(str)) | set(d.drug_b_id.astype(str))
    assert len(partial) == 800, f"expected 800 partial drugs, got {len(partial)}"
    # full = all drugs in ddi_edges
    full = set(edges.a) | set(edges.b)

    for group, ds in (("ddi800", partial), ("ddi_full", full)):
        st = build_latest(group, ds, edges, smap)
        print(f"[latest] {unified.DATASET_DIRS[group]}  drugs={st['drugs']} "
              f"types={st['types']} K={K} dropped(cross-unseen)={st['dropped']}", flush=True)


if __name__ == "__main__":
    main()

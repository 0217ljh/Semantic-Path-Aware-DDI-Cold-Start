"""Regenerate ddi800 binary cold_s2 with the legacy-style 640-train protocol (Option A).

Rationale
---------
The uniform 5-fold generator (``data_utils.folds`` + ``prepare_ddi800.py``) builds S1
and S2 from ONE drug partition where fold ``j`` uses ``test=G[j]``, ``val=G[(j+1)%k]``,
``train = the other 3 groups`` -> only 480 of 800 drugs train. That sacrifices a whole
group to a fully-disjoint cold val. Our adapter (a pure-structural cold-start model) was
developed/validated on the LEGACY 640-train regime (``splits.py`` ``_build_s2``:
``train = G1xG1`` with ``G1`` = 640 seen drugs, ``val/test = G2xG2`` split 50/50 over the
160 held-out drugs). Training on 480 vs 640 drugs systematically depresses the number.

This script regenerates ONLY ``binary_cls/drugbank_latest_partial/inductive/S2`` with the
legacy-style split, to recover the training set and test whether the method gains:

  * SAME 5 disjoint drug groups as the current benchmark (``drug_cv_partition``,
    ``PART_SEED=42``) -> each drug is the held-out (test) group exactly once, and the
    per-fold TEST DRUGS are IDENTICAL to the current S2 (so comparison to already-run
    baselines like EmerGNN fold0 stays on the same test drugs).
  * fold ``j``: held-out ``= G[j]`` (160), ``train = the other 4 groups`` (640).
  * S2 positives: ``train = train_d x train_d``; the held-out group's ``G[j] x G[j]``
    both-unseen positives are split ``VAL_RATIO`` (val) / ``1-VAL_RATIO`` (test) by
    canonical pair (deterministic seed) -> val is cold and shares the held-out drug pool
    with test (the legacy compromise, accepted by the user).
  * Negatives: 1:1, pool-matched (train negs from ``train_d x train_d``; val/test negs from
    ``G[j] x G[j]``), global-positive-excluded, non-overlapping across train/val/test --
    identical policy to ``prepare_ddi800.bin_leaf``, reusing ``UniformNegativeSampler``.

Scope / non-goals
-----------------
* ONLY ``binary`` / ``drugbank_latest_partial`` (ddi800) / ``cold_s2`` is regenerated.
  S0, S1, multiclass, and the ``drugbank_latest_full`` universe are NOT touched.
* Reuses ``data_utils.{unified,folds,negatives}`` unchanged (no existing function is
  modified); this is a NEW file per the repo conventions.
* Read-only on sources; overwrites the target leaf dir (a backup is expected to be made
  by the caller before running -- see the companion review note).

drug_split / audit caveat (codex 019f2143, accepted)
-----------------------------------------------------
Under Option A the val pairs and test pairs are BOTH drawn from the same held-out drug
group ``G[j]`` (they differ only at the pair level). The ``drug_split.parquet`` role
table assigns exactly one role per drug from ``{train,val,test,unused}``
(``folds.build_drug_split_cv``), so it CANNOT express "val drugs == test drugs". We mark
the held-out group as ``"test"`` (val drugs are a pair-level sub-slice, left implicit) --
the most honest single-role encoding of "these 160 are the held-out unseen eval pool".
Runtime consumers are correct: ``leaf_adapter`` merges ``{test,val,eval_unseen}`` into
the unseen pool ``g2`` and the runners read the train/val/test PARQUETS directly, not the
``val`` drug role. HOWEVER the generic ``analyze_unified_audit.py`` assumes the 480-train
protocol's DISJOINT val/test drug groups (each drug ``"val"`` exactly once), which does
NOT hold here -- so that audit will flag this leaf as invalid. That audit models a
different protocol and does not apply to this S2-640 leaf; do not run it against this leaf.
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
GROUP = "ddi800"
K = 5
PART_SEED = 42          # drug-partition seed -- MUST match prepare_ddi800.py so the
#                         5 held-out groups (and thus per-fold test drugs) are identical.
NEG_BASE = 70000        # negative-sampler seed base (matches prepare_ddi800.py S2 offset)
S2_TOFF = 5000          # S2 negative seed offset (matches prepare_ddi800.py bin_leaf "S2")
HELD_SPLIT_SEED = 73000  # seed base for splitting held-out G[j]xG[j] pairs into val/test
VAL_RATIO = 0.5         # fraction of held-out pairs used for val (legacy used ~50/50)


def _canon(a, b):
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def _bin_split(pos_pairs, neg_df) -> pd.DataFrame:
    """Assemble a 1:1 binary leaf frame (mirrors prepare_ddi800._bin_split)."""
    pdf = pd.DataFrame({"drug_a_id": [p[0] for p in pos_pairs],
                        "drug_b_id": [p[1] for p in pos_pairs], "y_bin": 1})
    ndf = pd.DataFrame({"drug_a_id": neg_df.drug_a_id.astype(str),
                        "drug_b_id": neg_df.drug_b_id.astype(str), "y_bin": 0})
    df = pd.concat([pdf, ndf], ignore_index=True)
    df.insert(0, "pair_id", df.index.astype(np.int64))
    return df[["pair_id", "drug_a_id", "drug_b_id", "y_bin"]]


def _load_partial_universe() -> set:
    partial = set()
    for f in ("train", "val_s2", "test_s2"):
        d = pd.read_parquet(LEGACY / f"{f}.parquet")
        partial |= set(d.drug_a_id.astype(str)) | set(d.drug_b_id.astype(str))
    assert len(partial) == 800, f"expected 800 partial drugs, got {len(partial)}"
    return partial


def main() -> None:
    edges = pd.read_csv(DDI_EDGES)
    edges["a"] = edges.drug_a_id.astype(str)
    edges["b"] = edges.drug_b_id.astype(str)
    s = pd.read_csv(SMI)
    cols = {c.lower(): c for c in s.columns}
    idc = cols.get("drug_id") or s.columns[0]
    smc = cols.get("smiles") or s.columns[1]
    s = s.rename(columns={idc: "drug_id", smc: "smiles"}).drop_duplicates("drug_id")
    smap = s.set_index(s.drug_id.astype(str)).smiles

    du = _load_partial_universe()
    drugs_sorted = sorted(du)
    e = edges[edges.a.isin(du) & edges.b.isin(du)]
    types = sorted(e.ddi_type.astype(str).unique())
    type2idx = {t: i for i, t in enumerate(types)}
    emap_sets: dict[tuple[str, str], set[str]] = {}
    for a, b, t in zip(e.a, e.b, e.ddi_type):
        emap_sets.setdefault(_canon(a, b), set()).add(str(t))
    conflict = {k: v for k, v in emap_sets.items() if len(v) > 1}
    if conflict:
        raise ValueError(f"[{GROUP}] {len(conflict)} canonical pairs conflict on type")
    cps = sorted(emap_sets)
    mc_pos = pd.DataFrame({"drug_a_id": [p[0] for p in cps],
                           "drug_b_id": [p[1] for p in cps],
                           "y_cls": [type2idx[next(iter(emap_sets[p]))] for p in cps]})
    global_pos = set(cps)
    miss = du - set(smap.index)
    if miss:
        raise ValueError(f"[{GROUP}] {len(miss)} drugs lack SMILES: {sorted(miss)[:5]}")
    drugs = unified.make_drugs_frame(drugs_sorted,
                                     smiles=[str(smap[d]) for d in drugs_sorted],
                                     smiles_source="drug_smiles__seed42.csv")
    kg = {"scope": "dataset", "source": MERGED_KG, "drug_node_key": "drug_id"}
    sampler = UniformNegativeSampler(drug_pool_a=drugs_sorted, drug_pool_b=drugs_sorted)

    # SAME partition as prepare_ddi800.py (PART_SEED) -> identical held-out groups.
    groups = folds.drug_cv_partition(drugs_sorted, K, PART_SEED)
    folds.assert_cv_coverage(groups, GROUP)

    a_all = mc_pos["drug_a_id"].astype(str)
    b_all = mc_pos["drug_b_id"].astype(str)

    s2_bin: dict[str, dict] = {}
    s2_ds: dict[str, pd.DataFrame] = {}
    for j in range(K):
        held_d = set(groups[j])
        train_d = set().union(*[set(groups[m]) for m in range(K) if m != j])
        assert not (train_d & held_d), f"[{GROUP} f{j}] train/held overlap"
        assert train_d | held_d == du, f"[{GROUP} f{j}] train+held != universe"

        train_pos = mc_pos[a_all.isin(train_d) & b_all.isin(train_d)]
        held_pos = mc_pos[a_all.isin(held_d) & b_all.isin(held_d)]

        # split held-out G[j]xG[j] positives into val / test by canonical pair.
        held_cps = sorted(folds.canonical_pairs(held_pos))
        rng = np.random.default_rng(HELD_SPLIT_SEED + j)
        perm = rng.permutation(len(held_cps))
        n_val = int(round(VAL_RATIO * len(held_cps)))
        val_cps = sorted(held_cps[i] for i in perm[:n_val])
        test_cps = sorted(held_cps[i] for i in perm[n_val:])
        train_cps = sorted(folds.canonical_pairs(train_pos))

        # negatives: pool-matched, 1:1, global-pos-excluded, non-overlapping.
        sf: dict[str, pd.DataFrame] = {}
        used: set = set()
        specs = [("train", train_cps, (list(train_d), list(train_d))),
                 ("val", val_cps, (list(held_d), list(held_d))),
                 ("test", test_cps, (list(held_d), list(held_d)))]
        for sp, pos_cps, (pa, pb) in specs:
            neg = sampler.sample(drug_pool_a=list(pa), drug_pool_b=list(pb),
                                 n_pairs=len(pos_cps), exclude=global_pos | used,
                                 seed=NEG_BASE + S2_TOFF + j * 3
                                 + {"train": 0, "val": 1, "test": 2}[sp])
            used |= folds.canonical_pairs(neg)
            sf[sp] = _bin_split(pos_cps, neg)
            folds.assert_no_dup_pairs(sf[sp], f"{GROUP} S2-640 bin f{j}/{sp}")
            folds.assert_neg_excluded(neg, global_pos, f"{GROUP} S2-640 bin f{j}/{sp}")
        folds.assert_pair_disjoint(sf, f"{GROUP} S2-640 bin f{j}")
        unified.check_cold_s2(sf["train"], [("val", sf["val"]), ("test", sf["test"])])

        s2_bin[f"fold{j}"] = sf
        # val drugs == test drugs (== held_d); no separate val drug group under Option A.
        s2_ds[f"fold{j}"] = folds.build_drug_split_cv(train_d, set(), held_d, du)

    unified.write_dataset(
        unified.layout_dir(OUT, GROUP, "binary", "cold_s2"),
        unified.base_meta(unified.DATASET_DIRS[GROUP], GROUP, "binary", "cold_s2",
                          [f"fold{j}" for j in range(K)], kg=kg,
                          smiles_source="drug_smiles__seed42.csv",
                          n_labels=None, pair_format="canonical"),
        drugs, None, s2_bin, s2_ds)

    # summary
    for j in range(K):
        sf = s2_bin[f"fold{j}"]
        ntr = int((sf["train"].y_bin == 1).sum())
        nva = int((sf["val"].y_bin == 1).sum())
        nte = int((sf["test"].y_bin == 1).sum())
        trd = set(sf["train"].drug_a_id) | set(sf["train"].drug_b_id)
        print(f"[ddi800 S2-640 fold{j}] train_pos={ntr} (ndrug={len(trd)}) "
              f"val_pos={nva} test_pos={nte}", flush=True)
    print(f"[ddi800 S2-640] wrote {unified.layout_dir(OUT, GROUP, 'binary', 'cold_s2')}",
          flush=True)


if __name__ == "__main__":
    main()

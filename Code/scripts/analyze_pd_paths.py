"""Path-structure analysis of PD (pharmacodynamic) DDI pairs vs random drug pairs.

Read-only. Answers three questions on the real merged KG:
  1. What mediators / paths do PD DDI pairs have (shared neighbours by type)?
  2. Which star (hub) nodes do those paths route through (degree of mediators)?
  3. How separable are PD-positive pairs from random drug pairs on simple path
     features — the most basic binary-cls signal.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_paths.py --n-pd 200 --n-rand 200
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from my_code.models.spmn_v1.retrieval import (
    KIND_ORDER,
    MergedKG,
    REL_BUCKET,
)

REL_NAME = {0: "target", 1: "enzyme", 2: "transporter", 3: "carrier",
            4: "db_pathway", 5: "gene_reg", 6: "side_effect",
            7: "indication", 8: "contraindication", 9: "other", 10: "2hop"}

DDI_EDGES = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PK_PD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"


def one_hop(kg: MergedKG, d: int) -> np.ndarray:
    nodes, dist = kg.neighborhood(d, 1)
    return nodes[(dist == 1) & ~kg.is_drug[nodes]]


def within_k(kg: MergedKG, d: int, k: int) -> np.ndarray:
    nodes, dist = kg.neighborhood(d, k)
    return nodes[(dist >= 1) & ~kg.is_drug[nodes]]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pair_features(kg: MergedKG, a: int, b: int) -> dict:
    na, nb = one_hop(kg, a), one_hop(kg, b)
    shared = np.intersect1d(na, nb, assume_unique=False)
    feat: dict = {}
    types = kg.type_id[shared]
    degs = kg.degree[shared]
    feat["n_shared"] = len(shared)
    # per-type 2-star counts
    for ti, tname in enumerate(KIND_ORDER):
        feat[f"sh_{tname}"] = int((types == ti).sum())
    # degree / hub involvement of shared mediators
    feat["sh_maxdeg"] = int(degs.max()) if len(degs) else 0
    feat["sh_meddeg"] = int(np.median(degs)) if len(degs) else 0
    feat["sh_hub_ge1000"] = int((degs >= 1000).sum())
    feat["aa"] = float(np.sum(1.0 / np.log(np.clip(degs, 2, None)))) if len(degs) else 0.0
    # relation buckets on a->m / b->m
    ra, rb = kg.drug_rel.get(a, {}), kg.drug_rel.get(b, {})
    rab = Counter()
    for m in shared.tolist():
        rab[(REL_NAME.get(ra.get(m, 10)), REL_NAME.get(rb.get(m, 10)))] += 1
    feat["_relpairs"] = rab
    feat["_shared"] = shared
    # broader context: shared within 2 hops (AND support l=2)
    wa, wb = within_k(kg, a, 2), within_k(kg, b, 2)
    sh2 = np.intersect1d(wa, wb, assume_unique=False)
    t2 = kg.type_id[sh2]
    feat["n_shared_2hop"] = len(sh2)
    feat["sh2_protein"] = int((t2 == KIND_ORDER.index("protein_gene")).sum())
    feat["sh2_pathway"] = int((t2 == KIND_ORDER.index("pathway")).sum())
    feat["sh2_disease"] = int((t2 == KIND_ORDER.index("disease")).sum())
    return feat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pd", type=int, default=200)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading KG ...", flush=True)
    kg = MergedKG.from_parquet()
    names = pd.read_parquet(NODES, columns=["id", "name", "kind"]).set_index("id")

    # PD ddi types
    pk_pd = pd.read_csv(PK_PD)
    pd_types = set(pk_pd.loc[pk_pd["pk_pd_label"] == "PD", "ddi_type"])
    ddi = pd.read_csv(DDI_EDGES)
    ddi = ddi[ddi["ddi_type"].isin(pd_types)].copy()

    # map to KG idx, keep pairs with both drugs present
    i2x = kg.id_to_idx
    ddi["ai"] = ddi["drug_a_id"].map(i2x)
    ddi["bi"] = ddi["drug_b_id"].map(i2x)
    ddi = ddi.dropna(subset=["ai", "bi"])
    ddi["ai"] = ddi["ai"].astype(int)
    ddi["bi"] = ddi["bi"].astype(int)
    print(f"PD edges in KG: {len(ddi)}  | unique PD drugs: "
          f"{len(set(ddi.ai)|set(ddi.bi))}", flush=True)

    # drug pool = drugs appearing in PD edges; positive set for neg exclusion
    pool = sorted(set(ddi.ai) | set(ddi.bi))
    pos_set = set(map(frozenset, zip(ddi.ai, ddi.bi)))

    pd_idx = rng.choice(len(ddi), size=min(args.n_pd, len(ddi)), replace=False)
    pd_pairs = list(zip(ddi.ai.to_numpy()[pd_idx], ddi.bi.to_numpy()[pd_idx]))

    rand_pairs = []
    while len(rand_pairs) < args.n_rand:
        a, b = rng.choice(pool, 2, replace=False)
        if a != b and frozenset((a, b)) not in pos_set:
            rand_pairs.append((int(a), int(b)))

    print("computing path features ...", flush=True)
    feats = {"PD": [pair_features(kg, a, b) for a, b in pd_pairs],
             "RAND": [pair_features(kg, a, b) for a, b in rand_pairs]}

    def col(grp, key):
        return np.array([f[key] for f in feats[grp]], dtype=float)

    # ---- report 1: shared mediators by type ----
    print("\n=== [1] mean shared 1-hop mediators by type (PD vs RAND) ===")
    print(f"{'type':<22}{'PD':>8}{'RAND':>8}")
    for tname in KIND_ORDER:
        k = f"sh_{tname}"
        mp, mr = col("PD", k).mean(), col("RAND", k).mean()
        if mp > 0.01 or mr > 0.01:
            print(f"{tname:<22}{mp:>8.2f}{mr:>8.2f}")
    print(f"{'TOTAL 1-hop shared':<22}{col('PD','n_shared').mean():>8.2f}"
          f"{col('RAND','n_shared').mean():>8.2f}")
    print(f"{'2-hop protein':<22}{col('PD','sh2_protein').mean():>8.1f}"
          f"{col('RAND','sh2_protein').mean():>8.1f}")
    print(f"{'2-hop pathway':<22}{col('PD','sh2_pathway').mean():>8.1f}"
          f"{col('RAND','sh2_pathway').mean():>8.1f}")
    print(f"{'2-hop disease':<22}{col('PD','sh2_disease').mean():>8.1f}"
          f"{col('RAND','sh2_disease').mean():>8.1f}")

    # ---- report 2: hub involvement ----
    print("\n=== [2] hub (star) involvement of shared mediators ===")
    for grp in ("PD", "RAND"):
        md = col(grp, "sh_maxdeg")
        hub = col(grp, "sh_hub_ge1000")
        frac_pairs_with_hub = (hub > 0).mean()
        print(f"{grp}: median(max mediator deg)={np.median(md):.0f}  "
              f"mean #shared with deg>=1000={hub.mean():.2f}  "
              f"frac pairs routing through a deg>=1000 hub={frac_pairs_with_hub:.2f}")

    # top star nodes across PD pairs
    star_ctr = Counter()
    for f in feats["PD"]:
        for m in f["_shared"].tolist():
            if kg.degree[m] >= 500:
                star_ctr[m] += 1
    print("\ntop star mediators in PD pairs (deg>=500), appearances / 200:")
    for m, c in star_ctr.most_common(12):
        nid = kg.node_ids[m]
        nm = names.loc[nid, "name"] if nid in names.index else "?"
        if isinstance(nm, pd.Series):
            nm = nm.iloc[0]
        print(f"  {c:>3}x  deg={kg.degree[m]:>5}  {KIND_ORDER[kg.type_id[m]]:<14}"
              f"{str(nm)[:40]}")

    # relation-pair profile
    print("\n=== relation buckets on (a->m, b->m) for PD shared mediators ===")
    agg = Counter()
    for f in feats["PD"]:
        agg.update(f["_relpairs"])
    tot = sum(agg.values())
    for (rpa, rpb), c in agg.most_common(10):
        print(f"  {c/max(tot,1)*100:5.1f}%  ({rpa}, {rpb})")

    # ---- report 3: separability (basic binary cls signal) ----
    print("\n=== [3] separability PD vs RAND (1-feature AUROC) ===")
    y = np.r_[np.ones(len(pd_pairs)), np.zeros(len(rand_pairs))]
    for key in ["n_shared", "sh_protein_gene", "aa", "sh_pathway",
                "sh_disease", "n_shared_2hop", "sh2_pathway", "sh2_disease"]:
        s = np.r_[col("PD", key), col("RAND", key)]
        print(f"  {key:<18} AUROC={auroc(s, y):.3f}   "
              f"PD mean={col('PD',key).mean():.2f}  RAND mean={col('RAND',key).mean():.2f}")
    # fraction with ZERO shared 1-hop (the "no path" failure)
    print(f"\n  frac PD pairs with 0 shared 1-hop mediators : "
          f"{(col('PD','n_shared')==0).mean():.2f}")
    print(f"  frac RAND pairs with 0 shared 1-hop mediators: "
          f"{(col('RAND','n_shared')==0).mean():.2f}")


if __name__ == "__main__":
    main()

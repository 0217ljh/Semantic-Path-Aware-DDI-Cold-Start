"""Longer-metapath analysis of PD DDI pairs vs random drug pairs.

Read-only. Tests whether the PD signal lives in length-3/4 mechanistic
metapaths that the 2-star (common-neighbor) readout discards:

  L3 PPI bridge : drug -> protein(u) -- protein(v) <- drug   (u != v, u~v)
  L4 pathway    : drug -> protein -- pathway(w) -- protein <- drug
  L4 disease    : drug -> protein -- disease(w) -- protein <- drug
  L4 bioproc    : drug -> protein -- bio_process(w) -- protein <- drug

For each metapath we report raw count, Adamic-Adar (1/log deg(w)) discounted
count, the connector-node degree (are the bridges themselves hubs?), the
PD-vs-RAND separability (1-feature AUROC), and coverage on the subset of PD
pairs that have NO shared 1-hop mediator at all.

Usage (WSL conda env project_1):
    PYTHONPATH=Code python -u Code/scripts/analyze_pd_metapaths.py --n-pd 200 --n-rand 200
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np
import pandas as pd

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

DDI_EDGES = "Code/data/KG/drugbank/filtered/ddi_edges.csv"
PK_PD = "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"

PROT = KIND_ORDER.index("protein_gene")
PATH = KIND_ORDER.index("pathway")
DIS = KIND_ORDER.index("disease")
BP = KIND_ORDER.index("biological_process")


def nbrs(kg: MergedKG, u: int) -> np.ndarray:
    return kg.indices[kg.indptr[u]:kg.indptr[u + 1]]


def typed_nbrs(kg: MergedKG, u: int, tid: int) -> np.ndarray:
    nb = nbrs(kg, u)
    return nb[kg.type_id[nb] == tid]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
        i = j + 1
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def metapaths(kg: MergedKG, a: int, b: int) -> dict:
    PA = typed_nbrs(kg, a, PROT)
    PB = typed_nbrs(kg, b, PROT)
    PB_set = set(PB.tolist())
    f: dict = {"n_protA": len(PA), "n_protB": len(PB)}

    # L3 PPI bridge: u in PA, v in PB, u~v (distinct proteins)
    ppi = 0
    for u in PA.tolist():
        nb = typed_nbrs(kg, u, PROT)
        ppi += int(np.isin(nb, PB).sum())
    f["L3_ppi"] = ppi

    # L4 bridges via a typed connector w reachable from BOTH sides' proteins
    for cname, ctid in [("pathway", PATH), ("disease", DIS), ("bioproc", BP)]:
        WA: set[int] = set()
        for u in PA.tolist():
            WA.update(typed_nbrs(kg, u, ctid).tolist())
        WB: set[int] = set()
        for v in PB.tolist():
            WB.update(typed_nbrs(kg, v, ctid).tolist())
        conn = WA & WB
        f[f"L4_{cname}"] = len(conn)
        f[f"L4_{cname}_aa"] = sum(
            1.0 / math.log(max(int(kg.degree[w]), 2)) for w in conn)
        f[f"_conn_{cname}"] = conn
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pd", type=int, default=200)
    ap.add_argument("--n-rand", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("loading KG ...", flush=True)
    kg = MergedKG.from_parquet()
    names = pd.read_parquet(NODES, columns=["id", "name"]).set_index("id")

    pk_pd = pd.read_csv(PK_PD)
    pd_types = set(pk_pd.loc[pk_pd["pk_pd_label"] == "PD", "ddi_type"])
    ddi = pd.read_csv(DDI_EDGES)
    ddi = ddi[ddi["ddi_type"].isin(pd_types)].copy()
    i2x = kg.id_to_idx
    ddi["ai"] = ddi["drug_a_id"].map(i2x)
    ddi["bi"] = ddi["drug_b_id"].map(i2x)
    ddi = ddi.dropna(subset=["ai", "bi"]).astype({"ai": int, "bi": int})
    pool = sorted(set(ddi.ai) | set(ddi.bi))
    pos_set = set(map(frozenset, zip(ddi.ai, ddi.bi)))

    pd_idx = rng.choice(len(ddi), size=min(args.n_pd, len(ddi)), replace=False)
    pd_pairs = list(zip(ddi.ai.to_numpy()[pd_idx], ddi.bi.to_numpy()[pd_idx]))
    rand_pairs = []
    while len(rand_pairs) < args.n_rand:
        a, b = rng.choice(pool, 2, replace=False)
        if a != b and frozenset((a, b)) not in pos_set:
            rand_pairs.append((int(a), int(b)))

    # need 1-hop shared (any non-drug) to flag the "no 2-star" subset
    def has_2star(a, b):
        na = nbrs(kg, a)[~kg.is_drug[nbrs(kg, a)]]
        nb = nbrs(kg, b)[~kg.is_drug[nbrs(kg, b)]]
        return len(np.intersect1d(na, nb)) > 0

    print("computing metapaths ...", flush=True)
    F = {"PD": [metapaths(kg, a, b) for a, b in pd_pairs],
         "RAND": [metapaths(kg, a, b) for a, b in rand_pairs]}
    no2star = np.array([not has_2star(a, b) for a, b in pd_pairs])

    def col(grp, k):
        return np.array([f[k] for f in F[grp]], dtype=float)

    y = np.r_[np.ones(len(pd_pairs)), np.zeros(len(rand_pairs))]
    keys = ["L3_ppi", "L4_pathway", "L4_pathway_aa", "L4_disease",
            "L4_disease_aa", "L4_bioproc", "L4_bioproc_aa"]
    print("\n=== metapath separability PD vs RAND ===")
    print(f"{'metapath':<18}{'AUROC':>8}{'PD mean':>10}{'RAND mean':>11}"
          f"{'PD>0%':>8}")
    for k in keys:
        sp, sr = col("PD", k), col("RAND", k)
        s = np.r_[sp, sr]
        print(f"{k:<18}{auroc(s, y):>8.3f}{sp.mean():>10.2f}{sr.mean():>11.2f}"
              f"{(sp > 0).mean()*100:>7.0f}%")

    # connector-degree: are the pathway/disease bridges themselves hubs?
    print("\n=== degree of L4 connector nodes (PD pairs) ===")
    for cname in ("pathway", "disease", "bioproc"):
        degs = []
        for f in F["PD"]:
            degs += [int(kg.degree[w]) for w in f[f"_conn_{cname}"]]
        if degs:
            degs = np.array(degs)
            print(f"{cname:<10} n_connectors={len(degs):>6}  "
                  f"median deg={np.median(degs):>5.0f}  p90={np.percentile(degs,90):>6.0f}  "
                  f"frac deg>=1000={np.mean(degs>=1000):.2f}")

    # coverage on the no-2-star PD subset
    print(f"\n=== coverage on PD pairs WITH NO shared 1-hop mediator "
          f"(n={int(no2star.sum())}/{len(pd_pairs)}) ===")
    for k in ["L3_ppi", "L4_pathway", "L4_disease", "L4_bioproc"]:
        sub = col("PD", k)[no2star]
        if len(sub):
            print(f"  {k:<14} frac with >=1 metapath: {(sub>0).mean():.2f}  "
                  f"mean={sub.mean():.2f}")

    # top disease/pathway connectors (are they generic hubs or specific?)
    for cname in ("disease", "pathway"):
        ctr = Counter()
        for f in F["PD"]:
            for w in f[f"_conn_{cname}"]:
                ctr[w] += 1
        print(f"\ntop L4 {cname} connectors in PD pairs (appearances/200):")
        for w, c in ctr.most_common(8):
            nid = kg.node_ids[w]
            nm = names.loc[nid, "name"] if nid in names.index else "?"
            if isinstance(nm, pd.Series):
                nm = nm.iloc[0]
            print(f"  {c:>3}x  deg={kg.degree[w]:>5}  {str(nm)[:44]}")


if __name__ == "__main__":
    main()

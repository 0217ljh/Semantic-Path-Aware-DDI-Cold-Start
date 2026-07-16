"""Do SMPDB pathways and Reactome pathways describe the SAME biology?

Read-only. If we unify all pathway sources into ONE meso layer, we need to know
whether SMPDB and Reactome are REDUNDANT (same protein sets -> need within-layer
alignment) or COMPLEMENTARY (different content -> just coexist). Measures, per
SMPDB pathway, the max Jaccard of its protein set (Entrez) against any Reactome
pathway's protein set, via an inverted index.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_pathway_smpdb_reactome_overlap.py
"""
from __future__ import annotations

import csv
from collections import defaultdict

import numpy as np
import pandas as pd

EDGES = "Code/data/KG/_merged_kg_dedup_v2/edges__dedup.parquet"
NODES = "Code/data/KG/_merged_kg_dedup_v2/nodes__dedup.parquet"
SMPDB = "Code/data/_cache/smpdb_pathways.parquet"
HGNC = "Code/data/_cache/hgnc_complete_set.txt"


def main() -> None:
    # Reactome pathway -> set(entrez) from prime:pathway_protein edges
    ed = pd.read_parquet(EDGES, columns=["src", "dst", "relation"])
    pp = ed[ed["relation"] == "prime:pathway_protein"]
    rea = defaultdict(set)
    for s, d in zip(pp["src"].astype(str), pp["dst"].astype(str)):
        pw = s if s.startswith("prime:pathway:") else d
        pr = d if s.startswith("prime:pathway:") else s
        if pr.startswith("prot:"):
            rea[pw].add(pr.split(":")[-1])
    print(f"Reactome pathways with proteins: {len(rea)}")

    # UniProt -> Entrez
    uni2ent = {}
    with open(HGNC, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ent = (row.get("entrez_id") or "").strip()
            if ent:
                for acc in (row.get("uniprot_ids") or "").split("|"):
                    if acc.strip():
                        uni2ent[acc.strip().upper()] = ent

    smp = pd.read_parquet(SMPDB)
    smp["entrez"] = smp["proteins"].map(
        lambda ps: set(uni2ent[p.upper()] for p in ps if p.upper() in uni2ent))

    # inverted index entrez -> reactome pathways
    inv = defaultdict(list)
    for pw, es in rea.items():
        for e in es:
            inv[e].append(pw)

    rows = []
    for _, r in smp.iterrows():
        es = r["entrez"]
        if not es:
            rows.append((r["smpdb_id"], r["category"], r["name"], len(es), 0.0, None, 0))
            continue
        cand = set()
        for e in es:
            cand.update(inv.get(e, []))
        best_j, best_pw, best_inter = 0.0, None, 0
        for pw in cand:
            rs = rea[pw]
            inter = len(es & rs)
            j = inter / len(es | rs)
            if j > best_j:
                best_j, best_pw, best_inter = j, pw, inter
        rows.append((r["smpdb_id"], r["category"], r["name"], len(es), best_j, best_pw, best_inter))

    res = pd.DataFrame(rows, columns=["smpdb_id", "category", "name", "n_ent",
                                      "max_jaccard", "best_reactome", "inter"])
    have = res[res["n_ent"] > 0]
    print(f"\n=== SMPDB vs Reactome protein-set overlap ({len(have)} SMPDB pathways w/ entrez) ===")
    for thr in [0.0, 0.1, 0.3, 0.5, 0.7]:
        print(f"  max Jaccard > {thr}: {int((have['max_jaccard'] > thr).sum())} "
              f"({(have['max_jaccard'] > thr).mean()*100:.0f}%)")
    print(f"  median max-Jaccard: {have['max_jaccard'].median():.2f}")
    print("\n  by category (median max-Jaccard with best Reactome twin):")
    for c, sub in have.groupby("category"):
        print(f"    {c:<16} n={len(sub):>4} | median maxJ {sub['max_jaccard'].median():.2f} | "
              f">0.5: {int((sub['max_jaccard']>0.5).sum())}")
    print("\n  examples (SMPDB -> best Reactome twin):")
    for _, r in have.sort_values("max_jaccard", ascending=False).head(6).iterrows():
        print(f"    J={r['max_jaccard']:.2f} [{r['category']}] '{r['name']}' "
              f"~ {r['best_reactome']} (inter {r['inter']}/{r['n_ent']})")
    print("  low-overlap examples (SMPDB-unique biology):")
    for _, r in have[have["max_jaccard"] < 0.1].head(4).iterrows():
        print(f"    J={r['max_jaccard']:.2f} [{r['category']}] '{r['name']}' ({r['n_ent']} proteins)")


if __name__ == "__main__":
    main()

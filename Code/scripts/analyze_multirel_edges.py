"""Audit multi-relation node pairs in the dedup KG (before collapsing to simple graph).

Read-only. The plan is to collapse all parallel edges between the same node pair
into ONE edge (simple graph). Before doing that, check whether any multi-relation
pair is SEMANTICALLY IMPORTANT (the multiple relations carry genuinely different
mechanism, e.g. drug->protein target vs enzyme vs transporter) vs merely redundant
(cross-source synonyms like het:GpBP + prime:bioprocess_protein).

Integer-factorized for memory safety on the 7.1M-edge table.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_multirel_edges.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

EDGES = "Code/data/KG/_merged_kg_dedup/edges__dedup.parquet"
NODES = "Code/data/KG/_merged_kg_dedup/nodes__dedup.parquet"


def main() -> None:
    ed = pd.read_parquet(EDGES, columns=["src", "dst", "relation"])
    nd = pd.read_parquet(NODES, columns=["id", "kind"])

    # factorize node ids (shared code space for src+dst)
    codes, uniq = pd.factorize(pd.concat([ed["src"], ed["dst"]], ignore_index=True))
    n = len(ed)
    sc = codes[:n].astype(np.int64)
    dc = codes[n:].astype(np.int64)
    u = np.minimum(sc, dc)
    v = np.maximum(sc, dc)
    rcode, runiq = pd.factorize(ed["relation"])
    del ed, codes

    df = pd.DataFrame({"u": u, "v": v, "r": rcode.astype(np.int32)}).drop_duplicates()
    df["pid"] = df["u"].values * (len(uniq) + 1) + df["v"].values

    cnt = df.groupby("pid")["r"].size()
    total_pairs = len(cnt)
    dist = cnt.value_counts().sort_index()
    print("=== distinct relations per undirected node pair ===")
    for k, c in dist.items():
        print(f"  {k} relation(s): {c} pairs ({c/total_pairs*100:.2f}%)")
    n_multi = int((cnt > 1).sum())
    n_now = len(df)
    print(f"  multi-relation pairs: {n_multi} ({n_multi/total_pairs*100:.2f}%)")
    print(f"\n  distinct (pair,relation) edges : {n_now}")
    print(f"  pairs (simple-graph edges)     : {total_pairs}")
    print(f"  edges removed by collapse      : {n_now - total_pairs} "
          f"({(n_now-total_pairs)/n_now*100:.1f}% of distinct-relation edges)")

    # combos among multi-rel pairs
    df = df.merge(cnt.rename("nr"), left_on="pid", right_index=True)
    dm = df[df["nr"] > 1]
    combo = dm.groupby("pid")["r"].agg(lambda x: tuple(sorted(set(x))))

    def name_combo(t):
        return " + ".join(runiq[i] for i in t)

    print("\n=== top relation-combinations among multi-relation pairs ===")
    for t, c in Counter(combo).most_common(25):
        print(f"  {c:>7}  {name_combo(t)}")

    # drug-incident multi-rel pairs
    drug_ids = set(nd[nd["kind"].astype(str).isin(["Drug", "drug"])]["id"])
    drug_codes = {i for i, nid in enumerate(uniq) if nid in drug_ids}
    dm2 = dm[dm["u"].isin(drug_codes) | dm["v"].isin(drug_codes)]
    dcombo = dm2.groupby("pid")["r"].agg(lambda x: tuple(sorted(set(x))))
    print(f"\n=== drug-incident multi-relation pairs: {dcombo.nunique() if len(dcombo) else 0} pairs ===")
    for t, c in Counter(dcombo).most_common(20):
        print(f"  {c:>7}  {name_combo(t)}")


if __name__ == "__main__":
    main()

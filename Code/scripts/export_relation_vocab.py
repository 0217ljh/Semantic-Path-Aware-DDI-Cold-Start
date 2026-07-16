"""List the full relation vocabulary of the merged KG with counts + endpoints.

Read-only. Prints every distinct `relation` value, its edge count, the typical
(src_kind -> dst_kind), directedness, and source KG.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/export_relation_vocab.py
"""
from __future__ import annotations

import pandas as pd

EDGES = "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def main() -> None:
    ed = pd.read_parquet(EDGES, columns=["src_kind", "dst_kind", "relation",
                                         "source_kg", "directed"])
    print("total distinct relations:", ed["relation"].nunique())
    g = (ed.groupby("relation")
           .agg(n=("relation", "size"), src=("src_kind", "first"),
                dst=("dst_kind", "first"), kg=("source_kg", "first"),
                directed=("directed", "first"))
           .sort_values("n", ascending=False))
    for rel, r in g.iterrows():
        d = "dir" if r["directed"] else "undir"
        print(f"{r['n']:>9}  {str(rel):<34} {str(r['src'])[:20]:<20} -> "
              f"{str(r['dst'])[:20]:<20} {d:<5} [{r['kg']}]")


if __name__ == "__main__":
    main()

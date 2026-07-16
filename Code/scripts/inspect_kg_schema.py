"""Quick inspection of merged KG edge / node parquet schema (for PMP cache builder)."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"

nodes = pd.read_parquet(NODES)
edges = pd.read_parquet(EDGES)

print("=" * 60)
print("NODES")
print("=" * 60)
print("columns:", nodes.columns.tolist())
print("dtype:")
print(nodes.dtypes)
print("head:")
print(nodes.head(3))
print(f"\nn_nodes = {len(nodes)}")
print(f"n_kinds = {nodes['kind'].nunique()}")
print(f"sample kinds: {nodes['kind'].value_counts().head(15)}")

print("\n" + "=" * 60)
print("EDGES")
print("=" * 60)
print("columns:", edges.columns.tolist())
print("dtype:")
print(edges.dtypes)
print("head:")
print(edges.head(3))
print(f"\nn_edges = {len(edges)}")
if "rel" in edges.columns:
    print(f"n_rel_types = {edges['rel'].nunique()}")
    print(f"sample rels: {edges['rel'].value_counts().head(20)}")
if "directed" in edges.columns:
    print(f"directed values: {edges['directed'].value_counts()}")

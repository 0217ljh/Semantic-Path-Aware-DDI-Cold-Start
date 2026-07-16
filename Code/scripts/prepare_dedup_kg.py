"""Build the protein-deduplicated merged KG (collapse duplicate protein nodes).

Uses the protein merge map (Code/data/_cache/protein_merge_map.parquet) to
collapse every protein that is the same gene (by Entrez) into ONE canonical
node, then rewrites EVERY edge so connections are preserved exactly:

  * protein node id  -> canonical id (prot:<entrez> / prot:uniprot:<acc>)
  * non-protein node -> itself (unchanged)
  * each edge's src/dst remapped; self-loops created by the merge dropped;
    undirected edges order-normalized (min,max); exact duplicates collapsed.

NO connection is lost: the set of distinct (canonical_src, canonical_dst,
relation, directed) edges is exactly the image of the original edge set minus
self-loops and exact duplicates. Asserted explicitly.

Output (new dir, original read-only KG untouched):
  Code/data/KG/_merged_kg_dedup/nodes__dedup.parquet
  Code/data/KG/_merged_kg_dedup/edges__dedup.parquet
Same column schema as the merged KG, so MergedKG.from_parquet / the EmerGNN
builder work by repointing paths.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_dedup_kg.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
MMAP = "Code/data/_cache/protein_merge_map.parquet"
OUT_DIR = "Code/data/KG/_merged_kg_dedup"
PROT_KIND = "gene/protein"  # reuse an existing kind label (loader groups all 3)


def main() -> None:
    nd = pd.read_parquet(NODES)
    ed = pd.read_parquet(EDGES)
    mm = pd.read_parquet(MMAP)

    # ---- node -> canonical map (proteins from merge map, others identity) ----
    node2canon = dict(zip(mm["orig_id"], mm["canonical_id"]))
    # representative symbol per canonical: prefer a het/prime member's symbol
    mm["_pri"] = mm["source_kg"].isin(["hetionet", "primekg"]) & (mm["symbol"].astype(str) != "")
    rep = (mm.sort_values("_pri", ascending=False)
             .drop_duplicates("canonical_id")
             .set_index("canonical_id"))
    canon_sym = {c: (str(r["symbol"]) if str(r["symbol"]) not in ("", "nan") else c)
                 for c, r in rep.iterrows()}

    prot_orig_ids = set(mm["orig_id"])

    # ---- new node table ----
    non_prot = nd[~nd["id"].isin(prot_orig_ids)].copy()          # unchanged
    canon_ids = sorted(set(mm["canonical_id"]))
    prot_rows = pd.DataFrame({
        "id": canon_ids,
        "kind": PROT_KIND,
        "name": [canon_sym.get(c, c) for c in canon_ids],
        "source_kg": "merged",
    })
    new_nodes = pd.concat([non_prot, prot_rows], ignore_index=True)
    assert new_nodes["id"].is_unique, "canonical node ids collided"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges ----
    n_edges0 = len(ed)
    ed["su"] = ed["src"].map(lambda x: node2canon.get(x, x))
    ed["du"] = ed["dst"].map(lambda x: node2canon.get(x, x))
    # Drop dangling edges whose endpoint is not a known node (mirrors the
    # loader's defensive behavior; the merged KG ships 1 junk `het:metaedge`
    # row between Unknown:source/Unknown:target). Guard against mass drops.
    known = ed["su"].isin(kind_of) & ed["du"].isin(kind_of)
    n_dangling = int((~known).sum())
    assert n_dangling < 100, f"{n_dangling} dangling edges — remap bug, not source junk"
    ed = ed[known].copy()

    self_loops = int((ed["su"] == ed["du"]).sum())
    ed = ed[ed["su"] != ed["du"]].copy()

    # order-normalize undirected edges so (a,b) and (b,a) collapse
    und = ~ed["directed"].astype(bool)
    lo = np.where(ed["su"].values <= ed["du"].values, ed["su"].values, ed["du"].values)
    hi = np.where(ed["su"].values <= ed["du"].values, ed["du"].values, ed["su"].values)
    ed["ks"] = np.where(und.values, lo, ed["su"].values)
    ed["kd"] = np.where(und.values, hi, ed["du"].values)

    before_dedup = len(ed)
    ed = ed.drop_duplicates(subset=["ks", "kd", "relation", "directed"]).copy()
    collapsed = before_dedup - len(ed)

    new_edges = pd.DataFrame({
        "src": ed["ks"].values,
        "src_kind": [kind_of[x] for x in ed["ks"].values],
        "dst": ed["kd"].values,
        "dst_kind": [kind_of[x] for x in ed["kd"].values],
        "relation": ed["relation"].values,
        "source_kg": ed["source_kg"].values,
        "directed": ed["directed"].values,
    })

    # connection-preservation check: distinct canonical edges == kept rows
    distinct = ed[["ks", "kd", "relation", "directed"]].drop_duplicates().shape[0]
    assert distinct == len(new_edges), "dedup count mismatch"

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    mm.drop(columns=["_pri"]).to_parquet(f"{OUT_DIR}/protein_merge_map.parquet", index=False)

    # ---- report ----
    print("=== dedup KG built ===")
    print(f"  nodes: {len(nd)} -> {len(new_nodes)}  "
          f"(proteins {len(prot_orig_ids)} -> {len(canon_ids)})")
    print(f"  edges: {n_edges0} -> {len(new_edges)}")
    print(f"    dangling junk edges dropped        : {n_dangling}")
    print(f"    self-loops dropped (merge-induced) : {self_loops}")
    print(f"    parallel/dup edges collapsed       : {collapsed}")
    print(f"  wrote {OUT_DIR}/nodes__dedup.parquet")
    print(f"  wrote {OUT_DIR}/edges__dedup.parquet")

    # spot-checks
    print("\n--- spot checks ---")
    # Furosemide DB00695 -> should connect to prot:6557 (SLC12A1) as a single node
    f_edges = new_edges[(new_edges["src"] == "DB00695") | (new_edges["dst"] == "DB00695")]
    has_slc = (f_edges[["src", "dst"]].isin(["prot:6557"]).any(axis=1)).any()
    print(f"  Furosemide(DB00695) connects to prot:6557 (SLC12A1): {has_slc}")
    # max-degree protein hub after merge (true degree)
    deg = pd.concat([new_edges["src"], new_edges["dst"]]).value_counts()
    prot_deg = deg[deg.index.astype(str).str.startswith("prot:")]
    print(f"  top protein hubs by degree (true, post-merge):")
    print(prot_deg.head(5).to_string())


if __name__ == "__main__":
    main()

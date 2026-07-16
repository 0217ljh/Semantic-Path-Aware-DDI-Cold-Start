"""Pathway renormalization — STEP 2: PathwayCommons (Hetionet) axis (v3 -> v4).

Option (b), codex-approved: do NOT collapse PC into Reactome by name (false-merge
risk). Instead:
  * reformat het:Pathway:Pathway::PC7_x  ->  path:PC7_x  (id reformat, NOT a merge)
  * rename relation  het:GpPW -> prime:pathway_protein   (unify protein->pathway)
  * mark source=pathwaycommons; PC pathways get NO Reactome hierarchy / macro roll-up
  * record the name matches to Reactome as a `candidate_reactome` provenance list
    (a candidate alignment, not an identity collapse)
  * keep ALL het:GpPW protein edges (connection preservation)

Reactome (path:R-HSA) and SMPDB (db:pathway) nodes are UNTOUCHED.
Acceptance = correctness/completeness of representation (NOT discriminability).

Output: Code/data/KG/_merged_kg_dedup_v4/{nodes__dedup,edges__dedup,path_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_pathway_pc.py
"""
from __future__ import annotations

import os
import re

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v3"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v4"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    meta = pd.read_parquet(f"{IN_DIR}/path_meta.parquet")
    print(f"v3 in: {len(nd)} nodes, {len(ed)} edges, {len(meta)} path_meta")

    # ---- het:Pathway -> path:PC7 rename map ----
    is_pc = nd["id"].astype(str).str.startswith("het:Pathway")
    pc_nodes = nd[is_pc].copy()
    ren = {x: f"path:{str(x).split('::')[-1]}" for x in pc_nodes["id"]}   # ...::PC7_x -> path:PC7_x
    print(f"PC pathway nodes: {len(pc_nodes)}")

    # Hetionet's Pathway nodes mix two sub-sources: Pathway Commons (PC7_*) and
    # WikiPathways (WP*). Record the ACCURATE sub-source (WP ids are explicitly
    # WikiPathways; PC7 ids are Pathway Commons aggregator, ultimate source unknown).
    def src_of(pid: str) -> str:
        return "wikipathways" if pid.startswith("path:WP") else "pathwaycommons"

    # ---- new node table ----
    non_pc = nd[~is_pc].copy()
    pc_ids = [ren[x] for x in pc_nodes["id"]]
    pc_rows = pd.DataFrame({
        "id": pc_ids,
        "kind": "pathway",
        "name": pc_nodes["name"].values,
        "source_kg": [src_of(p) for p in pc_ids],
    })
    new_nodes = pd.concat([non_pc, pc_rows], ignore_index=True)
    assert new_nodes["id"].is_unique, "node id collision (PC vs Reactome?)"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- new edges: remap PC endpoints + rename het:GpPW ----
    n_gpw = int((ed["relation"] == "het:GpPW").sum())
    e = ed.copy()
    e["src"] = e["src"].map(lambda x: ren.get(x, x))
    e["dst"] = e["dst"].map(lambda x: ren.get(x, x))
    e["relation"] = e["relation"].map(lambda r: "prime:pathway_protein" if r == "het:GpPW" else r)
    # refresh kinds for renamed pathway endpoints
    e["src_kind"] = [kind_of.get(s, k) for s, k in zip(e["src"], e["src_kind"])]
    e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(e["dst"], e["dst_kind"])]
    new_edges = e
    valid = set(kind_of)
    bad = ~(new_edges["src"].isin(valid) & new_edges["dst"].isin(valid))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ---- candidate_reactome (name match to Reactome path:R-HSA) ----
    rea = new_nodes[new_nodes["id"].astype(str).str.startswith("path:R-HSA")]
    rea_name2ids = {}
    for i, nm in zip(rea["id"], rea["name"]):
        rea_name2ids.setdefault(norm(nm), []).append(i)
    # n_protein per PC pathway (after rename)
    pp = new_edges[new_edges["relation"] == "prime:pathway_protein"]
    from collections import Counter
    nprot = Counter()
    for s, d in zip(pp["src"].astype(str), pp["dst"].astype(str)):
        pw = s if s.startswith("path:") else d
        nprot[pw] += 1

    pc_meta = []
    n_cand = 0
    for x in pc_nodes["id"]:
        pid = ren[x]
        nm = new_nodes.loc[new_nodes["id"] == pid, "name"].iloc[0]
        cand = [c.split(":")[-1] for c in rea_name2ids.get(norm(nm), [])]   # R-HSA ids
        if cand:
            n_cand += 1
        pc_meta.append({"id": pid, "source": src_of(pid), "source_ids": [pid.split(":")[-1]],
                        "level": "na", "macro_ancestors": [], "is_connector_only": False,
                        "obsolete": False, "n_protein": int(nprot.get(pid, 0)),
                        "candidate_reactome": cand})
    pc_meta = pd.DataFrame(pc_meta)
    meta = meta.copy()
    meta["candidate_reactome"] = [[] for _ in range(len(meta))]   # Reactome rows: no candidate
    new_meta = pd.concat([meta, pc_meta], ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    new_meta.to_parquet(f"{OUT_DIR}/path_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v4 built (PC axis) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} (het:Pathway {len(pc_nodes)} -> path:PC7 reformatted)")
    print(f"  edges {len(ed)} -> {len(new_edges)} (unchanged count; relations relabeled)")
    print(f"  het:GpPW {n_gpw} -> prime:pathway_protein (preserved): "
          f"{'OK' if int((new_edges['relation']=='prime:pathway_protein').sum()) == 85292 + n_gpw else 'MISMATCH'}")
    print(f"  total prime:pathway_protein now: {int((new_edges['relation']=='prime:pathway_protein').sum())} "
          f"(Reactome 85292 + PC {n_gpw})")
    print(f"  het:GpPW remaining (should be 0): {int((new_edges['relation']=='het:GpPW').sum())}")
    print(f"  het:Pathway nodes remaining (should be 0): {int(new_nodes['id'].astype(str).str.startswith('het:Pathway').sum())}")
    print(f"  PC pathways with candidate_reactome (name match): {n_cand}/{len(pc_nodes)}")
    print(f"  PC pathways with NO macro roll-up (correct): {len(pc_meta)} all level='na', macro_ancestors=[]")
    print(f"  Reactome path:R-HSA untouched: {int(new_nodes['id'].astype(str).str.startswith('path:R-HSA').sum())} | "
          f"SMPDB db:pathway untouched: {int(new_nodes['id'].astype(str).str.startswith('db:pathway').sum())}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,path_meta).parquet")


if __name__ == "__main__":
    main()

"""Exposure axis canonicalization (v9 -> v10): flat MeSH-chemical association layer.

Codex-locked call (thread 019f1728): exposure is NOT a true is_a meso ontology axis
like protein/pathway/GO/anatomy/disease/phenotype. Its 818 nodes are MeSH chemicals
(430 D-records + 388 C-records, environmental exposures from CTD), and
prime:exposure_exposure is a DENSE many-to-many chemical-relatedness graph (mean
degree 3.3, max 98), NOT a taxonomy. So: canonicalize + keep flat, NO hierarchy,
NO macro. Precedent: GO MF/CC oriented-but-no-macro; flat endpoint types stay typed
without a forced ontology.

Rules:
  * prime:exposure:<X> -> exp:<X> (distinct prefix, NOT bare mesh: to avoid collision
    with the phenotype axis' sym:<mesh>; mesh_id kept in sidecar). kind stays 'exposure'.
  * prime:exposure_exposure -> exp:chemical_related, kept UNDIRECTED (directed=False);
    no fabricated parent-child direction.
  * keep exposure_disease / _bioprocess / _protein / _molfunc / _cellcomp (endpoints remapped).
  * sidecar exp_meta {id, name, record_type(D/C), mesh_id, n_related, n_disease,
    n_protein, n_bioprocess}. NO is_a, NO macro. 88 drug-name overlaps NOT folded
    (name overlap != identity).

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v10/{nodes__dedup,edges__dedup,path_meta,go_meta,
        anat_meta,dis_meta,dis_fold_map,hp_meta,exp_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_exposure.py
"""
from __future__ import annotations

import os
from collections import Counter

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v9"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v10"
OLD_REL = "prime:exposure_exposure"
NEW_REL = "exp:chemical_related"
EXP_EXT = ["prime:exposure_disease", "prime:exposure_bioprocess", "prime:exposure_protein",
           "prime:exposure_molfunc", "prime:exposure_cellcomp"]


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v9 in: {len(nd)} nodes, {len(ed)} edges")

    # ---- rename exposure nodes prime:exposure:X -> exp:X ----
    is_exp = nd["kind"].astype(str) == "exposure"
    exp_nodes = nd[is_exp].copy()
    ren = {x: f"exp:{str(x).split(':')[-1]}" for x in exp_nodes["id"].astype(str)}
    print(f"exposure nodes: {len(exp_nodes)} -> exp:")

    non_exp = nd[~is_exp].copy()
    exp_rows = pd.DataFrame({"id": [ren[x] for x in exp_nodes["id"].astype(str)], "kind": "exposure",
                             "name": list(exp_nodes["name"]), "source_kg": "ctd_mesh"})
    new_nodes = pd.concat([non_exp, exp_rows], ignore_index=True).drop_duplicates("id")
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges; rename exposure_exposure relation ----
    e = ed.copy()
    e["src"] = e["src"].map(lambda x: ren.get(str(x), x))
    e["dst"] = e["dst"].map(lambda x: ren.get(str(x), x))
    e.loc[e["relation"] == OLD_REL, "relation"] = NEW_REL          # honest relatedness, keep directed=False
    # canonicalize undirected endpoints (src<=dst) so any mirrored (A,B)/(B,A) collapses in dedup
    m = e["relation"] == NEW_REL
    if m.any():
        sd = [tuple(sorted((str(s), str(d)))) for s, d in zip(e.loc[m, "src"], e.loc[m, "dst"])]
        e.loc[m, "src"] = [p[0] for p in sd]
        e.loc[m, "dst"] = [p[1] for p in sd]
    e["src_kind"] = [kind_of.get(s, k) for s, k in zip(e["src"], e["src_kind"])]
    e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(e["dst"], e["dst_kind"])]
    n_before = len(e)
    e = e.drop_duplicates(["src", "dst", "relation", "directed"])
    new_edges = e
    bad = ~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ---- exp_meta sidecar (NO hierarchy / NO macro) ----
    rel = new_edges["relation"].astype(str)
    n_related, n_dis, n_prot, n_bp = Counter(), Counter(), Counter(), Counter()
    for r, cnt, want_exp in ((NEW_REL, n_related, True),
                             ("prime:exposure_disease", n_dis, False),
                             ("prime:exposure_protein", n_prot, False),
                             ("prime:exposure_bioprocess", n_bp, False)):
        sub = new_edges[rel == r]
        for s, d in zip(sub["src"].astype(str), sub["dst"].astype(str)):
            for x in (s, d):
                if x.startswith("exp:"):
                    cnt[x] += 1
    rows = []
    for x in exp_rows["id"]:
        mesh = x.split(":")[-1]
        rows.append({"id": x, "name": new_nodes.loc[new_nodes["id"] == x, "name"].iloc[0],
                     "record_type": mesh[0], "mesh_id": mesh,
                     "n_related": int(n_related.get(x, 0)), "n_disease": int(n_dis.get(x, 0)),
                     "n_protein": int(n_prot.get(x, 0)), "n_bioprocess": int(n_bp.get(x, 0))})
    exp_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in ("path_meta", "go_meta", "anat_meta", "dis_meta", "dis_fold_map", "hp_meta"):
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)
    exp_meta.to_parquet(f"{OUT_DIR}/exp_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v10 built (exposure axis, flat) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | exposure {int(is_exp.sum())} -> exp: {len(exp_rows)}")
    print(f"  edges {len(ed)} -> {len(new_edges)} (dedup {n_before-len(new_edges)})")
    print(f"  {OLD_REL} -> {NEW_REL}: {int((rel==NEW_REL).sum())} (undirected)")
    for r in EXP_EXT:
        print(f"  {r:<28} v9 {int((ed['relation']==r).sum())} -> v10 {int((new_edges['relation']==r).sum())}")
    print(f"  no prime:exposure:* left: {int(new_nodes['id'].astype(str).str.startswith('prime:exposure:').sum())==0}")
    print(f"  record_type dist: {exp_meta['record_type'].value_counts().to_dict()}")
    print(f"  exp_meta {len(exp_meta)} | NO hierarchy, NO macro (not an ontology-backed meso axis)")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,+sidecars,exp_meta).parquet")


if __name__ == "__main__":
    main()

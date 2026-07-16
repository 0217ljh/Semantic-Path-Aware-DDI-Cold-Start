"""Pharmacologic Class canonicalization (v13 -> v14): flat NDF-RT drug-grouping axis.

Codex-locked (thread 019f1747 audit + 019f1a15 confirm). The 345 'Pharmacologic Class'
het nodes (id het:...::N0000000069, NDF-RT/NUI codes, names like 'P-Glycoprotein
Inhibitors', 'Cytochrome P450 2D6 Inducers', 'Calcium Channel Antagonists') are a real
DDI-relevant drug-grouping signal (724/8048 drugs attached, the CYP/P-gp classes are the
PK-interaction mechanism groupings). NORMALIZE, do not drop. Flat (0 class-class edges) ->
NO hierarchy, NO macro, like the exposure axis. This is a MIXED PK/PD grouping axis, not a
pure mechanism ontology.

Rules:
  * rename het:...::N0000000069 -> `pclass:N0000000069`, kind='pharmacologic_class',
    source_kg='ndfrt'; ndfrt_id kept in sidecar.
  * rename het:PCiC -> `pclass:includes_drug`, canonical direction class -> drug.
  * NO hierarchy / NO macro. sidecar pclass_meta {id, name, ndfrt_id, n_drugs}.
  * remap + dedup. After this ZERO capitalized het:* kinds remain (full canonical coverage).

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v14/{nodes__dedup,edges__dedup,+all sidecars,pclass_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_pharmclass.py
"""
from __future__ import annotations

import os
from collections import Counter

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v13"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v14"
OLD_REL = "het:PCiC"
NEW_REL = "pclass:includes_drug"
SIDECARS = ("path_meta", "go_meta", "anat_meta", "dis_meta", "dis_fold_map", "hp_meta", "exp_meta", "drug_meta")


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v13 in: {len(nd)} nodes, {len(ed)} edges")

    is_pc = nd["kind"].astype(str) == "Pharmacologic Class"
    pc_nodes = nd[is_pc].copy()
    ren = {x: f"pclass:{str(x).split('::')[-1]}" for x in pc_nodes["id"].astype(str)}
    print(f"Pharmacologic Class nodes: {len(pc_nodes)} -> pclass:")

    non_pc = nd[~is_pc].copy()
    pc_rows = pd.DataFrame({"id": [ren[x] for x in pc_nodes["id"].astype(str)], "kind": "pharmacologic_class",
                            "name": list(pc_nodes["name"]), "source_kg": "ndfrt"})
    new_nodes = pd.concat([non_pc, pc_rows], ignore_index=True).drop_duplicates("id")
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))
    pcset = set(ren.values())

    # ---- remap + canonicalize relation direction (class -> drug) ----
    e = ed.copy()
    e["src"] = e["src"].map(lambda x: ren.get(str(x), x))
    e["dst"] = e["dst"].map(lambda x: ren.get(str(x), x))
    m = e["relation"].astype(str) == OLD_REL
    # ensure src=pclass, dst=drug for the membership relation
    if m.any():
        sd = []
        for s, d in zip(e.loc[m, "src"].astype(str), e.loc[m, "dst"].astype(str)):
            sd.append((s, d) if s in pcset else (d, s))
        e.loc[m, "src"] = [p[0] for p in sd]
        e.loc[m, "dst"] = [p[1] for p in sd]
        e.loc[m, "relation"] = NEW_REL
        e.loc[m, "directed"] = True
    e["src_kind"] = [kind_of.get(s, k) for s, k in zip(e["src"], e["src_kind"])]
    e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(e["dst"], e["dst_kind"])]
    n_before = len(e)
    e = e.drop_duplicates(["src", "dst", "relation", "directed"])
    new_edges = e
    bad = ~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"
    # defensive: every membership edge must be exactly pclass(src) -> drug(dst)
    mem = new_edges[new_edges["relation"] == NEW_REL]
    assert mem["src"].isin(pcset).all() and (mem["dst_kind"].astype(str) == "drug").all(), \
        "pclass:includes_drug has a non-(pclass->drug) endpoint"

    # ---- pclass_meta sidecar ----
    n_drugs = Counter()
    sub = new_edges[new_edges["relation"] == NEW_REL]
    for s in sub["src"].astype(str):
        if s.startswith("pclass:"):
            n_drugs[s] += 1
    rows = [{"id": ren[x], "name": nm, "ndfrt_id": str(x).split("::")[-1], "n_drugs": int(n_drugs.get(ren[x], 0))}
            for x, nm in zip(pc_nodes["id"].astype(str), pc_nodes["name"])]
    pclass_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in SIDECARS:
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)
    pclass_meta.to_parquet(f"{OUT_DIR}/pclass_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v14 built (Pharmacologic Class) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | Pharmacologic Class {len(pc_nodes)} -> pclass: {len(pc_rows)}")
    print(f"  edges {len(ed)} -> {len(new_edges)} (dedup {n_before-len(new_edges)})")
    print(f"  {OLD_REL} -> {NEW_REL}: {int((new_edges['relation']==NEW_REL).sum())} (class->drug)")
    print(f"  no het:Pharmacologic left: {int(new_nodes['id'].astype(str).str.startswith('het:Pharmacologic').sum())==0}")
    cap = [k for k in new_nodes['kind'].astype(str).unique() if k[:1].isupper()]
    assert not cap, f"non-canonical capitalized kinds still present: {cap}"   # hard postcondition (last axis)
    print(f"  remaining capitalized (non-canonical) kinds: NONE — full canonical coverage")
    print(f"  pclass_meta {len(pclass_meta)} | n_drugs: median {int(pclass_meta['n_drugs'].median())} max {int(pclass_meta['n_drugs'].max())}")
    nm = dict(zip(new_nodes['id'], new_nodes['name']))
    for s, d in zip(sub['src'].astype(str).head(3), sub['dst'].astype(str).head(3)):
        print(f"  sample: {nm.get(s)} -includes_drug-> {nm.get(d)}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,+{len(SIDECARS)} sidecars,pclass_meta).parquet")


if __name__ == "__main__":
    main()

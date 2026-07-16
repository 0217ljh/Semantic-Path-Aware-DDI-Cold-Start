"""Functional-family cross-links (v10 -> v11): connect GO-MF<->GO-BP<->Reactome pathway.

Codex-locked call (thread 019f19bb): the functional axes (GO-BP/MF, pathway) are
underconnected above the protein bottleneck (physiological axes already got cross-links,
functional did not). Add ONLY authoritative structural edges that biologically exist;
do NOT fabricate macro/semantic-equivalence alignment. Governing rule:
"authoritative topology yes, semantic equivalence no."

Two authoritative additions, both from the FULL go.obo (Code/data/_cache/go-full.obo,
which the namespace-segregated go-basic stripped):
  * MF part_of BP cross-namespace: `go:mf_part_of_bp` (MF -> BP, directed part->whole).
    Only DIRECT asserted part_of among already-present go: nodes. No node expansion, no
    transitive closure, no `consider`/inferred links. Existing same-namespace go:part_of
    invariant left UNTOUCHED (this is a separately-named supplement).
  * GO-BP -> Reactome pathway: `go:reactome_bp`, from authoritative `xref: Reactome:R-HSA-*`
    in go.obo (NOT name matching). Only where both the GO term and the path:R-HSA node are
    already present. Connects the pathway axis directly to GO-BP (the missing go<->path link).

Cross-FAMILY macro equivalence (functional vs physiological) is deliberately NOT added
(no authoritative edge -> left to the model: parallel feature views / cross-attention).

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v11/{nodes__dedup,edges__dedup,+all sidecars}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_functional_links.py
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v10"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v11"
GO_FULL = "Code/data/_cache/go-full.obo"
SIDECARS = ("path_meta", "go_meta", "anat_meta", "dis_meta", "dis_fold_map", "hp_meta", "exp_meta")


def gid(s):
    m = re.search(r"GO:(\d+)", s)
    return int(m.group(1)) if m else None


def parse_go_full(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"id": None, "ns": None, "obs": False, "alt": [], "part_of": set(), "reactome": set()}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: GO:"):
                cur["id"] = gid(line); terms[cur["id"]] = cur
            elif line.startswith("namespace:"):
                cur["ns"] = line.split(":", 1)[1].strip()
            elif line.startswith("is_obsolete: true"):
                cur["obs"] = True
            elif line.startswith("alt_id: GO:"):
                cur["alt"].append(gid(line))
            elif line.startswith("relationship: part_of GO:"):
                u = gid(line)
                if u is not None:
                    cur["part_of"].add(u)
            elif line.startswith("xref: Reactome:R-HSA-"):
                m = re.search(r"R-HSA-(\d+)", line)
                if m:
                    cur["reactome"].add("R-HSA-" + m.group(1))
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v10 in: {len(nd)} nodes, {len(ed)} edges")
    terms, altmap = parse_go_full(GO_FULL)
    ns = {n: t["ns"] for n, t in terms.items()}
    print(f"go-full terms {len(terms)} | alt_id {len(altmap)}")

    def canon(N):
        return altmap.get(N, N)

    # present nodes (do NOT expand the node set)
    go_present = {int(str(i).split(":")[-1]) for i in nd["id"].astype(str) if i.startswith("go:")}
    path_present = {str(i).split(":", 1)[-1] for i in nd["id"].astype(str) if i.startswith("path:")}
    valid = set(nd["id"])

    # ---- (1) MF part_of BP (cross-namespace), present-only, direct asserted ----
    mf_bp = []
    for n, t in terms.items():
        if t["obs"] or ns.get(n) != "molecular_function":
            continue
        nc = canon(n)
        if nc not in go_present:
            continue
        for p in t["part_of"]:
            pc = canon(p)
            if ns.get(pc) == "biological_process" and pc in go_present and pc != nc:
                mf_bp.append((f"go:{nc}", f"go:{pc}"))
    mf_bp = sorted(set(mf_bp))
    mf_df = pd.DataFrame({"src": [s for s, _ in mf_bp], "src_kind": "molecular_function",
                          "dst": [d for _, d in mf_bp], "dst_kind": "biological_process",
                          "relation": "go:mf_part_of_bp", "source_kg": "go", "directed": True})

    # ---- (2) GO-BP -> Reactome pathway, authoritative xref, present-only ----
    go_path = []
    for n, t in terms.items():
        if t["obs"] or ns.get(n) != "biological_process":   # BP-only sources (contract: BP -> pathway)
            continue
        nc = canon(n)
        if nc not in go_present:
            continue
        for r in t["reactome"]:
            if r in path_present:
                go_path.append((f"go:{nc}", f"path:{r}"))
    go_path = sorted(set(go_path))
    gp_df = pd.DataFrame({"src": [s for s, _ in go_path], "src_kind": "biological_process",
                          "dst": [d for _, d in go_path], "dst_kind": "pathway",
                          "relation": "go:reactome_bp", "source_kg": "go", "directed": True})

    new_edges = pd.concat([ed, mf_df, gp_df], ignore_index=True)
    new_edges = new_edges.drop_duplicates(["src", "dst", "relation", "directed"])
    bad = ~(new_edges["src"].isin(valid) & new_edges["dst"].isin(valid))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    os.makedirs(OUT_DIR, exist_ok=True)
    nd.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in SIDECARS:
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)

    # ---- verification ----
    nm = dict(zip(nd["id"], nd["name"]))
    print(f"\n=== v11 built (functional-family cross-links) ===")
    print(f"  nodes unchanged: {len(nd)} | edges {len(ed)} -> {len(new_edges)} (+{len(new_edges)-len(ed)})")
    print(f"  +go:mf_part_of_bp (MF->BP): {len(mf_df)} | +go:reactome_bp (BP->pathway): {len(gp_df)}")
    print(f"  node set unchanged (no expansion): {len(nd)==len(pd.read_parquet(f'{OUT_DIR}/nodes__dedup.parquet'))}")
    # connectivity check: is GO-MF now reachable to GO-BP and to pathway without going through protein?
    print(f"  distinct MF terms now linked to BP: {mf_df['src'].nunique()} | distinct BP linked to pathway: {gp_df['src'].nunique()}")
    print("  sample MF->BP:")
    for s, d in mf_bp[:3]:
        print(f"    {nm.get(s)} -part_of-> {nm.get(d)}")
    print("  sample BP->pathway:")
    for s, d in go_path[:3]:
        print(f"    {nm.get(s)} -reactome_bp-> {nm.get(d)}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,+{len(SIDECARS)} sidecars).parquet")


if __name__ == "__main__":
    main()

"""Disease -> GO authoritative cross-family links (v14 -> v15).

Completeness add the user caught: when building the disease axis I added
disease_has_location->uberon and disease_has_feature->hp, but MISSED the
authoritative MONDO disease->GO axioms. These are the FIRST direct functional<->
physiological meso bridge (previously the two families joined only via shared protein).
Same justification/pattern as disease_has_location (codex thread 019f1a3f).

3 distinct authoritative MONDO predicates (allowlist; long tail realized_in/site_of/
disease_triggers excluded), parsed from BOTH relationship: and intersection_of:, GO
canon'd via go-basic alt_id, PRESENT-ONLY (both endpoints already in v14, no node
expansion), directed disease->GO, deduped. Targets span GO BP/MF/CC (all valid).

Output: Code/data/KG/_merged_kg_dedup_v15/{nodes__dedup,edges__dedup,+all sidecars}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_disease_go_links.py
"""
from __future__ import annotations

import os
import re
from collections import Counter

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v14"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v15"
MONDO = "Code/data/_cache/mondo.obo"
GO_BASIC = "Code/data/_cache/go-basic.obo"
PREDS = {"disease_has_basis_in_disruption_of": "mondo:disease_has_basis_in_disruption_of",
         "disease_disrupts": "mondo:disease_disrupts",
         "disease_has_basis_in_dysfunction_of": "mondo:disease_has_basis_in_dysfunction_of"}
SIDECARS = ("path_meta", "go_meta", "anat_meta", "dis_meta", "dis_fold_map",
            "hp_meta", "exp_meta", "drug_meta", "pclass_meta")


def gid(s):
    m = re.search(r"GO:(\d+)", s); return int(m.group(1)) if m else None


def mid(s):
    m = re.search(r"MONDO:(\d+)", s); return int(m.group(1)) if m else None


def go_altmap(path):
    alt, cur = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: GO:"):
            cur = gid(line)
        elif line.startswith("alt_id: GO:") and cur is not None:
            a = gid(line)
            if a is not None:
                alt[a] = cur
    return alt


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v14 in: {len(nd)} nodes, {len(ed)} edges")
    galt = go_altmap(GO_BASIC)
    gcanon = lambda N: galt.get(N, N)
    kind_of = dict(zip(nd["id"], nd["kind"].astype(str)))
    mondo_present = {f"mondo:{int(str(i).split(':')[-1])}" for i in nd['id'].astype(str) if i.startswith("mondo:")}
    go_present = {f"go:{int(str(i).split(':')[-1])}" for i in nd['id'].astype(str) if i.startswith("go:")}

    # ---- parse MONDO disease->GO (relationship + intersection_of, allowlisted preds) ----
    raw, cur = [], None
    for line in open(MONDO, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = None
        elif line.startswith("id: MONDO:"):
            cur = mid(line)
        elif cur is not None and (line.startswith("relationship: ") or line.startswith("intersection_of: ")):
            p = line.split(None, 2)
            if len(p) >= 3 and p[1] in PREDS and "GO:" in p[2]:
                g = gid(p[2])
                if g is not None:
                    raw.append((f"mondo:{cur}", PREDS[p[1]], f"go:{gcanon(g)}"))
    edges = [(s, rel, d) for s, rel, d in set(raw) if s in mondo_present and d in go_present]
    print(f"disease->GO edges: raw {len(set(raw))} -> present-only {len(edges)} "
          f"| by relation {dict(Counter(rel for _, rel, _ in edges))}")

    df = pd.DataFrame({"src": [s for s, _, _ in edges], "src_kind": "disease",
                       "dst": [d for _, _, d in edges], "dst_kind": [kind_of.get(d, "biological_process") for _, _, d in edges],
                       "relation": [rel for _, rel, _ in edges], "source_kg": "mondo", "directed": True})
    n_before = len(ed) + len(df)
    new_edges = pd.concat([ed, df], ignore_index=True).drop_duplicates(["src", "dst", "relation", "directed"])
    bad = ~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    os.makedirs(OUT_DIR, exist_ok=True)
    nd.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in SIDECARS:
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)

    # ---- verification ----
    nm = dict(zip(nd["id"], nd["name"]))
    print(f"\n=== v15 built (disease->GO cross-family links) ===")
    print(f"  nodes unchanged {len(nd)} | edges {len(ed)} -> {len(new_edges)} (+{len(new_edges)-len(ed)})")
    print(f"  distinct diseases linked to GO: {df['src'].nunique()} | distinct GO targets: {df['dst'].nunique()}")
    print(f"  GO namespace of targets: {dict(Counter(kind_of.get(d) for d in df['dst']))}")
    print(f"  functional<->physiological now bridged at disease<->GO (not only via protein)")
    for s, rel, d in edges[:4]:
        print(f"   {nm.get(s)} --{rel.split(':')[1].replace('disease_has_basis_in_','').replace('disease_','')}--> {nm.get(d)}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,+{len(SIDECARS)} sidecars).parquet")


if __name__ == "__main__":
    main()

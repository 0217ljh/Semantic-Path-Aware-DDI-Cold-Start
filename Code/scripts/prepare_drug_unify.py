"""Drug entity normalization (v11 -> v12): unify 3 source-conditioned drug kinds.

Codex-locked call (thread 019f1747 audit + 019f19d3 design). The DDI entities were
never normalized: 8048 distinct DrugBank drugs split across 3 inconsistent kinds
('drug' 5854 PrimeKG-only edges / 'Drug' 1900 all-source-rich / 'Compound' 294
Hetionet-mostly), all bare DrugBank IDs (DBxxxxx), disjoint by DB id. CONCRETE HAZARD:
the DDI prediction datasets (ddi_unified) reference drugs that SPAN ALL 3 KINDS (e.g.
drugbank_ryu 1700 = 1436 'Drug' + 108 'drug' + 145 'Compound'), so any code assuming
"DDI drug == kind 'Drug'" is wrong.

Rules:
  * rename all drug/Drug/Compound -> id `drug:DBxxxxx`, kind='drug' (disjoint DB ids ->
    no collision; unique DrugBank identity -> safe merge). Remap all edges.
  * kind = entity class, NOT annotation density. Edge-richness asymmetry preserved as
    drug_meta FEATURES, not as type.
  * sidecar drug_meta {id, drugbank_id, legacy_origin(primekg/drugbank/hetionet from old
    kind, = where the node came from before canon, NOT current edge provenance),
    n_edges, has_db_pharmacology, has_het, has_prime}.
  * in_ddi_pool is NOT baked into the KG (it spans kinds + varies by dataset -> stays
    dataset-side; loader joins drugs.parquet -> KG by DB id with kg_id = "drug:"+drug_id).

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v12/{nodes__dedup,edges__dedup,+all sidecars,drug_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_drug_unify.py
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v11"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v12"
DRUG_KINDS = {"drug", "Drug", "Compound"}
LEGACY = {"drug": "primekg", "Drug": "drugbank", "Compound": "hetionet"}
SIDECARS = ("path_meta", "go_meta", "anat_meta", "dis_meta", "dis_fold_map", "hp_meta", "exp_meta")


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v11 in: {len(nd)} nodes, {len(ed)} edges")

    is_drug = nd["kind"].astype(str).isin(DRUG_KINDS)
    drug_nodes = nd[is_drug].copy()
    # self-guard the disjoint-DB-id invariant: a DrugBank id must not appear under >1 kind
    assert drug_nodes["id"].is_unique, "duplicate drug node id (same DrugBank id in >1 kind) — merge would clobber"
    print(f"drug nodes (3 kinds): {len(drug_nodes)} | {drug_nodes['kind'].astype(str).value_counts().to_dict()}")
    ren = {x: f"drug:{x}" for x in drug_nodes["id"].astype(str)}
    legacy = {x: LEGACY[k] for x, k in zip(drug_nodes["id"].astype(str), drug_nodes["kind"].astype(str))}
    drugset = set(drug_nodes["id"].astype(str))
    assert len(set(ren.values())) == len(ren), "drug id collision after prefixing"

    # ---- per-drug edge stats (on v11 ids, before remap) ----
    n_edges = Counter(); has = defaultdict(set)
    for a, b, rel in zip(ed["src"].astype(str), ed["dst"].astype(str), ed["relation"].astype(str)):
        pfx = rel.split(":")[0]
        if a in drugset:
            n_edges[a] += 1; has[a].add(pfx)
        if b in drugset and b != a:
            n_edges[b] += 1; has[b].add(pfx)

    # ---- node table ----
    non_drug = nd[~is_drug].copy()
    drug_rows = pd.DataFrame({"id": [ren[x] for x in drug_nodes["id"].astype(str)], "kind": "drug",
                              "name": list(drug_nodes["name"]),
                              "source_kg": [legacy[x] for x in drug_nodes["id"].astype(str)]})
    new_nodes = pd.concat([non_drug, drug_rows], ignore_index=True).drop_duplicates("id")
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges ----
    e = ed.copy()
    e["src"] = e["src"].map(lambda x: ren.get(str(x), x))
    e["dst"] = e["dst"].map(lambda x: ren.get(str(x), x))
    e["src_kind"] = [kind_of.get(s, k) for s, k in zip(e["src"], e["src_kind"])]
    e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(e["dst"], e["dst_kind"])]
    n_before = len(e)
    e = e.drop_duplicates(["src", "dst", "relation", "directed"])
    new_edges = e
    bad = ~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ---- drug_meta sidecar ----
    rows = []
    for x in drug_nodes["id"].astype(str):
        pfx = has[x]
        rows.append({"id": ren[x], "drugbank_id": x, "legacy_origin": legacy[x],
                     "n_edges": int(n_edges.get(x, 0)),
                     "has_db_pharmacology": "db" in pfx, "has_het": "het" in pfx, "has_prime": "prime" in pfx})
    drug_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in SIDECARS:
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)
    drug_meta.to_parquet(f"{OUT_DIR}/drug_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v12 built (drug unification) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | 3 drug kinds -> kind='drug' {len(drug_rows)}")
    print(f"  edges {len(ed)} -> {len(new_edges)} (dedup {n_before-len(new_edges)})")
    leftover = int(new_nodes["kind"].astype(str).isin({"Drug", "Compound"}).sum())
    print(f"  leftover Drug/Compound kinds: {leftover} | all drug nodes drug:-prefixed: "
          f"{int((new_nodes[new_nodes['kind']=='drug']['id'].astype(str).str.startswith('drug:')).all())}")
    print(f"  drug_meta {len(drug_meta)} | legacy_origin: {drug_meta['legacy_origin'].value_counts().to_dict()}")
    print(f"  edge-richness: has_db_pharmacology {int(drug_meta['has_db_pharmacology'].sum())} | "
          f"has_het {int(drug_meta['has_het'].sum())} | has_prime {int(drug_meta['has_prime'].sum())}")
    print(f"  n_edges: median {int(drug_meta['n_edges'].median())} | max {int(drug_meta['n_edges'].max())} | "
          f"min {int(drug_meta['n_edges'].min())}")
    # sanity: a drug edge still walks
    nm = dict(zip(new_nodes["id"], new_nodes["name"]))
    dt = new_edges[new_edges["relation"] == "prime:drug_protein"].head(1)
    for s, d in zip(dt["src"].astype(str), dt["dst"].astype(str)):
        print(f"  sample drug->protein: {s}({nm.get(s)}) -> {d}({nm.get(d)})")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,+{len(SIDECARS)} sidecars,drug_meta).parquet")


if __name__ == "__main__":
    main()

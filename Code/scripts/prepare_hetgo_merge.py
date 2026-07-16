"""Hetionet GO fold into go: (v12 -> v13): alt_id-merge + obsolete-keep.

Codex-locked (thread 019f1747 audit + 019f19f8 confirm). The 673 Hetionet GO nodes
(kind Biological Process 471 / Molecular Function 166 / Cellular Component 36, ids
het:...::GO:N) are NOT new GO terms — after go-basic alt_id resolution they are either
alt_id ALIASES of GO terms already present as go: (494, merge) or OBSOLETE GO terms
(179, keep-flagged). This is the SAME canonical-identity policy prepare_go.py used for
PrimeKG GO; these het-sourced GO nodes were simply missed. They carry 36,372
prime:{bioprocess|molfunc|cellcomp}_protein membership edges (other endpoint = protein).

Rules (no false merge — alt_id is curated GO identity, not heuristic):
  * het GO node het:...::GO:N -> go:canon(N).
  * if go:canon(N) already present -> MERGE: remap its protein edges onto go:c, drop the
    duplicate het node.
  * if go:canon(N) absent (obsolete term not yet in go:) -> materialize go:c flagged
    obsolete (kind=namespace, no hierarchy/slim), consistent with prepare_go.py's
    keep-flagged obsolete policy. The obsolete term is NOT redirected/folded into its
    replacement; replaced_by is stored only as an ADVISORY field in go_meta (same as
    prepare_go.py).
  * namespace guard: prefer the authoritative go-basic namespace; LOG (not block)
    mismatches vs the looser het kind; fall back to het kind only when go-basic lacks it.
  * remap edges + exact dedup (memberships = same fact, no per-edge provenance added).
  * remove the 3 het GO kinds. go_meta gets rows for the 179 new obsolete terms;
    n_protein refreshed for all affected go: nodes.

Acceptance = representation correctness/completeness (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v13/{nodes__dedup,edges__dedup,+all sidecars}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_hetgo_merge.py
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v12"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v13"
GO_BASIC = "Code/data/_cache/go-basic.obo"
HET_GO_KINDS = {"Biological Process": "biological_process",
                "Molecular Function": "molecular_function",
                "Cellular Component": "cellular_component"}
GO_PROT_RELS = ("prime:bioprocess_protein", "prime:molfunc_protein", "prime:cellcomp_protein")
SIDECARS = ("path_meta", "anat_meta", "dis_meta", "dis_fold_map", "hp_meta", "exp_meta", "drug_meta")


def gid(s):
    m = re.search(r"GO:(\d+)", s)
    return int(m.group(1)) if m else None


def parse_go_basic(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"id": None, "ns": None, "name": None, "obs": False, "rb": None, "alt": []}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: GO:"):
                cur["id"] = gid(line); terms[cur["id"]] = cur
            elif line.startswith("namespace:"):
                cur["ns"] = line.split(":", 1)[1].strip()
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                cur["obs"] = True
            elif line.startswith("replaced_by: GO:"):
                cur["rb"] = gid(line)
            elif line.startswith("alt_id: GO:"):
                cur["alt"].append(gid(line))
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    gm = pd.read_parquet(f"{IN_DIR}/go_meta.parquet")
    print(f"v12 in: {len(nd)} nodes, {len(ed)} edges")
    terms, altmap = parse_go_basic(GO_BASIC)

    def canon(N):
        return altmap.get(N, N)

    go_present = {int(str(i).split(":")[-1]) for i in nd["id"].astype(str) if i.startswith("go:")}

    # ---- map het GO nodes -> go:canon ----
    is_hetgo = nd["kind"].astype(str).isin(HET_GO_KINDS)
    hetgo = nd[is_hetgo].copy()
    ren, materialize = {}, {}     # materialize: c -> (namespace, het_kind)
    ns_mismatch = 0
    for i, k in zip(hetgo["id"].astype(str), hetgo["kind"].astype(str)):
        g = gid(i.split("::")[-1])
        if g is None:
            continue
        c = canon(g)
        ren[i] = f"go:{c}"
        het_ns = HET_GO_KINDS[k]
        basic_ns = terms[c]["ns"] if c in terms else None
        if basic_ns and basic_ns != het_ns:
            ns_mismatch += 1
        ns = basic_ns or het_ns                       # guard: prefer go-basic, fall back to het kind
        if c not in go_present:
            materialize[c] = (ns, terms[c]["name"] if c in terms else None, terms[c]["rb"] if c in terms else None)
    n_merge = sum(1 for i in ren if f"go:{canon(gid(i.split('::')[-1]))}" and canon(gid(i.split('::')[-1])) in go_present)
    print(f"het GO {len(hetgo)} -> merge-into-existing {len(hetgo)-len(materialize)} | materialize-new-obsolete {len(materialize)} "
          f"| namespace mismatches {ns_mismatch}")

    # ---- node table: drop het GO, add materialized go: ----
    non_het = nd[~is_hetgo].copy()
    mat_rows = pd.DataFrame({"id": [f"go:{c}" for c in materialize],
                             "kind": [materialize[c][0] for c in materialize],
                             "name": [materialize[c][1] for c in materialize],
                             "source_kg": "hetionet"})
    new_nodes = pd.concat([non_het, mat_rows], ignore_index=True).drop_duplicates("id")
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges + dedup ----
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

    # ---- go_meta: add 179 new obsolete rows + refresh n_protein for affected go nodes ----
    affected = {ren[i] for i in ren}                  # go:c that received het edges
    npr = Counter()
    for rel in GO_PROT_RELS:
        sub = new_edges[new_edges["relation"] == rel]
        for s, d in zip(sub["src"].astype(str), sub["dst"].astype(str)):
            go = s if s.startswith("go:") else d
            if go.startswith("go:"):
                npr[go] += 1
    new_meta_rows = []
    for c, (ns, name, rb) in materialize.items():
        gidp = f"go:{c}"
        new_meta_rows.append({"id": gidp, "namespace": ns, "is_slim": False, "obsolete": True,
                              "replaced_by": (f"go:{canon(rb)}" if rb else None),
                              "n_protein": int(npr.get(gidp, 0)), "slim_ancestors": None})
    gm2 = pd.concat([gm, pd.DataFrame(new_meta_rows)], ignore_index=True)
    # refresh n_protein for affected existing rows
    aff_idx = gm2["id"].isin(affected)
    gm2.loc[aff_idx, "n_protein"] = gm2.loc[aff_idx, "id"].map(lambda x: int(npr.get(x, 0)))
    assert gm2["id"].is_unique, "go_meta id collision"

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    gm2.to_parquet(f"{OUT_DIR}/go_meta.parquet", index=False)
    for sc in SIDECARS:
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v13 built (Hetionet GO fold) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | het GO {len(hetgo)} removed, +{len(mat_rows)} materialized obsolete go:")
    print(f"  edges {len(ed)} -> {len(new_edges)} (dedup-merged {n_before-len(new_edges)} het-vs-existing membership collisions)")
    print(f"  no het:Biological/Molecular/Cellular nodes left: "
          f"{int(new_nodes['id'].astype(str).str.startswith('het:Biological').sum() + new_nodes['id'].astype(str).str.startswith('het:Molecular').sum() + new_nodes['id'].astype(str).str.startswith('het:Cellular').sum())==0}")
    print(f"  go_meta {len(gm)} -> {len(gm2)} (+{len(new_meta_rows)} obsolete) | affected go nodes n_protein refreshed: {int(aff_idx.sum())}")
    # remaining het: residue
    rem = Counter(k for k in new_nodes["kind"].astype(str) if k in ("Biological Process","Molecular Function","Cellular Component","Side Effect","Symptom","Pharmacologic Class","Compound","Drug"))
    print(f"  remaining capitalized het kinds: {dict(rem)}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,go_meta,+{len(SIDECARS)} sidecars).parquet")


if __name__ == "__main__":
    main()

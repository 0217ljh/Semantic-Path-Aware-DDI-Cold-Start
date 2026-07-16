"""Pathway renormalization — STEP 1: Reactome axis (on top of v2 -> v3).

Builds the Reactome pathway backbone with a complete, oriented meso->macro
hierarchy. Per the codex-reviewed spec:
  * rename Reactome pathway nodes  prime:pathway:R-HSA-x -> path:R-HSA-x
  * SUPPLEMENT the bridging hierarchy nodes (ancestors of our pathways that are
    missing from the KG) so every pathway traces to >=1 macro root
  * REPLACE the symmetrized prime:pathway_pathway with DIRECTED child->parent
    edges (asserted DIRECT edges only; no transitive closure materialized)
  * keep ALL prime:pathway_protein edges (connection preservation), remapped ids
  * sidecar path_meta.parquet: source, source_ids, level{macro,meso,leaf},
    macro_ancestors (SET-VALUED, Reactome is a DAG), is_connector_only, obsolete,
    n_protein
  * het:Pathway (PC) and db:pathway (SMPDB) nodes are UNTOUCHED this step.

Acceptance = correctness/completeness of representation (NOT discriminability).

Output: Code/data/KG/_merged_kg_dedup_v3/{nodes__dedup,edges__dedup,path_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_pathway_reactome.py
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v2"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v3"
REL = "Code/data/_cache/ReactomePathwaysRelation.txt"
PWN = "Code/data/_cache/ReactomePathways.txt"


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v2 in: {len(nd)} nodes, {len(ed)} edges")

    # ---- Reactome directed hierarchy (human) ----
    par2ch, ch2par, hier = defaultdict(set), defaultdict(set), set()
    with open(REL, encoding="utf-8") as f:
        for line in f:
            p, c = line.rstrip("\n").split("\t")
            if p.startswith("R-HSA") and c.startswith("R-HSA"):
                par2ch[p].add(c); ch2par[c].add(p); hier |= {p, c}
    id2name = {}
    with open(PWN, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].startswith("R-HSA"):
                id2name[parts[0]] = parts[1]
    roots = {n for n in hier if n not in ch2par}          # global macro roots

    def walk_up(n):
        seen, q = set(), deque([n])
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x)
            q.extend(ch2par.get(x, ()))
        return seen - {n}

    # ---- our Reactome pathway nodes ----
    is_rea = nd["id"].astype(str).str.startswith("prime:pathway:")
    our_rhsa = {x.split(":")[-1] for x in nd[is_rea]["id"]}
    our_v2name = {x.split(":")[-1]: nm for x, nm in zip(nd[is_rea]["id"], nd[is_rea]["name"])}

    # bridging nodes = ancestors of our pathways that are missing from KG
    needed = set()
    for r in our_rhsa & hier:
        needed |= walk_up(r)
    bridging = (needed - our_rhsa) & hier
    included = (our_rhsa & hier) | bridging
    obsolete = our_rhsa - hier                 # in KG but not in current Reactome
    print(f"our Reactome pathways: {len(our_rhsa)} | in hierarchy {len(our_rhsa & hier)} | "
          f"obsolete {len(obsolete)} | bridging to add {len(bridging)}")

    # ---- rename map prime:pathway:X -> path:X ----
    ren = {f"prime:pathway:{x}": f"path:{x}" for x in our_rhsa}

    # ---- new node table ----
    non_rea = nd[~is_rea].copy()               # everything else untouched
    keep_ids = sorted(our_rhsa | bridging)     # all path:* Reactome nodes (incl obsolete + bridging)
    rea_rows = pd.DataFrame({
        "id": [f"path:{x}" for x in keep_ids],
        "kind": "pathway",
        "name": [str(our_v2name.get(x) or id2name.get(x) or x) for x in keep_ids],
        "source_kg": "reactome",
    })
    new_nodes = pd.concat([non_rea, rea_rows], ignore_index=True)
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- new edges ----
    keep = ed[ed["relation"] != "prime:pathway_pathway"].copy()   # drop old symmetrized
    keep["src"] = keep["src"].map(lambda x: ren.get(x, x))
    keep["dst"] = keep["dst"].map(lambda x: ren.get(x, x))
    # directed child->parent hierarchy among included nodes only
    hp = [(c, p) for p in par2ch for c in par2ch[p] if c in included and p in included]
    hier_edges = pd.DataFrame({
        "src": [f"path:{c}" for c, _ in hp],
        "src_kind": "pathway",
        "dst": [f"path:{p}" for _, p in hp],
        "dst_kind": "pathway",
        "relation": "prime:pathway_pathway",
        "source_kg": "reactome",
        "directed": True,
    })
    # refresh kinds for renamed pathway endpoints in the kept edges
    keep["src_kind"] = [kind_of.get(s, k) for s, k in zip(keep["src"], keep["src_kind"])]
    keep["dst_kind"] = [kind_of.get(d, k) for d, k in zip(keep["dst"], keep["dst_kind"])]
    new_edges = pd.concat([keep, hier_edges], ignore_index=True)

    # connection-preservation: every endpoint exists
    valid = set(kind_of)
    bad = ~(new_edges["src"].isin(valid) & new_edges["dst"].isin(valid))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ---- path_meta sidecar ----
    incl_set = included
    # protein-edge count per pathway (after rename)
    pp = new_edges[new_edges["relation"] == "prime:pathway_protein"]
    nprot = defaultdict(int)
    for s, d in zip(pp["src"].astype(str), pp["dst"].astype(str)):
        pwn = s if s.startswith("path:") else d
        nprot[pwn] += 1
    meta_rows = []
    for x in keep_ids:
        pid = f"path:{x}"
        in_h = x in incl_set
        children_in = [c for c in par2ch.get(x, ()) if c in incl_set]
        is_root = x in roots
        if not in_h:
            level = "leaf"            # obsolete: no hierarchy
        elif is_root:
            level = "macro"
        elif not children_in:
            level = "leaf"
        else:
            level = "meso"
        macro_anc = sorted({a for a in walk_up(x) if a in roots}) if in_h else []
        meta_rows.append({
            "id": pid, "source": "reactome", "source_ids": [x],
            "level": level, "macro_ancestors": macro_anc,
            "is_connector_only": (x in bridging), "obsolete": (x in obsolete),
            "n_protein": nprot.get(pid, 0),
        })
    meta = pd.DataFrame(meta_rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    meta.to_parquet(f"{OUT_DIR}/path_meta.parquet", index=False)

    # ---- verification (acceptance checks) ----
    print(f"\n=== v3 built ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} (Reactome pathway {len(our_rhsa)} -> {len(keep_ids)}; +{len(bridging)} bridging)")
    print(f"  edges {len(ed)} -> {len(new_edges)}")
    old_pp = int((ed["relation"] == "prime:pathway_protein").sum())
    new_pp = int((new_edges["relation"] == "prime:pathway_protein").sum())
    print(f"  pathway_protein edges preserved: {old_pp} -> {new_pp}  ({'OK' if old_pp==new_pp else 'LOST!'})")
    print(f"  pathway_pathway: symmetrized {int((ed['relation']=='prime:pathway_pathway').sum())} "
          f"-> directed {len(hier_edges)}")
    # ancestry via the ACTUAL output hierarchy edges (not the full source), counting
    # a macro root as self-tracing; obsolete pathways are excluded (they legitimately
    # have no current-Reactome ancestry and are flagged in path_meta).
    ch2par_out = defaultdict(set)
    for c, p in hp:
        ch2par_out[c].add(p)

    def traces_out(x):
        if x in roots:
            return True
        seen, q = set(), deque([x])
        while q:
            n = q.popleft()
            for pa in ch2par_out.get(n, ()):
                if pa in roots:
                    return True
                if pa not in seen:
                    seen.add(pa); q.append(pa)
        return False

    nonobs_prot = {x for x in (our_rhsa & hier) if nprot.get(f"path:{x}", 0) > 0}
    traced = sum(1 for x in nonobs_prot if traces_out(x))
    obs_prot = sum(1 for x in obsolete if nprot.get(f"path:{x}", 0) > 0)
    print(f"  NON-OBSOLETE protein pathways tracing to macro (via v3 edges): "
          f"{traced}/{len(nonobs_prot)} ({traced/max(len(nonobs_prot),1)*100:.1f}%)")
    print(f"  obsolete pathways carrying protein edges (kept, flagged, no roll-up): {obs_prot}")
    print(f"  level dist: {meta['level'].value_counts().to_dict()}")
    print(f"  multi-macro pathways (>1 macro ancestor): {int(meta['macro_ancestors'].map(len).gt(1).sum())}")
    print(f"  connector-only: {int(meta['is_connector_only'].sum())} | obsolete: {int(meta['obsolete'].sum())}")
    # cycle check on directed hierarchy
    import collections
    indeg = collections.Counter()
    g = defaultdict(list)
    for c, p in hp:
        g[c].append(p); indeg[p] += 1; indeg.setdefault(c, indeg.get(c, 0))
    # Kahn
    q = deque([n for n in set([x for e in hp for x in e]) if indeg[n] == 0])
    seen = 0
    while q:
        n = q.popleft(); seen += 1
        for m in g[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    nnodes_h = len({x for e in hp for x in e})
    print(f"  hierarchy acyclic? {'YES' if seen==nnodes_h else 'NO (CYCLE!)'} ({seen}/{nnodes_h})")
    # spot check: Hemostasis macro -> descendants present
    hemo = [x for x in keep_ids if str(id2name.get(x, '')).lower() == 'hemostasis']
    if hemo:
        print(f"  spot: Hemostasis = path:{hemo[0]} level="
              f"{meta[meta['id']==f'path:{hemo[0]}']['level'].iloc[0]}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,path_meta).parquet")


if __name__ == "__main__":
    main()

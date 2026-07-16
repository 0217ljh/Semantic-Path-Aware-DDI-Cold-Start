"""GO axis renormalization (v5 -> v6): orient+type GO hierarchy + BP slim macro.

Codex-locked policy + alt_id fix:
  * rename GO nodes prime:{biological_process|molecular_function|cellular_component}:<N>
    -> go:<C> where C = canonical primary id (RESOLVE go-basic alt_id: an old/merged
    GO id N that is an alt_id of primary C is MERGED into go:<C>, proteins/edges joined).
    This fixes mislabeling merged terms (e.g. GO:0016021 'integral component of membrane'
    is an alt_id of GO:0016020 'membrane') as obsolete.
  * REPLACE symmetrized prime:*_* hierarchies with DIRECTED TYPED edges from go-basic.obo:
    go:is_a (child->parent), go:part_of (part->whole), go:regulates / positively / negatively
    (regulator->regulated). Only between current-KG canonical GO nodes, same namespace,
    non-obsolete. Node universe = current KG GO terms (canonicalized); not expanded.
  * BP slim macro: is_slim = canonical node in goslim_generic; slim_ancestors = most-specific
    reachable slim via is_a+part_of (BP non-obsolete only). MF/CC: is_slim kept, no slim_ancestors.
  * Genuinely obsolete (is_obsolete in go-basic, or absent): keep node, no hierarchy edges,
    obsolete=true, record replaced_by (advisory, GO `consider` is NOT used = not equivalence).
  * Keep ALL GO terms (no human pruning).

Output: Code/data/KG/_merged_kg_dedup_v6/{nodes__dedup,edges__dedup,path_meta,go_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_go.py
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v5"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v6"
GOBASIC = "Code/data/_cache/go-basic.obo"
GOSLIM = "Code/data/_cache/goslim_generic.obo"
NS_KINDS = {"biological_process", "molecular_function", "cellular_component"}
REL_MAP = {"part_of": "go:part_of", "regulates": "go:regulates",
           "positively_regulates": "go:positively_regulates",
           "negatively_regulates": "go:negatively_regulates"}
MEMB = {"prime:bioprocess_protein", "prime:molfunc_protein", "prime:cellcomp_protein"}
OLD_HIER = {"prime:bioprocess_bioprocess", "prime:molfunc_molfunc", "prime:cellcomp_cellcomp"}


def goid(s: str) -> int:
    return int(s.split(":")[1])


def parse_obo(path):
    """primary id -> attrs; alt_id -> primary map."""
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"ns": None, "name": None, "obsolete": False, "replaced_by": None, "alt": [],
                   "is_a": set(), "part_of": set(), "regulates": set(),
                   "positively_regulates": set(), "negatively_regulates": set()}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: GO:"):
                cur["id"] = goid(line[4:]); terms[cur["id"]] = cur
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("namespace:"):
                cur["ns"] = line[11:]
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("replaced_by: GO:"):
                cur["replaced_by"] = goid(line[13:])
            elif line.startswith("alt_id: GO:"):
                cur["alt"].append(goid(line[8:]))
            elif line.startswith("is_a: GO:"):
                cur["is_a"].add(goid(line[6:].split("!")[0].strip()))
            elif line.startswith("relationship:"):
                p = line.split()
                if len(p) >= 3 and p[1] in REL_MAP:
                    cur[p[1]].add(goid(p[2]))
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def parse_slim(path):
    return {goid(l[4:]) for l in open(path, encoding="utf-8") if l.startswith("id: GO:")}


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v5 in: {len(nd)} nodes, {len(ed)} edges")
    terms, altmap = parse_obo(GOBASIC)
    slim = {altmap.get(s, s) for s in parse_slim(GOSLIM)}      # canonicalize slim ids too
    print(f"go-basic primary {len(terms)} | alt_id map {len(altmap)} | goslim {len(slim)}")

    def canon(N):
        return altmap.get(N, N)

    def is_obs(C):
        t = terms.get(C)
        return t is None or t["obsolete"]

    # ---- KG GO nodes: prime:<ns>:<N> -> go:<C(anonical)> (alt_id merged into primary) ----
    is_go = nd["kind"].isin(NS_KINDS)
    go_nodes = nd[is_go].copy()
    go_nodes["N"] = go_nodes["id"].map(lambda x: int(str(x).split(":")[-1]))
    go_nodes["C"] = go_nodes["N"].map(canon)
    ren = {x: f"go:{c}" for x, c in zip(go_nodes["id"], go_nodes["C"])}
    go_nodes["is_primary"] = go_nodes["N"] == go_nodes["C"]
    rep = (go_nodes.sort_values("is_primary", ascending=False).drop_duplicates("C").set_index("C"))
    canon_ids = sorted(set(go_nodes["C"]))
    n_merged = int((go_nodes["N"] != go_nodes["C"]).sum())
    print(f"KG GO nodes {len(go_nodes)} -> canonical {len(canon_ids)} (alt_id merged: {n_merged})")

    def ckind(C):
        return terms[C]["ns"] if C in terms else str(rep.loc[C, "kind"])

    def cname(C):
        return terms[C]["name"] if C in terms else str(rep.loc[C, "name"])

    kg_go_C = {C: ckind(C) for C in canon_ids}
    canon_set = set(canon_ids)

    # ---- new node table ----
    non_go = nd[~is_go].copy()
    go_rows = pd.DataFrame({"id": [f"go:{c}" for c in canon_ids], "kind": [ckind(c) for c in canon_ids],
                            "name": [cname(c) for c in canon_ids],
                            "source_kg": [str(rep.loc[c, "source_kg"]) for c in canon_ids]})
    new_nodes = pd.concat([non_go, go_rows], ignore_index=True)
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges; drop old symmetrized hierarchy; dedup (alt+primary merges) ----
    keep = ed[~ed["relation"].isin(OLD_HIER)].copy()
    keep["src"] = keep["src"].map(lambda x: ren.get(x, x))
    keep["dst"] = keep["dst"].map(lambda x: ren.get(x, x))
    keep = keep.drop_duplicates(["src", "dst", "relation", "directed"])
    keep["src_kind"] = [kind_of.get(s, k) for s, k in zip(keep["src"], keep["src_kind"])]
    keep["dst_kind"] = [kind_of.get(d, k) for d, k in zip(keep["dst"], keep["dst_kind"])]

    # ---- directed typed hierarchy from go-basic (canonical, same-ns, non-obsolete) ----
    new_h = []
    type_counts = defaultdict(int)
    for C in canon_ids:
        t = terms.get(C)
        if t is None or t["obsolete"]:
            continue
        ns = kg_go_C[C]
        for kind_rel, parents in (("go:is_a", t["is_a"]), ("go:part_of", t["part_of"])):
            for p in parents:
                pc = canon(p)
                if pc in canon_set and kg_go_C.get(pc) == ns and not is_obs(pc) and pc != C:
                    new_h.append((C, pc, kind_rel))
        for rk in ("regulates", "positively_regulates", "negatively_regulates"):
            for tg in t[rk]:
                tc = canon(tg)
                if tc in canon_set and kg_go_C.get(tc) == ns and not is_obs(tc) and tc != C:
                    new_h.append((C, tc, REL_MAP[rk]))
    new_h = list(set(new_h))
    for _, _, r in new_h:
        type_counts[r] += 1
    hdf = pd.DataFrame({"src": [f"go:{c}" for c, _, _ in new_h], "src_kind": [kg_go_C[c] for c, _, _ in new_h],
                        "dst": [f"go:{p}" for _, p, _ in new_h], "dst_kind": [kg_go_C[p] for _, p, _ in new_h],
                        "relation": [r for _, _, r in new_h], "source_kg": "go", "directed": True})
    new_edges = pd.concat([keep, hdf], ignore_index=True)
    assert not (~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))).any(), "unknown endpoint"

    # ---- n_protein per canonical GO node ----
    nprot = defaultdict(int)
    mm = new_edges[new_edges["relation"].isin(MEMB)]
    for s, d in zip(mm["src"].astype(str), mm["dst"].astype(str)):
        nprot[s if s.startswith("go:") else d] += 1

    # ---- BP slim_ancestors via is_a+part_of closure ----
    bp_parent = defaultdict(set)
    for c, p, r in new_h:
        if r in ("go:is_a", "go:part_of") and kg_go_C.get(c) == "biological_process":
            bp_parent[c].add(p)
    slim_bp = {n for n in slim if kg_go_C.get(n) == "biological_process"}
    anc_cache = {}

    def ancestors(n):
        if n in anc_cache:
            return anc_cache[n]
        seen, q = set(), deque(bp_parent.get(n, ()))
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x); q.extend(bp_parent.get(x, ()))
        anc_cache[n] = seen
        return seen

    def minimal_slim(n):
        R = (ancestors(n) | ({n} if n in slim_bp else set())) & slim_bp
        return sorted(s for s in R if not any(sp != s and s in (ancestors(sp) | {sp}) for sp in R))

    # ---- go_meta ----
    rows = []
    for C in canon_ids:
        t = terms.get(C); obs = (t is None or t["obsolete"]); ns = kg_go_C[C]
        rb = f"go:{canon(t['replaced_by'])}" if (t and t["obsolete"] and t["replaced_by"]) else None
        sa = minimal_slim(C) if (ns == "biological_process" and not obs) else None
        rows.append({"id": f"go:{C}", "namespace": ns, "is_slim": bool(C in slim),
                     "obsolete": bool(obs), "replaced_by": rb,
                     "n_protein": int(nprot.get(f"go:{C}", 0)), "slim_ancestors": sa})
    go_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    pd.read_parquet(f"{IN_DIR}/path_meta.parquet").to_parquet(f"{OUT_DIR}/path_meta.parquet", index=False)
    go_meta.to_parquet(f"{OUT_DIR}/go_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v6 built (GO axis, alt_id-fixed) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | GO {len(go_nodes)} -> canonical {len(canon_ids)}")
    print(f"  edges {len(ed)} -> {len(new_edges)} | typed GO hierarchy {dict(type_counts)} (total {len(hdf)})")
    for r in MEMB:
        print(f"  {r}: {int((ed['relation']==r).sum())} -> {int((new_edges['relation']==r).sum())}")
    print(f"  go_meta {len(go_meta)} | is_slim {int(go_meta['is_slim'].sum())} | obsolete {int(go_meta['obsolete'].sum())} "
          f"| with replaced_by {int(go_meta['replaced_by'].notna().sum())}")
    # was: 7073 obsolete before alt_id fix
    print(f"  (alt_id merged {n_merged} nodes that were wrongly counted obsolete before)")
    # spot: integral component of membrane should now be go:16020 (membrane), NOT obsolete
    m = go_meta[go_meta['id'] == 'go:16020']
    if len(m):
        print(f"  spot go:16020(membrane): obsolete={m['obsolete'].iloc[0]} n_protein={m['n_protein'].iloc[0]} (was wrongly obsolete via alt 16021)")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,path_meta,go_meta).parquet")


if __name__ == "__main__":
    main()

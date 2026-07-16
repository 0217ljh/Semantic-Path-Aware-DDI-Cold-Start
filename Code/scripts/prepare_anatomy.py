"""Anatomy (UBERON) axis renormalization (v6 -> v7): orient+type hierarchy + organ-system macro.

Codex-locked policy:
  * rename prime:anatomy:<N> -> uberon:<C> (C = canonical primary UBERON id after
    alt_id resolution; merge alt_ids onto primary, join edges).
  * REPLACE prime:anatomy_anatomy with DIRECTED TYPED edges from uberon-basic.obo:
    uberon:is_a (child->parent), uberon:part_of (part->whole), uberon:develops_from
    (derived->precursor, stored as cross-link, EXCLUDED from roll-up). Drop metadata
    relations (taxon/contributor). Only between current-KG canonical anatomy nodes.
  * MACRO = organ systems = filtered transitive is_a descendants of 'anatomical system'
    (UBERON:0000467) present + non-obsolete in KG (granular subdivisions, non-human/
    embryonic, vascular sub-systems, and UBERON oddities excluded). macro_ancestors(term)
    = most-specific reachable organ systems via is_a+part_of (DAG multi-valued; allow empty).
  * Keep ALL anatomy_protein_present (membership; pruning is a downstream decision) and
    record n_protein_present/absent in sidecar for later down-weighting.
  * Keep anatomy_protein_absent SEPARATE (opposite of present; not in roll-up).
  * Obsolete (absent from uberon-basic or is_obsolete): keep node + memberships, no
    hierarchy/macro, flag, record replaced_by advisory (canonicalized).

Output: Code/data/KG/_merged_kg_dedup_v7/{nodes__dedup,edges__dedup,path_meta,go_meta,anat_meta}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_anatomy.py
"""
from __future__ import annotations

import os
import re
from collections import defaultdict, deque

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v6"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v7"
UBERON = "Code/data/_cache/uberon-basic.obo"
ANATOMICAL_SYSTEM = 467           # UBERON:0000467
PRESENT = "prime:anatomy_protein_present"
ABSENT = "prime:anatomy_protein_absent"
OLD_HIER = "prime:anatomy_anatomy"


def ubid(s: str):
    m = re.search(r"UBERON:(\d+)", s)   # robust to trailing {source=...} / ! comment / non-UBERON refs
    return int(m.group(1)) if m else None


def parse_obo(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"name": None, "obsolete": False, "replaced_by": None, "alt": [],
                   "is_a": set(), "part_of": set(), "develops_from": set()}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: UBERON:"):
                cur["id"] = ubid(line[4:]); terms[cur["id"]] = cur
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("replaced_by: UBERON:"):
                cur["replaced_by"] = ubid(line[13:])
            elif line.startswith("alt_id: UBERON:"):
                cur["alt"].append(ubid(line[8:]))
            elif line.startswith("is_a: UBERON:"):
                u = ubid(line[6:].split("!")[0].strip())
                if u is not None:
                    cur["is_a"].add(u)
            elif line.startswith("relationship:"):
                p = line.split()
                if len(p) >= 3 and p[1] in ("part_of", "develops_from"):
                    u = ubid(p[2])
                    if u is not None:
                        cur[p[1]].add(u)
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    print(f"v6 in: {len(nd)} nodes, {len(ed)} edges")
    terms, altmap = parse_obo(UBERON)
    print(f"UBERON terms {len(terms)} | alt_id map {len(altmap)}")

    def canon(N):
        return altmap.get(N, N)

    def is_obs(C):
        t = terms.get(C)
        return t is None or t["obsolete"]

    # macro = is_a-transitive descendants of 'anatomical system' (UBERON:0000467), named
    # "...system", minus granular subdivisions (codex guardrail). Direct-subclass-only was
    # too narrow (kidney's renal system is an indirect descendant -> missed).
    child_isa = defaultdict(set)
    for n, t in terms.items():
        for p in t["is_a"]:
            child_isa[p].add(n)
    sys_desc, dq = set(), deque([ANATOMICAL_SYSTEM])
    while dq:
        x = dq.popleft()
        for c in child_isa.get(x, ()):
            if c not in sys_desc:
                sys_desc.add(c); dq.append(c)
    # codex guardrail: drop granular subdivisions, non-human/embryonic clutter, and
    # vascular sub-systems (so vessels roll up to cardiovascular, not arterial/venous).
    BAD = ("subdivision", "regional part", "division", "system of",
           "insect", "larval", "extraembryonic", "tracheal", "water vascular",
           "open circulatory", "closed circulatory", "one-pass", "two-pass", "embryonic cardiovascular")
    EXCLUDE_EXACT = {"arterial system", "venous system", "vascular system",
                     "pulmonary vascular system", "systemic arterial system", "hindbrain venous system",
                     # codex final-acceptance: UBERON oddities, not human organ systems —
                     # molecular system (subset non_informative), tectorial restraint system
                     # (PHENOSCAPE inner-ear ciliary construct), dermatological-muscosal system
                     # (integumental + all mucosae composite).
                     "molecular system", "tectorial restraint system", "dermatological-muscosal system"}
    organ_systems = {n for n in sys_desc if not terms[n]["obsolete"] and terms[n]["name"]
                     and "system" in terms[n]["name"].lower()
                     and not any(b in terms[n]["name"].lower() for b in BAD)
                     and terms[n]["name"].lower() not in EXCLUDE_EXACT}
    print(f"organ-system macro set (filtered transitive subclasses of UBERON:0000467): {len(organ_systems)}")

    # ---- KG anatomy nodes: prime:anatomy:N -> uberon:C ----
    is_an = nd["kind"].astype(str).str.lower() == "anatomy"
    an = nd[is_an].copy()
    an["N"] = an["id"].map(lambda x: int(str(x).split(":")[-1]))
    an["C"] = an["N"].map(canon)
    ren = {x: f"uberon:{c}" for x, c in zip(an["id"], an["C"])}
    an["is_primary"] = an["N"] == an["C"]
    rep = an.sort_values("is_primary", ascending=False).drop_duplicates("C").set_index("C")
    canon_ids = sorted(set(an["C"]))
    canon_set = set(canon_ids)
    n_merged = int((an["N"] != an["C"]).sum())
    print(f"KG anatomy {len(an)} -> canonical {len(canon_ids)} (alt_id merged {n_merged})")

    def cname(C):
        return terms[C]["name"] if (C in terms and terms[C]["name"]) else str(rep.loc[C, "name"])

    # ---- new node table ----
    non_an = nd[~is_an].copy()
    an_rows = pd.DataFrame({"id": [f"uberon:{c}" for c in canon_ids], "kind": "anatomy",
                            "name": [cname(c) for c in canon_ids],
                            "source_kg": [str(rep.loc[c, "source_kg"]) for c in canon_ids]})
    new_nodes = pd.concat([non_an, an_rows], ignore_index=True)
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap edges; drop old hierarchy; dedup alt-merges ----
    keep = ed[ed["relation"] != OLD_HIER].copy()
    keep["src"] = keep["src"].map(lambda x: ren.get(x, x))
    keep["dst"] = keep["dst"].map(lambda x: ren.get(x, x))
    keep = keep.drop_duplicates(["src", "dst", "relation", "directed"])
    keep["src_kind"] = [kind_of.get(s, k) for s, k in zip(keep["src"], keep["src_kind"])]
    keep["dst_kind"] = [kind_of.get(d, k) for d, k in zip(keep["dst"], keep["dst_kind"])]

    # ---- directed typed hierarchy from UBERON ----
    new_h = []
    for C in canon_ids:
        t = terms.get(C)
        if t is None or t["obsolete"]:
            continue
        for rel, parents in (("uberon:is_a", t["is_a"]), ("uberon:part_of", t["part_of"]),
                             ("uberon:develops_from", t["develops_from"])):
            for p in parents:
                pc = canon(p)
                if pc in canon_set and not is_obs(pc) and pc != C:
                    new_h.append((C, pc, rel))
    new_h = list(set(new_h))
    type_counts = defaultdict(int)
    for _, _, r in new_h:
        type_counts[r] += 1
    hdf = pd.DataFrame({"src": [f"uberon:{c}" for c, _, _ in new_h], "src_kind": "anatomy",
                        "dst": [f"uberon:{p}" for _, p, _ in new_h], "dst_kind": "anatomy",
                        "relation": [r for _, _, r in new_h], "source_kg": "uberon", "directed": True})
    new_edges = pd.concat([keep, hdf], ignore_index=True)
    assert not (~(new_edges["src"].isin(kind_of) & new_edges["dst"].isin(kind_of))).any(), "unknown endpoint"

    # ---- macro_ancestors via is_a+part_of (organ systems, most-specific) ----
    parent = defaultdict(set)
    for c, p, r in new_h:
        if r in ("uberon:is_a", "uberon:part_of"):
            parent[c].add(p)
    sys_kg = organ_systems & canon_set
    anc_cache = {}

    def ancestors(n):
        if n in anc_cache:
            return anc_cache[n]
        seen, q = set(), deque(parent.get(n, ()))
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x); q.extend(parent.get(x, ()))
        anc_cache[n] = seen
        return seen

    def macro_anc(n):
        R = (ancestors(n) | ({n} if n in sys_kg else set())) & sys_kg
        return sorted(s for s in R if not any(sp != s and s in (ancestors(sp) | {sp}) for sp in R))

    # ---- n_protein present/absent per anatomy node ----
    npres, nabs = defaultdict(int), defaultdict(int)
    for rel, dd in ((PRESENT, npres), (ABSENT, nabs)):
        e = new_edges[new_edges["relation"] == rel]
        for s, d in zip(e["src"].astype(str), e["dst"].astype(str)):
            dd[s if s.startswith("uberon:") else d] += 1

    # ---- anat_meta ----
    rows = []
    for C in canon_ids:
        t = terms.get(C); obs = (t is None or t["obsolete"]); aid = f"uberon:{C}"
        rb = f"uberon:{canon(t['replaced_by'])}" if (t and t["obsolete"] and t["replaced_by"]) else None
        rows.append({"id": aid, "is_system": bool(C in organ_systems), "obsolete": bool(obs),
                     "replaced_by": rb, "n_protein_present": int(npres.get(aid, 0)),
                     "n_protein_absent": int(nabs.get(aid, 0)),
                     "macro_ancestors": (macro_anc(C) if not obs else None)})
    anat_meta = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    for sc in ("path_meta", "go_meta"):
        pd.read_parquet(f"{IN_DIR}/{sc}.parquet").to_parquet(f"{OUT_DIR}/{sc}.parquet", index=False)
    anat_meta.to_parquet(f"{OUT_DIR}/anat_meta.parquet", index=False)

    # ---- verification ----
    print(f"\n=== v7 built (anatomy axis) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} | anatomy {len(an)} -> canonical {len(canon_ids)} (alt merged {n_merged})")
    print(f"  edges {len(ed)} -> {len(new_edges)} | typed hierarchy {dict(type_counts)} (total {len(hdf)})")
    print(f"  old prime:anatomy_anatomy dropped: {int((ed['relation']==OLD_HIER).sum())}")
    print(f"  present {int((new_edges['relation']==PRESENT).sum())} | absent {int((new_edges['relation']==ABSENT).sum())} (kept)")
    print(f"  anat_meta {len(anat_meta)} | organ-system nodes {int(anat_meta['is_system'].sum())} | obsolete {int(anat_meta['obsolete'].sum())} (replaced_by {int(anat_meta['replaced_by'].notna().sum())})")
    has_macro = anat_meta['macro_ancestors'].map(lambda x: bool(x) if x is not None else False)
    print(f"  anatomy terms reaching >=1 organ system: {int(has_macro.sum())}/{len(anat_meta)} "
          f"({has_macro.mean()*100:.0f}%) | multi-system {int(anat_meta['macro_ancestors'].map(lambda x: len(x)>1 if x else False).sum())}")
    nm = dict(zip(new_nodes['id'], new_nodes['name']))
    for q in ['heart', 'brain', 'kidney']:
        r = anat_meta[anat_meta['id'].map(lambda i: str(nm.get(i)).lower()) == q]
        if len(r):
            print(f"  spot '{q}' {r['id'].iloc[0]} -> systems {[nm.get(s) for s in (r['macro_ancestors'].iloc[0] or [])]}")
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,path_meta,go_meta,anat_meta).parquet")


if __name__ == "__main__":
    main()

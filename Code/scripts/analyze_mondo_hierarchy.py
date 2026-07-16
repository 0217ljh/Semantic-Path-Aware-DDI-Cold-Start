"""Show the real MONDO disease hierarchy (read-only investigation, pre-build).

Parses Code/data/_cache/mondo.obo (authoritative) + the PrimeKG disease nodes in
_merged_kg_dedup_v7, and reports the SHAPE of the disease hierarchy so we can
decide the disease-axis design:
  * is_a chains (a few real diseases climbing to root)
  * top-level children of 'disease' (MONDO:0000001)
  * depth / branching distribution over the 22,205 terms PrimeKG actually uses
  * the 'disease_grouping' subset (macro candidates) + coverage
  * DOID xref coverage for the 137 Hetionet disease nodes
  * how a multi-term PrimeKG BERT group sits across is_a levels

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_mondo_hierarchy.py
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque

import pandas as pd

MONDO = "Code/data/_cache/mondo.obo"
DIS_FEAT = "Code/data/KG/primekg/disease_features.tab"
V7 = "Code/data/KG/_merged_kg_dedup_v7"
DISEASE_ROOT = 1   # MONDO:0000001 'disease'


def mid(s):
    m = re.search(r"MONDO:(\d+)", s)
    return int(m.group(1)) if m else None


def parse_mondo(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"name": None, "obsolete": False, "is_a": set(), "alt": [],
                   "subset": set(), "doid": set()}
        elif line == "[Typedef]":
            cur = None
        elif cur is not None:
            if line.startswith("id: MONDO:"):
                cur["id"] = mid(line); terms[cur["id"]] = cur
            elif line.startswith("name:"):
                cur["name"] = line[6:]
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
            elif line.startswith("alt_id: MONDO:"):
                cur["alt"].append(mid(line))
            elif line.startswith("is_a: MONDO:"):
                u = mid(line)
                if u is not None:
                    cur["is_a"].add(u)
            elif line.startswith("subset:"):
                cur["subset"].add(line.split()[1])
            elif line.startswith("xref: DOID:"):
                cur["doid"].add(line.split()[1])
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def main():
    terms, altmap = parse_mondo(MONDO)
    print(f"MONDO terms {len(terms)} | alt_id {len(altmap)} | "
          f"obsolete {sum(t['obsolete'] for t in terms.values())}")

    def canon(N):
        return altmap.get(N, N)

    nm = {n: t["name"] for n, t in terms.items()}
    parents = {n: {canon(p) for p in t["is_a"] if canon(p) in terms} for n, t in terms.items()}
    child = defaultdict(set)
    for n, ps in parents.items():
        for p in ps:
            child[p].add(n)

    def chain_to_root(n, maxhops=20):
        path = [n]
        seen = {n}
        while path[-1] != DISEASE_ROOT and maxhops > 0:
            ps = parents.get(path[-1], set()) - seen
            if not ps:
                break
            nxt = sorted(ps)[0]
            path.append(nxt); seen.add(nxt); maxhops -= 1
        return path

    # ---- the disease_grouping subset (macro candidates) ----
    grouping = {n for n, t in terms.items() if "disease_grouping" in t["subset"] and not t["obsolete"]}
    print(f"\n'disease_grouping' subset (macro candidates): {len(grouping)}")
    # top-level children of MONDO:0000001
    print(f"\nTop-level children of 'disease' (MONDO:0000001): {len(child.get(DISEASE_ROOT, set()))}")
    for c in sorted(child.get(DISEASE_ROOT, set())):
        tag = " [grouping]" if c in grouping else ""
        print(f"   MONDO:{c:07d}  {nm.get(c)}{tag}")

    # ---- our terms: load the 22,205 mondo_ids PrimeKG uses ----
    feat = pd.read_csv(DIS_FEAT, sep="\t", usecols=["mondo_id", "group_id_bert", "group_name_bert"], dtype=str)
    used = {int(x) for x in feat["mondo_id"].dropna().unique()}
    used_canon = {canon(u) for u in used if canon(u) in terms}
    print(f"\nPrimeKG uses {len(used)} MONDO ids | resolve to current MONDO terms: {len(used_canon)} "
          f"| not in mondo.obo: {len(used) - len([u for u in used if canon(u) in terms])}")

    # ---- is_a chains for real diseases in our KG ----
    print("\n=== REAL is_a CHAINS (atomic MONDO climbing to root) ===")
    for q in [13924, 4, 24290, 9260]:   # osteogenesis imperfecta t13, adrenocortical insuff, enuresis, GM1
        c = canon(q)
        if c not in terms:
            print(f"   MONDO:{q:07d} not in mondo.obo"); continue
        ch = chain_to_root(c)
        macro_on_path = [x for x in ch if x in grouping]
        print(f"   {nm.get(c)} (MONDO:{c:07d}):")
        print("      " + " -> ".join(f"{nm.get(x)}" for x in ch))
        print(f"      grouping(macro) terms on path: {[nm.get(x) for x in macro_on_path] or 'NONE'}")

    # ---- depth + branching over used terms ----
    depth_cache = {}

    def depth(n):
        if n in depth_cache:
            return depth_cache[n]
        if n == DISEASE_ROOT or not parents.get(n):
            depth_cache[n] = 0
            return 0
        d = 1 + min((depth(p) for p in parents[n]), default=0)
        depth_cache[n] = d
        return d

    used_list = [u for u in used_canon]
    depths = Counter(depth(u) for u in used_list)
    multi_parent = sum(1 for u in used_list if len(parents.get(u, set())) > 1)
    no_parent = sum(1 for u in used_list if not parents.get(u))
    print(f"\n=== HIERARCHY SHAPE over {len(used_list)} used terms ===")
    print(f"   depth distribution (hops to root): {dict(sorted(depths.items()))}")
    print(f"   terms with >1 parent (DAG, not tree): {multi_parent}")
    print(f"   terms with NO is_a parent in MONDO: {no_parent}")

    # macro coverage: each used term -> most-specific reachable grouping ancestor(s)
    anc_cache = {}

    def ancestors(n):
        if n in anc_cache:
            return anc_cache[n]
        seen, q = set(), deque(parents.get(n, ()))
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x); q.extend(parents.get(x, ()))
        anc_cache[n] = seen
        return seen

    reach = sum(1 for u in used_list if (ancestors(u) | {u}) & grouping)
    print(f"   used terms reaching >=1 grouping(macro) term: {reach}/{len(used_list)} "
          f"({reach/len(used_list)*100:.0f}%)")

    # ---- how a multi-term BERT group spans is_a levels ----
    print("\n=== a multi-term PrimeKG BERT group across is_a levels ===")
    osteo = feat[feat["group_name_bert"] == "osteogenesis imperfecta"]
    members = sorted({canon(int(x)) for x in osteo["mondo_id"].dropna().unique() if canon(int(x)) in terms})
    print(f"   group 'osteogenesis imperfecta': {len(members)} member MONDO terms")
    mdep = Counter(depth(m) for m in members)
    print(f"   member depth distribution: {dict(sorted(mdep.items()))}")
    print(f"   member is_a edges INTERNAL to the group (would self-loop if projected): "
          f"{sum(1 for m in members for p in parents.get(m, set()) if p in set(members))}")

    # ---- DOID xref coverage for the 137 Hetionet disease nodes ----
    nd = pd.read_parquet(f"{V7}/nodes__dedup.parquet")
    hd = nd[nd["kind"].astype(str) == "Disease"]
    het_doids = {str(i).split("::")[-1] for i in hd["id"].astype(str)}   # 'DOID:xxx'
    doid2mondo = defaultdict(set)
    for n, t in terms.items():
        for d in t["doid"]:
            doid2mondo[d].add(n)
    matched = sum(1 for d in het_doids if d in doid2mondo)
    print(f"\n=== DOID xref bridge ===")
    print(f"   Hetionet DOID nodes: {len(het_doids)} | have a MONDO xref: {matched} "
          f"| no xref: {len(het_doids) - matched}")
    ex = [d for d in het_doids if d in doid2mondo][:4]
    for d in ex:
        ms = list(doid2mondo[d])
        print(f"      {d} -> MONDO:{ms[0]:07d} ({nm.get(ms[0])})  [{len(ms)} mondo match]")


if __name__ == "__main__":
    main()

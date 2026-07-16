"""Can PrimeKG's BERT disease groups be dropped and folded into the standard MONDO
is_a tree? (read-only feasibility probe, pre-build)

The 1,267 multi-term BERT groups carry all disease edges at GROUP granularity.
To drop the group layer we must re-key those edges to a MONDO term. That is only
sound if a group maps cleanly to a single MONDO 'umbrella' term (one in-group
is_a root that is an ancestor of all other members = a clean subtree). This script
measures how many groups are clean subtrees vs cross-branch name-merges, and
whether the umbrella name matches the BERT group name.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_bert_group_vs_mondo.py
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict, deque

import pandas as pd

MONDO = "Code/data/_cache/mondo.obo"
DIS_FEAT = "Code/data/KG/primekg/disease_features.tab"


def mid(s):
    m = re.search(r"MONDO:(\d+)", s)
    return int(m.group(1)) if m else None


def parse_mondo(path):
    terms, altmap = {}, {}
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"name": None, "obsolete": False, "is_a": set(), "alt": []}
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
    for n, t in terms.items():
        for a in t["alt"]:
            altmap[a] = n
    return terms, altmap


def main():
    terms, altmap = parse_mondo(MONDO)

    def canon(N):
        return altmap.get(N, N)

    nm = {n: t["name"] for n, t in terms.items()}
    parents = {n: {canon(p) for p in t["is_a"] if canon(p) in terms} for n, t in terms.items()}

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

    feat = pd.read_csv(DIS_FEAT, sep="\t",
                       usecols=["node_index", "mondo_id", "group_id_bert", "group_name_bert"], dtype=str)
    # group -> set of canonical member MONDO ids
    grp_members = defaultdict(set)
    grp_name = {}
    for _, r in feat.iterrows():
        g = r["group_id_bert"]
        m = canon(int(r["mondo_id"]))
        if m in terms:
            grp_members[g].add(m)
            grp_name[g] = r["group_name_bert"]

    multi = {g: ms for g, ms in grp_members.items() if len(ms) > 1}
    print(f"total groups {len(grp_members)} | singletons {len(grp_members)-len(multi)} | multi-term {len(multi)}")

    clean1 = []          # exactly 1 in-group root AND it dominates all members (clean subtree)
    clean_root_dominates = []
    multiroot = []       # >1 in-group root = cross-branch name-merge
    name_match = 0
    for g, ms in multi.items():
        # in-group root = member with no other member as ancestor
        roots = [m for m in ms if not (ancestors(m) & ms)]
        if len(roots) == 1:
            r = roots[0]
            dominated = all((m == r) or (r in ancestors(m)) for m in ms)
            clean1.append(g)
            if dominated:
                clean_root_dominates.append(g)
                if str(nm.get(r)).strip().lower() == str(grp_name[g]).strip().lower():
                    name_match += 1
        else:
            multiroot.append((g, roots))

    print(f"\n=== multi-term group shape vs MONDO is_a ===")
    print(f"  single in-group root: {len(clean1)}/{len(multi)} ({len(clean1)/len(multi)*100:.0f}%)")
    print(f"    of which root is_a-DOMINATES all members (clean subtree): {len(clean_root_dominates)}")
    print(f"      of which umbrella MONDO name == BERT group name: {name_match}")
    print(f"  >1 in-group root (cross-branch name-merge): {len(multiroot)}")

    print(f"\n=== sample CLEAN groups (root dominates -> can fold to umbrella) ===")
    for g in clean_root_dominates[:5]:
        ms = multi[g]
        r = [m for m in ms if not (ancestors(m) & ms)][0]
        print(f"  group '{grp_name[g]}' ({len(ms)} members) -> umbrella MONDO:{r:07d} '{nm.get(r)}'")

    print(f"\n=== sample CROSS-BRANCH groups (>1 root -> NOT a clean subtree) ===")
    for g, roots in multiroot[:8]:
        ms = multi[g]
        print(f"  group '{grp_name[g]}' ({len(ms)} members), {len(roots)} roots: "
              f"{[nm.get(r) for r in roots[:4]]}")


if __name__ == "__main__":
    main()

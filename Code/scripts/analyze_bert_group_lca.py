"""Feasibility of folding BERT groups into MONDO via the member LCA (umbrella).

For each real multi-term BERT group (exclude singletons whose group_id_bert is
empty), drop obsolete members, then compute the most-specific common ancestor(s)
(LCA in the is_a DAG, including members themselves). Report whether the umbrella
is a specific term (deep) or too general (shallow), and whether it is unique.
This decides if we can drop the dgrp layer and attach edges to a real MONDO term.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_bert_group_lca.py
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

    def anc_incl(n):   # ancestors including self
        if n in anc_cache:
            return anc_cache[n]
        seen, q = {n}, deque(parents.get(n, ()))
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x); q.extend(parents.get(x, ()))
        anc_cache[n] = seen
        return seen

    depth_cache = {}

    def depth(n):
        if n in depth_cache:
            return depth_cache[n]
        if not parents.get(n):
            depth_cache[n] = 0; return 0
        d = 1 + min((depth(p) for p in parents[n]), default=0)
        depth_cache[n] = d; return d

    feat = pd.read_csv(DIS_FEAT, sep="\t",
                       usecols=["mondo_id", "group_id_bert", "group_name_bert"], dtype=str)
    feat = feat[feat["group_id_bert"].notna() & (feat["group_id_bert"].str.strip() != "")]
    grp_members, grp_name = defaultdict(set), {}
    for g, m, gn in zip(feat["group_id_bert"], feat["mondo_id"], feat["group_name_bert"]):
        c = canon(int(m))
        if c in terms:
            grp_members[g].add(c); grp_name[g] = gn
    multi = {g: ms for g, ms in grp_members.items() if len(ms) > 1}
    print(f"real multi-term groups: {len(multi)}")

    single_umbrella = 0
    member_lca = 0           # LCA is itself a member (clean subtree)
    external_umbrella = 0    # LCA is a real term, not a member (sibling merge)
    too_general = 0          # LCA only at depth<=3 (disease/by-system level) -> not usable
    no_common = 0            # disjoint, only root in common
    lca_depths = Counter()
    obsolete_dropped = 0
    samples_ext, samples_bad = [], []

    for g, ms_all in multi.items():
        ms = {m for m in ms_all if not terms[m]["obsolete"]}
        obsolete_dropped += len(ms_all) - len(ms)
        if len(ms) <= 1:
            single_umbrella += 1
            continue
        common = set.intersection(*[anc_incl(m) for m in ms])
        if not common:
            no_common += 1
            continue
        # most-specific common ancestors (no other common term is a descendant)
        msca = [c for c in common if not any(c in anc_incl(o) and o != c for o in common)]
        d = max(depth(c) for c in msca)
        lca_depths[d] += 1
        is_member = any(c in ms for c in msca)
        if d <= 3:
            too_general += 1
            if len(samples_bad) < 8:
                samples_bad.append((grp_name[g], len(ms), [nm.get(c) for c in msca[:3]], d))
        elif is_member:
            member_lca += 1
        else:
            external_umbrella += 1
            if len(samples_ext) < 8:
                samples_ext.append((grp_name[g], len(ms), [nm.get(c) for c in msca[:2]], d))

    usable = member_lca + external_umbrella
    print(f"\n=== fold-to-umbrella feasibility (multi-term groups) ===")
    print(f"  collapsed to single after dropping obsolete: {single_umbrella}")
    print(f"  LCA is a MEMBER (clean subtree -> fold to it):        {member_lca}")
    print(f"  LCA is an EXTERNAL real umbrella (sibling merge):     {external_umbrella}")
    print(f"  LCA too general (depth<=3, NOT usable as umbrella):   {too_general}")
    print(f"  no common ancestor at all:                            {no_common}")
    print(f"  --> USABLE specific umbrella (member or external): {usable}/{len(multi)} "
          f"({usable/len(multi)*100:.0f}%)  [+{single_umbrella} trivially single after obsolete drop]")
    print(f"  obsolete member terms dropped across all groups: {obsolete_dropped}")
    print(f"  LCA depth distribution: {dict(sorted(lca_depths.items()))}")

    print(f"\n  sample EXTERNAL-umbrella folds (sibling subtypes -> real parent):")
    for gn, k, lca, d in samples_ext:
        print(f"    '{gn}' ({k} members) -> {lca} (depth {d})")
    print(f"\n  sample TOO-GENERAL (LCA shallow -> folding would over-generalize):")
    for gn, k, lca, d in samples_bad:
        print(f"    '{gn}' ({k} members) -> {lca} (depth {d})")


if __name__ == "__main__":
    main()

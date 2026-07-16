"""Census of the Reactome pathway hierarchy vs our KG pathway nodes.

Read-only. Uses Reactome's DIRECTED hierarchy (ReactomePathwaysRelation.txt:
parent<TAB>child) + names (ReactomePathways.txt) to orient the meso->macro
hierarchy, then answers, BEFORE any convergence re-test:
  * how many high-level (macro / top-level) and low-level (leaf) pathways exist
  * how many of OUR KG pathway nodes map into the hierarchy, and to a top-level
  * which hierarchy nodes are MISSING from our KG (need supplementing) — split by
    top-level / internal / leaf
  * whether the macro (top-level) nodes carry direct protein edges or are reachable
    only via hierarchy

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_pathway_hierarchy.py
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

REL = "Code/data/_cache/ReactomePathwaysRelation.txt"
PWN = "Code/data/_cache/ReactomePathways.txt"
NODES = "Code/data/KG/_merged_kg_dedup_v2/nodes__dedup.parquet"
EDGES = "Code/data/KG/_merged_kg_dedup_v2/edges__dedup.parquet"


def main() -> None:
    # directed hierarchy (human only)
    par2ch = defaultdict(set)
    ch2par = defaultdict(set)
    nodes = set()
    with open(REL, encoding="utf-8") as f:
        for line in f:
            p, c = line.rstrip("\n").split("\t")
            if p.startswith("R-HSA") and c.startswith("R-HSA"):
                par2ch[p].add(c); ch2par[c].add(p); nodes |= {p, c}
    id2name = {}
    with open(PWN, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].startswith("R-HSA"):
                id2name[parts[0]] = parts[1]
    # also count human pathways that exist in names but maybe isolated (no relation)
    all_hsa = set(id2name)
    roots = {n for n in nodes if n not in ch2par}        # top-level macro
    leaves = {n for n in nodes if n not in par2ch}       # most specific
    isolated = all_hsa - nodes                            # in names, not in any relation
    print("=== Reactome human pathway hierarchy ===")
    print(f"  pathways in names file      : {len(all_hsa)}")
    print(f"  pathways in hierarchy (rel) : {len(nodes)}")
    print(f"  isolated (no parent/child)  : {len(isolated)}")
    print(f"  TOP-LEVEL (macro) roots     : {len(roots)}")
    print(f"  LEAF (most specific)        : {len(leaves)}")
    print(f"  internal (meso)             : {len(nodes) - len(roots) - len(leaves)}")

    def ancestors_top(n):
        seen, q, tops = set(), deque([n]), set()
        while q:
            x = q.popleft()
            if x in seen:
                continue
            seen.add(x)
            ps = ch2par.get(x)
            if not ps:
                if x in nodes:
                    tops.add(x)
            else:
                q.extend(ps)
        return tops

    print("\n  top-level macro categories + #descendant pathways:")
    # count descendants per root
    desc = defaultdict(set)
    for n in nodes:
        for t in ancestors_top(n):
            desc[t].add(n)
    for t in sorted(roots, key=lambda x: -len(desc[x]))[:30]:
        print(f"    {len(desc[t]):>4}  {id2name.get(t,'?')}  ({t})")

    # ---- our KG pathway nodes ----
    nd = pd.read_parquet(NODES, columns=["id", "kind"])
    ourpw = nd[nd["kind"] == "pathway"]["id"].astype(str)
    our_hsa = {x.split(":")[-1] for x in ourpw if "R-HSA" in x}
    print(f"\n=== our KG prime pathway nodes ===")
    print(f"  total prime pathway nodes   : {len(ourpw)}")
    print(f"  R-HSA ids                   : {len(our_hsa)}")
    print(f"  in Reactome hierarchy       : {len(our_hsa & nodes)}")
    print(f"  in names but isolated       : {len(our_hsa & isolated)}")
    print(f"  NOT in Reactome at all      : {len(our_hsa - all_hsa)}")
    traceable = {n for n in our_hsa & nodes if ancestors_top(n)}
    print(f"  traceable to a top-level    : {len(traceable)} "
          f"({len(traceable)/max(len(our_hsa),1)*100:.0f}% of ours)")

    # ---- missing hierarchy nodes (need supplement) ----
    in_kg = our_hsa
    missing = nodes - in_kg
    miss_top = missing & roots
    miss_leaf = missing & leaves
    miss_int = missing - roots - leaves
    print(f"\n=== hierarchy nodes MISSING from our KG (supplement candidates) ===")
    print(f"  total hierarchy nodes       : {len(nodes)}")
    print(f"  present in our KG           : {len(nodes & in_kg)}")
    print(f"  MISSING from our KG         : {len(missing)}  "
          f"(top-level {len(miss_top)} | internal {len(miss_int)} | leaf {len(miss_leaf)})")
    print("  missing TOP-LEVEL macro categories:")
    for t in sorted(miss_top, key=lambda x: -len(desc[x])):
        print(f"    {len(desc[t]):>4} desc  {id2name.get(t,'?')} ({t})")

    # ---- do macro nodes carry direct protein edges? ----
    ed = pd.read_parquet(EDGES, columns=["src", "dst", "relation"])
    pp = ed[ed["relation"] == "prime:pathway_protein"]
    pw_with_prot = set()
    for s, d in zip(pp["src"].astype(str), pp["dst"].astype(str)):
        pw = s if s.startswith("prime:pathway:") else d
        pw_with_prot.add(pw.split(":")[-1])
    print(f"\n=== protein-edge coverage by hierarchy level ===")
    print(f"  roots(macro) with direct protein edges : {len(roots & pw_with_prot)}/{len(roots)}")
    print(f"  leaves with direct protein edges       : {len(leaves & pw_with_prot)}/{len(leaves)}")
    print(f"  internal with direct protein edges     : "
          f"{len((nodes-roots-leaves) & pw_with_prot)}/{len(nodes-roots-leaves)}")


if __name__ == "__main__":
    main()

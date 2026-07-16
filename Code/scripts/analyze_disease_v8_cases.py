"""Disease v8 — edge conservation + fold cases + meta-path cases (read-only).

Verifies the disease axis fold did not lose associations (only dedup collisions),
shows real fold examples (singleton / external-LCA / residual), DOID bridge, and
the disease meta-paths with concrete instances.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_disease_v8_cases.py
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

V7 = "Code/data/KG/_merged_kg_dedup_v7"
V8 = "Code/data/KG/_merged_kg_dedup_v8"
DIS_EXT = ["prime:disease_protein", "prime:disease_phenotype_positive",
           "prime:disease_phenotype_negative", "prime:contraindication",
           "prime:indication", "prime:off-label use", "prime:exposure_disease"]


def endpoints(ed, rel, pref):
    e = ed[ed["relation"] == rel]
    s = set()
    for a, b in zip(e["src"].astype(str), e["dst"].astype(str)):
        s.add(a if a.startswith(pref) else b)
    return s


def main():
    e7 = pd.read_parquet(f"{V7}/edges__dedup.parquet")
    e8 = pd.read_parquet(f"{V8}/edges__dedup.parquet")
    n8 = pd.read_parquet(f"{V8}/nodes__dedup.parquet")
    dm = pd.read_parquet(f"{V8}/dis_meta.parquet")
    fm = pd.read_parquet(f"{V8}/dis_fold_map.parquet")
    nm = dict(zip(n8["id"], n8["name"]))
    valid = set(n8["id"])

    print("=" * 66)
    print("(1) EDGE CONSERVATION — disease external relations (v7 -> v8)")
    print("=" * 66)
    for rel in DIS_EXT:
        c7 = int((e7["relation"] == rel).sum())
        c8 = int((e8["relation"] == rel).sum())
        print(f"  {rel:<34} {c7:>8} -> {c8:>8}  (dedup-merged {c7-c8})")
    # no disease external edge points to a dangling/old prime:disease node
    de8 = e8[e8["relation"].isin(DIS_EXT)]
    orphan = (~de8["src"].isin(valid) | ~de8["dst"].isin(valid)).sum()
    leftover_prime = (de8["src"].astype(str).str.startswith("prime:disease:") |
                      de8["dst"].astype(str).str.startswith("prime:disease:")).sum()
    print(f"  orphan endpoints: {int(orphan)} | leftover prime:disease endpoints: {int(leftover_prime)}")

    print("\n" + "=" * 66)
    print("(2) FOLD CASES")
    print("=" * 66)
    for reason in ["external_lca", "member_lca", "residual_multi_lca", "residual_too_general"]:
        sub = fm[fm["reason"] == reason].head(3)
        print(f"\n  [{reason}]")
        for _, r in sub.iterrows():
            tgt = r["target"]
            tgt_nm = nm.get(tgt, "(residual group)")
            print(f"    {r['prime_id'][:40]} ({r['n_members']} members) -> {tgt} '{tgt_nm}'")

    # Fanconi anemia complementation group -> Fanconi anemia (external_lca expected)
    fanc = fm[fm["prime_id"].str.contains("_") & fm["members"].map(lambda m: len(m) > 15)]
    ext = fm[fm["reason"] == "external_lca"]
    print(f"\n  external_lca total {len(ext)} | member_lca {int((fm['reason']=='member_lca').sum())} "
          f"| residual dgrp groups {fm['target'].astype(str).str.startswith('dgrp:').sum()}")

    print("\n" + "=" * 66)
    print("(3) DOID BRIDGE")
    print("=" * 66)
    xref = e8[e8["relation"] == "doid:xref_mondo"]
    print(f"  doid:xref_mondo edges: {len(xref)}")
    for s, d in zip(xref["src"].astype(str).head(4), xref["dst"].astype(str).head(4)):
        print(f"    {s} ({nm.get(s)}) -> {d} ({nm.get(d)})")

    print("\n" + "=" * 66)
    print("(4) DISEASE META-PATH CASES")
    print("=" * 66)
    # build maps
    dp = e8[e8["relation"] == "prime:disease_protein"]
    prot2dis, dis2prot = defaultdict(set), defaultdict(set)
    for a, b in zip(dp["src"].astype(str), dp["dst"].astype(str)):
        dis, pr = (a, b) if a.startswith("mondo:") or a.startswith("dgrp:") else (b, a)
        if pr.startswith("prot:"):
            prot2dis[pr].add(dis); dis2prot[dis].add(pr)
    # is_a parent map
    isa = e8[e8["relation"] == "mondo:is_a"]
    parent = defaultdict(set)
    for s, d in zip(isa["src"].astype(str), isa["dst"].astype(str)):
        parent[s].add(d)
    macro_bs = set(dm[dm["is_macro_body_system"]]["id"])

    def climb(n):
        seen, q = set(), deque([n])
        while q:
            x = q.popleft()
            for p in parent.get(x, ()):
                if p not in seen:
                    seen.add(p); q.append(p)
        return seen

    # M_dis: protein -> disease -> body-system macro
    print("\n  [M_dis] protein -disease_protein-> disease -is_a*-> body-system macro")
    shown = 0
    for pr, ds in prot2dis.items():
        for d in ds:
            ms = (climb(d) | {d}) & macro_bs
            if ms and len(dis2prot[d]) < 40:
                print(f"    {pr}({nm.get(pr)}) -> {nm.get(d)} -> {[nm.get(m) for m in list(ms)[:2]]}")
                shown += 1; break
        if shown >= 3:
            break

    # drug -> disease (indication/contraindication)
    ind = e8[e8["relation"].isin(["prime:indication", "prime:contraindication"])]
    drug2dis = defaultdict(set)
    for a, b in zip(ind["src"].astype(str), ind["dst"].astype(str)):
        dis, dr = (a, b) if (a.startswith("mondo:") or a.startswith("dgrp:")) else (b, a)
        drug2dis[dr].add(dis)
    drugs = list(drug2dis)[:300]
    # C_dis_meso: two drugs share a disease
    print("\n  [C_dis_meso] two drugs share a disease (indication/contraindication)")
    shown = 0
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            sh = drug2dis[drugs[i]] & drug2dis[drugs[j]]
            if sh:
                d = next(iter(sh))
                print(f"    {nm.get(drugs[i])} + {nm.get(drugs[j])} share-> {nm.get(d)}")
                shown += 1; break
        if shown >= 3:
            break

    # C_dis_macro: two drugs share a body-system macro
    def drug_macros(dr):
        ms = set()
        for d in drug2dis.get(dr, ()):
            ms |= (climb(d) | {d}) & macro_bs
        return ms
    print("\n  [C_dis_macro] two drugs roll up to the same body-system disease macro")
    shown = 0
    for i in range(0, 40):
        for j in range(i + 1, 40):
            if i >= len(drugs) or j >= len(drugs):
                continue
            sh = drug_macros(drugs[i]) & drug_macros(drugs[j])
            if sh:
                print(f"    {nm.get(drugs[i])} + {nm.get(drugs[j])} share-> {[nm.get(m) for m in list(sh)[:3]]}")
                shown += 1; break
        if shown >= 3:
            break


if __name__ == "__main__":
    main()

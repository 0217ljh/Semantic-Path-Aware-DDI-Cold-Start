"""Summary of the v6 GO layer: stats + meta-paths + cases.

Read-only. Reports the GO layer as built in v6 (Code/data/KG/_merged_kg_dedup_v6),
the GO meta-paths now walkable, and concrete cases (protein->BP->slim chain; a
2-drug convergence on a BP slim; a regulates cross-link).

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_go_v6_summary.py
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

V6 = "Code/data/KG/_merged_kg_dedup_v6"
PD_CSV = "Code/data/_cache/pd_mechanism_200.csv"


def main() -> None:
    nd = pd.read_parquet(f"{V6}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{V6}/edges__dedup.parquet")
    gm = pd.read_parquet(f"{V6}/go_meta.parquet")
    nm = dict(zip(nd["id"], nd["name"]))

    # ---- STATS ----
    print("=" * 68)
    print("v6 GO LAYER — statistics")
    print("=" * 68)
    print(f"  GO nodes by namespace: {gm['namespace'].value_counts().to_dict()}")
    print(f"  is_slim {int(gm['is_slim'].sum())} | obsolete {int(gm['obsolete'].sum())} "
          f"(with replaced_by {int(gm['replaced_by'].notna().sum())})")
    hier = {r: int((ed['relation'] == r).sum()) for r in
            ['go:is_a', 'go:part_of', 'go:regulates', 'go:positively_regulates', 'go:negatively_regulates']}
    print(f"  typed hierarchy edges: {hier}")
    memb = {r: int((ed['relation'] == r).sum()) for r in
            ['prime:bioprocess_protein', 'prime:molfunc_protein', 'prime:cellcomp_protein']}
    print(f"  protein->GO membership: {memb}")

    # ---- protein <-> BP membership; BP -> slim ----
    bpp = ed[ed["relation"] == "prime:bioprocess_protein"]
    prot2bp, bp2prot = defaultdict(set), defaultdict(set)
    for s, d in zip(bpp["src"].astype(str), bpp["dst"].astype(str)):
        bp, pr = (s, d) if s.startswith("go:") else (d, s)
        prot2bp[pr].add(bp); bp2prot[bp].add(pr)
    bp2slim = {r["id"]: (list(r["slim_ancestors"]) if r["slim_ancestors"] is not None else [])
               for _, r in gm[gm["namespace"] == "biological_process"].iterrows()}

    # n_protein per BP term distribution (hub-ness)
    npr = gm[gm["namespace"] == "biological_process"]["n_protein"]
    print(f"\n  BP term n_protein: median {int(npr.median())} | mean {npr.mean():.1f} | "
          f"95th {int(npr.quantile(.95))} | max {int(npr.max())}")
    print("  biggest BP hubs (most proteins):")
    for _, r in gm[gm['namespace']=='biological_process'].sort_values('n_protein', ascending=False).head(5).iterrows():
        print(f"    {r['n_protein']:>5}  {nm.get(r['id'])}")

    # slim macro: proteins reachable per slim BP term
    slim_prot = defaultdict(set)
    for pr, bps in prot2bp.items():
        for bp in bps:
            for s in bp2slim.get(bp, []):
                slim_prot[f"go:{s}"].add(pr)
    print("\n  BP SLIM macro categories — #proteins reachable (the 72-term macro layer):")
    for s, ps in sorted(slim_prot.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"    {len(ps):>5}  {nm.get(s)}")

    # ---- META-PATHS ----
    print("\n" + "=" * 68)
    print("GO META-PATHS (v6) — GO joins drugs only via proteins (no drug->GO edge)")
    print("=" * 68)
    print("  M_go   P -bioprocess_protein-> BP -is_a/part_of*-> BP_slim         (micro->macro)")
    print("  C_go_meso   D_A->P_A->BP <-P_B<-D_B           (shared GO-BP term)")
    print("  C_go_macro  D_A->...->BP_a ->* SLIM <-* BP_b<-...<-D_B   (shared BP slim)")
    print("  (regulates family = BP-regulates-BP cross-links, NOT roll-up)")

    # drug -> proteins
    drug_kind = set(nd[nd["kind"].astype(str).isin(["Drug", "drug"])]["id"])
    drug2prot = defaultdict(set)
    for s, d in zip(ed["src"].astype(str), ed["dst"].astype(str)):
        if s in drug_kind and d.startswith("prot:"):
            drug2prot[s].add(d)
        elif d in drug_kind and s.startswith("prot:"):
            drug2prot[d].add(s)

    def drug_slims(dru):
        ms = set()
        for pr in drug2prot.get(dru, ()):
            for bp in prot2bp.get(pr, ()):
                for s in bp2slim.get(bp, []):
                    ms.add(s)
        return ms

    # ---- CASES ----
    print("\n" + "=" * 68)
    print("CASES")
    print("=" * 68)
    # 1. full chain protein -> BP -> slim
    ex_bp = next(b for b in bp2slim if bp2slim[b] and bp2prot[b])
    pr = next(iter(bp2prot[ex_bp]))
    print(f"[1] micro->macro: {pr}({nm.get(pr)}) -[bioprocess_protein]-> {nm.get(ex_bp)} "
          f"-[is_a/part_of*]-> {[nm.get('go:'+str(s)) for s in bp2slim[ex_bp][:2]]}")
    # 2. 2-drug convergence on BP slim (PD pairs)
    pdp = pd.read_csv(PD_CSV)
    print("[2] 2-drug GO-slim convergence (PD pairs, shared BP slim):")
    shown = 0
    for _, r in pdp.iterrows():
        a, b = r["drug_a_id"], r["drug_b_id"]
        sa, sb = drug_slims(a), drug_slims(b)
        sh = sa & sb
        if sh and len(sa) <= 8 and shown < 4:
            print(f"    {r['drug_a_name']} + {r['drug_b_name']}  ({str(r['ddi_type'])[:32]})")
            print(f"      shared BP slim: {[nm.get('go:'+str(s)) for s in list(sh)[:4]]}")
            shown += 1
    # 3. a regulates cross-link
    reg = ed[ed["relation"] == "go:negatively_regulates"].iloc[0]
    print(f"[3] regulates cross-link: {nm.get(reg['src'])} -[negatively_regulates]-> {nm.get(reg['dst'])}")


if __name__ == "__main__":
    main()

"""Summary of the v3 KG after the Reactome pathway axis: stats + meta-paths + cases.

Read-only. Reports the Reactome pathway layer as built in v3
(Code/data/KG/_merged_kg_dedup_v3), the meta-paths now walkable, and concrete
cases (full micro->meso->macro chain; a 2-drug macro convergence; a multi-macro
DAG pathway).

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_pathway_v3_summary.py
"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

V3 = "Code/data/KG/_merged_kg_dedup_v3"
PD_CSV = "Code/data/_cache/pd_mechanism_200.csv"


def main() -> None:
    nd = pd.read_parquet(f"{V3}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{V3}/edges__dedup.parquet")
    meta = pd.read_parquet(f"{V3}/path_meta.parquet")
    nm = dict(zip(nd["id"], nd["name"]))

    # hierarchy maps
    hp = ed[ed["relation"] == "prime:pathway_pathway"]
    par2ch, ch2par = defaultdict(set), defaultdict(set)
    for s, d in zip(hp["src"], hp["dst"]):       # src=child dst=parent
        ch2par[s].add(d); par2ch[d].add(s)
    macro_ids = set(meta[meta["level"] == "macro"]["id"])
    path2macro = {r["id"]: set(f"path:{a}" for a in r["macro_ancestors"])
                  for _, r in meta.iterrows()}
    for m in macro_ids:                          # a macro traces to itself
        path2macro.setdefault(m, set()).add(m)

    # protein <-> pathway
    pp = ed[ed["relation"] == "prime:pathway_protein"]
    prot2paths, path2prots = defaultdict(set), defaultdict(set)
    for s, d in zip(pp["src"].astype(str), pp["dst"].astype(str)):
        pw, pr = (s, d) if s.startswith("path:") else (d, s)
        prot2paths[pr].add(pw); path2prots[pw].add(pr)

    # ---- STATS ----
    print("=" * 70)
    print("v3 PATHWAY LAYER — Reactome axis (statistics)")
    print("=" * 70)
    print(f"  Reactome path: nodes {len(meta)} | level: "
          f"{meta['level'].value_counts().to_dict()}")
    print(f"  connector-only {int(meta['is_connector_only'].sum())} | "
          f"obsolete {int(meta['obsolete'].sum())} | multi-macro {int(meta['macro_ancestors'].map(len).gt(1).sum())}")
    print(f"  edges: prime:pathway_protein {len(pp)} | prime:pathway_pathway(directed) {len(hp)}")

    def descendants(m):
        seen, q = set(), deque([m])
        while q:
            x = q.popleft()
            for c in par2ch.get(x, ()):
                if c not in seen:
                    seen.add(c); q.append(c)
        return seen

    print("\n  macro categories — #descendant pathways | #distinct proteins reachable:")
    rows = []
    for m in macro_ids:
        desc = descendants(m)
        prots = set().union(*[path2prots[p] for p in desc | {m}]) if (desc or m) else set()
        rows.append((len(desc), len(prots), nm.get(m)))
    for ndesc, nprot, name in sorted(rows, reverse=True)[:12]:
        print(f"    {ndesc:>4} paths | {nprot:>5} prot  {name}")

    # ---- META-PATHS active with Reactome only ----
    print("\n" + "=" * 70)
    print("META-PATHS now walkable in v3 (Reactome side; SMPDB cross-modes pending)")
    print("=" * 70)
    print("  M3  P -pathway_protein-> PW_leaf -pathway_pathway*-> PW_macro      (micro->macro)")
    print("  C_meso   D_A->P_A->PW <-P_B<-D_B           (shared meso pathway)")
    print("  C_macro  D_A->...->PW_a ->* MACRO <-* PW_b<-...<-D_B   (shared macro system)")

    # drug -> proteins (from any drug-incident edge to a prot node)
    drug_kind = set(nd[nd["kind"].astype(str).isin(["Drug", "drug"])]["id"])
    drug2prot = defaultdict(set)
    for s, d, sk, dk in zip(ed["src"].astype(str), ed["dst"].astype(str),
                            ed["src_kind"].astype(str), ed["dst_kind"].astype(str)):
        if s in drug_kind and d.startswith("prot:"):
            drug2prot[s].add(d)
        elif d in drug_kind and s.startswith("prot:"):
            drug2prot[d].add(s)

    def drug_macros(dru):
        ms = set()
        for pr in drug2prot.get(dru, ()):
            for pw in prot2paths.get(pr, ()):
                ms |= path2macro.get(pw, set())
        return ms

    # ---- CASE 1: full micro->meso->macro chain ----
    print("\n" + "=" * 70)
    print("CASES")
    print("=" * 70)
    hemo = next(m for m in macro_ids if str(nm.get(m, "")).lower() == "hemostasis")
    leaf = next(p for p in descendants(hemo) if path2prots[p] and meta.set_index('id').loc[p, 'level'] == 'leaf')
    pr = next(iter(path2prots[leaf]))
    chain = " -> ".join([f"{nm.get(p)}" for p in [pr, leaf]] + [nm.get(m) for m in list(path2macro[leaf])[:2]])
    print(f"[1] full chain: {pr}({nm.get(pr)}) -[pathway_protein]-> {leaf}({nm.get(leaf)}) "
          f"-[hierarchy*]-> {[nm.get(m) for m in path2macro[leaf]]}")

    # ---- CASE 2: 2-drug macro convergence on a PD pair ----
    pdp = pd.read_csv(PD_CSV)
    shown = 0
    print("[2] 2-drug MACRO convergence (PD pairs, shared macro via different pathways):")
    for _, r in pdp.iterrows():
        a, b = r["drug_a_id"], r["drug_b_id"]
        ma, mb = drug_macros(a), drug_macros(b)
        shared = ma & mb
        if shared and len(ma) <= 6 and len(mb) <= 6 and shown < 4:
            print(f"    {r['drug_a_name']} + {r['drug_b_name']}  ({str(r['ddi_type'])[:40]})")
            print(f"      shared macro: {[nm.get(m) for m in shared]}")
            shown += 1

    # ---- CASE 3: multi-macro (DAG) pathway ----
    mm = meta[meta["macro_ancestors"].map(len) > 1].iloc[0]
    print(f"[3] multi-macro DAG pathway: {mm['id']}({nm.get(mm['id'])}) "
          f"rolls up to {[nm.get('path:'+a) for a in mm['macro_ancestors']]}")


if __name__ == "__main__":
    main()

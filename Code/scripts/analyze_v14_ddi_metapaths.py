"""Final v14 substrate analysis: node-level census + DDI drug-drug meta-path catalog.

Read-only. Two parts:
  (A) node census of the completed canonical KG (kinds, prefixes, macro layers).
  (B) the meta-path TEMPLATES that connect two drugs (the signal a DDI model can use),
      with coverage over REAL interacting pairs (drugbank_latest binary positives) and
      a concrete example per template. Templates by hop depth:
        L1 (shared direct neighbor): shared protein / pharm_class / disease / effect
        L2 (shared meso via proteins): shared GO-BP / pathway / anatomy / disease / phenotype
        L3 (shared macro): shared body-system / organ-system / abnormality / GO-slim / Reactome-macro

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_v14_ddi_metapaths.py
"""
from __future__ import annotations

import glob
from collections import Counter, defaultdict

import pandas as pd

V14 = "Code/data/KG/_merged_kg_dedup_v14"
DDI_GLOB = "Code/data/ddi_unified/binary_cls/drugbank_deng/inductive/S1/fold0/train.parquet"
DRUG_PROT_RELS = ("db:target", "db:enzyme", "db:transporter", "db:carrier",
                  "prime:drug_protein", "het:CdG", "het:CuG", "het:CbG")


def main() -> None:
    nd = pd.read_parquet(f"{V14}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{V14}/edges__dedup.parquet")
    nm = dict(zip(nd["id"], nd["name"])); kind = dict(zip(nd["id"], nd["kind"].astype(str)))
    s = ed["src"].astype(str); d = ed["dst"].astype(str); r = ed["relation"].astype(str)

    # ---------- (A) NODE CENSUS ----------
    print("=" * 70)
    print("(A) v14 NODE CENSUS")
    print("=" * 70)
    print(f"  total nodes {len(nd)} | total edges {len(ed)}")
    print("  by kind:")
    for k, c in nd["kind"].astype(str).value_counts().items():
        print(f"    {k:<22} {c}")
    pref = nd["id"].astype(str).str.split(":").str[0]
    print("  by canonical prefix:")
    for k, c in pref.value_counts().items():
        if c > 40:
            print(f"    {k:<16} {c}")

    # ---------- build drug adjacency ----------
    drugs = set(nd[nd["kind"].astype(str) == "drug"]["id"])
    d2prot, d2class, d2dis, d2eff = (defaultdict(set) for _ in range(4))
    for a, b, rel in zip(s, d, r):
        if a in drugs or b in drugs:
            dr = a if a in drugs else b; o = b if dr == a else a
            if o.startswith("prot:"):
                d2prot[dr].add(o)
            elif o.startswith("pclass:"):
                d2class[dr].add(o)
            elif o.startswith("mondo:") or o.startswith("dgrp:"):
                if rel in ("prime:indication", "prime:contraindication", "prime:off-label use", "het:CtD", "het:CpD"):
                    d2dis[dr].add(o)
            elif o.startswith("hp:") or o.startswith("se:"):
                d2eff[dr].add(o)

    # protein -> meso maps
    def pmap(rel, pref_):
        mp = defaultdict(set)
        sub = ed[r == rel]
        for a, b in zip(sub["src"].astype(str), sub["dst"].astype(str)):
            p, o = (a, b) if a.startswith("prot:") else (b, a)
            if p.startswith("prot:") and o.startswith(pref_):
                mp[p].add(o)
        return mp
    p2bp = pmap("prime:bioprocess_protein", "go:")
    p2path = pmap("prime:pathway_protein", "path:")
    p2anat = pmap("prime:anatomy_protein_present", "uberon:")
    p2dis = pmap("prime:disease_protein", "mondo:")
    p2phe = pmap("prime:phenotype_protein", "hp:")

    # macro lookups from sidecars
    gm = pd.read_parquet(f"{V14}/go_meta.parquet")
    bp_slim = {x: tuple(v) if v is not None else () for x, v in zip(gm["id"], gm["slim_ancestors"])}
    dm = pd.read_parquet(f"{V14}/dis_meta.parquet")
    dis_macro = {x: tuple(v) if v is not None else () for x, v in zip(dm["id"], dm["macro_body_system_ancestors"])}
    am = pd.read_parquet(f"{V14}/anat_meta.parquet")
    anat_macro = {x: tuple(v) if v is not None else () for x, v in zip(am["id"], am["macro_ancestors"])}
    hm = pd.read_parquet(f"{V14}/hp_meta.parquet")
    hp_macro = {x: tuple(v) if v is not None else () for x, v in zip(hm["id"], hm["macro_top_level_abnormality"])}

    def drug_meso(dr, p2x):
        out = set()
        for p in d2prot.get(dr, ()):
            out |= p2x.get(p, set())
        return out

    def drug_macro(dr, p2x, macro, pfx):
        out = set()
        for o in drug_meso(dr, p2x):
            for m in macro.get(o, ()):
                out.add(f"{pfx}{m}")
        return out

    # ---------- (B) DDI META-PATH CATALOG over real interacting pairs ----------
    print("\n" + "=" * 70)
    print("(B) DDI META-PATH TEMPLATES — coverage over real interacting pairs")
    print("=" * 70)
    cand = sorted(glob.glob(DDI_GLOB)) or sorted(glob.glob("Code/data/ddi_unified/binary_cls/*/inductive/S1/fold0/train.parquet"))
    pos = pd.read_parquet(cand[0])
    print(f"  pairs source: {cand[0]} | cols {list(pos.columns)}")
    dbcols = [c for c in pos.columns if pos[c].astype(str).str.startswith("DB").any()]
    a_col, b_col = dbcols[0], dbcols[1]
    lc = next((c for c in ("label", "y", "Y") if c in pos.columns), None)
    rows = pos[pos[lc] == 1] if lc else pos
    pairs = [(f"drug:{a}", f"drug:{b}") for a, b in zip(rows[a_col].astype(str), rows[b_col].astype(str))]
    pairs = [(a, b) for a, b in pairs if a in drugs and b in drugs][:4000]
    print(f"  interacting pairs sampled: {len(pairs)} ({cand[0].split('/')[-4]} S1 fold0 train positives)")

    templates = {
        "L1 shared protein     (D-P-D)": lambda a, b: d2prot[a] & d2prot[b],
        "L1 shared pharm_class (D-PC-D)": lambda a, b: d2class[a] & d2class[b],
        "L1 shared disease     (D-Dis-D)": lambda a, b: d2dis[a] & d2dis[b],
        "L1 shared effect      (D-Eff-D)": lambda a, b: d2eff[a] & d2eff[b],
        "L2 shared GO-BP       (D-P-BP-P-D)": lambda a, b: drug_meso(a, p2bp) & drug_meso(b, p2bp),
        "L2 shared pathway     (D-P-Path-P-D)": lambda a, b: drug_meso(a, p2path) & drug_meso(b, p2path),
        "L2 shared anatomy     (D-P-Anat-P-D)": lambda a, b: drug_meso(a, p2anat) & drug_meso(b, p2anat),
        "L2 shared disease     (D-P-Dis-P-D)": lambda a, b: drug_meso(a, p2dis) & drug_meso(b, p2dis),
        "L2 shared phenotype   (D-P-Phe-P-D)": lambda a, b: drug_meso(a, p2phe) & drug_meso(b, p2phe),
        "L3 shared body-system  macro": lambda a, b: drug_macro(a, p2dis, dis_macro, "mondo:") & drug_macro(b, p2dis, dis_macro, "mondo:"),
        "L3 shared organ-system macro": lambda a, b: drug_macro(a, p2anat, anat_macro, "uberon:") & drug_macro(b, p2anat, anat_macro, "uberon:"),
        "L3 shared abnormality  macro": lambda a, b: drug_macro(a, p2phe, hp_macro, "hp:") & drug_macro(b, p2phe, hp_macro, "hp:"),
        "L3 shared GO-slim      macro": lambda a, b: drug_macro(a, p2bp, bp_slim, "go:") & drug_macro(b, p2bp, bp_slim, "go:"),
    }
    cov = Counter(); example = {}
    for a, b in pairs:
        for name, fn in templates.items():
            sh = fn(a, b)
            if sh:
                cov[name] += 1
                if name not in example:
                    one = sorted(sh)[0]
                    example[name] = (nm.get(a), nm.get(b), nm.get(one) or one)
    print(f"\n  {'template':<38} {'pairs covered':>13}   example")
    for name in templates:
        c = cov[name]; ex = example.get(name)
        exs = f"{ex[0]} + {ex[1]} -> {ex[2]}" if ex else "-"
        print(f"  {name:<38} {c:>6} ({c/len(pairs)*100:>4.0f}%)   {exs[:60]}")
    # any-path coverage
    anyc = sum(1 for a, b in pairs if any(fn(a, b) for fn in templates.values()))
    print(f"\n  pairs with >=1 meta-path: {anyc}/{len(pairs)} ({anyc/len(pairs)*100:.0f}%)")


if __name__ == "__main__":
    main()

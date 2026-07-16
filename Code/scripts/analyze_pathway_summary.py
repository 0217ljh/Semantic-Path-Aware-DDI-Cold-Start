"""Consolidated census of ALL pathway sources in the KG (Reactome / PathwayCommons / SMPDB).

Read-only. Single merged table: per source — node count in the current v2 KG, id
system, what it connects to (proteins / drugs), proteins-per-pathway, plus the
Reactome hierarchy census and the SMPDB content available from the DrugBank XML
but not yet in the KG.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_pathway_summary.py
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

NODES = "Code/data/KG/_merged_kg_dedup_v2/nodes__dedup.parquet"
EDGES = "Code/data/KG/_merged_kg_dedup_v2/edges__dedup.parquet"
SMPDB = "Code/data/_cache/smpdb_pathways.parquet"
REL = "Code/data/_cache/ReactomePathwaysRelation.txt"


def pstats(pairs_pw):  # dict pw->set(proteins) -> median/with-protein
    sizes = [len(v) for v in pairs_pw.values()]
    return len(pairs_pw), (int(np.median(sizes)) if sizes else 0)


def main() -> None:
    nd = pd.read_parquet(NODES, columns=["id", "kind", "name", "source_kg"])
    ed = pd.read_parquet(EDGES, columns=["src", "dst", "relation"])

    pw = nd[nd["kind"].astype(str).str.lower() == "pathway"]
    cnt = pw["source_kg"].value_counts().to_dict()

    # protein/drug membership per source (in current KG)
    def members(relation, pw_prefix, nbr_prefix):
        e = ed[ed["relation"] == relation]
        m = defaultdict(set)
        for s, d in zip(e["src"].astype(str), e["dst"].astype(str)):
            p = s if s.startswith(pw_prefix) else d
            o = d if s.startswith(pw_prefix) else s
            if o.startswith(nbr_prefix):
                m[p].add(o)
        return m

    rea_prot = members("prime:pathway_protein", "prime:pathway:", "prot:")
    pc_prot = members("het:GpPW", "het:Pathway", "prot:")
    smp_drug = members("db:pathway", "db:pathway:", "DB")

    print("=" * 78)
    print("CONSOLIDATED PATHWAY CENSUS — current v2 KG")
    print("=" * 78)
    print(f"{'source':<16}{'nodes':>7}{'id system':>16}{'connects to':>22}{'med #prot':>10}")
    rea_n, rea_med = pstats(rea_prot)
    pc_n, pc_med = pstats(pc_prot)
    print(f"{'Reactome(prime)':<16}{cnt.get('primekg',0):>7}{'R-HSA-nnn':>16}{'proteins+hierarchy':>22}{rea_med:>10}")
    print(f"{'PathwayCommons':<16}{cnt.get('hetionet',0):>7}{'PC7_nnn':>16}{'proteins(genes)':>22}{pc_med:>10}")
    print(f"{'SMPDB(drugbank)':<16}{cnt.get('drugbank',0):>7}{'SMP-nnnnnnn':>16}{'DRUGS only (!)':>22}{0:>10}")
    print(f"  -> Reactome pathways w/ protein edges: {rea_n} | PathwayCommons w/ protein: {pc_n}")
    print(f"  -> SMPDB in current KG has NO protein edges (incomplete extraction)")

    # name overlap PC <-> Reactome
    def norm(s): return " ".join(str(s).strip().lower().split()).strip(" .")
    rea_names = set(norm(x) for x in pw[pw["source_kg"] == "primekg"]["name"])
    pc_names = set(norm(x) for x in pw[pw["source_kg"] == "hetionet"]["name"])
    print(f"\n  PathwayCommons ∩ Reactome by name: {len(pc_names & rea_names)} "
          f"({len(pc_names & rea_names)/max(len(pc_names),1)*100:.0f}% of PC are really Reactome)")

    # SMPDB available from XML (cache) but not in KG
    smp = pd.read_parquet(SMPDB)
    links = sum(len(p) for p in smp["proteins"])
    print("\n" + "-" * 78)
    print("SMPDB content AVAILABLE from DrugBank XML (not yet in KG as protein edges)")
    print("-" * 78)
    print(f"  SMPDB pathways (XML)      : {len(smp)}  | with proteins: {int((smp['n_prot']>0).sum())}")
    print(f"  pathway->protein links    : {links} | distinct UniProt: "
          f"{len(set().union(*smp['proteins']))}")
    print(f"  median proteins/pathway   : {int(smp['n_prot'].median())}")
    print("  by category:")
    for c, sub in smp.groupby("category"):
        print(f"     {c:<16} {len(sub):>4} | mean prot {sub['n_prot'].mean():.0f}")

    # Reactome hierarchy census
    par2ch, ch2par, hnodes = defaultdict(set), defaultdict(set), set()
    with open(REL, encoding="utf-8") as f:
        for line in f:
            p, c = line.rstrip("\n").split("\t")
            if p.startswith("R-HSA") and c.startswith("R-HSA"):
                par2ch[p].add(c); ch2par[c].add(p); hnodes |= {p, c}
    roots = {n for n in hnodes if n not in ch2par}
    leaves = {n for n in hnodes if n not in par2ch}
    ourhsa = {x.split(":")[-1] for x in pw[pw["source_kg"] == "primekg"]["id"]}
    miss = hnodes - ourhsa
    print("\n" + "-" * 78)
    print("REACTOME HIERARCHY (the meso->macro layering)")
    print("-" * 78)
    print(f"  levels: {len(roots)} macro(top) / {len(hnodes)-len(roots)-len(leaves)} meso(internal) / {len(leaves)} leaf")
    print(f"  our KG covers {len(ourhsa & hnodes)}/{len(hnodes)} hierarchy nodes")
    print(f"  MISSING (supplement): {len(miss)} = "
          f"{len(miss&roots)} macro + {len(miss-roots-leaves)} meso + {len(miss&leaves)} leaf")
    print(f"  our pathways with direct protein edges sit at LEAVES; macro roots have ~0 protein edges")


if __name__ == "__main__":
    main()

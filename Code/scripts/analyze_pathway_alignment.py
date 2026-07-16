"""Pathway-node alignability audit across DrugBank / Hetionet / PrimeKG.

Read-only. Pathway nodes use THREE different id systems (no shared numeric id like
Entrez/GO), so direct id-alignment is impossible. This quantifies:
  1. counts per source + id format/meaning
  2. how many nodes carry a usable name
  3. NAME-based overlap across sources (the practical bridge: e.g. a Reactome
     "Platelet degranulation" pathway vs a Pathway-Commons one with the same name)
  4. whether the local raw source files carry any cross-reference to a shared db

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_pathway_alignment.py
"""
from __future__ import annotations

import re

import pandas as pd

NODES = "Code/data/KG/_merged_kg_dedup_v2/nodes__dedup.parquet"
DB_PATHWAYS = "Code/data/KG/drugbank/filtered/drug_pathways.csv"
HET_NODES = "Code/data/KG/hetionet/hetionet-v1.0-nodes.tsv"
PRIME_KG = "Code/data/KG/primekg/kg.csv"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def main() -> None:
    nd = pd.read_parquet(NODES)
    pw = nd[nd["kind"].astype(str).str.lower() == "pathway"].copy()
    print("=== [1] pathway nodes per source + id format ===")
    for src in pw["source_kg"].unique():
        s = pw[pw["source_kg"] == src]
        named = (s["name"].astype(str).str.strip().replace("nan", "") != "").sum()
        print(f"  {src:<10} {len(s):>5} nodes | named {named:>5} | id e.g. {s['id'].iloc[0]}")
    print(f"  TOTAL pathway nodes: {len(pw)}")

    # DrugBank SMPDB names come from the raw table (node-table name is blank)
    dbp = pd.read_csv(DB_PATHWAYS)
    print(f"\n=== [2] DrugBank drug_pathways.csv columns: {list(dbp.columns)} ===")
    print(dbp.head(3).to_string())

    # build name sets per source
    het_names = set(norm(x) for x in pw[pw["source_kg"] == "hetionet"]["name"] if str(x).strip())
    prime_names = set(norm(x) for x in pw[pw["source_kg"] == "primekg"]["name"] if str(x).strip())
    # smpdb names: pathway_name column if present
    namecol = next((c for c in dbp.columns if "name" in c.lower()), None)
    smp_names = set(norm(x) for x in dbp[namecol]) if namecol else set()

    print("\n=== [3] NAME-based overlap across sources (the practical bridge) ===")
    print(f"  het(PC) names    : {len(het_names)}")
    print(f"  prime(Reactome)  : {len(prime_names)}")
    print(f"  drugbank(SMPDB)  : {len(smp_names)}")
    print(f"  het ∩ prime      : {len(het_names & prime_names)}")
    print(f"  het ∩ smpdb      : {len(het_names & smp_names)}")
    print(f"  prime ∩ smpdb    : {len(prime_names & smp_names)}")
    print(f"  all three        : {len(het_names & prime_names & smp_names)}")
    print("  sample het∩prime name matches:")
    for n in list(het_names & prime_names)[:8]:
        print(f"     {n}")

    # raw cross-ref hunt
    print("\n=== [4] do raw source files carry a pathway cross-reference? ===")
    try:
        hn = pd.read_csv(HET_NODES, sep="\t")
        hpw = hn[hn["kind"].astype(str).str.contains("Pathway", na=False)] if "kind" in hn.columns else hn
        print(f"  hetionet nodes.tsv cols: {list(hn.columns)} | pathway rows sample id: "
              f"{hpw.iloc[0].to_dict() if len(hpw) else 'n/a'}")
    except Exception as e:  # noqa: BLE001
        print(f"  hetionet nodes.tsv: {e}")
    print(f"  -> PC7_* = Pathway Commons v7 (aggregates Reactome/WikiPathways/PID/KEGG);"
          " R-HSA-* = Reactome; SMP* = SMPDB. Cross-ref to a shared db is NOT in the id.")


if __name__ == "__main__":
    main()

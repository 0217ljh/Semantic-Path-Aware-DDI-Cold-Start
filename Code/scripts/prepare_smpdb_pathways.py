"""Extract SMPDB pathways (drug + protein members) from DrugBank XML and analyze.

The merged KG kept only drug->pathway edges, dropping each SMPDB pathway's protein
(enzyme) members. SMPDB pathways are curated drug-mechanism subgraphs: a pathway
groups a drug with its target/enzyme/transporter proteins. This streams the XML,
extracts the full tripartite drug -- pathway -- protein(UniProt), persists it, and
analyzes:
  * structure / levels (micro protein  <-  meso pathway  ->  drug)
  * hub-ness: proteins-per-pathway, pathways-per-protein (hub proteins), drugs-per-pathway
  * how SMPDB proteins map to the KG canonical prot:<entrez> nodes (via HGNC UniProt->Entrez)

Persists Code/data/_cache/smpdb_pathways.parquet (reusable for a later KG build).
Read-only on sources.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_smpdb_pathways.py
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

XML = "/mnt/f/Datasets/Formal_DDI/Durg_bank/Raw_data/full database.xml"
HGNC = "Code/data/_cache/hgnc_complete_set.txt"
NODES_V2 = "Code/data/KG/_merged_kg_dedup_v2/nodes__dedup.parquet"
OUT = "Code/data/_cache/smpdb_pathways.parquet"


def ln(t):
    return t.split("}")[-1]


def kids(e, name):
    return [c for c in e if ln(c.tag) == name]


def main() -> None:
    rows = {}
    for _, el in ET.iterparse(XML, events=("end",)):
        if ln(el.tag) != "pathway":
            continue
        smp = kids(el, "smpdb-id")
        sid = smp[0].text if smp else None
        if not sid:
            el.clear(); continue
        cat = kids(el, "category"); nm = kids(el, "name")
        drugs = kids(el, "drugs"); enz = kids(el, "enzymes")
        dset = set(d.text for d in kids(drugs[0], "drug") for d in kids(d, "drugbank-id")) if drugs else set()
        # drugbank-id sits one level under <drug>
        dids = set()
        if drugs:
            for d in kids(drugs[0], "drug"):
                for x in kids(d, "drugbank-id"):
                    dids.add(x.text)
        uni = set(u.text for u in enz[0]) if enz else set()
        if sid not in rows:
            rows[sid] = {"smpdb_id": sid, "category": cat[0].text if cat else "NA",
                         "name": nm[0].text if nm else "", "drugs": set(), "proteins": set()}
        rows[sid]["drugs"] |= dids
        rows[sid]["proteins"] |= uni
        el.clear()

    df = pd.DataFrame([{**r, "drugs": sorted(r["drugs"]), "proteins": sorted(r["proteins"]),
                        "n_drug": len(r["drugs"]), "n_prot": len(r["proteins"])}
                       for r in rows.values()])
    df.to_parquet(OUT, index=False)
    print(f"extracted {len(df)} SMPDB pathways -> {OUT}")

    # ---- UniProt -> Entrez (HGNC) and KG prot node resolution ----
    uni2ent = {}
    with open(HGNC, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ent = (row.get("entrez_id") or "").strip()
            if not ent:
                continue
            for acc in (row.get("uniprot_ids") or "").split("|"):
                if acc.strip():
                    uni2ent[acc.strip().upper()] = ent
    kg_prot = set(pd.read_parquet(NODES_V2, columns=["id"])["id"])

    # ---- [1] STRUCTURE: a concrete coagulation example ----
    print("\n=== [1] structure: drug -- pathway -- protein (concrete) ===")
    ex = df[df["name"].str.contains("Bivalirudin Action", na=False)].iloc[0]
    print(f"  pathway: {ex['smpdb_id']} '{ex['name']}' [{ex['category']}]")
    print(f"    drugs ({ex['n_drug']}): {ex['drugs']}")
    print(f"    proteins ({ex['n_prot']}): {ex['proteins'][:8]} ...")

    # ---- [2] HUB: proteins-per-pathway (pathway size) ----
    npz = df["n_prot"].to_numpy()
    print("\n=== [2] hub — pathway SIZE (proteins per pathway) ===")
    print(f"  min {npz.min()} | median {int(np.median(npz))} | mean {npz.mean():.1f} | "
          f"90th {int(np.quantile(npz,0.9))} | max {npz.max()}")
    print("  biggest pathways:")
    for _, r in df.sort_values("n_prot", ascending=False).head(5).iterrows():
        print(f"    {r['n_prot']:>3} proteins  [{r['category']}]  {r['name']}")

    # ---- [3] HUB: pathways-per-protein (protein hub-ness) ----
    prot_deg = Counter()
    for ps in df["proteins"]:
        for p in ps:
            prot_deg[p] += 1
    degs = np.array(list(prot_deg.values()))
    print("\n=== [3] hub — protein DEGREE (how many SMPDB pathways each protein is in) ===")
    print(f"  distinct proteins {len(prot_deg)} | median {int(np.median(degs))} | "
          f"mean {degs.mean():.1f} | 95th {int(np.quantile(degs,0.95))} | max {degs.max()}")
    sym = {}
    with open(HGNC, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for acc in (row.get("uniprot_ids") or "").split("|"):
                if acc.strip():
                    sym[acc.strip().upper()] = row.get("symbol", "")
    print("  top hub proteins (in most pathways):")
    for p, d in prot_deg.most_common(10):
        print(f"    {d:>3} pathways  {p} ({sym.get(p.upper(),'?')})")
    print(f"  proteins in >=20 pathways (hubs): {int((degs>=20).sum())} | "
          f">=50: {int((degs>=50).sum())}")

    # ---- resolution to KG ----
    allp = set().union(*df["proteins"]) if len(df) else set()
    res_ent = {p for p in allp if p.upper() in uni2ent}
    in_kg = {p for p in res_ent if f"prot:{uni2ent[p.upper()]}" in kg_prot}
    print(f"\n=== resolution: SMPDB UniProt -> Entrez -> KG prot node ===")
    print(f"  distinct SMPDB proteins {len(allp)} | UniProt->Entrez {len(res_ent)} | "
          f"present as prot:<entrez> in KG {len(in_kg)}")
    print("\n=== category x drug-sharing (already know drug-private; now WITH proteins) ===")
    for c, sub in df.groupby("category"):
        shared = int((sub["n_drug"] >= 2).sum())
        print(f"  {c:<16} {len(sub):>4} paths | shared>=2drug {shared:>4} | "
              f"mean_prot {sub['n_prot'].mean():.0f}")


if __name__ == "__main__":
    main()

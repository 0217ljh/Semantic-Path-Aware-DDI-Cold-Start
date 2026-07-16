"""Build a DrugBank BE-code -> gene-symbol / UniProt / HGNC crosswalk.

Streams the DrugBank full database XML (~1.6 GB) ONCE with iterparse (never
loads it whole) and extracts, for every target/enzyme/transporter/carrier
polypeptide, the mapping:

    BE id (db partner id)  ->  gene-name (official HGNC symbol)
                               polypeptide@id (UniProt accession)
                               HGNC id, full protein name, kind

This is the crosswalk that lets the Entrez-less `db:*` protein nodes in the
merged KG be aligned to the canonical Entrez space (gene symbol bridges to the
het/prime nodes, which already carry Entrez). One-time; result cached to a
small parquet+csv. Read-only on the XML.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_drugbank_protein_crosswalk.py \
        --xml "/mnt/f/Datasets/Formal_DDI/Durg_bank/Raw_data/full database.xml"
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import pandas as pd

DEFAULT_XML = "/mnt/f/Datasets/Formal_DDI/Durg_bank/Raw_data/full database.xml"
OUT_PARQUET = "Code/data/_cache/drugbank_be_crosswalk.parquet"
OUT_CSV = "Code/data/_cache/drugbank_be_crosswalk.csv"
NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
KINDS = ("target", "enzyme", "transporter", "carrier")


def ln(tag: str) -> str:
    return tag.split("}")[-1]


def kids(el, name: str) -> list:
    return [c for c in el if ln(c.tag) == name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=DEFAULT_XML)
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    n_poly = 0
    for _, el in ET.iterparse(args.xml, events=("end",)):
        t = ln(el.tag)
        if t in KINDS:
            be = kids(el, "id")
            nm = kids(el, "name")
            poly = kids(el, "polypeptide")
            if be and poly:
                be_id = be[0].text
                p = poly[0]
                gn = kids(p, "gene-name")
                ext = kids(p, "external-identifiers")
                hgnc = ""
                if ext:
                    for ei in kids(ext[0], "external-identifier"):
                        r = kids(ei, "resource")
                        idv = kids(ei, "identifier")
                        if r and "HGNC" in (r[0].text or "") and idv:
                            hgnc = idv[0].text or ""
                            break
                n_poly += 1
                # keep first occurrence (same BE recurs across many drugs)
                if be_id not in rows:
                    rows[be_id] = {
                        "be_id": be_id,
                        "kind": t,
                        "gene_symbol": (gn[0].text if gn else "") or "",
                        "uniprot": p.get("id") or "",
                        "uniprot_source": p.get("source") or "",
                        "hgnc": hgnc,
                        "protein_name": (nm[0].text if nm else "") or "",
                    }
            el.clear()

    df = pd.DataFrame(list(rows.values()))
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    print(f"polypeptide occurrences scanned : {n_poly}")
    print(f"distinct BE proteins in crosswalk: {len(df)}")
    print(f"  with gene_symbol               : {(df['gene_symbol'] != '').sum()}")
    print(f"  with UniProt                    : {(df['uniprot'] != '').sum()}")
    print(f"  with HGNC                       : {(df['hgnc'] != '').sum()}")

    # coverage of the merged KG's db:* protein nodes
    nd = pd.read_parquet(NODES)
    dbp = nd[(nd["source_kg"] == "drugbank")
             & nd["kind"].astype(str).str.contains("protein|Protein", na=False)].copy()
    dbp["be"] = dbp["id"].astype(str).str.split(":").str[-1]
    be_have = set(df["be_id"])
    cov = dbp["be"].isin(be_have)
    print(f"\nmerged-KG db:* protein nodes      : {len(dbp)}")
    print(f"  covered by crosswalk            : {cov.sum()} ({cov.mean()*100:.0f}%)")
    # of covered, how many resolve to a symbol that exists in het/prime (-> Entrez)
    prot = nd[nd["kind"].astype(str).str.contains("gene|Gene|protein|Protein", na=False)]
    sym_in_kg = set(prot[prot["source_kg"].isin(["hetionet", "primekg"])]["name"]
                    .dropna().astype(str).str.upper())
    sym_by_be = dict(zip(df["be_id"], df["gene_symbol"].astype(str).str.upper()))
    resolvable = sum(1 for b in dbp.loc[cov, "be"]
                     if sym_by_be.get(b, "") in sym_in_kg)
    print(f"  of covered, symbol also in het/prime (-> Entrez): {resolvable} "
          f"({resolvable/max(cov.sum(),1)*100:.0f}%)")
    print(f"\nwrote {OUT_PARQUET}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

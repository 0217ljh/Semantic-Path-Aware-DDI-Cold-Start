"""Build the authoritative protein merge map (orig node id -> canonical Entrez).

Reconciliation step "B": resolve every protein node in the merged KG to a
canonical Entrez gene id using, in priority order:

  het/prime nodes  : Entrez is already in the id suffix (route=id)
  DrugBank db nodes: BE -> (crosswalk) gene_symbol / UniProt / HGNC, then
    1. gene_symbol present as a het/prime node            (route=kg-symbol)  [the original 82%]
    2. HGNC id        -> Entrez via HGNC complete set      (route=hgnc-id)
    3. UniProt        -> Entrez via HGNC complete set      (route=uniprot)
    4. gene_symbol    -> Entrez via HGNC approved/alias/prev(route=hgnc-symbol)
    else: no Entrez -> provisional `prot:uniprot:<acc>` fallback (route=uniprot-fallback)

Ambiguous mappings (a symbol/alias resolving to >1 Entrez) are QUARANTINED:
kept as their own node, never merged (a bad merge is worse than a missed merge).

Canonical id = `prot:<entrez>` when resolved, else `prot:uniprot:<acc>`, else the
original id. Emits a merge-map table + a coverage report. Read-only on inputs.

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_protein_merge_map.py
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

import pandas as pd

NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
XWALK = "Code/data/_cache/drugbank_be_crosswalk.parquet"
HGNC = "Code/data/_cache/hgnc_complete_set.txt"
OUT_PARQUET = "Code/data/_cache/protein_merge_map.parquet"
OUT_CSV = "Code/data/_cache/protein_merge_map.csv"


def het_entrez(nid: str):
    return nid.split("::")[-1] if nid.startswith("het:Gene") else None


def prime_entrez(nid: str):
    return nid.split(":")[-1] if nid.startswith("prime:gene/protein:") else None


def load_hgnc():
    """Return (hgncid2ent, uniprot2ent, sym2ent) with ambiguity collapsed to None."""
    hgncid2ent: dict[str, str] = {}
    uni2ent_multi: dict[str, set] = defaultdict(set)
    sym2ent_multi: dict[str, set] = defaultdict(set)
    with open(HGNC, encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for row in rd:
            ent = (row.get("entrez_id") or "").strip()
            if not ent:
                continue
            hid = (row.get("hgnc_id") or "").strip()
            if hid:
                hgncid2ent[hid] = ent
            for acc in (row.get("uniprot_ids") or "").split("|"):
                acc = acc.strip()
                if acc:
                    uni2ent_multi[acc.upper()].add(ent)
            approved = (row.get("symbol") or "").strip().upper()
            if approved:
                sym2ent_multi[approved].add(ent)
            for col in ("alias_symbol", "prev_symbol"):
                for s in (row.get(col) or "").split("|"):
                    s = s.strip().upper()
                    if s:
                        sym2ent_multi[s].add(ent)
    uni2ent = {k: next(iter(v)) for k, v in uni2ent_multi.items() if len(v) == 1}
    sym2ent = {k: next(iter(v)) for k, v in sym2ent_multi.items() if len(v) == 1}
    return hgncid2ent, uni2ent, sym2ent


def main() -> None:
    nd = pd.read_parquet(NODES)
    xw = pd.read_parquet(XWALK)
    prot = nd[nd["kind"].astype(str).str.contains("gene|Gene|protein|Protein", na=False)].copy()

    # het/prime: id -> Entrez ; build the in-KG symbol -> Entrez (the 82% route)
    prot["entrez_id"] = prot["id"].map(lambda x: het_entrez(x) or prime_entrez(x))
    hp = prot[prot["source_kg"].isin(["hetionet", "primekg"])].dropna(subset=["entrez_id"])
    kg_sym_multi: dict[str, set] = defaultdict(set)
    for s, e in zip(hp["name"].astype(str).str.upper(), hp["entrez_id"]):
        if s and s != "NAN":
            kg_sym_multi[s].add(e)
    kg_sym2ent = {k: next(iter(v)) for k, v in kg_sym_multi.items() if len(v) == 1}
    kg_entrez = set(hp["entrez_id"])

    hgncid2ent, uni2ent, hgnc_sym2ent = load_hgnc()

    be = xw.set_index("be_id")  # BE -> symbol/uniprot/hgnc

    rows = []
    route_ct = Counter()
    for _, r in prot.iterrows():
        nid = str(r["id"])
        src = str(r["source_kg"])
        entrez, route, sym = None, None, str(r["name"] or "")
        if src in ("hetionet", "primekg"):
            entrez = r["entrez_id"]; route = "id"
        else:  # drugbank BE
            be_code = nid.split(":")[-1]
            info = be.loc[be_code] if be_code in be.index else None
            gsym = str(info["gene_symbol"]).upper() if info is not None else ""
            uacc = str(info["uniprot"]).upper() if info is not None else ""
            hid = str(info["hgnc"]) if info is not None else ""
            sym = gsym or sym
            if gsym and gsym in kg_sym2ent:
                entrez, route = kg_sym2ent[gsym], "kg-symbol"
            elif hid and hid in hgncid2ent:
                entrez, route = hgncid2ent[hid], "hgnc-id"
            elif uacc and uacc in uni2ent:
                entrez, route = uni2ent[uacc], "uniprot"
            elif gsym and gsym in hgnc_sym2ent:
                entrez, route = hgnc_sym2ent[gsym], "hgnc-symbol"
            else:
                route = "uniprot-fallback"
        route_ct[route] += 1
        if entrez is not None:
            canonical = f"prot:{entrez}"
        elif src == "drugbank":
            be_code = nid.split(":")[-1]
            uacc = str(be.loc[be_code]["uniprot"]) if be_code in be.index else ""
            canonical = f"prot:uniprot:{uacc}" if uacc and uacc != "nan" else nid
        else:
            canonical = nid
        rows.append({"orig_id": nid, "source_kg": src, "canonical_id": canonical,
                     "entrez": entrez, "symbol": sym, "route": route,
                     "bridges_to_hetprime": bool(entrez is not None and entrez in kg_entrez)})

    mm = pd.DataFrame(rows)
    mm.to_parquet(OUT_PARQUET, index=False)
    mm.to_csv(OUT_CSV, index=False)

    # ---- report ----
    db = mm[mm["source_kg"] == "drugbank"]
    db_res = db["entrez"].notna()
    print(f"=== protein merge map ({len(mm)} protein nodes) ===")
    print(f"  het/prime (id route)        : {(mm['source_kg']!='drugbank').sum()}")
    print(f"  DrugBank db nodes           : {len(db)}")
    print("  --- db resolution routes ---")
    for rte, n in route_ct.most_common():
        if rte != "id":
            print(f"      {rte:<18}: {n}")
    print(f"  db resolved to Entrez       : {db_res.sum()} ({db_res.mean()*100:.0f}%)  "
          f"[was 82% with kg-symbol only]")
    print(f"  db NOT resolved (fallback)  : {(~db_res).sum()}")
    db_bridge = db["bridges_to_hetprime"].sum()
    print(f"  db that bridge to a het/prime node (true merge): {db_bridge}")
    print(f"  db with Entrez but NO het/prime partner (standalone): {db_res.sum()-db_bridge}")

    # final de-duplicated protein count
    distinct = mm["canonical_id"].nunique()
    print(f"\n  raw protein nodes           : {len(mm)}")
    print(f"  distinct canonical proteins : {distinct}  (inflation {len(mm)/distinct:.2f}x)")
    print(f"\nwrote {OUT_PARQUET}\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

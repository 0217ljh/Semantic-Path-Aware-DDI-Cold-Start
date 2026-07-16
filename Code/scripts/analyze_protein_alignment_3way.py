"""Three-way protein node de-duplication + PD shared-target, all via Entrez.

Read-only. Uses the DrugBank BE crosswalk (Code/data/_cache/drugbank_be_crosswalk.*)
to give the Entrez-less `db:*` protein nodes a gene symbol -> Entrez, then unifies
all three namespaces (DrugBank / Hetionet / PrimeKG) on Entrez and reports:

  Part A — node-level fragmentation: how many distinct proteins (by Entrez)
    appear in 1 / 2 / 3 namespaces; raw protein-node count vs de-duplicated
    protein count (inflation factor from un-merged duplicates).

  Part B — per-drug target inflation (200 PD drugs): raw protein-neighbor nodes
    vs distinct Entrez targets per drug (how much fragmentation inflates a
    drug's apparent target set).

  Part C — PD shared-target rate under unified Entrez: for the 200 PD pairs,
    do the two drugs share at least one target Entrez once all namespaces are
    unified (vs the earlier db-bucket-only view that missed het/prime)?

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/analyze_protein_alignment_3way.py
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
import pandas as pd

from my_code.models.spmn_v1.retrieval import KIND_ORDER, MergedKG

NODES = "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
XWALK = "Code/data/_cache/drugbank_be_crosswalk.parquet"
CHAINS = "Code/runs/pd_mechanism/chains.jsonl"
PROT = KIND_ORDER.index("protein_gene")


def het_entrez(nid: str):
    return nid.split("::")[-1] if nid.startswith("het:Gene") else None


def prime_entrez(nid: str):
    return nid.split(":")[-1] if nid.startswith("prime:gene/protein:") else None


def main() -> None:
    kg = MergedKG.from_parquet()
    nd = pd.read_parquet(NODES)
    xw = pd.read_parquet(XWALK)

    # symbol -> Entrez from het/prime node names (these carry both)
    prot = nd[nd["kind"].astype(str).str.contains("gene|Gene|protein|Protein", na=False)].copy()
    hp = prot[prot["source_kg"].isin(["hetionet", "primekg"])].copy()
    hp["entrez"] = hp["id"].map(lambda x: het_entrez(x) or prime_entrez(x))
    sym2ent: dict[str, str] = {}
    for _, r in hp.dropna(subset=["entrez"]).iterrows():
        s = str(r["name"]).strip().upper()
        if s and s != "NAN":
            sym2ent.setdefault(s, r["entrez"])

    # db BE node id -> Entrez (via crosswalk symbol)
    be2sym = dict(zip(xw["be_id"], xw["gene_symbol"].astype(str).str.upper()))
    db_node2ent: dict[str, str] = {}
    for nid in prot[prot["source_kg"] == "drugbank"]["id"]:
        be = str(nid).split(":")[-1]
        sym = be2sym.get(be, "")
        e = sym2ent.get(sym)
        if e:
            db_node2ent[nid] = e

    def node_entrez(nid: str):
        return het_entrez(nid) or prime_entrez(nid) or db_node2ent.get(nid)

    # ---- Part A: node-level fragmentation across namespaces ----
    ns_of_entrez: dict[str, set] = {}
    raw_nodes = 0
    for _, r in prot.iterrows():
        nid = str(r["id"])
        e = node_entrez(nid)
        if e is None:
            continue
        raw_nodes += 1
        ns_of_entrez.setdefault(e, set()).add(str(r["source_kg"]))
    span = Counter(len(v) for v in ns_of_entrez.values())
    distinct = len(ns_of_entrez)
    print("=== [A] protein-node fragmentation (canonical = Entrez, all 3 sources) ===")
    print(f"  raw Entrez-resolvable protein nodes : {raw_nodes}")
    print(f"  distinct proteins (by Entrez)       : {distinct}")
    print(f"  inflation factor (nodes / distinct) : {raw_nodes / max(distinct,1):.2f}x")
    print(f"  proteins in exactly 1 namespace     : {span.get(1,0)}")
    print(f"  proteins in exactly 2 namespaces    : {span.get(2,0)}  <- duplicated")
    print(f"  proteins in all 3 namespaces        : {span.get(3,0)}  <- triplicated")
    dup = span.get(2, 0) + span.get(3, 0)
    print(f"  duplicated across >=2 namespaces    : {dup} ({dup/max(distinct,1)*100:.0f}% of distinct proteins)")

    # ---- load PD pairs ----
    recs = []
    for line in pathlib.Path(CHAINS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("parsed"):
                recs.append(r)

    def drug_target_entrez(d_idx):
        nb = kg.indices[kg.indptr[d_idx]:kg.indptr[d_idx + 1]]
        raw, ent = 0, set()
        for n in nb:
            if kg.type_id[n] == PROT:
                raw += 1
                e = node_entrez(kg.node_ids[n])
                if e:
                    ent.add(e)
        return raw, ent

    # ---- Part B: per-drug target inflation ----
    drug_ids = set()
    for r in recs:
        a, b = r["key"].split("|")
        drug_ids.update([a, b])
    raw_tot, ent_tot, ndrug = 0, 0, 0
    for d in drug_ids:
        di = kg.id_to_idx.get(d)
        if di is None:
            continue
        raw, ent = drug_target_entrez(di)
        if raw:
            raw_tot += raw; ent_tot += len(ent); ndrug += 1
    print(f"\n=== [B] per-drug target inflation ({ndrug} PD drugs) ===")
    print(f"  mean raw protein-neighbor nodes / drug : {raw_tot/max(ndrug,1):.1f}")
    print(f"  mean distinct Entrez targets / drug    : {ent_tot/max(ndrug,1):.1f}")
    print(f"  per-drug fragmentation inflation       : {raw_tot/max(ent_tot,1):.2f}x")

    # ---- Part C: PD shared-target under unified Entrez ----
    share_uni, share_dbonly, n = 0, 0, 0
    for r in recs:
        a, b = r["key"].split("|")
        ai, bi = kg.id_to_idx.get(a), kg.id_to_idx.get(b)
        if ai is None or bi is None:
            continue
        n += 1
        _, ea = drug_target_entrez(ai)
        _, eb = drug_target_entrez(bi)
        if ea & eb:
            share_uni += 1
        # db-only view: shared raw db node id
        da = {kg.node_ids[x] for x in kg.indices[kg.indptr[ai]:kg.indptr[ai+1]]
              if kg.type_id[x] == PROT and kg.node_ids[x].startswith("db:")}
        db = {kg.node_ids[x] for x in kg.indices[kg.indptr[bi]:kg.indptr[bi+1]]
              if kg.type_id[x] == PROT and kg.node_ids[x].startswith("db:")}
        if da & db:
            share_dbonly += 1
    print(f"\n=== [C] PD shared-target rate ({n} pairs) ===")
    print(f"  share >=1 target (db raw node only)     : {share_dbonly} ({share_dbonly/max(n,1)*100:.0f}%)")
    print(f"  share >=1 target (unified Entrez, 3 src) : {share_uni} ({share_uni/max(n,1)*100:.0f}%)")


if __name__ == "__main__":
    main()

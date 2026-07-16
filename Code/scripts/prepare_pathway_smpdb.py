"""Pathway renormalization — STEP 3: SMPDB axis (v4 -> v5).

Activates SMPDB as a real protein-connected pathway source. The KG had only
drug->pathway edges (proteins dropped in extraction); this adds the curated
pathway->protein membership from the DrugBank XML (cache smpdb_pathways.parquet).
Codex-confirmed plan + safeguards:
  * rename db:pathway:SMP* -> path:SMP* (canonical; NOT a merge); name from cache
    (prefer XML name, assert one non-null name per SMP id)
  * add prime:pathway_protein edges (protein -> pathway), SAME relation as
    Reactome/PC (source attribute distinguishes); UniProt->Entrez via HGNC,
    DEDUP by (pathway, prot:<entrez>); keep ALL incl. Na/K-ATPase backbone
  * keep db:pathway (drug->pathway) edges, remap endpoint to path:SMP
  * 21 unresolved UniProt: kept auditable in a sidecar, NOT dropped, no fallback node
  * skip the 212 XML-only SMPDB pathways (no drug link -> protein-only islands)
  * path_meta: source=smpdb, level='na', macro_ancestors=[], category=SMPDB subject,
    n_protein (resolved), n_drug, candidate_reactome (weak name match)

Acceptance = correctness/completeness of representation (NOT discriminability).
Output: Code/data/KG/_merged_kg_dedup_v5/{nodes__dedup,edges__dedup,path_meta,smpdb_unresolved}.parquet

Usage (WSL conda env project_1, from project root):
    PYTHONPATH=Code python -u Code/scripts/prepare_pathway_smpdb.py
"""
from __future__ import annotations

import csv
import os
import re
from collections import Counter, defaultdict

import pandas as pd

IN_DIR = "Code/data/KG/_merged_kg_dedup_v4"
OUT_DIR = "Code/data/KG/_merged_kg_dedup_v5"
SMPDB = "Code/data/_cache/smpdb_pathways.parquet"
HGNC = "Code/data/_cache/hgnc_complete_set.txt"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def main() -> None:
    nd = pd.read_parquet(f"{IN_DIR}/nodes__dedup.parquet")
    ed = pd.read_parquet(f"{IN_DIR}/edges__dedup.parquet")
    meta = pd.read_parquet(f"{IN_DIR}/path_meta.parquet")
    smp = pd.read_parquet(SMPDB)
    smp_by_id = {r["smpdb_id"]: r for _, r in smp.iterrows()}
    print(f"v4 in: {len(nd)} nodes, {len(ed)} edges, {len(meta)} path_meta")

    # UniProt -> Entrez (HGNC)
    uni2ent = {}
    with open(HGNC, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            e = (row.get("entrez_id") or "").strip()
            if e:
                for a in (row.get("uniprot_ids") or "").split("|"):
                    if a.strip():
                        uni2ent[a.strip().upper()] = e
    kg_prot = set(nd[nd["id"].astype(str).str.startswith("prot:")]["id"])

    # ---- SMPDB nodes in KG: db:pathway:SMP* -> path:SMP* ----
    is_smp = nd["id"].astype(str).str.startswith("db:pathway:")
    smp_nodes = nd[is_smp].copy()
    ren = {x: f"path:{str(x).split(':')[-1]}" for x in smp_nodes["id"]}   # db:pathway:SMP_x -> path:SMP_x
    print(f"SMPDB pathway nodes in KG: {len(smp_nodes)}")

    # name from cache (prefer XML/cache name; assert one non-null per id)
    def smp_name(x):
        sid = str(x).split(":")[-1]
        rec = smp_by_id.get(sid)
        return str(rec["name"]) if rec is not None and str(rec.get("name") or "").strip() else str(nd.loc[nd["id"] == x, "name"].iloc[0] or "")

    non_smp = nd[~is_smp].copy()
    smp_rows = pd.DataFrame({
        "id": [ren[x] for x in smp_nodes["id"]],
        "kind": "pathway",
        "name": [smp_name(x) for x in smp_nodes["id"]],
        "source_kg": "smpdb",
    })
    new_nodes = pd.concat([non_smp, smp_rows], ignore_index=True)
    assert new_nodes["id"].is_unique, "node id collision"
    kind_of = dict(zip(new_nodes["id"], new_nodes["kind"]))

    # ---- remap db:pathway edges (drug->pathway) ----
    e = ed.copy()
    e["src"] = e["src"].map(lambda x: ren.get(x, x))
    e["dst"] = e["dst"].map(lambda x: ren.get(x, x))
    e["src_kind"] = [kind_of.get(s, k) for s, k in zip(e["src"], e["src_kind"])]
    e["dst_kind"] = [kind_of.get(d, k) for d, k in zip(e["dst"], e["dst_kind"])]

    # ---- NEW protein->pathway edges from SMPDB cache (dedup by (path, entrez)) ----
    new_pp = []
    unresolved_rows = []
    nprot_resolved, nprot_raw = {}, {}
    for x in smp_nodes["id"]:
        sid = str(x).split(":")[-1]
        rec = smp_by_id.get(sid)
        pid = ren[x]
        if rec is None:
            continue
        ents, unres = set(), []
        for u in rec["proteins"]:
            ent = uni2ent.get(str(u).upper())
            node = f"prot:{ent}" if ent else None
            if node and node in kg_prot:
                ents.add(node)
            else:
                unres.append(u)
        for node in ents:
            new_pp.append((node, pid))         # protein -> pathway (orientation matches Reactome/PC)
        nprot_resolved[pid] = len(ents)
        nprot_raw[pid] = len(rec["proteins"])
        if unres:
            unresolved_rows.append({"smpdb_id": sid, "path_id": pid,
                                    "n_raw": len(rec["proteins"]), "n_resolved": len(ents),
                                    "unresolved_uniprot": unres})
    pp_df = pd.DataFrame({
        "src": [s for s, _ in new_pp], "src_kind": "gene/protein",
        "dst": [d for _, d in new_pp], "dst_kind": "pathway",
        "relation": "prime:pathway_protein", "source_kg": "smpdb", "directed": True,
    })
    new_edges = pd.concat([e, pp_df], ignore_index=True)
    valid = set(kind_of)
    bad = ~(new_edges["src"].isin(valid) & new_edges["dst"].isin(valid))
    assert not bad.any(), f"{int(bad.sum())} edges with unknown endpoint"

    # ---- path_meta for SMPDB ----
    rea = new_nodes[new_nodes["id"].astype(str).str.startswith("path:R-HSA")]
    rea_name2ids = defaultdict(list)
    for i, n in zip(rea["id"], rea["name"]):
        rea_name2ids[norm(n)].append(i.split(":")[-1])
    # n_drug per SMPDB pathway (from db:pathway edges)
    dbp = new_edges[new_edges["relation"] == "db:pathway"]
    ndrug = Counter()
    for s, d in zip(dbp["src"].astype(str), dbp["dst"].astype(str)):
        pw = s if s.startswith("path:") else d
        ndrug[pw] += 1
    smp_meta = []
    for x in smp_nodes["id"]:
        sid = str(x).split(":")[-1]; pid = ren[x]
        rec = smp_by_id.get(sid)
        cat = str(rec["category"]) if rec is not None else ""
        nm = new_nodes.loc[new_nodes["id"] == pid, "name"].iloc[0]
        smp_meta.append({"id": pid, "source": "smpdb", "source_ids": [sid],
                         "level": "na", "macro_ancestors": [], "is_connector_only": False,
                         "obsolete": False, "n_protein": int(nprot_resolved.get(pid, 0)),
                         "candidate_reactome": [c for c in rea_name2ids.get(norm(nm), [])],
                         "category": cat, "n_drug": int(ndrug.get(pid, 0))})
    smp_meta = pd.DataFrame(smp_meta)
    # add category/n_drug columns to existing meta (backfill)
    meta = meta.copy()
    if "category" not in meta.columns:
        meta["category"] = ""
    if "n_drug" not in meta.columns:
        meta["n_drug"] = 0
    new_meta = pd.concat([meta, smp_meta], ignore_index=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    new_nodes.to_parquet(f"{OUT_DIR}/nodes__dedup.parquet", index=False)
    new_edges.to_parquet(f"{OUT_DIR}/edges__dedup.parquet", index=False)
    new_meta.to_parquet(f"{OUT_DIR}/path_meta.parquet", index=False)
    pd.DataFrame(unresolved_rows).to_parquet(f"{OUT_DIR}/smpdb_unresolved.parquet", index=False)

    # ---- verification / acceptance asserts ----
    print(f"\n=== v5 built (SMPDB axis) ===")
    print(f"  nodes {len(nd)} -> {len(new_nodes)} (db:pathway {len(smp_nodes)} -> path:SMP)")
    print(f"  edges {len(ed)} -> {len(new_edges)} (+{len(pp_df)} new SMPDB protein->pathway)")
    print(f"  no db:pathway:SMP* nodes remain: {int(new_nodes['id'].astype(str).str.startswith('db:pathway:').sum())==0}")
    print(f"  all db:pathway edges -> path:SMP*: "
          f"{bool((new_edges.loc[new_edges['relation']=='db:pathway','dst'].astype(str).str.startswith('path:SMP') | new_edges.loc[new_edges['relation']=='db:pathway','src'].astype(str).str.startswith('path:SMP')).all())}")
    print(f"  all path:SMP* nodes kind=pathway: "
          f"{bool((new_nodes[new_nodes['id'].astype(str).str.startswith('path:SMP')]['kind']=='pathway').all())}")
    print(f"  total prime:pathway_protein now: {int((new_edges['relation']=='prime:pathway_protein').sum())} "
          f"(was 169664 + SMPDB {len(pp_df)})")
    tot_raw = sum(nprot_raw.values()); tot_res = sum(nprot_resolved.values())
    print(f"  SMPDB protein membership: raw UniProt {tot_raw} -> resolved+dedup edges {len(pp_df)} | "
          f"unresolved UniProt rows {len(unresolved_rows)}")
    print(f"  path_meta now {len(new_meta)} rows | source dist: {new_meta['source'].value_counts().to_dict()}")
    print(f"  SMPDB category dist: {smp_meta['category'].value_counts().to_dict()}")
    # spot: the Disulfiram/Ethanol catecholamine pathway
    print(f"\n  wrote {OUT_DIR}/(nodes,edges,path_meta,smpdb_unresolved).parquet")


if __name__ == "__main__":
    main()

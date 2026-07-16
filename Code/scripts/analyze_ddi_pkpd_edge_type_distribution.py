"""Edge-type distribution of PK vs PD DDI pairs over the full merged KG.

For every positive DDI pair we look at the 2-hop paths that connect the two
drugs through a shared mediator:  a --(r1)--> m <--(r2)-- b. We sweep ALL 59
merged-KG relation types and ask: which edge types carry the connection for
pharmacokinetic (PK) vs pharmacodynamic (PD) pairs?

This is exactly the relation signal a C-MPNN / NBFNet propagates over, and
literally the rel(a->m) / rel(m->b) embeddings that PMP v1.5/v1.6 feed into each
mediator message. So the PK vs PD edge-type histogram says what relation-labeled
path evidence the model actually sees for each regime.

Method (only ~1900 drugs, so dense 1900x1900 per relation is cheap):
  N        : drug x node binary incidence over ANY relation (a mediator = a
             common 1-hop neighbour of both drugs).
  B_r      : drug x node binary incidence over relation r only.
  M_r=B_rN^T: M_r[a,b] = #mediators m with a--(r)--m and m a neighbour of b.
  For each PK/PD pair set, accumulate over BOTH endpoints:
     count_r = sum_pairs ( M_r[a,b] + M_r[b,a] )
  i.e. every connecting 2-hop path contributes its a-side and b-side relation.

PK/PD bucket is the same keyword rule used in
analyze_ddi_mechanism_kg_separability.py.

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_ddi_pkpd_edge_type_distribution.py

Outputs (under --out, default Code/runs/analyze_ddi_pkpd_edge_type_distribution/):
  edge_type_distribution.parquet   per-relation PK/PD counts, shares, enrichment
  summary.json                     totals + connectivity stats
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"


def log(m: str) -> None:
    print(m, flush=True)


def pkpd_bucket(t: str) -> str:
    t = t.lower()
    if any(k in t for k in ("metabolism", "excretion", "serum concentration",
                            "absorption", "protein binding", "clearance")):
        return "PK"
    if any(k in t for k in ("risk or severity", "activities", "efficacy",
                            "cns depression", "qtc", "hypertension",
                            "hypotensive", "sedative", "adverse effects")):
        return "PD"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_pkpd_edge_type_distribution"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[load] {NODES_PARQUET}")
    nodes = pd.read_parquet(NODES_PARQUET)
    log(f"[load] {EDGES_PARQUET}")
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    kinds = nodes["kind"].to_numpy()

    drug_db = nodes["id"].to_numpy()[kinds == "Drug"]
    drug_idx = np.array([id2idx[d] for d in drug_db], dtype=np.int64)
    n_drug = len(drug_idx)
    g2l = {int(g): i for i, g in enumerate(drug_idx)}
    drug_set = set(int(g) for g in drug_idx)
    d2loc = {db: i for i, db in enumerate(drug_db)}

    pos = pd.read_csv(DDI_POS_CSV)
    pa = pos["drug_a_id"].map(d2loc).to_numpy()
    pb = pos["drug_b_id"].map(d2loc).to_numpy()
    ok = (~pd.isna(pa)) & (~pd.isna(pb))
    pa = pa[ok].astype(np.int64); pb = pb[ok].astype(np.int64)
    buck = np.array([pkpd_bucket(m) for m in pos["ddi_type"].to_numpy()[ok]])
    pk = buck == "PK"; pd_ = buck == "PD"
    log(f"[pos] pairs={len(pa)}  PK={pk.sum()}  PD={pd_.sum()}  other={(buck=='other').sum()}")

    # ---- drug x node incidence per relation + overall ----
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    rel = edges["relation"].to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64); rel = rel[keep]

    def drug_node_pairs(mask):
        """Return (drug_local, node_global) for edges (under mask) that touch a drug."""
        s, d = src[mask], dst[mask]
        s_is_drug = np.array([x in drug_set for x in s])
        d_is_drug = np.array([x in drug_set for x in d]) if len(d) else np.array([], bool)
        rows = np.concatenate([
            np.array([g2l[x] for x in s[s_is_drug]], dtype=np.int64),
            np.array([g2l[x] for x in d[d_is_drug]], dtype=np.int64),
        ])
        cols = np.concatenate([d[s_is_drug], s[d_is_drug]])
        return rows, cols

    def incidence(rows, cols):
        inc = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                            shape=(n_drug, n_nodes)).tocsr()
        inc.data[:] = 1.0
        return inc

    # overall N (any relation), exclude drug->drug? keep all; mediators can be drugs
    rN, cN = drug_node_pairs(np.ones(len(src), bool))
    N = incidence(rN, cN)
    NT = N.T.tocsr()
    log(f"[inc] overall drug-node incidence nnz={N.nnz}")

    relations = sorted(pd.unique(rel).tolist())
    log(f"[rel] sweeping {len(relations)} relation types")

    rows_out = []
    pk_total = 0.0; pd_total = 0.0
    for ri, r in enumerate(relations):
        m = rel == r
        rr, cc = drug_node_pairs(m)
        if len(rr) == 0:
            rows_out.append({"relation": r, "drug_incident_edges": 0,
                             "pk_count": 0.0, "pd_count": 0.0})
            continue
        Br = incidence(rr, cc)
        Mr = (Br @ NT).todense()
        Mr = np.asarray(Mr)
        # both endpoints: a-side (M[a,b]) + b-side (M[b,a])
        pk_c = float(Mr[pa[pk], pb[pk]].sum() + Mr[pb[pk], pa[pk]].sum())
        pd_c = float(Mr[pa[pd_], pb[pd_]].sum() + Mr[pb[pd_], pa[pd_]].sum())
        pk_total += pk_c; pd_total += pd_c
        rows_out.append({"relation": r, "drug_incident_edges": int(len(rr)),
                         "pk_count": pk_c, "pd_count": pd_c})
        if (ri + 1) % 10 == 0:
            log(f"  ...{ri + 1}/{len(relations)} relations")

    df = pd.DataFrame(rows_out)
    df["pk_share"] = df["pk_count"] / max(pk_total, 1.0)
    df["pd_share"] = df["pd_count"] / max(pd_total, 1.0)
    df["pk_per_pair"] = df["pk_count"] / max(int(pk.sum()), 1)
    df["pd_per_pair"] = df["pd_count"] / max(int(pd_.sum()), 1)
    # enrichment: PK share vs PD share (log2), smoothed
    eps = 1e-9
    df["log2_pk_over_pd"] = np.log2((df["pk_share"] + eps) / (df["pd_share"] + eps))
    df = df.sort_values("pk_count", ascending=False).reset_index(drop=True)
    df.to_parquet(out_dir / "edge_type_distribution.parquet")

    # ---- print: top connecting relations + PK/PD shares ----
    log("\n================= CONNECTING EDGE-TYPE DISTRIBUTION (PK vs PD) =================")
    log(f"total connecting-path edges  PK={pk_total:.0f}  PD={pd_total:.0f}")
    log(f"mean connecting edges/pair   PK={pk_total/max(pk.sum(),1):.2f}  PD={pd_total/max(pd_.sum(),1):.2f}")
    log(f"\n{'relation':24s} {'PKshare':>8s} {'PDshare':>8s} {'PK/pair':>8s} {'PD/pair':>8s} {'log2PK/PD':>9s}")
    for _, r in df.head(20).iterrows():
        if r["pk_count"] == 0 and r["pd_count"] == 0:
            continue
        log(f"{r['relation']:24s} {r['pk_share']*100:7.2f}% {r['pd_share']*100:7.2f}% "
            f"{r['pk_per_pair']:8.3f} {r['pd_per_pair']:8.3f} {r['log2_pk_over_pd']:9.2f}")

    # most PK-enriched and most PD-enriched (among non-trivial relations)
    sig = df[(df["pk_count"] + df["pd_count"]) >= 1000].copy()
    log("\n  most PK-enriched connecting relations (log2 PK/PD):")
    for _, r in sig.sort_values("log2_pk_over_pd", ascending=False).head(6).iterrows():
        log(f"    {r['log2_pk_over_pd']:+.2f}  {r['relation']:24s} PKshare={r['pk_share']*100:.2f}% PDshare={r['pd_share']*100:.2f}%")
    log("  most PD-enriched connecting relations:")
    for _, r in sig.sort_values("log2_pk_over_pd", ascending=True).head(6).iterrows():
        log(f"    {r['log2_pk_over_pd']:+.2f}  {r['relation']:24s} PKshare={r['pk_share']*100:.2f}% PDshare={r['pd_share']*100:.2f}%")

    summary = {
        "n_pk_pairs": int(pk.sum()), "n_pd_pairs": int(pd_.sum()),
        "pk_total_connecting_edges": pk_total, "pd_total_connecting_edges": pd_total,
        "pk_mean_per_pair": pk_total / max(int(pk.sum()), 1),
        "pd_mean_per_pair": pd_total / max(int(pd_.sum()), 1),
        "n_relations": len(relations),
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"\n[save] {out_dir/'edge_type_distribution.parquet'}")
    log(f"[save] {out_dir/'summary.json'}")
    log(f"[done] wall_time={summary['wall_time_s']}s")


if __name__ == "__main__":
    sys.exit(main())

"""PK vs PD mediator-type distribution within the meeting budget (dA+dB<=4).

Goal: NOT to classify PK vs PD, but to characterize how the two sample classes
differ in WHAT kind of mediators connect their drugs, to explain (and inform
fixes for) PD's poor cold-start prediction.

For 100 PK + 100 PD positive pairs (seed=42), enumerate mediators M (M!=A,B)
with dA=dist(A,M)>=1, dB=dist(B,M)>=1, dA+dB<=4, and break them down by:

  (1) node KIND  -- all ~23 merged-KG node kinds (Gene, Side Effect, ...).
                    within budget, and at the confluent (1,1) cell.
  (2) relation CHANNEL -- the drugbank/het/prime relation through which BOTH
                    drugs reach a shared 1-hop mediator (enzyme/transporter/
                    target/carrier vs side-effect/effect/gene-assoc/...).
                    node-kind alone collapses enzyme/target/transporter (all
                    are protein/gene); the relation channel resolves them.

Reports PK vs PD mean mediators/pair per type + the difference, under undirected
and directed reachability (directed ~= undirected was shown earlier; kept for
the kind table).

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_ddi_pkpd_mediator_types.py --n 100 --seed 42
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

MAX_DEPTH = 3
BUDGET = 4

# relation channels (both-drugs-via-same-relation shared 1-hop mediators)
CHANNELS = [
    "db:target", "db:enzyme", "db:transporter", "db:carrier", "db:pathway",
    "het:CcSE", "prime:drug_effect",
    "het:CdG", "het:CuG", "het:CbG", "prime:drug_protein",
    "prime:indication", "prime:contraindication", "het:CrC",
]


def log(m: str) -> None:
    print(m, flush=True)


def pkpd_bucket(t: str) -> str:
    t = str(t).lower()
    if any(k in t for k in ("metabolism", "excretion", "serum concentration",
                            "absorption", "protein binding", "clearance")):
        return "PK"
    if any(k in t for k in ("risk or severity", "activities", "efficacy",
                            "cns depression", "qtc", "hypertension",
                            "hypotensive", "sedative", "adverse effects")):
        return "PD"
    return "other"


def build_csrs(edges, id2idx, n_nodes):
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    directed = edges["directed"].to_numpy().astype(bool)
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64); directed = directed[keep]
    nz = src != dst
    src, dst, directed = src[nz], dst[nz], directed[nz]
    u_rows = np.concatenate([src, dst]); u_cols = np.concatenate([dst, src])
    A_und = sp.coo_matrix((np.ones(len(u_rows), np.int8), (u_rows, u_cols)), shape=(n_nodes, n_nodes)).tocsr()
    A_und.data[:] = 1
    und = ~directed
    d_rows = np.concatenate([src, dst[und]]); d_cols = np.concatenate([dst, src[und]])
    A_dir = sp.coo_matrix((np.ones(len(d_rows), np.int8), (d_rows, d_cols)), shape=(n_nodes, n_nodes)).tocsr()
    A_dir.data[:] = 1
    log(f"[csr] undirected nnz={A_und.nnz}  directed-out nnz={A_dir.nnz}")
    return A_und, A_dir


def build_channel_incidence(edges, id2idx, n_nodes, drug_idx):
    """drug_local x node binary incidence per channel relation."""
    n_drug = len(drug_idx)
    g2l = {int(g): i for i, g in enumerate(drug_idx)}
    dset = set(int(g) for g in drug_idx)
    src = edges["src"].map(id2idx).to_numpy(); dst = edges["dst"].map(id2idx).to_numpy()
    rel = edges["relation"].to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64); dst = dst[keep].astype(np.int64); rel = rel[keep]
    inc = {}
    for ch in CHANNELS:
        m = rel == ch; s, d = src[m], dst[m]
        sdrug = np.array([x in dset for x in s])
        ddrug = np.array([x in dset for x in d]) if len(d) else np.array([], bool)
        rows = np.concatenate([np.array([g2l[x] for x in s[sdrug]], dtype=np.int64),
                               np.array([g2l[x] for x in d[ddrug]], dtype=np.int64)])
        cols = np.concatenate([d[sdrug], s[ddrug]])
        M = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_drug, n_nodes)).tocsr()
        M.data[:] = 1
        inc[ch] = M
    return inc, g2l


def bfs_depth(A_csr, source, n_nodes, max_depth=MAX_DEPTH):
    dist = np.full(n_nodes, -1, dtype=np.int16)
    dist[source] = 0
    frontier = np.array([source], dtype=np.int64)
    for d in range(1, max_depth + 1):
        if len(frontier) == 0:
            break
        nbrs = np.unique(A_csr[frontier].indices)
        new = nbrs[dist[nbrs] == -1]
        if len(new) == 0:
            break
        dist[new] = d
        frontier = new
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_pkpd_mediator_types"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    kind_arr = nodes["kind"].to_numpy().astype(object)  # node kind per global idx
    all_kinds = sorted(pd.unique(kind_arr).tolist())
    kind2col = {k: i for i, k in enumerate(all_kinds)}
    kind_code = np.array([kind2col[k] for k in kind_arr], dtype=np.int32)
    n_kind = len(all_kinds)
    log(f"[kinds] {n_kind} node kinds")

    drug_db = nodes["id"].to_numpy()[nodes["kind"].to_numpy() == "Drug"]
    drug_idx = np.array([id2idx[d] for d in drug_db], dtype=np.int64)
    d2glob = {db: id2idx[db] for db in drug_db}

    A_und, A_dir = build_csrs(edges, id2idx, n_nodes)
    inc, g2l = build_channel_incidence(edges, id2idx, n_nodes, drug_idx)

    pos = pd.read_csv(DDI_POS_CSV)
    pos = pos[pos["drug_a_id"].isin(d2glob) & pos["drug_b_id"].isin(d2glob)].reset_index(drop=True)
    bk_arr = np.array([pkpd_bucket(t) for t in pos["ddi_type"].to_numpy()])
    samples = {}
    for bk in ["PK", "PD"]:
        idx = np.where(bk_arr == bk)[0]
        sel = rng.choice(idx, size=min(args.n, len(idx)), replace=False)
        samples[bk] = pos.iloc[sel][["drug_a_id", "drug_b_id"]].to_numpy()
        log(f"[sample] {bk}: {len(sel)} pairs")

    cache: dict = {}

    def get_dist(cname, A, g):
        key = (cname, g)
        if key not in cache:
            cache[key] = bfs_depth(A, g, n_nodes)
        return cache[key]

    results = {}
    # ---- (1)+(2) node-kind distribution within budget and at (1,1) ----
    for cname, A in [("undirected", A_und), ("directed", A_dir)]:
        for bk in ["PK", "PD"]:
            budget_kind = np.zeros(n_kind, dtype=np.float64)   # mediators within budget, by kind
            conflu_kind = np.zeros(n_kind, dtype=np.float64)   # mediators at (1,1), by kind
            npairs = len(samples[bk])
            for (a, b) in samples[bk]:
                dA = get_dist(cname, A, d2glob[a]); dB = get_dist(cname, A, d2glob[b])
                mask = (dA >= 1) & (dB >= 1) & ((dA + dB) <= BUDGET)
                kc = kind_code[mask]
                np.add.at(budget_kind, kc, 1.0)
                m11 = mask & (dA == 1) & (dB == 1)
                np.add.at(conflu_kind, kind_code[m11], 1.0)
            results[f"{cname}__{bk}"] = {
                "n_pairs": npairs,
                "budget_kind_per_pair": (budget_kind / npairs),
                "conflu11_kind_per_pair": (conflu_kind / npairs),
            }

    # ---- (3) relation-channel shared confluent (both drugs via same relation r) ----
    chan_res = {}
    for bk in ["PK", "PD"]:
        per_chan = {ch: 0.0 for ch in CHANNELS}
        cov_chan = {ch: 0 for ch in CHANNELS}
        npairs = len(samples[bk])
        for (a, b) in samples[bk]:
            la, lb = g2l[d2glob[a]], g2l[d2glob[b]]
            for ch in CHANNELS:
                M = inc[ch]
                shared = int(M[la].multiply(M[lb]).sum())
                per_chan[ch] += shared
                cov_chan[ch] += int(shared > 0)
        chan_res[bk] = {"mean_per_pair": {ch: per_chan[ch] / npairs for ch in CHANNELS},
                        "coverage": {ch: cov_chan[ch] / npairs for ch in CHANNELS}}

    # ===================== PRINT =====================
    def kind_table(metric_key, title):
        log(f"\n================= {title} =================")
        log(f"{'node kind':22s} {'PK/pair':>10s} {'PD/pair':>10s} {'PD-PK':>10s} {'PD/PK':>8s}")
        pk = results[f"undirected__PK"][metric_key]
        pd_ = results[f"undirected__PD"][metric_key]
        order = np.argsort(-(pk + pd_))
        for i in order:
            if pk[i] + pd_[i] < 0.01:
                continue
            ratio = (pd_[i] + 1e-9) / (pk[i] + 1e-9)
            log(f"{all_kinds[i]:22s} {pk[i]:10.2f} {pd_[i]:10.2f} {pd_[i]-pk[i]:10.2f} {ratio:8.2f}")

    kind_table("budget_kind_per_pair", "mediators within dA+dB<=4, by NODE KIND (undirected)")
    kind_table("conflu11_kind_per_pair", "mediators at CONFLUENT (1,1), by NODE KIND (undirected)")

    log(f"\n================= shared CONFLUENT mediators by RELATION CHANNEL (both drugs via same r) =================")
    log(f"{'channel':22s} {'PK/pair':>9s} {'PKcov':>7s} {'PD/pair':>9s} {'PDcov':>7s} {'PD/PK':>8s}")
    for ch in CHANNELS:
        pk = chan_res["PK"]["mean_per_pair"][ch]; pkc = chan_res["PK"]["coverage"][ch]
        pdv = chan_res["PD"]["mean_per_pair"][ch]; pdc = chan_res["PD"]["coverage"][ch]
        ratio = (pdv + 1e-9) / (pk + 1e-9)
        log(f"{ch:22s} {pk:9.2f} {pkc*100:6.0f}% {pdv:9.2f} {pdc*100:6.0f}% {ratio:8.2f}")

    # directed vs undirected sanity for kind (one line)
    log("\n[note] directed vs undirected kind totals (within budget, mean/pair):")
    for bk in ["PK", "PD"]:
        u = results[f"undirected__{bk}"]["budget_kind_per_pair"].sum()
        d = results[f"directed__{bk}"]["budget_kind_per_pair"].sum()
        log(f"  {bk}: undirected={u:.1f}  directed={d:.1f}")

    # ---- save ----
    save = {
        "all_kinds": all_kinds,
        "kind_within_budget": {bk: results[f"undirected__{bk}"]["budget_kind_per_pair"].tolist() for bk in ["PK", "PD"]},
        "kind_confluent11": {bk: results[f"undirected__{bk}"]["conflu11_kind_per_pair"].tolist() for bk in ["PK", "PD"]},
        "relation_channel": chan_res,
    }
    (out_dir / "summary.json").write_text(json.dumps(save, indent=2))
    log(f"\n[save] {out_dir/'summary.json'}")
    log(f"[done] wall_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

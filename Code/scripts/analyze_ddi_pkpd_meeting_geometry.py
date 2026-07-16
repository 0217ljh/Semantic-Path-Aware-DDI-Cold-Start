"""PK vs PD meeting geometry: the (dA, dB) double-radius distribution.

For a DDI pair (A, B), a *mediator* M (M != A, B) is a common meeting node with
  dA = dist(A, M) >= 1,  dB = dist(B, M) >= 1,  dA + dB <= 4.
Each mediator falls into one of 6 cells:
  (1,1) | (1,2),(2,1) | (2,2),(1,3),(3,1).
The per-pair histogram over these cells is the pair's structural signature
(confluent = (1,1); bridged = asymmetric / sum-4 cells; severed = no mediator).

Two edge-direction conditions:
  cond1 "directed"  : respect merged-KG edge directions; M must be reachable from
                      BOTH A and B following edge directions (common directed
                      successor, A->...->M<-...<-B). Directed edges forward only;
                      natively-undirected edges (directed=False) both ways.
  cond2 "undirected": ignore direction, all edges both ways.

Samples 100 PK + 100 PD positive pairs (seed=42), reports per cell the mean
mediators/pair and the coverage (% pairs with >=1 mediator), plus the severed
rate (% pairs with no mediator within dA+dB<=4).

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_ddi_pkpd_meeting_geometry.py --n 100 --seed 42

Outputs (under --out, default Code/runs/analyze_ddi_pkpd_meeting_geometry/):
  summary.json            per (bucket, condition) cell grids + severed rate
  per_pair.parquet        per-pair per-cell mediator counts
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

MAX_DEPTH = 3              # max single-side distance (dA or dB), paired with >=1 on the other
BUDGET = 4                # dA + dB <= BUDGET
CELLS = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]  # dA>=1,dB>=1,dA+dB<=4


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
    """Return (A_undirected, A_directed_out) CSR adjacency matrices.

    A_undirected: symmetric, every edge both ways.
    A_directed_out: out-adjacency respecting direction. directed=True -> src->dst
        only; directed=False -> src->dst and dst->src. BFS forward in this graph
        = directed reachability following allowed edge directions.
    """
    src = edges["src"].map(id2idx).to_numpy()
    dst = edges["dst"].map(id2idx).to_numpy()
    directed = edges["directed"].to_numpy().astype(bool)
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    src = src[keep].astype(np.int64)
    dst = dst[keep].astype(np.int64)
    directed = directed[keep]
    nz = src != dst
    src, dst, directed = src[nz], dst[nz], directed[nz]

    # undirected: both ways for everything
    u_rows = np.concatenate([src, dst])
    u_cols = np.concatenate([dst, src])
    A_und = sp.coo_matrix((np.ones(len(u_rows), np.int8), (u_rows, u_cols)),
                          shape=(n_nodes, n_nodes)).tocsr()
    A_und.data[:] = 1

    # directed-out: src->dst always; dst->src only when natively undirected
    und_mask = ~directed
    d_rows = np.concatenate([src, dst[und_mask]])
    d_cols = np.concatenate([dst, src[und_mask]])
    A_dir = sp.coo_matrix((np.ones(len(d_rows), np.int8), (d_rows, d_cols)),
                          shape=(n_nodes, n_nodes)).tocsr()
    A_dir.data[:] = 1
    log(f"[csr] undirected nnz={A_und.nnz}  directed-out nnz={A_dir.nnz}")
    return A_und, A_dir


def bfs_depth(A_csr, source, n_nodes, max_depth=MAX_DEPTH):
    """Depth-limited BFS. Returns int16 dist array (-1 = unreached within depth)."""
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


def pair_cells(distA, distB):
    """Counts per (dA,dB) cell for one pair given two dist arrays."""
    mask = (distA >= 1) & (distB >= 1) & ((distA + distB) <= BUDGET)
    a = distA[mask]; b = distB[mask]
    counts = {}
    for (ca, cb) in CELLS:
        counts[(ca, cb)] = int(np.sum((a == ca) & (b == cb)))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="pairs per bucket")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/analyze_ddi_pkpd_meeting_geometry"))
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    log(f"[load] {NODES_PARQUET}")
    nodes = pd.read_parquet(NODES_PARQUET)
    log(f"[load] {EDGES_PARQUET}")
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    kinds = nodes["kind"].to_numpy()
    drug_db = nodes["id"].to_numpy()[kinds == "Drug"]
    d2glob = {db: id2idx[db] for db in drug_db}

    A_und, A_dir = build_csrs(edges, id2idx, n_nodes)
    conds = {"directed": A_dir, "undirected": A_und}

    pos = pd.read_csv(DDI_POS_CSV)
    pos = pos[pos["drug_a_id"].isin(d2glob) & pos["drug_b_id"].isin(d2glob)].reset_index(drop=True)
    bucket = np.array([pkpd_bucket(t) for t in pos["ddi_type"].to_numpy()])

    samples = {}
    for bk in ["PK", "PD"]:
        idx = np.where(bucket == bk)[0]
        sel = rng.choice(idx, size=min(args.n, len(idx)), replace=False)
        samples[bk] = pos.iloc[sel][["drug_a_id", "drug_b_id"]].to_numpy()
        log(f"[sample] {bk}: {len(sel)} pairs")

    # cache dist arrays per (cond, drug_global)
    dist_cache: dict[tuple[str, int], np.ndarray] = {}

    def get_dist(cname, A, g):
        key = (cname, g)
        if key not in dist_cache:
            dist_cache[key] = bfs_depth(A, g, n_nodes)
        return dist_cache[key]

    rows = []
    summary = {}
    for bk in ["PK", "PD"]:
        for cname, A in conds.items():
            cell_tot = {c: 0 for c in CELLS}      # total mediators across pairs
            cell_cov = {c: 0 for c in CELLS}      # #pairs with >=1 mediator in cell
            severed = 0
            n_pairs = len(samples[bk])
            for (da_id, db_id) in samples[bk]:
                gA = d2glob[da_id]; gB = d2glob[db_id]
                distA = get_dist(cname, A, gA)
                distB = get_dist(cname, A, gB)
                counts = pair_cells(distA, distB)
                tot = sum(counts.values())
                if tot == 0:
                    severed += 1
                for c in CELLS:
                    cell_tot[c] += counts[c]
                    cell_cov[c] += int(counts[c] > 0)
                rows.append({"bucket": bk, "cond": cname, "a": da_id, "b": db_id,
                             "total_mediators": tot,
                             **{f"cell_{c[0]}_{c[1]}": counts[c] for c in CELLS}})
            summary[f"{bk}__{cname}"] = {
                "n_pairs": n_pairs,
                "severed_rate": severed / n_pairs,
                "coverage_any": 1 - severed / n_pairs,
                "mean_mediators_per_pair": {f"{c[0]},{c[1]}": cell_tot[c] / n_pairs for c in CELLS},
                "coverage_per_cell": {f"{c[0]},{c[1]}": cell_cov[c] / n_pairs for c in CELLS},
            }

    # ---- print ----
    def grid_str(d, fmt):
        # rows dA=1..3, cols dB=1..3, only valid cells
        out = ["        dB=1     dB=2     dB=3"]
        for da in [1, 2, 3]:
            cells = []
            for db in [1, 2, 3]:
                k = f"{da},{db}"
                if (da, db) in CELLS:
                    cells.append(fmt.format(d[k]))
                else:
                    cells.append("    -   ")
            out.append(f"  dA={da}  " + " ".join(cells))
        return "\n".join(out)

    for bk in ["PK", "PD"]:
        for cname in ["directed", "undirected"]:
            s = summary[f"{bk}__{cname}"]
            log(f"\n================= {bk}  |  {cname}  (n={s['n_pairs']}) =================")
            log(f"  coverage (>=1 mediator within dA+dB<=4) = {s['coverage_any']*100:5.1f}%   "
                f"severed = {s['severed_rate']*100:5.1f}%")
            log("  mean mediators / pair per cell:")
            log(grid_str(s["mean_mediators_per_pair"], "{:7.2f} "))
            log("  coverage (% pairs with >=1) per cell:")
            log(grid_str({k: v * 100 for k, v in s["coverage_per_cell"].items()}, "{:6.1f}% "))

    # ---- save ----
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_parquet(out_dir / "per_pair.parquet")
    log(f"\n[save] {out_dir/'summary.json'}")
    log(f"[save] {out_dir/'per_pair.parquet'}")
    log(f"[done] wall_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())

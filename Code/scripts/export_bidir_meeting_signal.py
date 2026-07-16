"""Simplest bidirectional-NBFNet structural signal (no phase, no learning).

Two BFS wavefronts meet in the middle: from A (depth) and from B (depth). The
transferable structural signal of a pair (A,B) is the meeting-mediator structure
binned by the double-radius cell (dA,dB) = (dist(A,V), dist(B,V)), with two
aggregations per cell:
  cnt_(dA,dB)  = number of meeting mediators in that cell
  spec_(dA,dB) = sum of 1/log1p(1+deg(V))  (hub-down-weighted "specificity mass")

This is the boolean/node meeting semiring of a bidirectional generalized
Bellman-Ford. It is purely structural (no drug-id embedding) -> inductive /
cold-start safe / transferable. The sum-product PATH-count version is the natural
next step (weight V by #A->V paths * #V->B paths) but is left out for simplicity.

Cells (dA>=1, dB>=1, dA+dB<=budget); budget=4 -> 6 cells:
  (1,1) | (1,2),(2,1) | (2,2),(1,3),(3,1).

Runs on the seed42 cold-start S2 pairs (test_s2 PK/PD positives + negatives),
saves per-pair features, and reports a quick LR 5-fold-CV AUROC sanity (does the
signal carry pos-vs-neg discriminative info, split by PK/PD).

Run (from project root, via WSL conda env project_1):
  python Code/scripts/export_bidir_meeting_signal.py --budget 4 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
sys.path.insert(0, str(ROOT))
NODES_PARQUET = ROOT / "data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES_PARQUET = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"


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


def cells_for_budget(budget: int):
    return [(da, db) for da in range(1, budget) for db in range(1, budget)
            if da + db <= budget]


def bfs_depth(A_csr, source, n_nodes, max_depth):
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
    ap.add_argument("--budget", type=int, default=4, help="dA+dB <= budget")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(ROOT / "runs/bidir_meeting_signal"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cells = cells_for_budget(args.budget)
    max_depth = args.budget - 1
    log(f"[cfg] budget={args.budget} cells={cells} max_depth={max_depth}")

    nodes = pd.read_parquet(NODES_PARQUET)
    edges = pd.read_parquet(EDGES_PARQUET)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"].to_numpy())}
    n_nodes = len(id2idx)
    drug_db = nodes["id"].to_numpy()[nodes["kind"].to_numpy() == "Drug"]
    d2g = {db: id2idx[db] for db in drug_db}

    # undirected binary adjacency + node degrees
    src = edges["src"].map(id2idx).to_numpy(); dst = edges["dst"].map(id2idx).to_numpy()
    keep = (~pd.isna(src)) & (~pd.isna(dst))
    s = src[keep].astype(np.int64); d = dst[keep].astype(np.int64)
    nz = s != d
    s, d = s[nz], d[nz]
    rows = np.concatenate([s, d]); cols = np.concatenate([d, s])
    A = sp.coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    inv_log_deg = 1.0 / np.log1p(1.0 + deg)   # hub down-weight; monotone decreasing
    log(f"[kg] n_nodes={n_nodes} adj_nnz={A.nnz}")

    # ----- pair set: seed42 cold-start S2 (test_s2 pos PK/PD + neg) -----
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(ROOT / f"data/coldddi_legacy/800drug/seed{args.seed}.pkl"))
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].astype(str)
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]].astype(str)
    # bucket positives via ddi_edges.csv
    ddi = pd.read_csv(DDI_POS_CSV)
    key2t = {frozenset((a, b)): t for a, b, t in
             zip(ddi["drug_a_id"].astype(str), ddi["drug_b_id"].astype(str), ddi["ddi_type"].astype(str))}
    pos_bucket = [pkpd_bucket(key2t.get(frozenset((a, b)), "")) for a, b in
                  zip(pos["drug_a_id"], pos["drug_b_id"])]

    pairs = pd.concat([
        pos.assign(label=1, bucket=pos_bucket),
        neg.assign(label=0, bucket="neg"),
    ], ignore_index=True)
    log(f"[pairs] pos={len(pos)} (PK={pos_bucket.count('PK')} PD={pos_bucket.count('PD')}) neg={len(neg)}")

    # ----- cache per-drug BFS, then extract per-pair features -----
    dist_cache: dict[int, np.ndarray] = {}

    def get_dist(g):
        if g not in dist_cache:
            dist_cache[g] = bfs_depth(A, g, n_nodes, max_depth)
        return dist_cache[g]

    feat_cols = [f"cnt_{a}_{b}" for (a, b) in cells] + [f"spec_{a}_{b}" for (a, b) in cells] + ["sev"]
    feats = np.zeros((len(pairs), len(feat_cols)), dtype=np.float64)
    n_cell = len(cells)
    for i, (a, b) in enumerate(zip(pairs["drug_a_id"], pairs["drug_b_id"])):
        if a not in d2g or b not in d2g:
            feats[i, -1] = 1.0  # severed / unknown
            continue
        dA = get_dist(d2g[a]); dB = get_dist(d2g[b])
        mask = (dA >= 1) & (dB >= 1) & ((dA + dB) <= args.budget)
        if not mask.any():
            feats[i, -1] = 1.0
            continue
        da = dA[mask]; db = dB[mask]; w = inv_log_deg[mask]
        for j, (ca, cb) in enumerate(cells):
            sel = (da == ca) & (db == cb)
            feats[i, j] = int(sel.sum())             # cnt
            feats[i, n_cell + j] = float(w[sel].sum())  # spec
        if (i + 1) % 2000 == 0:
            log(f"  ...{i + 1}/{len(pairs)} pairs")

    feat_df = pd.DataFrame(feats, columns=feat_cols)
    out = pd.concat([pairs.reset_index(drop=True), feat_df], axis=1)
    out.to_parquet(out_dir / "pair_features.parquet")
    log(f"[save] {out_dir / 'pair_features.parquet'}  ({len(out)} pairs, {len(feat_cols)} feats)")

    # ----- quick sanity: does the signal discriminate pos vs neg (per PK/PD)? -----
    # log1p on count features for scale; LR 5-fold CV AUROC.
    X_all = feats.copy()
    X_all[:, :n_cell] = np.log1p(X_all[:, :n_cell])  # log1p the counts
    y = pairs["label"].to_numpy()
    neg_mask = y == 0
    log("\n[sanity] LR 5-fold CV AUROC on the bidirectional meeting signal:")
    results = {}
    for bk in ["PK", "PD"]:
        sel = (pairs["bucket"].to_numpy() == bk) | neg_mask
        Xb = X_all[sel]; yb = y[sel]
        if len(np.unique(yb)) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        auc = cross_val_score(clf, Xb, yb, cv=5, scoring="roc_auc")
        results[bk] = float(auc.mean())
        log(f"  {bk} pos vs neg: AUROC = {auc.mean():.4f} (+/-{auc.std():.3f})  "
            f"[n_pos={int(yb.sum())} n_neg={int((yb==0).sum())}]")
    log("  (ref: NBFNet cold-start S2 PK=0.855, PD=0.727; this is a structure-only signal + LR)")

    (out_dir / "summary.json").write_text(json.dumps({
        "budget": args.budget, "cells": [list(c) for c in cells],
        "feat_cols": feat_cols, "n_pairs": int(len(pairs)),
        "cv_auroc": results,
    }, indent=2))
    log(f"[save] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    sys.exit(main())

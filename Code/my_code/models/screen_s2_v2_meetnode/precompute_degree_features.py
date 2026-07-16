"""Degree-only control features (codex round 14 #1 control).

Same 11 kinds x {1hop,2hop} = 22 dims, but instead of the SHARED-mediator
intersection count |N(a) ∩ N(b)|, we use a pair-symmetric combination of the
INDIVIDUAL per-drug degrees: min(deg_kind(a), deg_kind(b)) (log1p).

Rationale: this captures "how connected each drug is per kind" but DESTROYS the
pair-specific intersection (which specific mediators are SHARED). If MNAH with
these features recovers most of the gain, the signal is crude degree/topology.
If it collapses toward chance, the gain is genuinely the meeting-node overlap.

min() is the natural degree-only analog of intersection size: |A ∩ B| <= min(|A|,|B|),
so min is the tightest intersection-free upper bound and the fairest degree control.

Output: Code/data/_cache/degree_feat_drugbank_seed42_kgonly_v1.parquet
  same schema as meet_feat (drug_a_id, drug_b_id canonical, f_1hop_*, f_2hop_*)
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FILE = Path(__file__).resolve()
# reuse the shared-mediator precompute's helpers + constants
sys.path.insert(0, str(_FILE.parent))
from precompute_meet_features import (  # noqa: E402
    PROJECT_ROOT, NODES, EDGES, SPLITS, KIND_ORDER, N_KINDS, N_FEAT,
    build_neighbor_sets, canonical_pair,
)


def per_drug_degree(n1: dict, n2: dict, drug: str) -> tuple[dict, dict]:
    """Return (deg1, deg2): per-kind neighbor counts for one drug."""
    d1: dict[str, int] = defaultdict(int)
    d2: dict[str, int] = defaultdict(int)
    for _, g in n1.get(drug, ()):
        d1[g] += 1
    for _, g in n2.get(drug, ()):
        d2[g] += 1
    return d1, d2


def featurize_degree(pairs, n1, n2, deg_cache) -> np.ndarray:
    """min(deg_a, deg_b) per kind, log1p. pairs canonical (a<=b)."""
    X = np.zeros((len(pairs), N_FEAT), dtype=np.float32)
    for i, (da, db) in enumerate(pairs):
        if da not in deg_cache:
            deg_cache[da] = per_drug_degree(n1, n2, da)
        if db not in deg_cache:
            deg_cache[db] = per_drug_degree(n1, n2, db)
        a1, a2 = deg_cache[da]
        b1, b2 = deg_cache[db]
        for j, grp in enumerate(KIND_ORDER):
            X[i, j] = np.log1p(min(a1[grp], b1[grp]))
            X[i, N_KINDS + j] = np.log1p(min(a2[grp], b2[grp]))
    return X


def main():
    out_dir = PROJECT_ROOT / "Code/data/_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "degree_feat_drugbank_seed42_kgonly_v1.parquet"
    if out_path.exists():
        print(f"[deg-pre] cache EXISTS at {out_path}")
        return

    print("[deg-pre] Loading nodes/edges ...", flush=True)
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    n1, n2 = build_neighbor_sets(edges, id2kind, drug_set)

    all_drugs: set[str] = set()
    for sub in ["train.parquet", "val_s2.parquet", "test_s2.parquet"]:
        df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
        all_drugs |= set(df["drug_a_id"].astype(str)) | set(df["drug_b_id"].astype(str))
    for sub in ["negatives/val_s2.parquet", "negatives/test_s2.parquet",
                "train_negatives/epoch_0.parquet"]:
        try:
            df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
            all_drugs |= set(df["drug_a_id"].astype(str)) | set(df["drug_b_id"].astype(str))
        except Exception:
            pass
    all_drugs = sorted(all_drugs)
    n = len(all_drugs)
    print(f"[deg-pre] |unique drugs| = {n}; generating {n*(n+1)//2} canonical pairs", flush=True)
    pairs = [(all_drugs[i], all_drugs[j]) for i in range(n) for j in range(i, n)]

    deg_cache: dict = {}
    t0 = time.time()
    BATCH = 50000
    feats = []
    for s in range(0, len(pairs), BATCH):
        feats.append(featurize_degree(pairs[s:s+BATCH], n1, n2, deg_cache))
        if (s // BATCH) % 5 == 0:
            print(f"  {s}/{len(pairs)}  elapsed={time.time()-t0:.1f}s", flush=True)
    X = np.vstack(feats)
    print(f"[deg-pre] done shape={X.shape} time={time.time()-t0:.1f}s", flush=True)

    cols = [f"f_1hop_{g}" for g in KIND_ORDER] + [f"f_2hop_{g}" for g in KIND_ORDER]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "drug_a_id", [p[0] for p in pairs])
    df.insert(1, "drug_b_id", [p[1] for p in pairs])
    df.to_parquet(out_path, index=False)
    print(f"[deg-pre] wrote {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()

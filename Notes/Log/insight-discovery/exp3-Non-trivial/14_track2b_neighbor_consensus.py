"""exp3 Track 2b — Neighbor-Consensus Paradox.

For each cold drug d in test_s2:
  N_k(d) = top-k nearest training drugs (PubMedBERT cosine)
For pair (a, b) in test:
  N_a × N_b = k^2 candidate "twin pairs" in training space
  vote(a, b) = fraction of (n_i, m_j) ∈ N_a × N_b that are positive DDI in train
  consensus(a, b) = 1 - entropy of vote at thresholded levels OR std of per-neighbor agreement

Hypothesis (cold-start specific):
  High closeness + HIGH consensus (twins agree DDI exists) → easy
  High closeness + LOW consensus (twins disagree) → hard (worse than low-closeness baseline)

Operationalization:
- k = 10
- For each pair, compute:
  - vote: fraction of N_a × N_b training pairs that ARE in train DDI
  - vote_var: variance of binary indicator (mean × (1-mean))
  - high_consensus_pos: most twins agree positive (vote > 0.5 AND low var)
  - high_consensus_neg: most twins agree negative (vote < 0.5 AND low var)
  - low_consensus: split (vote ~ 0.5 with high var)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root():
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
CACHE = ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features"


def main():
    t0 = time.time()
    K = 10  # top-k neighbors
    print(f"[setup] K = {K}")

    # Load landscape features (already has closeness, ambiguity computed)
    df = pd.read_parquet(OUT_DIR / "pair_features_landscape.parquet")
    print(f"  pairs={len(df)}")

    # Need: pair_close_tx already in df; now compute pair_consensus
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    X_all = np.load(CACHE / "pubmedbert_real.npz")["X"]
    id2idx = dict(zip(nodes["id"], range(len(nodes))))

    train_pairs = pd.read_parquet(ROOT / "Code/data/KG/drugbank/splits/seed42/train.parquet")
    train_drugs = sorted(set(train_pairs["drug_a_id"]) | set(train_pairs["drug_b_id"]))
    train_drugs = [d for d in train_drugs if d in id2idx]
    test_drugs = sorted(set(df["drug_a_id"]) | set(df["drug_b_id"]))
    test_drugs = [d for d in test_drugs if d in id2idx]
    print(f"  train drugs: {len(train_drugs)}, test drugs: {len(test_drugs)}")

    # Build training DDI lookup: set of unordered (drug_a, drug_b) pairs
    train_pair_set = set()
    for a, b in zip(train_pairs["drug_a_id"], train_pairs["drug_b_id"]):
        if a in id2idx and b in id2idx:
            train_pair_set.add((a, b) if a <= b else (b, a))
    print(f"  train positive pairs (unique unordered): {len(train_pair_set)}")

    # Compute top-K nearest training drugs for each test drug (PubMedBERT cosine)
    E_train = np.stack([X_all[id2idx[d]] for d in train_drugs], axis=0)
    E_test = np.stack([X_all[id2idx[d]] for d in test_drugs], axis=0)
    E_all = np.vstack([E_train, E_test])
    mu = E_all.mean(0, keepdims=True)
    sd = E_all.std(0, keepdims=True) + 1e-8
    E_train = (E_train - mu) / sd
    E_test = (E_test - mu) / sd
    def l2norm(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    E_train = l2norm(E_train)
    E_test = l2norm(E_test)
    sim = E_test @ E_train.T  # (n_test, n_train)
    print(f"  sim shape: {sim.shape}")

    # Top-K indices per cold drug
    top_idx = np.argpartition(-sim, K, axis=1)[:, :K]
    # Sort within top-K by similarity descending
    for i in range(len(top_idx)):
        s_i = sim[i, top_idx[i]]
        order = np.argsort(-s_i)
        top_idx[i] = top_idx[i][order]
    drug_to_topk = {d: [train_drugs[j] for j in top_idx[i]] for i, d in enumerate(test_drugs)}
    print(f"  example: top-{K} of {test_drugs[0]} = {drug_to_topk[test_drugs[0]][:3]}...")

    # For each pair (a, b) in test_s2, compute neighbor consensus
    n = len(df)
    consensus = np.full(n, np.nan, dtype=np.float32)  # vote = fraction of N_a × N_b in train DDI
    vote_var = np.full(n, np.nan, dtype=np.float32)
    n_train_cross = np.zeros(n, dtype=np.int32)

    print("[compute] iterating pairs ...")
    t_iter = time.time()
    for i, (a, b) in enumerate(zip(df["drug_a_id"], df["drug_b_id"])):
        na = drug_to_topk.get(a)
        nb = drug_to_topk.get(b)
        if not na or not nb:
            continue
        # Count: how many N_a × N_b pairs are in train DDI?
        total = 0
        hits = 0
        for x in na:
            for y in nb:
                if x == y:
                    continue
                key = (x, y) if x <= y else (y, x)
                if key in train_pair_set:
                    hits += 1
                total += 1
        if total == 0:
            continue
        vote = hits / total
        consensus[i] = vote  # 0 or 1 = full agreement, 0.5 = max conflict
        vote_var[i] = vote * (1 - vote)
        n_train_cross[i] = total
        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t_iter
            print(f"  {i+1}/{n}, elapsed={elapsed:.1f}s")

    df["pair_consensus_vote"] = consensus
    df["pair_consensus_var"] = vote_var
    df["pair_n_cross"] = n_train_cross
    df["pair_consensus_strength"] = 1 - 2 * vote_var  # 0 = max conflict, 1 = full agreement either way
    print(f"  done. consensus mean={np.nanmean(consensus):.4f}, var mean={np.nanmean(vote_var):.4f}")

    # Stratified analysis
    print("\n=== Marginal: consensus_vote vs NLL ===")
    df["vote_q"] = pd.qcut(df["pair_consensus_vote"], 5, labels=["V1", "V2", "V3", "V4", "V5"], duplicates="drop")
    tbl = df.groupby("vote_q", observed=True).agg(
        n=("nll", "count"),
        vote_mean=("pair_consensus_vote", "mean"),
        nll=("nll", "mean"),
        label_rate=("label", "mean"),
    )
    print(tbl)

    print("\n=== Marginal: consensus_strength (1=full agreement, 0=max conflict) ===")
    df["str_q"] = pd.qcut(df["pair_consensus_strength"], 5, labels=False, duplicates="drop")
    tbl = df.groupby("str_q", observed=True).agg(
        n=("nll", "count"),
        str_mean=("pair_consensus_strength", "mean"),
        nll=("nll", "mean"),
        label_rate=("label", "mean"),
    )
    print(tbl)

    # 2D paradox check: high closeness × consensus
    print("\n=== 2D: pair_close_tx (top 30%) × consensus_strength (top 30% = full agreement) ===")
    df["hi_close"] = df["pair_close_tx"] > df["pair_close_tx"].quantile(0.7)
    df["hi_str"] = df["pair_consensus_strength"] > df["pair_consensus_strength"].quantile(0.7)
    df["q2d"] = np.where(df["hi_close"] & df["hi_str"], "high-close, agreement",
                np.where(df["hi_close"] & ~df["hi_str"], "high-close, CONFLICT",
                        np.where(~df["hi_close"] & df["hi_str"], "low-close, agreement",
                                "low-close, conflict")))
    print(df.groupby("q2d").agg(
        n=("nll", "count"),
        nll=("nll", "mean"),
        label_rate=("label", "mean"),
    ))

    print("\n[2D, label=1 only]")
    pos = df[df["label"] == 1]
    print(pos.groupby("q2d").agg(n=("nll", "count"), nll=("nll", "mean")))
    print("[2D, label=0 only]")
    neg = df[df["label"] == 0]
    print(neg.groupby("q2d").agg(n=("nll", "count"), nll=("nll", "mean")))

    # Within high-closeness, decile by consensus
    print("\n=== Within high-closeness subset: NLL deciled by consensus_var ===")
    hi = df[df["hi_close"]].copy()
    for lbl in [1, 0]:
        sub = hi[hi["label"] == lbl].copy()
        if len(sub) < 200 or sub["pair_consensus_var"].isna().sum() > 0.5*len(sub):
            print(f"  label={lbl} insufficient/many nan ({sub['pair_consensus_var'].isna().sum()} nan / {len(sub)})")
            continue
        sub = sub[sub["pair_consensus_var"].notna()]
        sub["var_dec"] = pd.qcut(sub["pair_consensus_var"], 10, labels=False, duplicates="drop")
        t = sub.groupby("var_dec").agg(
            var_mean=("pair_consensus_var", "mean"),
            vote_mean=("pair_consensus_vote", "mean"),
            nll=("nll", "mean"),
            n=("nll", "count"),
        )
        print(f"\n[label={lbl}, high-closeness] decile by consensus_var (higher = more conflict)")
        print(t)
        rho, p = spearmanr(sub["pair_consensus_var"], sub["nll"])
        print(f"  ρ(consensus_var, NLL): {rho:+.4f} p={p:.2e}")

    # Save
    df.to_parquet(OUT_DIR / "pair_features_consensus.parquet")
    print(f"\nsaved. elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""exp3 M3 fix — Explicit pair alignment verification.

Codex critique: only verified label sums, not pair-by-pair identity.
This script independently reloads pair order from disk in E3's exact pattern,
then compares with our pair_features.parquet pair order at the (drug_a, drug_b)
level. If any single pair is misaligned, every NLL claim is suspect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
SPLITS = ROOT / "Code/data/KG/drugbank/splits/seed42"


# Replay E3's exact loading code (copied from 05_init_comparison.py:220-225)
def e3_load_pairs(pos_path, neg_path):
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]]
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]]
    pos["lab"] = 1
    neg["lab"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    return df


def exp3_load_pairs(pos_path, neg_path):
    """Mirror of 01_compute_features.py's load_pairs_and_pred."""
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
    pos["label"] = 1.0
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    neg["label"] = 0.0
    df = pd.concat([pos, neg], ignore_index=True)
    return df


def main():
    pos_path = SPLITS / "test_s2.parquet"
    neg_path = SPLITS / "negatives/test_s2.parquet"

    e3_df = e3_load_pairs(pos_path, neg_path)
    exp3_df = exp3_load_pairs(pos_path, neg_path)
    feat_df = pd.read_parquet(OUT_DIR / "pair_features.parquet")

    print(f"E3 loader:    {len(e3_df)} rows")
    print(f"exp3 loader:  {len(exp3_df)} rows")
    print(f"pair_features: {len(feat_df)} rows")

    # Pair-by-pair identity check
    assert len(e3_df) == len(exp3_df) == len(feat_df)

    mismatches_e3_vs_exp3 = ((e3_df["drug_a_id"].values != exp3_df["drug_a_id"].values) |
                              (e3_df["drug_b_id"].values != exp3_df["drug_b_id"].values)).sum()
    mismatches_exp3_vs_feat = ((exp3_df["drug_a_id"].values != feat_df["drug_a_id"].values) |
                                (exp3_df["drug_b_id"].values != feat_df["drug_b_id"].values)).sum()

    print(f"\nE3-loader vs exp3-loader pair mismatches: {mismatches_e3_vs_exp3}")
    print(f"exp3-loader vs pair_features pair mismatches: {mismatches_exp3_vs_feat}")

    # Also verify label consistency in all three
    lbl_e3 = e3_df["lab"].astype(float).values
    lbl_exp3 = exp3_df["label"].values
    lbl_feat = feat_df["label"].values
    print(f"E3 vs exp3 label mismatches: {(lbl_e3 != lbl_exp3).sum()}")
    print(f"exp3 vs feat label mismatches: {(lbl_exp3 != lbl_feat).sum()}")

    # Verify cached prediction's `y` matches our `label` pair-by-pair (which we did before)
    cache = np.load(ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features/pred_real_pubmedbert_pca_seed42.npz")
    y_cached = cache["y"]
    print(f"\ncached prediction y length: {len(y_cached)}")
    print(f"label mismatch with cached y: {(y_cached != lbl_feat).sum()}")

    # Stronger: hash the pair sequences and compare
    import hashlib
    def pair_hash(df, a_col, b_col, lbl_col=None):
        cols = [a_col, b_col]
        if lbl_col is not None:
            cols.append(lbl_col)
        s = df[cols].astype(str).agg("|".join, axis=1).str.cat(sep="\n")
        return hashlib.md5(s.encode()).hexdigest()
    h_e3 = pair_hash(e3_df, "drug_a_id", "drug_b_id", "lab")
    h_exp3 = pair_hash(exp3_df, "drug_a_id", "drug_b_id", "label")
    h_feat = pair_hash(feat_df, "drug_a_id", "drug_b_id", "label")
    print(f"\nHash of full (drug_a, drug_b, label) sequence:")
    print(f"  E3 loader:   {h_e3}")
    print(f"  exp3 loader: {h_exp3}")
    print(f"  feat parquet:{h_feat}")
    print(f"  match: {h_e3 == h_exp3 == h_feat}")

    # Confirm: positives block is exactly 54369 rows, negatives 54369
    n_pos = (lbl_feat == 1.0).sum()
    n_neg = (lbl_feat == 0.0).sum()
    print(f"\nPositives count: {n_pos}, Negatives count: {n_neg}")

    # Cross-check: first 5 and last 5 pair_features pairs
    print(f"\nFirst 3 pairs in feat:    {feat_df.head(3)[['drug_a_id','drug_b_id','label']].values.tolist()}")
    print(f"First 3 pairs in E3:      {e3_df.head(3)[['drug_a_id','drug_b_id','lab']].values.tolist()}")
    print(f"Last 3 pairs in feat:     {feat_df.tail(3)[['drug_a_id','drug_b_id','label']].values.tolist()}")
    print(f"Last 3 pairs in E3:       {e3_df.tail(3)[['drug_a_id','drug_b_id','lab']].values.tolist()}")
    print(f"Pair-54369 (boundary) feat: {feat_df.iloc[54369][['drug_a_id','drug_b_id','label']].tolist()}")
    print(f"Pair-54369 (boundary) E3:   {e3_df.iloc[54369][['drug_a_id','drug_b_id','lab']].tolist()}")

    # FINAL DECISION
    print("\n=== M3 verdict ===")
    if mismatches_e3_vs_exp3 == 0 and mismatches_exp3_vs_feat == 0 and h_e3 == h_exp3 == h_feat:
        print("PASS: pair iteration order is identical across E3 loader, exp3 loader, and pair_features.parquet.")
        print("      Cached predictions are correctly aligned with our features.")
    else:
        print("FAIL: pair order differs. Cached NLL values are misaligned!")


if __name__ == "__main__":
    main()

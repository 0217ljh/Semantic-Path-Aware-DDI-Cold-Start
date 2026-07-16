"""exp3 C4 step 2 — exploratory analysis of cross-modal disagreement boundary.

Hypothesis: when BOTH modalities (structure + text) show high similarity but
DISAGREE (rank gap), per-pair NLL is HIGHER than when both agree, even though
each modality individually says "supported".

Procedure:
1. Compute rank-percentile of S_st, S_tx across all pairs
2. Compute disagreement D_sttx = |rank(S_st) - rank(S_tx)|
3. 2D quadrant analysis: (high S_st, high S_tx) split into aligned vs discrepant
4. Within "both high" stratum, decile NLL by D_sttx → look for elbow

Print everything; save quadrant table & decile table to JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
df = pd.read_parquet(OUT_DIR / "pair_features.parquet")
print(f"[load] pairs={len(df)}")

# Drop rows with NaN features (S_kg has some); keep S_st/S_tx for primary
mask = df["S_st"].notna() & df["S_tx"].notna()
df = df[mask].copy()
print(f"[clean] after S_st & S_tx not-null: {len(df)}")

# Rank percentiles (0-1) by feature across all S2 pairs
def rank_pct(x: pd.Series) -> pd.Series:
    return x.rank(pct=True, method="average")

df["r_st"] = rank_pct(df["S_st"])
df["r_tx"] = rank_pct(df["S_tx"])
df["D_sttx"] = (df["r_st"] - df["r_tx"]).abs()

# By-label breakdown
print("\n[label balance & nll]")
print(df.groupby("label")["nll"].describe().T)

# Marginal: NLL by S_st quartile
print("\n[Marginal NLL by S_st quartile]")
df["q_st"] = pd.qcut(df["r_st"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
print(df.groupby("q_st", observed=True)["nll"].agg(["mean", "std", "count"]))

print("\n[Marginal NLL by S_tx quartile]")
df["q_tx"] = pd.qcut(df["r_tx"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
print(df.groupby("q_tx", observed=True)["nll"].agg(["mean", "std", "count"]))

# 2D quadrant: top 30% vs bottom 70% per modality
top_st = df["r_st"] >= 0.70
top_tx = df["r_tx"] >= 0.70
df["q2d"] = (
    np.where(top_st & top_tx, "HH",
             np.where(top_st & ~top_tx, "HL",
                      np.where(~top_st & top_tx, "LH", "LL")))
)
print("\n[2D quadrant — top30 cut]")
quad = df.groupby("q2d")["nll"].agg(["mean", "std", "count"])
quad["err_pos"] = df.groupby("q2d")["label"].mean()
print(quad)

# Within HH (both supported), is disagreement increasing NLL?
hh = df[df["q2d"] == "HH"].copy()
print(f"\n[HH subset] n={len(hh)}")
if len(hh) > 100:
    hh["D_dec"] = pd.qcut(hh["D_sttx"], 10, labels=False, duplicates="drop")
    decile_tbl = hh.groupby("D_dec").agg(
        D_mean=("D_sttx", "mean"),
        NLL_mean=("nll", "mean"),
        NLL_std=("nll", "std"),
        count=("nll", "count"),
        label_rate=("label", "mean"),
    )
    print("[HH decile of D_sttx → NLL]")
    print(decile_tbl)

# Also: in HH stratum, label-conditional NLL (since neg are easy when no similarity)
print("\n[HH subset: label-conditional NLL deciled by D_sttx]")
for lbl in [0, 1]:
    sub = hh[hh["label"] == lbl]
    if len(sub) < 200:
        print(f"  label={lbl} n={len(sub)} — skip")
        continue
    sub["D_dec"] = pd.qcut(sub["D_sttx"], 10, labels=False, duplicates="drop")
    t = sub.groupby("D_dec").agg(D_mean=("D_sttx", "mean"), NLL=("nll", "mean"), n=("nll", "count"))
    print(f"  label={lbl}")
    print(t)

# Correlation in HH between D_sttx and nll
from scipy.stats import spearmanr, pearsonr
for lbl in [None, 1, 0]:
    sub = hh if lbl is None else hh[hh["label"] == lbl]
    if len(sub) < 100:
        continue
    rho, p = spearmanr(sub["D_sttx"], sub["nll"])
    r, _ = pearsonr(sub["D_sttx"], sub["nll"])
    print(f"\n[HH lbl={lbl} n={len(sub)}] Spearman ρ={rho:.4f} p={p:.2e}, Pearson r={r:.4f}")

# Save
hh_dec_path = OUT_DIR / "_hh_disagreement_deciles.json"
out = {
    "quadrant": quad.to_dict(),
    "n_pairs": int(len(df)),
    "n_hh": int(len(hh)),
}
with open(hh_dec_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nsaved: {hh_dec_path}")

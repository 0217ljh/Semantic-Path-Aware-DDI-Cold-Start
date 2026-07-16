"""exp3 C4 step 2 (v2) — text × KG disagreement.

The E3 GCN uses KG+text features, not Morgan structure.
So 'structure × text' disagreement may be testing the wrong modality pair.
Switch primary to S_tx × S_kg.

Termination criteria (codex commit):
- ρ(D_txkg, NLL | label, HH) < 0.03
- decile curve no monotone / no elbow
- calibration flat
=> if all three hold, abandon C4 and switch to C2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
df = pd.read_parquet(OUT_DIR / "pair_features.parquet")
print(f"[load] pairs={len(df)}")

# Filter: require S_tx and S_kg
mask = df["S_tx"].notna() & df["S_kg"].notna()
df = df[mask].copy()
print(f"[clean] after S_tx & S_kg not-null: {len(df)}")

# Rank percentiles
df["r_tx"] = df["S_tx"].rank(pct=True, method="average")
df["r_kg"] = df["S_kg"].rank(pct=True, method="average")
df["D_txkg"] = (df["r_tx"] - df["r_kg"]).abs()

# Marginal
print("\n[Marginal NLL by S_tx quartile]")
df["q_tx"] = pd.qcut(df["r_tx"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
print(df.groupby("q_tx", observed=True)["nll"].agg(["mean", "std", "count"]))

# S_kg is heavily zero-inflated; check distribution first
print(f"\n[S_kg dist] zero pairs={int((df['S_kg']==0).sum())} ({(df['S_kg']==0).mean()*100:.1f}%)")
print(f"[S_kg dist] nonzero quantiles: q25={df.loc[df['S_kg']>0,'S_kg'].quantile(0.25):.4f} q50={df.loc[df['S_kg']>0,'S_kg'].quantile(0.5):.4f} q75={df.loc[df['S_kg']>0,'S_kg'].quantile(0.75):.4f}")
print(f"[r_kg dist] unique values approx (very few when many zeros): see dist")

print("\n[Marginal NLL by S_kg level (zero / low / mid / high)]")
def kg_bin(x):
    if x == 0: return "zero"
    if x < 0.05: return "low"
    if x < 0.2: return "mid"
    return "high"
df["q_kg"] = df["S_kg"].apply(kg_bin)
print(df.groupby("q_kg")["nll"].agg(["mean", "std", "count"]))

# Quadrant
top_tx = df["r_tx"] >= 0.70
top_kg = df["r_kg"] >= 0.70
df["q2d"] = (
    np.where(top_tx & top_kg, "HH",
             np.where(top_tx & ~top_kg, "HL",
                      np.where(~top_tx & top_kg, "LH", "LL")))
)
print("\n[2D quadrant tx×kg, top30 cut]")
quad = df.groupby("q2d").agg(
    NLL=("nll", "mean"),
    NLL_std=("nll", "std"),
    n=("nll", "count"),
    label_rate=("label", "mean"),
    pred_mean=("pred_p", "mean"),
    pred_std=("pred_p", "std"),
)
print(quad)

# HH stratum
hh = df[df["q2d"] == "HH"].copy()
print(f"\n[HH subset] n={len(hh)}")

if len(hh) > 100:
    hh["D_dec"] = pd.qcut(hh["D_txkg"], 10, labels=False, duplicates="drop")
    decile_tbl = hh.groupby("D_dec").agg(
        D_mean=("D_txkg", "mean"),
        NLL_mean=("nll", "mean"),
        NLL_std=("nll", "std"),
        n=("nll", "count"),
        label_rate=("label", "mean"),
        pred_p_mean=("pred_p", "mean"),
        pred_p_std=("pred_p", "std"),
    )
    print("[HH decile by D_txkg]")
    print(decile_tbl)

# Label-conditional within HH
print("\n[HH label-conditional NLL deciled by D_txkg]")
for lbl in [1, 0]:
    sub = hh[hh["label"] == lbl].copy()
    if len(sub) < 200:
        print(f"  label={lbl} n={len(sub)} skip")
        continue
    sub["D_dec"] = pd.qcut(sub["D_txkg"], 10, labels=False, duplicates="drop")
    t = sub.groupby("D_dec").agg(
        D_mean=("D_txkg", "mean"),
        NLL=("nll", "mean"),
        pred_p=("pred_p", "mean"),
        n=("nll", "count"),
    )
    print(f"  label={lbl}")
    print(t)
    rho_lbl, p_lbl = spearmanr(sub["D_txkg"], sub["nll"])
    r_lbl, _ = pearsonr(sub["D_txkg"], sub["nll"])
    print(f"  ρ_spearman={rho_lbl:.4f} p={p_lbl:.2e}, Pearson r={r_lbl:.4f}")

# Calibration: in HH, are predictions overconfident at high D?
print("\n[HH calibration check: pred_p variance by D_dec]")
hh["pos"] = (hh["label"] == 1).astype(int)
calib = hh.groupby("D_dec").agg(
    mean_pred=("pred_p", "mean"),
    mean_true=("pos", "mean"),
    p_var=("pred_p", "var"),
)
calib["cal_gap"] = (calib["mean_pred"] - calib["mean_true"]).abs()
print(calib)

# Final check: overall ρ within HH
rho_all, p_all = spearmanr(hh["D_txkg"], hh["nll"])
print(f"\n[HH all-label] ρ={rho_all:.4f} p={p_all:.2e}")

# Decision
print("\n=== Decision check ===")
ρ_pos, _ = spearmanr(hh[hh["label"] == 1]["D_txkg"], hh[hh["label"] == 1]["nll"])
ρ_neg, _ = spearmanr(hh[hh["label"] == 0]["D_txkg"], hh[hh["label"] == 0]["nll"])
print(f"ρ(D, NLL | label=1, HH) = {ρ_pos:.4f}")
print(f"ρ(D, NLL | label=0, HH) = {ρ_neg:.4f}")
print(f"Threshold to abandon: both |ρ| < 0.03 AND flat deciles")

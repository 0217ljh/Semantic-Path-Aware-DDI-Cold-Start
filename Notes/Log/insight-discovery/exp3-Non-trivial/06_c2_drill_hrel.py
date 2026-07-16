"""exp3 C2 step 3 — drill into H_rel finding.

Observed (step 2):
- Within Q3 (|M2|>30), H_rel shows U-shape with sharp cliff at decile 9
- Decile 0: NLL=0.60 (single template)
- Decile 1-3: NLL=0.52 (best, low-mid entropy)
- Decile 9: NLL=0.75 (CRASH, max entropy)
- label_rate also drops to 0.47 in decile 9

Question: is the decile 9 cliff a true phase transition, or just driven
by label_rate? Need label-conditional + control analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent
df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")
df = df[df["M2_count"] > 1].copy()

# Restrict to Q3 (mediator-rich)
q67 = df["M2_count"].quantile(0.67)
q3 = df[df["M2_count"] > q67].copy()
print(f"[Q3 rich subset] n={len(q3)}, |M2|>{q67:.0f}, label_rate={q3['label'].mean():.3f}")

# Detailed H_rel breakdown
q3["Hr_dec"] = pd.qcut(q3["H_rel"], 10, labels=False, duplicates="drop")

print("\n[H_rel decile: ALL labels]")
all_tbl = q3.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL=("nll", "mean"),
    pred_p=("pred_p", "mean"),
    n=("nll", "count"),
    label_rate=("label", "mean"),
    M2_mean=("M2_count", "mean"),
)
print(all_tbl)

# Label-conditional
print("\n[H_rel decile: label=1 ONLY]")
pos = q3[q3["label"] == 1].copy()
pos["Hr_dec"] = pd.qcut(pos["H_rel"], 10, labels=False, duplicates="drop")
pos_tbl = pos.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL=("nll", "mean"),
    pred_p=("pred_p", "mean"),
    n=("nll", "count"),
    M2_mean=("M2_count", "mean"),
)
print(pos_tbl)

print("\n[H_rel decile: label=0 ONLY]")
neg = q3[q3["label"] == 0].copy()
neg["Hr_dec"] = pd.qcut(neg["H_rel"], 10, labels=False, duplicates="drop")
neg_tbl = neg.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL=("nll", "mean"),
    pred_p=("pred_p", "mean"),
    n=("nll", "count"),
    M2_mean=("M2_count", "mean"),
)
print(neg_tbl)

# Correlations
print("\n[Correlations: H_rel vs NLL]")
for name, sub in [("ALL", q3), ("POS", pos), ("NEG", neg)]:
    rho, p = spearmanr(sub["H_rel"], sub["nll"])
    print(f"  {name} (n={len(sub)}): ρ={rho:.4f}, p={p:.2e}")

# Try: residual NLL after controlling for label and M2 count
# Using log-NLL ~ a + b*log(M2) + label*c + g(H_rel)
print("\n[Residualize NLL on (M2, label), then check H_rel effect]")
from sklearn.linear_model import LinearRegression

X = np.column_stack([np.log(q3["M2_count"].values), q3["label"].values, q3["label"].values * np.log(q3["M2_count"].values)])
y = q3["nll"].values
lr = LinearRegression().fit(X, y)
nll_pred = lr.predict(X)
q3["nll_resid"] = y - nll_pred
print(f"  baseline R^2: {lr.score(X, y):.4f}")

# Now decile by H_rel
print("\n[H_rel decile on residualized NLL]")
resid_tbl = q3.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL_resid=("nll_resid", "mean"),
    n=("nll", "count"),
)
print(resid_tbl)

# Test for nonlinearity: piecewise linear fit
from scipy.optimize import minimize_scalar

x = q3["H_rel"].values
y = q3["nll_resid"].values

def loss_breakpoint(tau):
    mask_l = x <= tau
    mask_r = ~mask_l
    if mask_l.sum() < 50 or mask_r.sum() < 50:
        return 1e6
    yl_mean = y[mask_l].mean()
    yr_mean = y[mask_r].mean()
    ssl = ((y[mask_l] - yl_mean) ** 2).sum()
    ssr = ((y[mask_r] - yr_mean) ** 2).sum()
    return ssl + ssr

x_range = (x.min() + 0.01, x.max() - 0.01)
res = minimize_scalar(loss_breakpoint, bounds=x_range, method="bounded")
tau_star = res.x
print(f"\n[Two-piece constant breakpoint] τ*={tau_star:.4f}")
mask_l = x <= tau_star
mask_r = ~mask_l
print(f"  left (H_rel ≤ {tau_star:.3f}): n={mask_l.sum()}, NLL_resid={y[mask_l].mean():.4f}")
print(f"  right(H_rel > {tau_star:.3f}): n={mask_r.sum()}, NLL_resid={y[mask_r].mean():.4f}")
print(f"  jump: {y[mask_r].mean() - y[mask_l].mean():.4f}")

# Within label=1 only, residualize on M2
print("\n[label=1: residualize NLL on log(M2), then decile by H_rel]")
lr_pos = LinearRegression().fit(np.log(pos["M2_count"].values).reshape(-1, 1), pos["nll"].values)
pos["nll_resid"] = pos["nll"].values - lr_pos.predict(np.log(pos["M2_count"].values).reshape(-1, 1))
print(pos.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL_resid=("nll_resid", "mean"),
    n=("nll", "count"),
))

# Same for label=0
print("\n[label=0: residualize NLL on log(M2), then decile by H_rel]")
lr_neg = LinearRegression().fit(np.log(neg["M2_count"].values).reshape(-1, 1), neg["nll"].values)
neg["nll_resid"] = neg["nll"].values - lr_neg.predict(np.log(neg["M2_count"].values).reshape(-1, 1))
print(neg.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL_resid=("nll_resid", "mean"),
    n=("nll", "count"),
))

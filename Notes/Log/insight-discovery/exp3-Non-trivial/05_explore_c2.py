"""exp3 C2 step 2 — explore mediator conflict transition.

Hypothesis: within mediator-rich pairs (Q3 of |M2|), high H_kind (mediator
kind-distribution entropy) predicts HIGHER NLL — i.e., diverse mediator support
hurts prediction more than concentrated support, despite having same count.

Procedure:
1. Stratify by |M2| tercile (Q1/Q2/Q3)
2. Within Q3 (mediator-rich), decile by H_kind → NLL
3. Label-conditional: same analysis per label
4. Robustness: H_rel and H_emb as alternative conflict measures
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
df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")
print(f"[load] pairs={len(df)}")

# Restrict to pairs with M2 > 1 (entropy meaningful)
df = df[df["M2_count"] > 1].copy()
print(f"[clean] pairs with M2>1: {len(df)}")

# |M2| terciles
q33 = df["M2_count"].quantile(0.33)
q67 = df["M2_count"].quantile(0.67)
df["M2_bin"] = np.where(df["M2_count"] <= q33, "Q1",
                np.where(df["M2_count"] <= q67, "Q2", "Q3"))
print(f"\n[M2 terciles] Q1≤{q33:.0f} | Q2≤{q67:.0f} | Q3>{q67:.0f}")
print(df.groupby("M2_bin").agg(n=("nll", "count"), M2_mean=("M2_count", "mean"), nll_mean=("nll", "mean"), label_rate=("label", "mean")))

# Marginal: does mediator count alone predict NLL?
print("\n[Marginal: |M2| vs NLL by tercile]")
print(df.groupby("M2_bin")["nll"].agg(["mean", "std", "count"]))

# Within Q3 (rich), decile H_kind
q3 = df[df["M2_bin"] == "Q3"].copy()
print(f"\n[Q3 rich] n={len(q3)}")
print(f"  H_kind quartiles: {q3['H_kind'].quantile([0.25, 0.5, 0.75]).to_dict()}")
print(f"  H_kind=0 (single-kind): {int((q3['H_kind']==0).sum())} ({(q3['H_kind']==0).mean()*100:.1f}%)")

# Deciles
q3["Hk_dec"] = pd.qcut(q3["H_kind"], 10, labels=False, duplicates="drop")
print("\n[Q3 H_kind decile → NLL]")
print(q3.groupby("Hk_dec").agg(
    Hk_mean=("H_kind", "mean"),
    NLL_mean=("nll", "mean"),
    NLL_std=("nll", "std"),
    n=("nll", "count"),
    label_rate=("label", "mean"),
    pred_p_mean=("pred_p", "mean"),
))

# Label-conditional
print("\n[Q3 label-conditional NLL by H_kind decile]")
for lbl in [1, 0]:
    sub = q3[q3["label"] == lbl].copy()
    if len(sub) < 200:
        continue
    sub["Hk_dec"] = pd.qcut(sub["H_kind"], 10, labels=False, duplicates="drop")
    t = sub.groupby("Hk_dec").agg(
        Hk_mean=("H_kind", "mean"),
        NLL=("nll", "mean"),
        pred=("pred_p", "mean"),
        n=("nll", "count"),
    )
    print(f"label={lbl} (n={len(sub)}):")
    print(t)
    rho, p = spearmanr(sub["H_kind"], sub["nll"])
    r, _ = pearsonr(sub["H_kind"], sub["nll"])
    print(f"  ρ_spearman={rho:.4f} p={p:.2e}, Pearson r={r:.4f}")

# Robustness: H_rel, H_emb
print("\n[Q3 H_rel decile → NLL]")
q3["Hr_dec"] = pd.qcut(q3["H_rel"], 10, labels=False, duplicates="drop")
print(q3.groupby("Hr_dec").agg(
    Hr_mean=("H_rel", "mean"),
    NLL=("nll", "mean"),
    n=("nll", "count"),
    label_rate=("label", "mean"),
))

q3_emb = q3[q3["H_emb"].notna()].copy()
if len(q3_emb) > 200:
    q3_emb["He_dec"] = pd.qcut(q3_emb["H_emb"], 10, labels=False, duplicates="drop")
    print("\n[Q3 H_emb decile → NLL]")
    print(q3_emb.groupby("He_dec").agg(
        He_mean=("H_emb", "mean"),
        NLL=("nll", "mean"),
        n=("nll", "count"),
        label_rate=("label", "mean"),
    ))

# Q3 label-conditional summary correlations
print("\n=== Q3 stratified correlations ===")
for col in ["H_kind", "H_rel", "H_emb"]:
    sub = q3.dropna(subset=[col])
    if len(sub) < 100:
        continue
    rho_all, _ = spearmanr(sub[col], sub["nll"])
    pos = sub[sub["label"] == 1]
    neg = sub[sub["label"] == 0]
    rho_pos, _ = spearmanr(pos[col], pos["nll"]) if len(pos) > 50 else (np.nan, None)
    rho_neg, _ = spearmanr(neg[col], neg["nll"]) if len(neg) > 50 else (np.nan, None)
    print(f"  {col}: ρ(all)={rho_all:.4f}, ρ(pos)={rho_pos:.4f}, ρ(neg)={rho_neg:.4f}, n={len(sub)}")

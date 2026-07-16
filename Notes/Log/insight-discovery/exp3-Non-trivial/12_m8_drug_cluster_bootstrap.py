"""exp3 M8 fix — drug-cluster bootstrap + leave-one-drug-out.

Codex critique: 108k pairs from only 634 drugs → pair-level bootstrap is
massively over-optimistic. Resample by DRUG, not by pair.

Procedure:
1. Sample 634 drugs with replacement
2. Take all pairs both of whose drugs are in the sample
3. Compute POS decile NLL_resid, NEG decile NLL_resid
4. Repeat 500 times
5. Check if POS U-shape and NEG inverted-U still hold under drug-cluster bootstrap

Also: leave-one-drug-out for top contributors.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent

df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")
df = df[df["M2_count"] > 1].copy()
q67 = df["M2_count"].quantile(0.67)
q3 = df[df["M2_count"] > q67].copy()
print(f"[Q3] n={len(q3)}, pos={(q3['label']==1).sum()}, neg={(q3['label']==0).sum()}")

unique_drugs = sorted(set(q3["drug_a_id"]) | set(q3["drug_b_id"]))
print(f"[Q3] unique drugs: {len(unique_drugs)}")

# Diagnostic: which drugs appear most often in decile-9 high-NLL_resid pairs?
pos = q3[q3["label"] == 1].copy()
neg = q3[q3["label"] == 0].copy()
lr_pos = LinearRegression().fit(np.log(pos["M2_count"].values).reshape(-1, 1), pos["nll"].values)
pos["nll_resid"] = pos["nll"].values - lr_pos.predict(np.log(pos["M2_count"].values).reshape(-1, 1))
pos["Hr_dec"] = pd.qcut(pos["H_rel"], 10, labels=False, duplicates="drop")
lr_neg = LinearRegression().fit(np.log(neg["M2_count"].values).reshape(-1, 1), neg["nll"].values)
neg["nll_resid"] = neg["nll"].values - lr_neg.predict(np.log(neg["M2_count"].values).reshape(-1, 1))
neg["Hr_dec"] = pd.qcut(neg["H_rel"], 10, labels=False, duplicates="drop")

# === Drug contributions in decile 9 ===
d9 = pos[pos["Hr_dec"] == 9]
all_drugs = list(d9["drug_a_id"]) + list(d9["drug_b_id"])
top_d9_drugs = Counter(all_drugs).most_common(10)
print(f"\n[decile-9 POS pairs] top contributing drugs (out of {len(d9)} pairs, {len(set(all_drugs))} unique drugs):")
for d, c in top_d9_drugs:
    pct = c / len(d9) * 100
    print(f"  {d}: {c} pairs ({pct:.1f}% of decile-9 pos)")

# === Drug-cluster bootstrap ===
print("\n=== Drug-cluster bootstrap (500 reps) ===")
rng = np.random.default_rng(42)
n_boot = 500
drugs_arr = np.array(unique_drugs)
n_drugs = len(unique_drugs)

# Use drug index lookup for fast intersection
drug_set_lookups = []
pos_d1 = pos["drug_a_id"].values
pos_d2 = pos["drug_b_id"].values
neg_d1 = neg["drug_a_id"].values
neg_d2 = neg["drug_b_id"].values
pos_dec = pos["Hr_dec"].values
neg_dec = neg["Hr_dec"].values
pos_resid = pos["nll_resid"].values
neg_resid = neg["nll_resid"].values

results_pos = np.full((n_boot, 10), np.nan)
results_neg = np.full((n_boot, 10), np.nan)
for b in range(n_boot):
    sampled = set(rng.choice(drugs_arr, size=n_drugs, replace=True))
    mask_p = np.array([(a in sampled and b_ in sampled) for a, b_ in zip(pos_d1, pos_d2)])
    mask_n = np.array([(a in sampled and b_ in sampled) for a, b_ in zip(neg_d1, neg_d2)])
    for d in range(10):
        mp = (pos_dec == d) & mask_p
        mn = (neg_dec == d) & mask_n
        if mp.sum() >= 20:
            results_pos[b, d] = pos_resid[mp].mean()
        if mn.sum() >= 20:
            results_neg[b, d] = neg_resid[mn].mean()
    if (b + 1) % 50 == 0:
        print(f"  bootstrap {b+1}/{n_boot} pos_sample_size_q3={mask_p.sum()}")

# Stats
print("\n[Drug-cluster bootstrap deciles]")
for d in range(10):
    pm = np.nanmean(results_pos[:, d])
    plo = np.nanpercentile(results_pos[:, d], 2.5)
    phi = np.nanpercentile(results_pos[:, d], 97.5)
    nm = np.nanmean(results_neg[:, d])
    nlo = np.nanpercentile(results_neg[:, d], 2.5)
    nhi = np.nanpercentile(results_neg[:, d], 97.5)
    print(f"  dec={d}: pos={pm:+.4f} [{plo:+.4f}, {phi:+.4f}]  neg={nm:+.4f} [{nlo:+.4f}, {nhi:+.4f}]")

# U-shape tests under drug-cluster bootstrap
pos_u = (results_pos[:, 0] > results_pos[:, 5]) & (results_pos[:, 9] > results_pos[:, 5])
neg_inv_u = (results_neg[:, 5] > results_neg[:, 0]) & (results_neg[:, 5] > results_neg[:, 9])
jump_89 = results_pos[:, 9] - results_pos[:, 8]
jump_01 = results_pos[:, 0] - results_pos[:, 1]

print("\n[U-shape stability under DRUG-cluster bootstrap]")
print(f"  POS U-shape (dec0 AND dec9 > dec5): {np.nanmean(pos_u)*100:.1f}% of bootstraps (was 100% under pair-bootstrap)")
print(f"  NEG inverted-U: {np.nanmean(neg_inv_u)*100:.1f}%")
print(f"  POS dec9-8 jump: mean={np.nanmean(jump_89):+.4f}, CI=[{np.nanpercentile(jump_89, 2.5):+.4f}, {np.nanpercentile(jump_89, 97.5):+.4f}]")
print(f"  POS dec0-1 jump: mean={np.nanmean(jump_01):+.4f}, CI=[{np.nanpercentile(jump_01, 2.5):+.4f}, {np.nanpercentile(jump_01, 97.5):+.4f}]")

# === Leave-top-K-drugs out ===
print("\n=== Leave-top-K-drugs-out ===")
all_q3_drugs = list(q3["drug_a_id"]) + list(q3["drug_b_id"])
top_drugs = [d for d, _ in Counter(all_q3_drugs).most_common(30)]
print(f"Top 30 most-frequent drugs in Q3 (across all pairs):")
for d in top_drugs[:5]:
    cnt = sum(1 for x in all_q3_drugs if x == d)
    print(f"  {d}: {cnt} pair-appearances")

for k in [1, 3, 5, 10]:
    drop = set(top_drugs[:k])
    mask = ~(q3["drug_a_id"].isin(drop) | q3["drug_b_id"].isin(drop))
    sub = q3[mask].copy()
    pos_s = sub[sub["label"] == 1].copy()
    neg_s = sub[sub["label"] == 0].copy()
    if len(pos_s) < 1000 or len(neg_s) < 500:
        print(f"  k={k}: sample too small after drop (pos={len(pos_s)}, neg={len(neg_s)})")
        continue
    lr_p = LinearRegression().fit(np.log(pos_s["M2_count"].values).reshape(-1, 1), pos_s["nll"].values)
    pos_s["resid"] = pos_s["nll"].values - lr_p.predict(np.log(pos_s["M2_count"].values).reshape(-1, 1))
    pos_s["Hr_dec"] = pd.qcut(pos_s["H_rel"], 10, labels=False, duplicates="drop")
    lr_n = LinearRegression().fit(np.log(neg_s["M2_count"].values).reshape(-1, 1), neg_s["nll"].values)
    neg_s["resid"] = neg_s["nll"].values - lr_n.predict(np.log(neg_s["M2_count"].values).reshape(-1, 1))
    neg_s["Hr_dec"] = pd.qcut(neg_s["H_rel"], 10, labels=False, duplicates="drop")
    p_dec = pos_s.groupby("Hr_dec")["resid"].mean()
    n_dec = neg_s.groupby("Hr_dec")["resid"].mean()
    u_pos = (p_dec.iloc[0] > p_dec.iloc[5]) and (p_dec.iloc[9] > p_dec.iloc[5])
    inv_n = (n_dec.iloc[5] > n_dec.iloc[0]) and (n_dec.iloc[5] > n_dec.iloc[9])
    j_89 = p_dec.iloc[9] - p_dec.iloc[8]
    j_01 = p_dec.iloc[0] - p_dec.iloc[1]
    print(f"  k={k} drugs dropped (drop {len(drop)} ids), pairs left: {mask.sum()}")
    print(f"    POS U-shape: {u_pos}, dec9-8 jump: {j_89:+.4f}, dec0-1: {j_01:+.4f}")
    print(f"    NEG inv-U: {inv_n}")

# Save
import json
res = {
    "drug_cluster_bootstrap": {
        "pos_U_pct": float(np.nanmean(pos_u) * 100),
        "neg_invU_pct": float(np.nanmean(neg_inv_u) * 100),
        "jump_89_mean": float(np.nanmean(jump_89)),
        "jump_89_ci": [float(np.nanpercentile(jump_89, 2.5)), float(np.nanpercentile(jump_89, 97.5))],
        "jump_01_mean": float(np.nanmean(jump_01)),
        "jump_01_ci": [float(np.nanpercentile(jump_01, 2.5)), float(np.nanpercentile(jump_01, 97.5))],
        "n_boot": n_boot,
    },
    "decile_9_top_drugs": [(d, c) for d, c in top_d9_drugs],
}
with open(OUT_DIR / "m8_drug_cluster_results.json", "w") as f:
    json.dump(res, f, indent=2)
print(f"\nsaved: {OUT_DIR/'m8_drug_cluster_results.json'}")

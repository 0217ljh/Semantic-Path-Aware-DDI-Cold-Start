"""exp3 C2 step 4 — robustness checks per codex.

(R1) Bootstrap CI on pos decile-8 vs decile-9 jump
(R2) Instance audit: dump example pairs from decile 0 / 5 / 9 with templates
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")
df = df[df["M2_count"] > 1].copy()

q67 = df["M2_count"].quantile(0.67)
q3 = df[df["M2_count"] > q67].copy()
pos = q3[q3["label"] == 1].copy()
neg = q3[q3["label"] == 0].copy()
print(f"[Q3 rich] n={len(q3)} pos={len(pos)} neg={len(neg)}")

# Residualize pos NLL on log(M2)
from sklearn.linear_model import LinearRegression
lr_pos = LinearRegression().fit(np.log(pos["M2_count"].values).reshape(-1, 1), pos["nll"].values)
pos["nll_resid"] = pos["nll"].values - lr_pos.predict(np.log(pos["M2_count"].values).reshape(-1, 1))
pos["Hr_dec"] = pd.qcut(pos["H_rel"], 10, labels=False, duplicates="drop")

lr_neg = LinearRegression().fit(np.log(neg["M2_count"].values).reshape(-1, 1), neg["nll"].values)
neg["nll_resid"] = neg["nll"].values - lr_neg.predict(np.log(neg["M2_count"].values).reshape(-1, 1))
neg["Hr_dec"] = pd.qcut(neg["H_rel"], 10, labels=False, duplicates="drop")

# ============= R1: Bootstrap CI =================
print("\n=== R1: Bootstrap 1000 CIs ===")
rng = np.random.default_rng(42)
n_boot = 1000

def boot_decile_means(sub: pd.DataFrame, col: str = "nll_resid", n_dec: int = 10):
    """Returns matrix of shape (n_boot, n_dec) of bootstrap mean per decile."""
    means = np.zeros((n_boot, n_dec))
    n = len(sub)
    sub_arr = sub[col].values
    sub_dec = sub["Hr_dec"].values
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bs_dec = sub_dec[idx]
        bs_val = sub_arr[idx]
        for d in range(n_dec):
            m = bs_dec == d
            means[b, d] = bs_val[m].mean() if m.any() else np.nan
    return means

print("[pos] bootstrapping pos deciles ...")
pos_boot = boot_decile_means(pos)
print("[neg] bootstrapping neg deciles ...")
neg_boot = boot_decile_means(neg)

for d in range(10):
    pm = pos_boot[:, d]
    nm = neg_boot[:, d]
    print(f"  dec={d}: pos NLL_resid={np.nanmean(pm):+.4f} [{np.nanpercentile(pm, 2.5):+.4f}, {np.nanpercentile(pm, 97.5):+.4f}]"
          f"  | neg={np.nanmean(nm):+.4f} [{np.nanpercentile(nm, 2.5):+.4f}, {np.nanpercentile(nm, 97.5):+.4f}]")

# Jump tests
print("\n[Jump tests]")
jumps_pos_89 = pos_boot[:, 9] - pos_boot[:, 8]
print(f"  POS decile 9-8 jump: mean={np.nanmean(jumps_pos_89):+.4f} CI=[{np.nanpercentile(jumps_pos_89, 2.5):+.4f}, {np.nanpercentile(jumps_pos_89, 97.5):+.4f}]")
jumps_pos_01 = pos_boot[:, 0] - pos_boot[:, 1]
print(f"  POS decile 0-1 jump: mean={np.nanmean(jumps_pos_01):+.4f} CI=[{np.nanpercentile(jumps_pos_01, 2.5):+.4f}, {np.nanpercentile(jumps_pos_01, 97.5):+.4f}]")

# Test for U-shape: pos decile-0 + decile-9 should both be higher than mid (5)
u_pos = ((pos_boot[:, 0] - pos_boot[:, 5]) > 0) & ((pos_boot[:, 9] - pos_boot[:, 5]) > 0)
print(f"  POS U-shape (both ends > middle): {u_pos.mean()*100:.1f}% of bootstraps")

# Test for inverted-U for neg
inv_u_neg = ((neg_boot[:, 5] - neg_boot[:, 0]) > 0) & ((neg_boot[:, 5] - neg_boot[:, 9]) > 0)
print(f"  NEG inverted-U (middle > both ends): {inv_u_neg.mean()*100:.1f}% of bootstraps")

# Mirror symmetry: in mid deciles, pos NLL_resid and neg NLL_resid should be opposite signs
sign_mismatch = ((pos_boot[:, 1:8].mean(axis=1) < 0) & (neg_boot[:, 1:8].mean(axis=1) > 0))
print(f"  Mirror pattern in mid (pos<0 & neg>0): {sign_mismatch.mean()*100:.1f}% of bootstraps")

# ============= R2: Instance audit =================
print("\n=== R2: Instance audit ===")
# Need to recover template histograms for sampled pairs
print("Loading edges + nodes for template recovery ...")
edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
KIND_CANON = {
    "Drug": "Drug", "drug": "Drug",
    "Protein": "Protein", "gene/protein": "Protein",
    "Gene": "Gene",
    "Side Effect": "SideEffect", "effect/phenotype": "Phenotype",
    "Disease": "Disease", "disease": "Disease",
    "Pathway": "Pathway", "pathway": "Pathway",
    "Anatomy": "Anatomy", "anatomy": "Anatomy",
    "biological_process": "BioProcess", "Biological Process": "BioProcess",
    "molecular_function": "MolFunction", "Molecular Function": "MolFunction",
    "cellular_component": "CellComp",
}
nodes["kind_canon"] = nodes["kind"].map(KIND_CANON).fillna("Other")
id2kind = dict(zip(nodes["id"], nodes["kind_canon"]))
id2name = dict(zip(nodes["id"], nodes["name"]))

MEDIATOR_KINDS = ["Protein", "Gene", "SideEffect", "Phenotype", "Disease", "Pathway", "Anatomy", "BioProcess", "MolFunction"]
drug_set = set(df["drug_a_id"]) | set(df["drug_b_id"])

e1 = edges[edges["src"].isin(drug_set)][["src", "dst", "relation"]].rename(columns={"src": "d", "dst": "m", "relation": "r"})
e2 = edges[edges["dst"].isin(drug_set)][["dst", "src", "relation"]].rename(columns={"dst": "d", "src": "m", "relation": "r"})
drug_edges = pd.concat([e1, e2], ignore_index=True)
drug_edges["m_kind"] = drug_edges["m"].map(id2kind)
drug_edges = drug_edges[drug_edges["m_kind"].isin(MEDIATOR_KINDS)]

drug_to_med: dict[str, dict[str, list[str]]] = {}
for d, grp in drug_edges.groupby("d"):
    m_to_rels = {m: list(sub["r"]) for m, sub in grp.groupby("m")}
    drug_to_med[d] = m_to_rels

def template_hist(a: str, b: str) -> Counter:
    ma = drug_to_med.get(a, {})
    mb = drug_to_med.get(b, {})
    shared = set(ma) & set(mb)
    tcnt = Counter()
    for m in shared:
        ra = ma[m][0]
        rb = mb[m][0]
        tcnt[(ra, rb, id2kind.get(m, "?"))] += 1
    return tcnt

# Sample pairs from pos decile 0, 5, 9
print("\n--- POS decile 0 (singleton template, high NLL_resid): ---")
sample = pos[pos["Hr_dec"] == 0].nlargest(3, "nll_resid")
for _, row in sample.iterrows():
    a, b = row["drug_a_id"], row["drug_b_id"]
    print(f"\n  Pair: {a}({id2name.get(a,'?')[:20]}) × {b}({id2name.get(b,'?')[:20]}) | M2={row['M2_count']} H_rel={row['H_rel']:.3f} NLL_resid={row['nll_resid']:+.3f}")
    th = template_hist(a, b)
    for t, c in th.most_common(5):
        print(f"    template={t} count={c}")

print("\n--- POS decile 5 (mid, low NLL_resid, best): ---")
sample = pos[pos["Hr_dec"] == 5].nsmallest(3, "nll_resid")
for _, row in sample.iterrows():
    a, b = row["drug_a_id"], row["drug_b_id"]
    print(f"\n  Pair: {a}({id2name.get(a,'?')[:20]}) × {b}({id2name.get(b,'?')[:20]}) | M2={row['M2_count']} H_rel={row['H_rel']:.3f} NLL_resid={row['nll_resid']:+.3f}")
    th = template_hist(a, b)
    for t, c in th.most_common(5):
        print(f"    template={t} count={c}")

print("\n--- POS decile 9 (max-entropy, high NLL_resid): ---")
sample = pos[pos["Hr_dec"] == 9].nlargest(3, "nll_resid")
for _, row in sample.iterrows():
    a, b = row["drug_a_id"], row["drug_b_id"]
    print(f"\n  Pair: {a}({id2name.get(a,'?')[:20]}) × {b}({id2name.get(b,'?')[:20]}) | M2={row['M2_count']} H_rel={row['H_rel']:.3f} NLL_resid={row['nll_resid']:+.3f}")
    th = template_hist(a, b)
    print(f"    template histogram size={len(th)} (top 8):")
    for t, c in th.most_common(8):
        print(f"    template={t} count={c}")

# Save bootstrap stats
import json
stats = {
    "pos_deciles": {
        "mean": [float(np.nanmean(pos_boot[:, d])) for d in range(10)],
        "ci_low": [float(np.nanpercentile(pos_boot[:, d], 2.5)) for d in range(10)],
        "ci_high": [float(np.nanpercentile(pos_boot[:, d], 97.5)) for d in range(10)],
    },
    "neg_deciles": {
        "mean": [float(np.nanmean(neg_boot[:, d])) for d in range(10)],
        "ci_low": [float(np.nanpercentile(neg_boot[:, d], 2.5)) for d in range(10)],
        "ci_high": [float(np.nanpercentile(neg_boot[:, d], 97.5)) for d in range(10)],
    },
    "pos_u_shape_pct": float(u_pos.mean() * 100),
    "neg_inv_u_pct": float(inv_u_neg.mean() * 100),
    "mirror_pct": float(sign_mismatch.mean() * 100),
    "pos_89_jump_mean": float(np.nanmean(jumps_pos_89)),
    "pos_89_jump_ci": [float(np.nanpercentile(jumps_pos_89, 2.5)), float(np.nanpercentile(jumps_pos_89, 97.5))],
}
with open(OUT_DIR / "c2_robustness.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nsaved: {OUT_DIR/'c2_robustness.json'}")

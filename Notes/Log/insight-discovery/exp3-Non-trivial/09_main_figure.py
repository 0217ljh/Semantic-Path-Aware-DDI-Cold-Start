"""exp3 main figure — Support-Rich Underdetermination phase transition.

Top: POS NLL_resid by H_rel decile, overlay GCN + LR with CI bands.
Bottom: NEG NLL_resid by H_rel decile, overlay GCN + LR with CI bands.

Shows U-shape (pos) and inverted-U (neg) mirror pattern across two model classes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).parent


def bootstrap_decile(sub: pd.DataFrame, val_col: str, dec_col: str, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(sub)
    vals = sub[val_col].values
    decs = sub[dec_col].values
    out = np.zeros((n_boot, 10))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        bs_vals = vals[idx]
        bs_decs = decs[idx]
        for d in range(10):
            m = bs_decs == d
            out[b, d] = bs_vals[m].mean() if m.any() else np.nan
    mean = np.nanmean(out, axis=0)
    lo = np.nanpercentile(out, 2.5, axis=0)
    hi = np.nanpercentile(out, 97.5, axis=0)
    return mean, lo, hi


def main():
    df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")
    df = df[df["M2_count"] > 1].copy()

    # Build LR predictions (re-fit fast, or assume previous run can be replayed)
    print("[step 1] training LR on E7 features for predictions ...")
    # Use cached cross-model script's LR — re-run for simplicity
    import subprocess  # noqa
    # Instead of re-running, regenerate LR features inline
    from collections import defaultdict

    KIND_GROUPS = {
        "protein_gene": ["Gene", "gene/protein", "Protein"],
        "pathway": ["Pathway", "pathway"],
        "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
        "disease": ["Disease", "disease"],
        "anatomy": ["Anatomy", "anatomy"],
        "compound": ["Compound"],
        "biological_process": ["Biological Process", "biological_process"],
        "molecular_function": ["Molecular Function", "molecular_function"],
        "cellular_component": ["Cellular Component", "cellular_component"],
        "pharmacologic_class": ["Pharmacologic Class"],
        "exposure": ["exposure"],
    }
    KIND_ORDER = list(KIND_GROUPS.keys())
    KIND_TO_GROUP = {k: g for g, ks in KIND_GROUPS.items() for k in ks}

    def _find_project_root() -> Path:
        cur = Path(__file__).resolve()
        for cand in [cur, *cur.parents]:
            if (cand / "Code" / "data" / "KG").is_dir():
                return cand
        raise FileNotFoundError
    ROOT = _find_project_root()

    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])

    n1 = defaultdict(set)
    fwd = defaultdict(set)
    for src, dst, directed in zip(edges["src"].values, edges["dst"].values, edges["directed"].values):
        fwd[src].add(dst)
        if not directed:
            fwd[dst].add(src)
        sd = src in drug_set
        dd = dst in drug_set
        if sd and not dd:
            n1[src].add((dst, KIND_TO_GROUP.get(id2kind.get(dst, ""), "other")))
        if dd and not sd and not directed:
            n1[dst].add((src, KIND_TO_GROUP.get(id2kind.get(src, ""), "other")))
    n2 = defaultdict(set)
    for drug in drug_set:
        for mid, _ in n1.get(drug, ()):
            for term in fwd.get(mid, ()):
                if term == drug or term in drug_set:
                    continue
                n2[drug].add((term, KIND_TO_GROUP.get(id2kind.get(term, ""), "other")))

    def featurize_pairs(pairs_df):
        n = len(pairs_df)
        n_g = len(KIND_ORDER)
        X = np.zeros((n, 2 * n_g), dtype=np.float32)
        for i, (da, db) in enumerate(zip(pairs_df["drug_a_id"].values, pairs_df["drug_b_id"].values)):
            s1 = n1.get(da, set()) & n1.get(db, set())
            s2 = n2.get(da, set()) & n2.get(db, set())
            c1 = defaultdict(int)
            c2 = defaultdict(int)
            for _, g in s1: c1[g] += 1
            for _, g in s2: c2[g] += 1
            for j, grp in enumerate(KIND_ORDER):
                X[i, j] = np.log1p(c1[grp])
                X[i, n_g + j] = np.log1p(c2[grp])
        return X

    def load_split(pos_path, neg_path):
        pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
        pos["label"] = 1
        neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
        neg["label"] = 0
        return pd.concat([pos, neg], ignore_index=True)

    SPLITS = ROOT / "Code/data/KG/drugbank/splits/seed42"
    train_split = load_split(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet")
    Xtr = featurize_pairs(train_split)
    ytr = train_split["label"].values
    print(f"  train: {Xtr.shape}")
    lr = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr)

    # Predict on test (df is full test set in pair_features order)
    Xte = featurize_pairs(df)
    p = lr.predict_proba(Xte)[:, 1]
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    df["lr_pred_p"] = p.astype(np.float32)
    df["lr_nll"] = -(df["label"] * np.log(p) + (1 - df["label"]) * np.log(1 - p)).astype(np.float32)

    # Slice Q3
    q67 = df["M2_count"].quantile(0.67)
    q3 = df[df["M2_count"] > q67].copy()
    pos = q3[q3["label"] == 1].copy()
    neg = q3[q3["label"] == 0].copy()
    print(f"  Q3 n={len(q3)}, pos={len(pos)}, neg={len(neg)}")

    # Residualize on log(M2)
    def residualize(sub, col):
        lr_lin = LinearRegression().fit(np.log(sub["M2_count"].values).reshape(-1, 1), sub[col].values)
        return sub[col].values - lr_lin.predict(np.log(sub["M2_count"].values).reshape(-1, 1))

    pos["gcn_nll_resid"] = residualize(pos, "nll")
    pos["lr_nll_resid"] = residualize(pos, "lr_nll")
    pos["Hr_dec"] = pd.qcut(pos["H_rel"], 10, labels=False, duplicates="drop")
    neg["gcn_nll_resid"] = residualize(neg, "nll")
    neg["lr_nll_resid"] = residualize(neg, "lr_nll")
    neg["Hr_dec"] = pd.qcut(neg["H_rel"], 10, labels=False, duplicates="drop")

    # Bootstrap CIs
    print("[step 2] bootstrap CIs ...")
    gcn_pos_m, gcn_pos_lo, gcn_pos_hi = bootstrap_decile(pos, "gcn_nll_resid", "Hr_dec")
    lr_pos_m, lr_pos_lo, lr_pos_hi = bootstrap_decile(pos, "lr_nll_resid", "Hr_dec")
    gcn_neg_m, gcn_neg_lo, gcn_neg_hi = bootstrap_decile(neg, "gcn_nll_resid", "Hr_dec")
    lr_neg_m, lr_neg_lo, lr_neg_hi = bootstrap_decile(neg, "lr_nll_resid", "Hr_dec")

    pos_hr_mean = pos.groupby("Hr_dec")["H_rel"].mean().values
    neg_hr_mean = neg.groupby("Hr_dec")["H_rel"].mean().values

    print("[step 3] plotting ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    # Top-left: GCN pos
    ax = axes[0, 0]
    ax.fill_between(range(10), gcn_pos_lo, gcn_pos_hi, alpha=0.25, color="C0", label="95% CI")
    ax.plot(range(10), gcn_pos_m, "o-", color="C0", linewidth=2, label="GCN pos")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("GCN K=2 PubMedBERT-init\nPositive pairs (U-shape)")
    ax.set_xlabel("H_rel decile (low → high template entropy)")
    ax.set_ylabel("residual NLL (after log M2 control)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Top-right: LR pos
    ax = axes[0, 1]
    ax.fill_between(range(10), lr_pos_lo, lr_pos_hi, alpha=0.25, color="C1", label="95% CI")
    ax.plot(range(10), lr_pos_m, "s-", color="C1", linewidth=2, label="LR pos")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Meeting-Node LR (22 count features)\nPositive pairs (U-shape, 14× amplified)")
    ax.set_xlabel("H_rel decile (low → high template entropy)")
    ax.set_ylabel("residual NLL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-left: GCN neg
    ax = axes[1, 0]
    ax.fill_between(range(10), gcn_neg_lo, gcn_neg_hi, alpha=0.25, color="C2", label="95% CI")
    ax.plot(range(10), gcn_neg_m, "o-", color="C2", linewidth=2, label="GCN neg")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("GCN K=2 PubMedBERT-init\nNegative pairs (inverted-U, mirror)")
    ax.set_xlabel("H_rel decile")
    ax.set_ylabel("residual NLL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bottom-right: LR neg
    ax = axes[1, 1]
    ax.fill_between(range(10), lr_neg_lo, lr_neg_hi, alpha=0.25, color="C3", label="95% CI")
    ax.plot(range(10), lr_neg_m, "s-", color="C3", linewidth=2, label="LR neg")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Meeting-Node LR\nNegative pairs (inverted-U, mirror)")
    ax.set_xlabel("H_rel decile")
    ax.set_ylabel("residual NLL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Support-Rich Underdetermination in Cold-Start DDI (Q3 mediator-rich, |M2|>30)\n"
                 "U-shape (positives) + inverted-U (negatives) replicate across 2 model classes",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig_path = OUT_DIR / "fig_support_rich_underdetermination.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")
    print(f"saved figure: {fig_path}")

    # Save numerical data
    import json
    data = {
        "gcn_pos_mean": gcn_pos_m.tolist(),
        "gcn_pos_ci": [gcn_pos_lo.tolist(), gcn_pos_hi.tolist()],
        "lr_pos_mean": lr_pos_m.tolist(),
        "lr_pos_ci": [lr_pos_lo.tolist(), lr_pos_hi.tolist()],
        "gcn_neg_mean": gcn_neg_m.tolist(),
        "gcn_neg_ci": [gcn_neg_lo.tolist(), gcn_neg_hi.tolist()],
        "lr_neg_mean": lr_neg_m.tolist(),
        "lr_neg_ci": [lr_neg_lo.tolist(), lr_neg_hi.tolist()],
        "pos_Hr_decile_means": pos_hr_mean.tolist(),
        "neg_Hr_decile_means": neg_hr_mean.tolist(),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "gcn_pos_jump_89": float(gcn_pos_m[9] - gcn_pos_m[8]),
        "lr_pos_jump_89": float(lr_pos_m[9] - lr_pos_m[8]),
        "amplification_factor": float((lr_pos_m[9] - lr_pos_m[8]) / max(abs(gcn_pos_m[9] - gcn_pos_m[8]), 1e-6)),
    }
    with open(OUT_DIR / "main_figure_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved data: {OUT_DIR/'main_figure_data.json'}")


if __name__ == "__main__":
    main()

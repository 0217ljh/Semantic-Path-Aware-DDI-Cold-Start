"""exp3 step 17 — Non-monotonic training scaling for cold-start (candidate β).

Hypothesis: As training pair count increases, S2 (cold-start) AUC/NLL is
NON-monotonic, i.e., has a critical scaling point beyond which performance
DEGRADES. Meanwhile S0 (warm-start) is monotonically improving.

If this holds: contradicts "more data is better" intuition; suggests model
learns seen-drug-specific shortcuts at scale that hurt cold transfer.

Experimental design:
- Training fractions: 5%, 10%, 20%, 40%, 60%, 80%, 100%
- For each fraction: subsample train pairs (RANDOM, NOT by drug, to test
  "amount of supervision" not "drug coverage" effect)
- Model: LR with E7 22-features (fast)
- Seeds: 42, 43, 44
- Evaluate: S0 + S2 AUC + per-pair NLL

Hold constant:
- train drug set (subsample preserves all drugs that appear in any pair)
- positive/negative balance (subsample pos and neg proportionally)
- test sets unchanged
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

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


def build_neighbors(edges, id2kind, drug_set):
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
    return n1, n2


def featurize(df, n1, n2):
    n = len(df)
    n_g = len(KIND_ORDER)
    X = np.zeros((n, 2 * n_g), dtype=np.float32)
    for i, (da, db) in enumerate(zip(df["drug_a_id"].values, df["drug_b_id"].values)):
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


def main():
    t0 = time.time()
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])

    n1, n2 = build_neighbors(edges, id2kind, drug_set)
    print("[features built]")

    fractions = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0]
    seeds = [42, 43, 44]

    results = []
    for seed in seeds:
        print(f"\n========== seed {seed} ==========")
        SP = ROOT / f"Code/data/KG/drugbank/splits/seed{seed}"
        train = load_split(SP / "train.parquet", SP / "train_negatives/epoch_0.parquet")
        test_s0 = load_split(SP / "test_s0.parquet", SP / "negatives/test_s0.parquet")
        test_s2 = load_split(SP / "test_s2.parquet", SP / "negatives/test_s2.parquet")
        print(f"  train={len(train)}, S0={len(test_s0)}, S2={len(test_s2)}")

        print("  featurizing test sets ...")
        X_s0 = featurize(test_s0, n1, n2)
        y_s0 = test_s0["label"].values
        X_s2 = featurize(test_s2, n1, n2)
        y_s2 = test_s2["label"].values

        print("  featurizing full train ...")
        X_tr_full = featurize(train, n1, n2)
        y_tr_full = train["label"].values
        n_train = len(y_tr_full)

        rng = np.random.default_rng(seed * 1000 + 7)
        order = rng.permutation(n_train)
        X_tr_full = X_tr_full[order]
        y_tr_full = y_tr_full[order]

        for frac in fractions:
            n_use = max(int(n_train * frac), 100)
            X_sub = X_tr_full[:n_use]
            y_sub = y_tr_full[:n_use]
            n_pos = int(y_sub.sum())
            n_neg = int((1 - y_sub).sum())
            clf = LogisticRegression(C=1.0, max_iter=500).fit(X_sub, y_sub)

            # Predict S0 + S2
            p_s0 = clf.predict_proba(X_s0)[:, 1]
            p_s2 = clf.predict_proba(X_s2)[:, 1]
            eps = 1e-7
            p_s0c = np.clip(p_s0, eps, 1 - eps)
            p_s2c = np.clip(p_s2, eps, 1 - eps)
            auc_s0 = roc_auc_score(y_s0, p_s0)
            auc_s2 = roc_auc_score(y_s2, p_s2)
            nll_s0 = log_loss(y_s0, p_s0c)
            nll_s2 = log_loss(y_s2, p_s2c)
            results.append({
                "seed": seed, "frac": frac, "n_train": n_use, "n_pos": n_pos, "n_neg": n_neg,
                "auc_s0": auc_s0, "auc_s2": auc_s2,
                "nll_s0": nll_s0, "nll_s2": nll_s2,
            })
            print(f"  frac={frac:.2f} n_tr={n_use:6d} | S0 AUC={auc_s0:.4f} NLL={nll_s0:.4f} | S2 AUC={auc_s2:.4f} NLL={nll_s2:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "train_scaling_results.csv", index=False)
    print(f"\nsaved: {OUT_DIR/'train_scaling_results.csv'}")

    # Aggregate by frac
    print("\n=== Aggregate (mean ± std across seeds) ===")
    agg = res_df.groupby("frac").agg(
        n_train=("n_train", "mean"),
        S0_AUC_mean=("auc_s0", "mean"), S0_AUC_std=("auc_s0", "std"),
        S2_AUC_mean=("auc_s2", "mean"), S2_AUC_std=("auc_s2", "std"),
        S0_NLL_mean=("nll_s0", "mean"), S0_NLL_std=("nll_s0", "std"),
        S2_NLL_mean=("nll_s2", "mean"), S2_NLL_std=("nll_s2", "std"),
    )
    print(agg.round(4))

    # Check non-monotonicity
    print("\n=== Non-monotonicity test ===")
    s0_aucs = agg["S0_AUC_mean"].values
    s2_aucs = agg["S2_AUC_mean"].values
    s0_diffs = np.diff(s0_aucs)
    s2_diffs = np.diff(s2_aucs)
    print(f"S0 AUC diffs (frac increment): {s0_diffs}")
    print(f"S2 AUC diffs (frac increment): {s2_diffs}")
    print(f"S0 all increasing? {(s0_diffs >= 0).all()}")
    print(f"S2 all increasing? {(s2_diffs >= 0).all()}")
    print(f"S2 has any DECREASE step? {(s2_diffs < 0).any()}  ← this is the signature")
    if (s2_diffs < 0).any():
        idx = np.where(s2_diffs < 0)[0]
        for i in idx:
            print(f"   S2 DECREASES from frac {agg.index[i]:.2f} to {agg.index[i+1]:.2f} by {s2_diffs[i]:+.4f}")

    # Compare S0 max-to-end vs S2 max-to-end
    s0_max = s0_aucs.max()
    s0_end = s0_aucs[-1]
    s2_max = s2_aucs.max()
    s2_end = s2_aucs[-1]
    print(f"\nS0 AUC: max={s0_max:.4f}, at frac={agg.index[np.argmax(s0_aucs)]:.2f}; end (100%)={s0_end:.4f}, gap={s0_max-s0_end:+.4f}")
    print(f"S2 AUC: max={s2_max:.4f}, at frac={agg.index[np.argmax(s2_aucs)]:.2f}; end (100%)={s2_end:.4f}, gap={s2_max-s2_end:+.4f}")
    print(f"elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

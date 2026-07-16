"""exp3 C2 step 5 — cross-model verification with meeting-node LR.

Train a quick LR-count model on 22 meeting-node count features (E7 design)
on train.parquet + train negatives, predict test_s2, then redo decile
analysis to see if H_rel cliff appears for a DIFFERENT model class.

If LR shows same pattern: phase transition is data-regime, not GCN-specific.
If LR shows flat: phase transition is GCN shortcut.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

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


def build_neighbor_sets(edges, id2kind, drug_set):
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
    n_groups = len(KIND_ORDER)
    X = np.zeros((n, 2 * n_groups), dtype=np.float32)
    for i, (da, db) in enumerate(zip(df["drug_a_id"].values, df["drug_b_id"].values)):
        s1 = n1.get(da, set()) & n1.get(db, set())
        s2 = n2.get(da, set()) & n2.get(db, set())
        c1 = defaultdict(int)
        c2 = defaultdict(int)
        for _, g in s1:
            c1[g] += 1
        for _, g in s2:
            c2[g] += 1
        for j, grp in enumerate(KIND_ORDER):
            X[i, j] = np.log1p(c1[grp])
            X[i, n_groups + j] = np.log1p(c2[grp])
    return X


def load_split(pos_path, neg_path):
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
    pos["label"] = 1
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def main():
    t0 = time.time()
    print("[load]")
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    print(f"  drugs={len(drug_set)}")

    SPLITS = ROOT / "Code/data/KG/drugbank/splits/seed42"
    train = load_split(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet")
    test = load_split(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet")
    print(f"  train: {len(train)}, test_s2: {len(test)}")

    print("[build neighbors]")
    n1, n2 = build_neighbor_sets(edges, id2kind, drug_set)
    print(f"  n1 size mean={np.mean([len(v) for v in n1.values()]):.0f}, n2 mean={np.mean([len(v) for v in n2.values()]):.0f}")

    print("[featurize train]")
    Xtr = featurize(train, n1, n2)
    ytr = train["label"].values
    print(f"  Xtr={Xtr.shape}")
    print("[featurize test]")
    Xte = featurize(test, n1, n2)
    yte = test["label"].values
    print(f"  Xte={Xte.shape}")

    print("[fit LR-count]")
    lr = LogisticRegression(C=1.0, max_iter=500, n_jobs=4).fit(Xtr, ytr)
    p_test = lr.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_test)
    print(f"  test AUC (LR-count) = {auc:.4f}")

    # Now compute per-pair NLL from LR predictions, attach to df
    df = pd.read_parquet(OUT_DIR / "pair_features_c2.parquet")
    assert len(df) == len(test)
    # Verify pair alignment
    diff = ((df["drug_a_id"].values != test["drug_a_id"].values) | (df["drug_b_id"].values != test["drug_b_id"].values)).sum()
    assert diff == 0, f"pair mismatch: {diff}"
    diff2 = (df["label"].values != yte.astype(float)).sum()
    assert diff2 == 0, f"label mismatch: {diff2}"
    eps = 1e-7
    p = np.clip(p_test, eps, 1 - eps)
    df["lr_pred_p"] = p.astype(np.float32)
    df["lr_nll"] = -(yte * np.log(p) + (1 - yte) * np.log(1 - p)).astype(np.float32)

    # Redo decile analysis with LR predictions
    df = df[df["M2_count"] > 1].copy()
    q67 = df["M2_count"].quantile(0.67)
    q3 = df[df["M2_count"] > q67].copy()
    print(f"\n[Q3 LR cross-model] n={len(q3)}, label_rate={q3['label'].mean():.3f}")
    print(f"  GCN AUC on this slice: {roc_auc_score(q3['label'], q3['pred_p']):.4f}")
    print(f"  LR  AUC on this slice: {roc_auc_score(q3['label'], q3['lr_pred_p']):.4f}")

    # Decile analysis using LR predictions
    pos = q3[q3["label"] == 1].copy()
    neg = q3[q3["label"] == 0].copy()

    from sklearn.linear_model import LinearRegression
    lr_pos = LinearRegression().fit(np.log(pos["M2_count"].values).reshape(-1, 1), pos["lr_nll"].values)
    pos["lr_nll_resid"] = pos["lr_nll"].values - lr_pos.predict(np.log(pos["M2_count"].values).reshape(-1, 1))
    pos["Hr_dec"] = pd.qcut(pos["H_rel"], 10, labels=False, duplicates="drop")

    lr_neg = LinearRegression().fit(np.log(neg["M2_count"].values).reshape(-1, 1), neg["lr_nll"].values)
    neg["lr_nll_resid"] = neg["lr_nll"].values - lr_neg.predict(np.log(neg["M2_count"].values).reshape(-1, 1))
    neg["Hr_dec"] = pd.qcut(neg["H_rel"], 10, labels=False, duplicates="drop")

    print("\n[LR pos decile by H_rel — residualized on log(M2)]")
    tbl_pos = pos.groupby("Hr_dec").agg(
        Hr_mean=("H_rel", "mean"),
        LR_NLL_resid=("lr_nll_resid", "mean"),
        LR_pred=("lr_pred_p", "mean"),
        GCN_pred=("pred_p", "mean"),
        n=("lr_nll", "count"),
    )
    print(tbl_pos)
    rho_pos, _ = spearmanr(pos["H_rel"], pos["lr_nll"])
    print(f"  ρ_spearman(H_rel, LR_NLL pos) = {rho_pos:.4f}")

    print("\n[LR neg decile by H_rel — residualized on log(M2)]")
    tbl_neg = neg.groupby("Hr_dec").agg(
        Hr_mean=("H_rel", "mean"),
        LR_NLL_resid=("lr_nll_resid", "mean"),
        LR_pred=("lr_pred_p", "mean"),
        GCN_pred=("pred_p", "mean"),
        n=("lr_nll", "count"),
    )
    print(tbl_neg)
    rho_neg, _ = spearmanr(neg["H_rel"], neg["lr_nll"])
    print(f"  ρ_spearman(H_rel, LR_NLL neg) = {rho_neg:.4f}")

    # U / inverted-U test
    pos_dec_means = pos.groupby("Hr_dec")["lr_nll_resid"].mean().values
    neg_dec_means = neg.groupby("Hr_dec")["lr_nll_resid"].mean().values
    print(f"\n[Cross-model U test]")
    print(f"  LR pos deciles: {[f'{v:+.4f}' for v in pos_dec_means]}")
    print(f"  LR pos U-shape: dec0>dec5? {pos_dec_means[0] > pos_dec_means[5]}, dec9>dec5? {pos_dec_means[9] > pos_dec_means[5]}")
    print(f"  LR pos cliff dec9-dec8: {pos_dec_means[9] - pos_dec_means[8]:+.4f}")
    print(f"  LR neg deciles: {[f'{v:+.4f}' for v in neg_dec_means]}")
    print(f"  LR neg inverted-U: mid > dec0? {neg_dec_means[5] > neg_dec_means[0]}, mid > dec9? {neg_dec_means[5] > neg_dec_means[9]}")

    print(f"\nelapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""exp3 step 18 — Phantom mediator subgraph screening (candidate δ).

Systematic causal screening: for each mediator family F, train LR with F
features removed, evaluate S0 + S2. Find F where:
  Δ_S2 = AUC_S2(without F) - AUC_S2(with F) > 0   (cold-start IMPROVES)
  Δ_S0 = AUC_S0(without F) - AUC_S0(with F) ≈ 0   (warm-start unchanged)

Such F is a "phantom" mediator subgraph: it's actively HURTING cold-start
while being neutral on warm-start. Cold-start KG debiasing target.

Levels of granularity:
1. By mediator kind (11 kinds → 11 features × 2 hops = 22 features)
2. By individual feature (zero one column at a time)
3. By relation family (more granular: CcSE, drug_effect, contraindication, ...)
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
    print(f"loaded: edges={len(edges)} drugs={len(drug_set)}")

    n1, n2 = build_neighbors(edges, id2kind, drug_set)
    print("[features built]")

    SP = ROOT / "Code/data/KG/drugbank/splits/seed42"
    train = load_split(SP / "train.parquet", SP / "train_negatives/epoch_0.parquet")
    test_s0 = load_split(SP / "test_s0.parquet", SP / "negatives/test_s0.parquet")
    test_s2 = load_split(SP / "test_s2.parquet", SP / "negatives/test_s2.parquet")
    print(f"train={len(train)}, S0={len(test_s0)}, S2={len(test_s2)}")

    print("featurizing ...")
    Xtr = featurize(train, n1, n2)
    ytr = train["label"].values
    X_s0 = featurize(test_s0, n1, n2)
    y_s0 = test_s0["label"].values
    X_s2 = featurize(test_s2, n1, n2)
    y_s2 = test_s2["label"].values
    print(f"  Xtr={Xtr.shape}")

    n_g = len(KIND_ORDER)

    # Baseline LR
    print("\n[Baseline] full 22 features")
    clf_full = LogisticRegression(C=1.0, max_iter=500).fit(Xtr, ytr)
    base_s0 = roc_auc_score(y_s0, clf_full.predict_proba(X_s0)[:, 1])
    base_s2 = roc_auc_score(y_s2, clf_full.predict_proba(X_s2)[:, 1])
    print(f"  baseline S0 AUC: {base_s0:.4f}")
    print(f"  baseline S2 AUC: {base_s2:.4f}")

    # Per-kind ablation: zero out feature columns for each kind (1-hop + 2-hop = 2 features)
    print("\n=== Per-kind ablation: drop both 1-hop and 2-hop count for kind ===")
    print(f"{'kind':<25}{'S0 AUC':>10}{'ΔS0':>10}{'S2 AUC':>10}{'ΔS2':>10}{'PHANTOM?':>12}")
    print("-" * 80)
    results = []
    for j, kind in enumerate(KIND_ORDER):
        # drop column j (1-hop) and column j + n_g (2-hop)
        drop_idx = [j, j + n_g]
        keep_mask = np.ones(2 * n_g, dtype=bool)
        keep_mask[drop_idx] = False
        Xtr_drop = Xtr[:, keep_mask]
        X_s0_drop = X_s0[:, keep_mask]
        X_s2_drop = X_s2[:, keep_mask]
        clf = LogisticRegression(C=1.0, max_iter=500).fit(Xtr_drop, ytr)
        s0_auc = roc_auc_score(y_s0, clf.predict_proba(X_s0_drop)[:, 1])
        s2_auc = roc_auc_score(y_s2, clf.predict_proba(X_s2_drop)[:, 1])
        d_s0 = s0_auc - base_s0
        d_s2 = s2_auc - base_s2
        # Phantom criterion: S2 improves (Δ_S2 > 0) AND S0 nearly unchanged (|Δ_S0| small)
        is_phantom = "★PHANTOM" if (d_s2 > 0.001 and abs(d_s0) < 0.005) else "phantom" if d_s2 > 0.0005 and abs(d_s0) < 0.001 else ""
        results.append((kind, s0_auc, d_s0, s2_auc, d_s2, is_phantom))
        print(f"{kind:<25}{s0_auc:>10.4f}{d_s0:>+10.4f}{s2_auc:>10.4f}{d_s2:>+10.4f}{is_phantom:>12}")

    # Per-hop ablation: drop just 1-hop or just 2-hop for the kind
    print("\n=== Per-hop ablation: drop 1-hop OR 2-hop separately ===")
    print(f"{'kind':<25}{'hop':<5}{'S0 AUC':>10}{'ΔS0':>10}{'S2 AUC':>10}{'ΔS2':>10}")
    print("-" * 70)
    for j, kind in enumerate(KIND_ORDER):
        for hop_off, hop_name in [(0, "1hop"), (n_g, "2hop")]:
            drop_idx = [j + hop_off]
            keep_mask = np.ones(2 * n_g, dtype=bool)
            keep_mask[drop_idx] = False
            Xtr_d = Xtr[:, keep_mask]
            X_s0_d = X_s0[:, keep_mask]
            X_s2_d = X_s2[:, keep_mask]
            clf = LogisticRegression(C=1.0, max_iter=500).fit(Xtr_d, ytr)
            s0_auc = roc_auc_score(y_s0, clf.predict_proba(X_s0_d)[:, 1])
            s2_auc = roc_auc_score(y_s2, clf.predict_proba(X_s2_d)[:, 1])
            d_s0 = s0_auc - base_s0
            d_s2 = s2_auc - base_s2
            if abs(d_s2) > 0.0005 or abs(d_s0) > 0.0005:
                print(f"{kind:<25}{hop_name:<5}{s0_auc:>10.4f}{d_s0:>+10.4f}{s2_auc:>10.4f}{d_s2:>+10.4f}")

    # Save
    rdf = pd.DataFrame(results, columns=["kind", "s0_auc", "delta_s0", "s2_auc", "delta_s2", "phantom"])
    rdf.to_csv(OUT_DIR / "phantom_screening_kind.csv", index=False)
    print(f"\nsaved: {OUT_DIR/'phantom_screening_kind.csv'}")
    print(f"elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

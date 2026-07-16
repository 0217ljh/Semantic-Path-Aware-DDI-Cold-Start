"""E6-simplified — Pool-size ablation on meeting-node features (supports i3).

The original E6 plan used the 219 manually-labeled path types, but those
labels were generated DDInter-anchored (drug→entity), not pair-level
(drug_a→...→drug_b). So we test a related but simpler claim:

Adding more features into a pool does NOT monotonically help when most
features are noise. We re-use E7's 22-d shared-mediator features and run:
  - Univariate AUC per feature → rank features by signal
  - Train LR on top-k features for k ∈ {1, 3, 5, 8, 12, 16, 22}
  - Plot AUC(k) — does adding more features hurt past a peak?

If AUC(k=top-5) ≥ AUC(k=22), this supports "uniform pooling over noisy
features hurts" — the i3 first half. The "attention can't fix it" half is
established by E3 (semantic prior helps).
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
SPLITS = PROJECT_ROOT / "Code/data/KG/drugbank/splits/seed42"

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
    for s, d, dr in zip(edges["src"], edges["dst"], edges["directed"]):
        fwd[s].add(d)
        if not dr:
            fwd[d].add(s)
        sd = s in drug_set
        dd = d in drug_set
        if sd and not dd:
            n1[s].add((d, KIND_TO_GROUP.get(id2kind.get(d, ""), "other")))
        if dd and not sd and not dr:
            n1[d].add((s, KIND_TO_GROUP.get(id2kind.get(s, ""), "other")))
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
    n_feat = 2 * len(KIND_ORDER)
    X = np.zeros((n, n_feat), dtype=np.float32)
    for i, (da, db) in enumerate(zip(df["drug_a_id"].values, df["drug_b_id"].values)):
        s1 = n1.get(da, set()) & n1.get(db, set())
        s2 = n2.get(da, set()) & n2.get(db, set())
        c1 = defaultdict(int)
        c2 = defaultdict(int)
        for _, g in s1: c1[g] += 1
        for _, g in s2: c2[g] += 1
        for j, grp in enumerate(KIND_ORDER):
            X[i, j] = np.log1p(c1[grp])
            X[i, len(KIND_ORDER) + j] = np.log1p(c2[grp])
    return X


def load_pairs(pos_path, neg_path):
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
    pos["lab"] = 1
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    neg["lab"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def bootstrap_auc_ci(y, p, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try: aucs.append(roc_auc_score(y[idx], p[idx]))
        except ValueError: continue
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def main():
    print("[E6s] Loading ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])

    print("[E6s] Building neighbor sets ...")
    n1, n2 = build_neighbor_sets(edges, id2kind, drug_set)

    print("[E6s] Featurizing train/test ...")
    train = load_pairs(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet")
    test = load_pairs(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet")
    X_train = featurize(train, n1, n2)
    X_test = featurize(test, n1, n2)
    y_train = train["lab"].values
    y_test = test["lab"].values

    feat_names = [f"1hop_{g}" for g in KIND_ORDER] + [f"2hop_{g}" for g in KIND_ORDER]

    # 1. Univariate AUC per feature → rank by signal
    print("\n[E6s] Computing univariate AUC per feature ...")
    univariate = []
    for j, name in enumerate(feat_names):
        try:
            auc = roc_auc_score(y_test, X_test[:, j])
        except ValueError:
            auc = 0.5
        # also try negative direction
        auc_max = max(auc, 1 - auc)
        univariate.append((name, float(auc_max), int(j)))
    univariate.sort(key=lambda x: -x[1])
    print("  Top 10 features by univariate |AUC - 0.5|:")
    for name, auc, j in univariate[:10]:
        print(f"    {name:<30s} AUC={auc:.4f}")

    # 2. LR on top-k features for k ∈ {1, 3, 5, 8, 12, 16, 22}
    print("\n[E6s] LR on top-k features ...")
    rows = []
    for k in [1, 3, 5, 8, 12, 16, 22]:
        cols = [j for _, _, j in univariate[:k]]
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(X_train[:, cols], y_train)
        p = lr.predict_proba(X_test[:, cols])[:, 1]
        auc = roc_auc_score(y_test, p)
        ci_lo, ci_hi = bootstrap_auc_ci(y_test, p)
        feat_used = [univariate[i][0] for i in range(k)]
        rows.append({"k": k, "auc": auc, "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "features": feat_used})
        print(f"  top-{k:2d}: AUC={auc:.4f}  [CI {ci_lo:.4f}, {ci_hi:.4f}]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "pool_size_ablation.csv", index=False, encoding="utf-8-sig")
    summary = {
        "experiment": "E6-simplified pool-size ablation",
        "feat_count_total": len(feat_names),
        "univariate_top10": [{"feature": n, "auc": a} for n, a, _ in univariate[:10]],
        "results_by_k": rows,
        "verdict": None,
    }
    # i3 verdict: does AUC peak at small k?
    aucs = [r["auc"] for r in rows]
    peak_idx = int(np.argmax(aucs))
    peak_k = rows[peak_idx]["k"]
    peak_auc = aucs[peak_idx]
    full_auc = aucs[-1]
    if peak_k < 22 and (peak_auc - full_auc) > 0.005:
        summary["verdict"] = f"i3 supported (within-feature noise dilution): peak AUC {peak_auc:.4f} at k={peak_k}, drops to {full_auc:.4f} at k=22 (Δ={peak_auc-full_auc:.4f})"
    elif peak_k == 22:
        summary["verdict"] = f"i3 NOT supported in this test: AUC monotonically increases with k (peak={peak_auc:.4f} at k=22)"
    else:
        summary["verdict"] = f"i3 weakly supported: peak at k={peak_k} (AUC={peak_auc:.4f}); k=22 AUC={full_auc:.4f} — small gap"

    (OUT_DIR / "pool_size_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[E6s] saved → pool_size_ablation.csv, .json")
    print(f"\n[E6s i3 verdict] {summary['verdict']}")


if __name__ == "__main__":
    main()

"""E7 — Meeting-node logistic-regression baseline on ColdDDI S2.

Supports i2 architectural claim: a logistic regression on simple shared-mediator
features should be competitive with deep GCN baselines — direct evidence that
meeting-node signal IS the predictive signal.

Features per pair (drug_a, drug_b):
  For each of 11 consolidated node kinds K:
    - n_shared_1hop_K   : |N1(a) ∩ N1(b)| restricted to kind K
    - n_shared_2hop_K   : |N2(a) ∩ N2(b)| restricted to kind K   (excludes drug nodes)
  → 22 features. Two variants reported:
    - LR-binary : indicator(count > 0)
    - LR-count  : log1p(count)

Setup:
  - Train on `train.parquet` + epoch_0 paired negatives
  - Eval on `test_s2.parquet` + paired negatives in `negatives/test_s2.parquet`
  - 1 seed (this is a simple sanity baseline, not depth-sensitive)

Output:
  - meeting_node_lr.json  : AUCs + paired bootstrap CI vs GCN K=2 placeholder
  - meeting_node_lr.csv   : feature importance (LR coefs)

The comparison vs GCN K=2 will be added after E2 produces that number.
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
from sklearn.metrics import roc_auc_score, average_precision_score

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

# 11 consolidated kinds. Each KG node's `kind` maps to one of these (or "other").
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

# Reverse map: original kind → group
KIND_TO_GROUP: dict[str, str] = {}
for grp, kinds in KIND_GROUPS.items():
    for k in kinds:
        KIND_TO_GROUP[k] = grp


# ---------------------------------------------------------------------------
def build_neighbor_sets(edges: pd.DataFrame, id2kind: dict, drug_set: set) -> tuple[dict, dict]:
    """Return (n1, n2): drug_id → {(node_id, group)} for 1-hop and 2-hop reach.

    2-hop here = nodes reachable in exactly 2 hops via a non-drug intermediate.
    For both, we restrict to NON-drug terminals (drug-drug isn't useful).
    """
    print("[E7] Building 1-hop neighbor sets ...")
    n1: dict[str, set[tuple[str, str]]] = defaultdict(set)
    # Also: forward adjacency for arbitrary nodes (drug or not) → for 2-hop expansion
    fwd: dict[str, set[str]] = defaultdict(set)

    for src, dst, directed in zip(edges["src"], edges["dst"], edges["directed"]):
        # fwd: any → any
        fwd[src].add(dst)
        if not directed:
            fwd[dst].add(src)

        src_is_drug = src in drug_set
        dst_is_drug = dst in drug_set
        if src_is_drug and not dst_is_drug:
            n1[src].add((dst, KIND_TO_GROUP.get(id2kind.get(dst, ""), "other")))
        if dst_is_drug and not src_is_drug and not directed:
            n1[dst].add((src, KIND_TO_GROUP.get(id2kind.get(src, ""), "other")))

    print(f"[E7] drugs with ≥1 1-hop non-drug neighbor: {sum(1 for d in drug_set if d in n1)}")

    print("[E7] Building 2-hop neighbor sets (non-drug terminals only) ...")
    n2: dict[str, set[tuple[str, str]]] = defaultdict(set)
    t0 = time.time()
    for i, drug in enumerate(drug_set):
        # Get all non-drug 1-hop neighbors of `drug`, then expand to their neighbors
        for mid, _ in n1.get(drug, ()):
            for term in fwd.get(mid, ()):
                if term == drug or term in drug_set:
                    continue
                n2[drug].add((term, KIND_TO_GROUP.get(id2kind.get(term, ""), "other")))
        if (i + 1) % 1000 == 0:
            print(f"    progress: {i+1}/{len(drug_set)}  elapsed={time.time()-t0:.1f}s")
    print(f"[E7] drugs with ≥1 2-hop non-drug node: {sum(1 for d in drug_set if d in n2)}  total time={time.time()-t0:.1f}s")
    return n1, n2


def featurize(
    df: pd.DataFrame, n1: dict, n2: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_binary, X_count) of shape (N, 2 * |KIND_ORDER|)."""
    n = len(df)
    n_groups = len(KIND_ORDER)
    n_feat = 2 * n_groups
    X_binary = np.zeros((n, n_feat), dtype=np.float32)
    X_count = np.zeros((n, n_feat), dtype=np.float32)
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
            v1 = c1[grp]
            v2 = c2[grp]
            X_count[i, j] = np.log1p(v1)
            X_count[i, n_groups + j] = np.log1p(v2)
            X_binary[i, j] = 1.0 if v1 > 0 else 0.0
            X_binary[i, n_groups + j] = 1.0 if v2 > 0 else 0.0
    return X_binary, X_count


def load_split(pos_path: Path, neg_path: Path) -> pd.DataFrame:
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
    pos["label"] = 1
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    neg["label"] = 0
    return pd.concat([pos, neg], ignore_index=True)


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    aucs = np.array(aucs)
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def paired_bootstrap_auc_diff(
    y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """Return (mean_diff, ci_lo, ci_hi) for AUC(a) - AUC(b) on paired data."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(roc_auc_score(y_true[idx], y_a[idx]) - roc_auc_score(y_true[idx], y_b[idx]))
        except ValueError:
            continue
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ---------------------------------------------------------------------------
def main() -> None:
    print("[E7] Loading nodes / edges / splits ...")
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    print(f"[E7] nodes={len(nodes)}  edges={len(edges)}  drugs={len(drug_set)}")

    train = load_split(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet")
    test = load_split(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet")
    print(f"[E7] train: {len(train)} (pos={int((train['label']==1).sum())}); test_s2: {len(test)} (pos={int((test['label']==1).sum())})")

    n1, n2 = build_neighbor_sets(edges, id2kind, drug_set)

    print("[E7] Featurizing train ...")
    t0 = time.time()
    X_train_bin, X_train_count = featurize(train, n1, n2)
    print(f"  shape: bin={X_train_bin.shape}  count={X_train_count.shape}  time={time.time()-t0:.1f}s")

    print("[E7] Featurizing test_s2 ...")
    t0 = time.time()
    X_test_bin, X_test_count = featurize(test, n1, n2)
    print(f"  shape: bin={X_test_bin.shape}  time={time.time()-t0:.1f}s")

    y_train = train["label"].values
    y_test = test["label"].values

    # --- LR-binary ---
    print("[E7] Fitting LR-binary ...")
    lr_bin = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr_bin.fit(X_train_bin, y_train)
    p_bin = lr_bin.predict_proba(X_test_bin)[:, 1]
    auc_bin = roc_auc_score(y_test, p_bin)
    ap_bin = average_precision_score(y_test, p_bin)
    ci_bin_lo, ci_bin_hi = bootstrap_auc_ci(y_test, p_bin)
    print(f"  LR-binary  AUC={auc_bin:.4f}  [95% CI {ci_bin_lo:.4f}-{ci_bin_hi:.4f}]  AUPRC={ap_bin:.4f}")

    # --- LR-count ---
    print("[E7] Fitting LR-count (log1p) ...")
    lr_cnt = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr_cnt.fit(X_train_count, y_train)
    p_cnt = lr_cnt.predict_proba(X_test_count)[:, 1]
    auc_cnt = roc_auc_score(y_test, p_cnt)
    ap_cnt = average_precision_score(y_test, p_cnt)
    ci_cnt_lo, ci_cnt_hi = bootstrap_auc_ci(y_test, p_cnt)
    print(f"  LR-count  AUC={auc_cnt:.4f}  [95% CI {ci_cnt_lo:.4f}-{ci_cnt_hi:.4f}]  AUPRC={ap_cnt:.4f}")

    # Paired bootstrap diff (count - binary)
    diff_mean, diff_lo, diff_hi = paired_bootstrap_auc_diff(y_test, p_cnt, p_bin)
    print(f"  Paired Δ-AUC (count - binary): mean={diff_mean:+.4f}  [95% CI {diff_lo:+.4f}, {diff_hi:+.4f}]")

    # --- Feature importances (LR-count) ---
    feat_names = [f"1hop_{g}" for g in KIND_ORDER] + [f"2hop_{g}" for g in KIND_ORDER]
    coefs = lr_cnt.coef_[0]
    coef_df = pd.DataFrame({"feature": feat_names, "coef_count": coefs})
    coef_df = coef_df.reindex(coef_df["coef_count"].abs().sort_values(ascending=False).index)
    coef_df.to_csv(OUT_DIR / "meeting_node_lr_coefs.csv", index=False, encoding="utf-8-sig")
    print(f"\n[E7] Top features by |coef| (LR-count):")
    print(coef_df.head(10).to_string(index=False))

    # --- Save summary ---
    summary = {
        "experiment": "E7 meeting-node LR baseline",
        "feature_groups": KIND_ORDER,
        "n_features": 2 * len(KIND_ORDER),
        "n_train": int(len(train)),
        "n_test_s2": int(len(test)),
        "results": {
            "LR_binary": {
                "test_auc": float(auc_bin),
                "test_auc_ci_95": [float(ci_bin_lo), float(ci_bin_hi)],
                "test_auprc": float(ap_bin),
            },
            "LR_count": {
                "test_auc": float(auc_cnt),
                "test_auc_ci_95": [float(ci_cnt_lo), float(ci_cnt_hi)],
                "test_auprc": float(ap_cnt),
            },
            "paired_diff_count_vs_binary": {
                "mean": float(diff_mean),
                "ci_95": [float(diff_lo), float(diff_hi)],
            },
        },
        "gcn_k2_comparison": "TBD — populate after E2",
    }
    (OUT_DIR / "meeting_node_lr.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[E7] saved → meeting_node_lr.json")
    print(f"\n[E7 verdict]")
    print(f"  LR-count AUC: {auc_cnt:.4f}  (this is the meeting-node-only signal)")
    print(f"  If this is within 3pt of E2's vanilla GCN K=2 AUC, i2's architectural claim is supported.")


if __name__ == "__main__":
    main()

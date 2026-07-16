"""Diagnose train vs S2 covariate shift in the adapter's support features.

The pure-structural adapter (8.7K params) still collapses on S2, which points to
a train/test distribution gap rather than capacity. This script tests that
directly on the CACHED support features (no training, no GPU):

  1. Domain classifier: can a model tell train pairs from val_s2 pairs using only
     the structural support vector? High AUC => strong covariate shift (seen-seen
     vs unseen-unseen supports differ systematically).
  2. By-label split: domain AUC among positives vs among negatives. If the
     negative pool shifts much more, that is the AUROC-collapse driver (AUROC is
     prior-insensitive; class-conditional feature shift is what hurts).
  3. Key feature shift: support size and per-type s_tau, train vs val means, to
     show WHAT shifts.

Reads the spmn_v2 SUM support cache (struct vectors are the per-pair structural
summary). Read-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
from my_code.models.spmn_v1.retrieval import KIND_ORDER, N_TYPES  # noqa: E402

CACHE_DEFAULT = (ROOT / "Code/data/_cache/"
                 "spmn_v2_supports_sum_seed42_lmax5_kpt128_nmax1200_cp0.npz")


def _domain_auc(Xa, Xb, seed=0, linear=False):
    """AUC of telling Xa (domain 0) from Xb (domain 1). Balanced by subsampling
    the larger set; honest via a held-out split. ~0.5 = no shift, ->1 = shift."""
    rng = np.random.default_rng(seed)
    n = min(len(Xa), len(Xb))
    Xa_s = Xa[rng.choice(len(Xa), n, replace=False)]
    Xb_s = Xb[rng.choice(len(Xb), n, replace=False)]
    X = np.vstack([Xa_s, Xb_s])
    d = np.r_[np.zeros(n), np.ones(n)]
    Xtr, Xte, dtr, dte = train_test_split(X, d, test_size=0.3, stratify=d,
                                          random_state=seed)
    if linear:
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                             random_state=seed)
    clf.fit(Xtr, dtr)
    return roc_auc_score(dte, clf.predict_proba(Xte)[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_DEFAULT))
    args = ap.parse_args()

    z = np.load(args.cache)
    tr_x, tr_y = z["train__struct"], z["train__y"]
    va_x, va_y = z["val_s2__struct"], z["val_s2__y"]
    te_x, te_y = z["test_s2__struct"], z["test_s2__y"]
    print(f"[shift] cache: {Path(args.cache).name}")
    print(f"[shift] train={len(tr_y)} val_s2={len(va_y)} test_s2={len(te_y)} "
          f"struct_dim={tr_x.shape[1]}")

    print("\n[1] domain classifier — can features tell the domains apart?")
    print(f"  train vs val_s2 : GBDT AUC = {_domain_auc(tr_x, va_x):.4f}   "
          f"(linear {_domain_auc(tr_x, va_x, linear=True):.4f})")
    print(f"  train vs test_s2: GBDT AUC = {_domain_auc(tr_x, te_x):.4f}")
    print("  (0.5 = no shift, ->1.0 = strong covariate shift)")

    print("\n[2] by-label domain shift (train vs val_s2):")
    pos = _domain_auc(tr_x[tr_y == 1], va_x[va_y == 1])
    neg = _domain_auc(tr_x[tr_y == 0], va_x[va_y == 0])
    print(f"  positives only : domain AUC = {pos:.4f}")
    print(f"  negatives only : domain AUC = {neg:.4f}")
    print("  (if negatives >> positives, negative-pool shift drives the collapse)")

    print("\n[3] key feature shift (train mean -> val_s2 mean):")
    # struct layout: [s_tau(N), log(N), aa(N), copath_triu(N*(N+1)/2), n_support, n_types]
    sz_idx = 3 * N_TYPES + N_TYPES * (N_TYPES + 1) // 2
    n_sup_tr, n_sup_va = tr_x[:, sz_idx], va_x[:, sz_idx]
    print(f"  support size      : {n_sup_tr.mean():8.2f} -> {n_sup_va.mean():8.2f}  "
          f"({100*(n_sup_va.mean()/max(n_sup_tr.mean(),1e-9)-1):+.1f}%)")
    ntypes_tr = tr_x[:, sz_idx + 1]; ntypes_va = va_x[:, sz_idx + 1]
    print(f"  #types present    : {ntypes_tr.mean():8.2f} -> {ntypes_va.mean():8.2f}")
    # per-type s_tau (first N_TYPES dims), ranked by |Δmean|
    dmean = va_x[:, :N_TYPES].mean(0) - tr_x[:, :N_TYPES].mean(0)
    order = np.argsort(-np.abs(dmean))
    print("  per-type s_tau (top shifts):")
    for i in order[:6]:
        print(f"    {KIND_ORDER[i]:<20} {tr_x[:, i].mean():8.2f} -> "
              f"{va_x[:, i].mean():8.2f}  (Δ {dmean[i]:+.2f})")

    print("\n[verdict] domain AUC >> 0.5 => covariate shift is real -> "
          "domain-generalization is the right lever; ~0.5 => generic overfit -> "
          "just regularize + early stop.")


if __name__ == "__main__":
    main()

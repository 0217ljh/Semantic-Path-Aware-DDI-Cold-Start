"""SPMN v1 — where do the explicit-structure predictions fail?

Stratifies test_s2 errors by structural properties of each pair, to localize
failure and point optimization. Uses the cached Phase-2 supports/struct
(``spmn_v1_phase2_supports_seed{seed}_lmax{l}.npz``) + a fast GBDT on the
standardized structural features (CPU, instant) — the failure PATTERN is
model-agnostic (NN and GBDT rely on the same KG structure), so this is a valid
diagnostic without GPU.

Strata reported (AUC + error rates):
  * empty vs non-empty corridor support
  * support-size buckets
  * co-path present vs absent
  * dominant mechanism type
  * FP vs FN characterization (hard negatives = high structure but label 0;
    missed positives = low structure but label 1)

Run: python Code/scripts/analyze_spmn_v1_failures.py --seed 42 --l-max 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.spmn_v1.retrieval import KIND_ORDER, N_TYPES  # noqa: E402

# struct layout: [s_tau(12) | log_s_tau(12) | s_tau_aa(12) | sym_copath_triu(78) | n_support | n_active]
S_TAU = slice(0, N_TYPES)
AA = slice(2 * N_TYPES, 3 * N_TYPES)
COPATH = slice(3 * N_TYPES, 3 * N_TYPES + N_TYPES * (N_TYPES + 1) // 2)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    args = ap.parse_args()

    cache = ROOT / "Code/data/_cache" / \
        f"spmn_v1_phase2_supports_v2aa_seed{args.seed}_lmax{args.l_max}.npz"
    z = np.load(cache)
    xtr, ytr = z["train__struct"], z["train__y"]
    xte, yte = z["test_s2__struct"], z["test_s2__y"]

    # standardize on train, fit fast GBDT (model-agnostic failure pattern)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                         learning_rate=0.1, random_state=args.seed)
    clf.fit((xtr - mu) / sd, ytr)
    p = clf.predict_proba((xte - mu) / sd)[:, 1]

    s_tau = xte[:, S_TAU]
    supp_total = s_tau.sum(1)                       # uncapped common-reach count
    copath_total = xte[:, COPATH].sum(1)
    n_active = (s_tau > 0).sum(1)
    dom_type = np.where(supp_total > 0, s_tau.argmax(1), -1)

    print(f"=== SPMN v1 failure analysis (test_s2, seed{args.seed} l{args.l_max}) ===")
    print(f"overall AUC={_auc(yte, p):.4f}  n={len(yte)}  pos_rate={yte.mean():.3f}")

    print("\n-- by corridor support --")
    for name, mask in [("empty (s_tau=0)", supp_total == 0),
                       ("non-empty", supp_total > 0)]:
        m = mask
        print(f"  {name:18s} n={m.sum():4d} pos_rate={yte[m].mean():.3f} "
              f"AUC={_auc(yte[m], p[m]):.4f}")

    print("\n-- by support size (non-empty) --")
    nz = supp_total > 0
    edges = [1, 5, 15, 40, 100, 1e9]
    lo = 0.0
    for hi in edges:
        m = nz & (supp_total > lo) & (supp_total <= hi)
        if m.sum() > 20:
            print(f"  ({lo:.0f},{hi:.0f}] n={m.sum():4d} pos_rate={yte[m].mean():.3f} "
                  f"AUC={_auc(yte[m], p[m]):.4f}")
        lo = hi

    print("\n-- by co-path presence (non-empty support) --")
    for name, m in [("co-path > 0", nz & (copath_total > 0)),
                    ("co-path == 0", nz & (copath_total == 0))]:
        print(f"  {name:14s} n={m.sum():4d} pos_rate={yte[m].mean():.3f} "
              f"AUC={_auc(yte[m], p[m]):.4f}")

    print("\n-- by dominant mechanism type --")
    for t in range(N_TYPES):
        m = dom_type == t
        if m.sum() > 30:
            print(f"  {KIND_ORDER[t]:20s} n={m.sum():4d} pos_rate={yte[m].mean():.3f} "
                  f"AUC={_auc(yte[m], p[m]):.4f}")

    # FP / FN characterization at a balanced threshold
    thr = np.quantile(p, 1 - yte.mean())   # predict top pos_rate fraction as positive
    pred = (p >= thr).astype(int)
    fp = (pred == 1) & (yte == 0)
    fn = (pred == 0) & (yte == 1)
    tp = (pred == 1) & (yte == 1)
    tn = (pred == 0) & (yte == 0)
    print(f"\n-- error characterization (thr={thr:.3f}) --")
    print(f"  TP={tp.sum()} FP={fp.sum()} FN={fn.sum()} TN={tn.sum()}")
    print(f"  FP (hard neg: structure but no DDI)  mean_supp={supp_total[fp].mean():.1f} "
          f"mean_copath={copath_total[fp].mean():.1f} empty_frac={(supp_total[fp]==0).mean():.2f}")
    print(f"  FN (missed pos)                      mean_supp={supp_total[fn].mean():.1f} "
          f"mean_copath={copath_total[fn].mean():.1f} empty_frac={(supp_total[fn]==0).mean():.2f}")
    print(f"  TP                                   mean_supp={supp_total[tp].mean():.1f} "
          f"mean_copath={copath_total[tp].mean():.1f}")
    print(f"  positives total={int(yte.sum())}  of which empty-support={int((yte==1)&(supp_total==0)).sum() if False else int(((yte==1)&(supp_total==0)).sum())}")


if __name__ == "__main__":
    main()

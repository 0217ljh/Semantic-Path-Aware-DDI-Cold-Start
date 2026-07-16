"""Decisive test: does v1.6-style support {d_a<=2 AND d_b<=2} beat the sum
budget {d_a+d_b<=3}? My sum budget drops the (2,2) mediators v1.6 keeps.

Builds symmetric struct features (s_tau + AA + co-path) for test_s2 + a train
sample under both support modes, fits a fast GBDT, compares S2 AUC + mean
support size + protein_gene-stratum AUC. ~1-2 min.

Run: python Code/scripts/analyze_spmn_v1_support_mode.py --seed 42 --l-max 3 --train-sample 30000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = sys.path and _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, build_pair_support,
)
from my_code.models.spmn_v1.struct_features import (  # noqa: E402
    compute_struct_features, symmetric_binary_vector, symmetric_binary_dim,
)

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _build(kg, frame, l_max, support_and, tag):
    n = len(frame)
    X = np.zeros((n, symmetric_binary_dim()), dtype=np.float32)
    y = frame["label"].to_numpy().astype(np.int64)
    a = frame["drug_a_id"].astype(str).to_numpy()
    b = frame["drug_b_id"].astype(str).to_numpy()
    supp = np.zeros(n)
    t0 = time.time()
    for i in range(n):
        ai = kg.id_to_idx.get(a[i]); bi = kg.id_to_idx.get(b[i])
        if ai is None or bi is None or ai == bi:
            continue
        s = build_pair_support(kg, ai, bi, l_max=l_max, support_and=support_and)
        f = compute_struct_features(kg, s, with_copath=True)
        X[i] = symmetric_binary_vector(f)
        supp[i] = f.s_tau.sum()
    print(f"  [{tag}] {n} pairs {time.time()-t0:.0f}s  mean_support={supp.mean():.1f}",
          flush=True)
    return X, y, supp


def _auc(y, p, m=None):
    from sklearn.metrics import roc_auc_score
    if m is not None:
        y, p = y[m], p[m]
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--train-sample", type=int, default=30000)
    args = ap.parse_args()

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    train = _binary_frame(ds.splits.train, ds.get_train_negatives())
    test = _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))
    rng = np.random.default_rng(args.seed)
    tr = train.iloc[rng.permutation(len(train))[:args.train_sample]].reset_index(drop=True)

    kg = MergedKG.from_parquet()
    from sklearn.ensemble import HistGradientBoostingClassifier

    print(f"=== support-mode test (train sample={len(tr)}) ===")
    for mode_name, sand in [("sum  {d_a+d_b<=l}", False), ("AND  {d_a<=l-1 & d_b<=l-1}", True)]:
        Xtr, ytr, _ = _build(kg, tr, args.l_max, sand, f"train/{mode_name}")
        Xte, yte, ste = _build(kg, test, args.l_max, sand, f"test /{mode_name}")
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                             learning_rate=0.1, random_state=args.seed)
        clf.fit((Xtr - mu) / sd, ytr)
        p = clf.predict_proba((Xte - mu) / sd)[:, 1]
        s_tau = Xte[:, :N_TYPES]
        pg = (s_tau.sum(1) > 0) & (s_tau.argmax(1) == 0)
        print(f"  >> [{mode_name}] test AUC={_auc(yte,p):.4f}  "
              f"protein_gene AUC={_auc(yte,p,pg):.4f} (n={pg.sum()})\n", flush=True)


if __name__ == "__main__":
    main()

"""Decisive test: is v1.6's per-drug cluster-affinity the Phase-2 gap?

v1.6 (pure PMP head, S2 AUC 0.779) routes cluster evidence by a per-drug
affinity ``affinity[d,k] = log1p(#type-k mediators within (l-1) hops of d)``.
Our Phase-2 head dropped this. Here we compute that exact per-drug affinity
profile, add ``[affinity_a (K) , affinity_b (K)]`` to the explicit structural
features, refit a fast GBDT, and compare S2 AUC + the protein_gene stratum
against the no-affinity baseline (0.705). A jump toward 0.779 confirms the gap.

Run: python Code/scripts/analyze_spmn_v1_affinity_gap.py --seed 42 --l-max 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import MergedKG, N_TYPES  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _drug_affinity(kg: MergedKG, depth: int) -> dict[str, np.ndarray]:
    """affinity[d] = log1p over types of (#type-k non-drug nodes within depth)."""
    out: dict[str, np.ndarray] = {}
    for nid, is_drug in zip(kg.node_ids, kg.is_drug):
        if not is_drug:
            continue
        src = kg.id_to_idx[nid]
        nodes, _ = kg.neighborhood(src, depth)
        # exclude the drug itself + other drug nodes (mediators only)
        t = kg.type_id[nodes]
        nd = ~kg.is_drug[nodes]
        cnt = np.bincount(t[nd].astype(np.int64), minlength=N_TYPES)
        out[nid] = np.log1p(cnt.astype(np.float64))
    return out


def _affinity_matrix(frame, aff, dim):
    A = np.zeros((len(frame), 2 * dim), dtype=np.float64)
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    z = np.zeros(dim)
    for i in range(len(frame)):
        A[i, :dim] = aff.get(a_ids[i], z)
        A[i, dim:] = aff.get(b_ids[i], z)
    return A


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    args = ap.parse_args()

    # reuse cached structural features (v2aa) for train/test
    cache = ROOT / "Code/data/_cache" / \
        f"spmn_v1_phase2_supports_v2aa_seed{args.seed}_lmax{args.l_max}.npz"
    z = np.load(cache)
    xtr, ytr = z["train__struct"], z["train__y"]
    xte, yte = z["test_s2__struct"], z["test_s2__y"]

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    train = _binary_frame(ds.splits.train, ds.get_train_negatives())
    test = _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))
    assert len(train) == len(xtr) and len(test) == len(xte), "row mismatch"

    print("[affinity-gap] computing per-drug affinity ...", flush=True)
    kg = MergedKG.from_parquet()
    aff = _drug_affinity(kg, depth=max(0, args.l_max - 1))
    atr = _affinity_matrix(train, aff, N_TYPES)
    ate = _affinity_matrix(test, aff, N_TYPES)

    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_eval(Xtr, Xte, tag):
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=6,
                                             learning_rate=0.1,
                                             random_state=args.seed)
        clf.fit((Xtr - mu) / sd, ytr)
        p = clf.predict_proba((Xte - mu) / sd)[:, 1]
        s_tau = xte[:, :N_TYPES]
        supp = s_tau.sum(1)
        dom = np.where(supp > 0, s_tau.argmax(1), -1)
        pg = dom == 0          # protein_gene
        print(f"  [{tag}] test AUC={_auc(yte, p):.4f}  "
              f"protein_gene AUC={_auc(yte[pg], p[pg]):.4f} (n={pg.sum()})",
              flush=True)

    print("=== affinity-gap test (vs no-affinity 0.705 / v1.6 0.779) ===")
    fit_eval(xtr, xte, "struct only          ")
    fit_eval(atr, ate, "affinity only        ")
    fit_eval(np.concatenate([xtr, atr], 1),
             np.concatenate([xte, ate], 1), "struct + affinity    ")


if __name__ == "__main__":
    main()

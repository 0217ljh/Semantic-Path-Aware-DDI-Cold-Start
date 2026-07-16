"""SPMN v1 — Phase 1 minimal-falsification probe.

Question this script answers (the Phase-1 go/no-go gate from
``Notes/Log/algorithm_design_debate.md`` R14): does the EXPLICIT pairwise
mechanism structure alone (``s_tau``, ``s_{tau,tau'}`` on the symmetrised
merged KG) carry cold-start S2 signal — enough to match v1.6 (S2 AUC 0.779 on
this exact 800-drug seed42 split) and challenge EmerGNN — BEFORE we build any
fragment alignment / propagation?

It is deliberately model-free on the head: a gradient-boosted tree on the
explicit features. That doubles as the R13-A1 reviewer baseline ("a GBDT on
your features would match you"): if the later neural model cannot beat this,
the neural part is not earning its keep.

Data: the SAME pkl v1.6 used — ``Code/data/coldddi_legacy/800drug/seed{seed}.pkl``
(binary DDI; positives in ``splits.train`` + ``get_train_negatives()``; S2
holds out both drugs). Features come ONLY from the merged KG
(``Code/data/KG/_merged_kg``). The co-path feature is symmetrised for this
undirected binary task.

Run (WSL conda env project_1, from project root):
  python Code/scripts/run_spmn_v1_phase1.py --seed 42 --l-max 3
  python Code/scripts/run_spmn_v1_phase1.py --seed 42 --l-max 3 --no-copath
"""
from __future__ import annotations

import argparse
import json
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
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, build_pair_support,
)
from my_code.models.spmn_v1.struct_features import compute_struct_features  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
CACHE_DIR = ROOT / "Code/data/_cache"


def _binary_frame(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    """Stack positives (label 1) and negatives (label 0) into one pair frame."""
    p = pos[["drug_a_id", "drug_b_id"]].copy()
    p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy()
    n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _feature_dim(with_copath: bool) -> int:
    # s_tau (N) + log_s_tau (N) + sym-copath upper-triangle incl diagonal + 2 scalars
    tri = N_TYPES * (N_TYPES + 1) // 2 if with_copath else 0
    return 2 * N_TYPES + tri + 2


_TRI_I, _TRI_J = np.triu_indices(N_TYPES)


def _vectorise(kg: MergedKG, a_idx: int, b_idx: int, l_max: int,
               with_copath: bool) -> np.ndarray:
    support = build_pair_support(kg, a_idx, b_idx, l_max=l_max)
    feat = compute_struct_features(kg, support, with_copath=with_copath)
    s_tau = feat.s_tau.astype(np.float64)
    parts = [s_tau, np.log1p(s_tau)]
    if with_copath:
        sym = feat.s_tau_tau + feat.s_tau_tau.T          # undirected binary task
        parts.append(sym[_TRI_I, _TRI_J].astype(np.float64))
    parts.append(np.array([float(feat.n_support),
                           float((feat.s_tau > 0).sum())], dtype=np.float64))
    return np.concatenate(parts)


def _build_matrix(kg: MergedKG, frame: pd.DataFrame, l_max: int,
                  with_copath: bool, tag: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(frame)
    dim = _feature_dim(with_copath)
    X = np.zeros((n, dim), dtype=np.float32)
    y = frame["label"].to_numpy().astype(np.int64)
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    t0 = time.time()
    n_empty = 0
    for i in range(n):
        ai = kg.id_to_idx.get(a_ids[i])
        bi = kg.id_to_idx.get(b_ids[i])
        if ai is None or bi is None or ai == bi:
            n_empty += 1
            continue
        v = _vectorise(kg, ai, bi, l_max, with_copath)
        X[i] = v
        if v[:N_TYPES].sum() == 0:
            n_empty += 1
        if (i + 1) % 20000 == 0:
            print(f"  [{tag}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    print(f"  [{tag}] done {n} pairs in {time.time() - t0:.0f}s "
          f"(empty/oov support: {n_empty})", flush=True)
    return X, y


def _fit_gbdt(x_tr: np.ndarray, y_tr: np.ndarray, seed: int):
    """Return (model, name). Prefer xgboost; fall back to sklearn."""
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=8, random_state=seed,
        )
        clf.fit(x_tr, y_tr)
        return clf, "xgboost"
    except Exception as exc:  # noqa: BLE001
        print(f"[gbdt] xgboost unavailable ({exc}); using sklearn HGB", flush=True)
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter=400, max_depth=6, learning_rate=0.1, random_state=seed,
        )
        clf.fit(x_tr, y_tr)
        return clf, "sklearn_hgb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--no-copath", action="store_true",
                    help="ablate the cross-type co-path feature (s_tau only)")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    with_copath = not args.no_copath
    pkl = PKL_DIR / f"seed{args.seed}.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(pkl)

    print(f"[phase1] dataset: {pkl}", flush=True)
    ds = PairDataset.from_pkl(str(pkl))
    train = _binary_frame(ds.splits.train, ds.get_train_negatives())
    val = _binary_frame(ds.splits.val_s2, ds.get_negatives("val_s2"))
    test = _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))
    print(f"[phase1] train={len(train)} val_s2={len(val)} test_s2={len(test)}",
          flush=True)

    cache_tag = f"spmn_v1_phase1_feats_seed{args.seed}_lmax{args.l_max}_" \
                f"{'copath' if with_copath else 'staunly'}.npz"
    cache_path = CACHE_DIR / cache_tag

    if cache_path.is_file() and not args.no_cache:
        print(f"[phase1] feature cache HIT: {cache_path}", flush=True)
        z = np.load(cache_path)
        x_tr, y_tr, x_va, y_va, x_te, y_te = (
            z["x_tr"], z["y_tr"], z["x_va"], z["y_va"], z["x_te"], z["y_te"]
        )
    else:
        print("[phase1] building features from merged KG ...", flush=True)
        kg = MergedKG.from_parquet()
        x_tr, y_tr = _build_matrix(kg, train, args.l_max, with_copath, "train")
        x_va, y_va = _build_matrix(kg, val, args.l_max, with_copath, "val_s2")
        x_te, y_te = _build_matrix(kg, test, args.l_max, with_copath, "test_s2")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, x_tr=x_tr, y_tr=y_tr, x_va=x_va,
                            y_va=y_va, x_te=x_te, y_te=y_te)
        print(f"[phase1] feature cache saved: {cache_path}", flush=True)

    from sklearn.metrics import average_precision_score, roc_auc_score
    clf, head = _fit_gbdt(x_tr, y_tr, args.seed)
    out: dict[str, dict[str, float]] = {}
    for name, x, y in [("val_s2", x_va, y_va), ("test_s2", x_te, y_te)]:
        p = clf.predict_proba(x)[:, 1]
        out[name] = {
            "auc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "n": int(len(y)),
        }

    print("\n=== SPMN v1 Phase-1 (explicit structure only) ===", flush=True)
    print(f"head={head}  l_max={args.l_max}  copath={with_copath}  "
          f"feat_dim={x_tr.shape[1]}", flush=True)
    for name, m in out.items():
        print(f"  {name}: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} (n={m['n']})",
              flush=True)
    print(f"  reference: v1.6 S2 AUC=0.779 (same 800-drug seed{args.seed} split)",
          flush=True)

    res_dir = ROOT / "Code/runs/spmn_v1_phase1"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"seed{args.seed}_lmax{args.l_max}_" \
                         f"{'copath' if with_copath else 'staunly'}.json"
    res_path.write_text(json.dumps({
        "seed": args.seed, "l_max": args.l_max, "with_copath": with_copath,
        "head": head, "feat_dim": int(x_tr.shape[1]),
        "n_train": int(len(y_tr)), "results": out,
        "reference_v1_6_s2_auc": 0.779,
    }, indent=2))
    print(f"[phase1] results -> {res_path}", flush=True)


if __name__ == "__main__":
    main()

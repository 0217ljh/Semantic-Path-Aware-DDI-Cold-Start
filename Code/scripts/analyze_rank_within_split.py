"""Within-split rank-sufficiency analysis (companion to analyze_rank_sufficiency.py).

Motivation (codex + user 2026-07-05): the TRANSFER protocol in
``analyze_rank_sufficiency.py`` fits PCA + the linear probe on Z_TRAIN (seen
drugs) and transfers to warm/cold. On EmerGNN's full KG that transfer INVERTS on
cold (train-fit probe full-rank ~0.28 while the model head is ~0.74) because the
eval KG's DDI edges make seen drugs high-degree and unseen drugs low-degree — a
seen/unseen domain gap, NOT an absence of cold signal.

This script answers the OTHER question the transfer probe cannot: *how much
linearly-usable DDI signal exists INSIDE each split's own representation, and at
what rank.* For each split independently it does stratified K-fold CV; within each
fold it standardizes + PCA on the TRAIN folds only (no leakage), fits a logistic
probe on the top-r PCs, and scores the held-out fold. r95 = smallest rank
reaching 95% of that split's own full-rank CV AUROC.

Report BOTH scripts together (per codex): the transfer probe demonstrates the
domain gap; this within-split probe demonstrates the intrinsic usable-info rank.
Neither replaces the other.

Run:
  python Code/scripts/analyze_rank_within_split.py --z Code/runs/<run_id>/pilot_Z.npz --key raw --title EmerGNN-FIXED
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _within_split_curve(Z: np.ndarray, y: np.ndarray, ranks: list[int],
                        n_splits: int = 5, seed: int = 0) -> tuple[list[int], dict]:
    """Stratified K-fold CV AUROC vs PCA-truncation rank, all INSIDE this split.

    PCA (standardize + SVD) is fit on the TRAIN folds only and applied to the
    held-out fold, so there is no rank/probe leakage across folds.
    """
    y = y.astype(int)
    dim = Z.shape[1]
    ranks = sorted(set([r for r in ranks if r < dim] + [dim]))
    if len(np.unique(y)) < 2:
        raise ValueError(f"single-class labels in split (counts={dict(zip(*np.unique(y, return_counts=True)))})")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_rank: dict[int, list[float]] = {r: [] for r in ranks}
    for tr_idx, te_idx in skf.split(Z, y):
        Ztr, ytr = Z[tr_idx], y[tr_idx]
        Zte, yte = Z[te_idx], y[te_idx]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        mu = Ztr.mean(0, keepdims=True)
        sd = Ztr.std(0, keepdims=True) + 1e-8
        Str = (Ztr - mu) / sd
        Ste = (Zte - mu) / sd
        center = Str.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Str - center, full_matrices=False)
        for r in ranks:
            Ptr = (Str - center) @ Vt[:r].T
            Pte = (Ste - center) @ Vt[:r].T
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Ptr, ytr)
            p = clf.predict_proba(Pte)[:, 1]
            per_rank[r].append(float(roc_auc_score(yte, p)))
    mean_auc = {r: float(np.mean(v)) for r, v in per_rank.items() if v}
    std_auc = {r: float(np.std(v)) for r, v in per_rank.items() if v}
    return ranks, {"mean": mean_auc, "std": std_auc}


def _rec(auc_r: float, auc_full: float, null: float = 0.5) -> float:
    denom = auc_full - null
    return (auc_r - null) / denom if abs(denom) > 1e-6 else 0.0


def _r95(recs: list[float], ranks: list[int], thr: float = 0.95) -> int:
    for r, v in zip(ranks, recs):
        if v >= thr:
            return r
    return ranks[-1]


def _balance(Z: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced subsample so AUROC/r95 are not distorted by prior skew."""
    y = y.astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    n = min(len(pos), len(neg))
    if n == 0:
        return Z, y
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(idx)
    return Z[idx], y[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True, help="path to pilot_Z.npz")
    ap.add_argument("--key", default="raw", help="'raw' (pre-scorer) or 'Z'")
    ap.add_argument("--ranks", type=int, nargs="*", default=RANKS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="EmerGNN")
    ap.add_argument("--no-balance", action="store_true",
                    help="skip class balancing (default: balance each split)")
    args = ap.parse_args()

    zp = Path(args.z)
    d = np.load(zp)
    k = args.key
    splits = {"train": (d[f"{k}_train"], d["y_train"]),
              "warm": (d[f"{k}_warm"], d["y_warm"]),
              "cold": (d[f"{k}_cold"], d["y_cold"])}
    dim = splits["train"][0].shape[1]
    print(f"[load] {zp.name} | key={k} dim={dim} | "
          + " ".join(f"{s}={splits[s][0].shape}" for s in splits), flush=True)

    res = {"z_path": str(zp), "dim": dim, "key": k, "folds": args.folds,
           "protocol": "within-split stratified K-fold CV (PCA+probe inside each split)",
           "auc": {}, "auc_std": {}, "rec": {}, "r95": {}, "rho95": {}, "head": {}}
    ranks_used = None
    for s, (Z, y) in splits.items():
        if not args.no_balance:
            Z, y = _balance(Z, y, args.seed)
        ranks, curve = _within_split_curve(Z, y, args.ranks, n_splits=args.folds, seed=args.seed)
        ranks_used = ranks
        aucs = [curve["mean"][r] for r in ranks]
        stds = [curve["std"][r] for r in ranks]
        full = aucs[-1]
        recs = [_rec(a, full) for a in aucs]
        res["auc"][s] = aucs
        res["auc_std"][s] = stds
        res["rec"][s] = recs
        res["r95"][s] = _r95(recs, ranks)
        res["rho95"][s] = res["r95"][s] / dim
        if s in ("warm", "cold"):
            res["head"][s] = float(roc_auc_score(d[f"y_{s}"], d[f"logit_{s}"]))
        print(f"  [{s:5s}] full_cv_auc={full:.4f}  r95={res['r95'][s]}  "
              f"rho95={res['rho95'][s]:.3f}"
              + (f"  head={res['head'][s]:.4f}" if s in res["head"] else ""), flush=True)
        for r, a, st in zip(ranks, aucs, stds):
            print(f"      r={r:4d}  cv_auc={a:.4f}±{st:.4f}  rec={_rec(a, full):.3f}", flush=True)
    res["ranks"] = ranks_used

    print("\n=== WITHIN-SPLIT SUMMARY ===")
    print(f"  full-rank CV AUROC:  warm={res['auc']['warm'][-1]:.4f}  cold={res['auc']['cold'][-1]:.4f}")
    print(f"  r95:  warm={res['r95']['warm']}  cold={res['r95']['cold']}  "
          f"(rho95 warm={res['rho95']['warm']:.3f} cold={res['rho95']['cold']:.3f})")
    print(f"  model head:  warm={res['head'].get('warm')}  cold={res['head'].get('cold')}")

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [r / dim for r in ranks_used]
    ax.plot(xs, res["rec"]["warm"], "o-", label="warm (entity-seen)", color="C0")
    ax.plot(xs, res["rec"]["cold"], "s--", label="cold (S2 unseen)", color="C3")
    ax.axhline(0.95, ls=":", c="gray", lw=0.8)
    ax.set_xlabel("rank fraction  r / d"); ax.set_ylabel("normalized recovery Rec(r)")
    ax.set_title(f"{args.title}  within-split rank recovery\n"
                 f"(full CV AUROC warm {res['auc']['warm'][-1]:.3f} / cold {res['auc']['cold'][-1]:.3f})")
    ax.legend(); ax.set_xscale("log", base=2); fig.tight_layout()
    out_png = zp.parent / f"rank_within_split_{k}.png"
    out_json = zp.parent / f"rank_within_split_{k}.json"
    fig.savefig(out_png, dpi=150)
    out_json.write_text(json.dumps(res, indent=2))
    print(f"\n[done] plot -> {out_png}\n       json -> {out_json}")


if __name__ == "__main__":
    main()

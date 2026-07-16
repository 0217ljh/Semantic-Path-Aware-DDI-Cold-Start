"""A1 rank-sufficiency analysis on a pilot's pair-bottleneck Z (from
train_gcn_rank_pilot.py). Tests the warm/cold rank-transfer divergence.

Protocol (codex-reviewed 2026-07-04):
- PCA fit on Z_train ONLY (train stats). For r in a sweep, project train/warm/cold onto
  the top-r principal components, fit a LINEAR probe (logistic regression) on train only,
  evaluate AUROC on warm and cold.
- Headline = normalized recovery Rec_s(r) = (AUROC_s(r) - 0.5) / (AUROC_s(d) - 0.5) and
  rho95 = r95 / d (smallest rank reaching 95% of that condition's own full-rank recovery,
  divided by d). Thesis prediction: warm keeps rising with r (large r95); cold saturates
  early (small r95). The divergence is the evidence.
- Sanity 1: full-rank probe AUROC(r=d) should ~match the model head AUROC (else we measure
  linear-recoverability, not the model's decision rank).
- Sanity 2 (degree-only control): fit a probe on the 2-d per-pair log-degrees; if this
  trivial control shows the SAME warm/cold gap, the divergence is degree-driven, not rank.
- Supplement: SVD stable-rank / 90%-energy rank of Z (warm/cold, class-balanced sample).

Run:
  python Code/scripts/analyze_rank_sufficiency.py --z Code/runs/<run_id>/pilot_Z.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _probe_auc(Xtr, ytr, Xev, yev) -> float:
    if len(np.unique(ytr)) < 2 or len(np.unique(yev)) < 2:
        raise ValueError(
            "single-class labels — cannot compute AUROC (a data bug, not a rank effect). "
            f"train counts={dict(zip(*np.unique(ytr, return_counts=True)))}, "
            f"eval counts={dict(zip(*np.unique(yev, return_counts=True)))}")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xev)[:, 1]
    return float(roc_auc_score(yev, p))


def _rec(auc_r, auc_full, null=0.5):
    denom = auc_full - null
    return (auc_r - null) / denom if abs(denom) > 1e-6 else 0.0


def _r95(recs, ranks, thr=0.95):
    for r, v in zip(ranks, recs):
        if v >= thr:
            return r
    return ranks[-1]


def _eff_rank(Z):
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    s2 = s ** 2
    stable = float(s2.sum() / (s2.max() + 1e-12))
    energy90 = int(np.searchsorted(np.cumsum(s2) / s2.sum(), 0.90) + 1)
    return stable, energy90


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z", required=True, help="path to pilot_Z.npz")
    ap.add_argument("--key", default="Z",
                    help="representation to analyze: 'Z' (post-MLP, collapsed) or 'raw' (pre-MLP, max info)")
    ap.add_argument("--ranks", type=int, nargs="*", default=RANKS)
    ap.add_argument("--title", default="GCN-DDI",
                    help="encoder name for the plot title (e.g. 'R-GCN-DDI')")
    args = ap.parse_args()

    zp = Path(args.z)
    d = np.load(zp)
    k = args.key
    Ztr, ytr = d[f"{k}_train"], d["y_train"]
    Zwarm, ywarm = d[f"{k}_warm"], d["y_warm"]
    Zcold, ycold = d[f"{k}_cold"], d["y_cold"]
    dim = Ztr.shape[1]
    # guarantee the largest rank == dim so Rec/r95/probe-vs-head use a true full-rank endpoint
    ranks = sorted(set([r for r in args.ranks if r < dim] + [dim]))
    print(f"[load] Z dim={dim} | train={Ztr.shape} warm={Zwarm.shape} cold={Zcold.shape}",
          flush=True)

    # standardize with TRAIN stats, PCA (SVD) on TRAIN only
    mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
    Str = (Ztr - mu) / sd
    Swarm = (Zwarm - mu) / sd
    Scold = (Zcold - mu) / sd
    _, _, Vt = np.linalg.svd(Str - Str.mean(0, keepdims=True), full_matrices=False)

    def proj(S, r):
        return S @ Vt[:r].T

    res = {"z_path": str(zp), "dim": dim, "ranks": ranks,
           "auc": {"warm": [], "cold": []}, "rec": {"warm": [], "cold": []}}
    for r in ranks:
        Ptr, Pw, Pc = proj(Str, r), proj(Swarm, r), proj(Scold, r)
        aw = _probe_auc(Ptr, ytr, Pw, ywarm)
        ac = _probe_auc(Ptr, ytr, Pc, ycold)
        res["auc"]["warm"].append(aw)
        res["auc"]["cold"].append(ac)
        print(f"  r={r:4d}  warm_auc={aw:.4f}  cold_auc={ac:.4f}", flush=True)

    aw_full, ac_full = res["auc"]["warm"][-1], res["auc"]["cold"][-1]
    res["rec"]["warm"] = [_rec(a, aw_full) for a in res["auc"]["warm"]]
    res["rec"]["cold"] = [_rec(a, ac_full) for a in res["auc"]["cold"]]
    res["r95"] = {"warm": _r95(res["rec"]["warm"], ranks),
                  "cold": _r95(res["rec"]["cold"], ranks)}
    res["rho95"] = {k: v / dim for k, v in res["r95"].items()}

    # sanity 1: full-rank probe vs model head
    res["sanity_probe_vs_head"] = {
        "warm": {"probe_full": aw_full, "head": float(roc_auc_score(ywarm, d["logit_warm"]))},
        "cold": {"probe_full": ac_full, "head": float(roc_auc_score(ycold, d["logit_cold"]))},
    }
    # sanity 2: degree-only control (2-d per-pair log-degrees)
    res["degree_only"] = {
        "warm": _probe_auc(d["deg_train"], ytr, d["deg_warm"], ywarm),
        "cold": _probe_auc(d["deg_train"], ytr, d["deg_cold"], ycold),
    }
    # supplement: effective rank of Z
    res["eff_rank"] = {
        "train": _eff_rank(Ztr), "warm": _eff_rank(Zwarm), "cold": _eff_rank(Zcold)}

    print("\n=== SUMMARY ===")
    print(f"  full-rank AUROC:  warm={aw_full:.4f}  cold={ac_full:.4f}")
    print(f"  r95:  warm={res['r95']['warm']}  cold={res['r95']['cold']}  "
          f"(rho95 warm={res['rho95']['warm']:.3f} cold={res['rho95']['cold']:.3f})")
    print(f"  sanity probe-vs-head: {res['sanity_probe_vs_head']}")
    print(f"  degree-only control: {res['degree_only']}")
    print(f"  eff_rank (stable,90%E): {res['eff_rank']}")

    # plot: normalized recovery vs r/d
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [r / dim for r in ranks]
    ax.plot(xs, res["rec"]["warm"], "o-", label="warm (entity-seen)", color="C0")
    ax.plot(xs, res["rec"]["cold"], "s--", label="cold (S2 unseen)", color="C3")
    ax.axhline(0.95, ls=":", c="gray", lw=0.8)
    ax.set_xlabel("rank fraction  r / d"); ax.set_ylabel("normalized recovery Rec(r)")
    ax.set_title(f"{args.title}  warm/cold rank recovery\n(full-rank AUROC warm {aw_full:.3f} / cold {ac_full:.3f})")
    ax.legend(); ax.set_xscale("log", base=2); fig.tight_layout()
    out_png = zp.parent / f"rank_sufficiency_{k}.png"
    out_json = zp.parent / f"rank_sufficiency_{k}.json"
    fig.savefig(out_png, dpi=150)
    out_json.write_text(json.dumps(res, indent=2))
    print(f"\n[done] plot -> {out_png}\n       json -> {out_json}")


if __name__ == "__main__":
    main()

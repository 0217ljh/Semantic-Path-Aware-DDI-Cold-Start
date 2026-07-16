"""MNAH 5-run score-ensemble analysis (codex r21).

Loads mnah_e0..e4 test_s2 logit dumps, aligns by canonical pair, averages PROBS,
reports per-run AUC + ensemble AUC. Tests if MNAH is variance-limited (ensemble
approaching 0.80 => variance was the gap; else 0.80 needs a stronger method).
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]
ENS = ROOT / "Code/runs/_ensemble"


def _canon(a, b): return (a, b) if a <= b else (b, a)


def main():
    files = sorted(glob.glob(str(ENS / "mnah_e*.npz")))
    print(f"[ens] {len(files)} runs: {[Path(f).name for f in files]}")
    if len(files) < 2:
        print("[ens] need >=2 runs"); return
    # reference order from first file (canonical pairs + labels)
    base = np.load(files[0], allow_pickle=True)
    ref_pairs = [_canon(str(a), str(b)) for a, b in zip(base["pair_a"], base["pair_b"])]
    y = base["y_true"].astype(int)
    ref_idx = {k: i for i, k in enumerate(ref_pairs)}

    prob_mat = []
    aucs = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        p = np.full(len(ref_pairs), np.nan)
        for a, b, pr in zip(d["pair_a"], d["pair_b"], d["mnah_prob"]):
            j = ref_idx.get(_canon(str(a), str(b)))
            if j is not None:
                p[j] = pr
        if np.isnan(p).any():
            print(f"[ens] WARN {Path(f).name}: {int(np.isnan(p).sum())} unmatched pairs")
            p = np.nan_to_num(p, nan=0.5)
        aucs.append(roc_auc_score(y, p))
        prob_mat.append(p)
    prob_mat = np.vstack(prob_mat)
    ens_prob = prob_mat.mean(axis=0)
    ens_auc = roc_auc_score(y, ens_prob)
    # rank-average variant (often better than prob-average)
    from scipy.stats import rankdata
    rank_avg = np.mean([rankdata(p) for p in prob_mat], axis=0)
    ens_rank_auc = roc_auc_score(y, rank_avg)

    print(f"[ens] per-run AUC: {[f'{a:.4f}' for a in aucs]}")
    print(f"[ens] per-run mean AUC = {np.mean(aucs):.4f} (best {max(aucs):.4f})")
    print(f"[ens] PROB-AVERAGE ensemble AUC = {ens_auc:.4f}")
    print(f"[ens] RANK-AVERAGE ensemble AUC = {ens_rank_auc:.4f}")
    print(f"[ens] ensemble gain over mean-single = {ens_auc-np.mean(aucs):+.4f}; "
          f"over best-single = {ens_auc-max(aucs):+.4f}")
    (ENS / "ensemble_analysis.json").write_text(json.dumps({
        "n_runs": len(files), "per_run_auc": aucs, "mean_single": float(np.mean(aucs)),
        "best_single": float(max(aucs)), "prob_avg_auc": float(ens_auc),
        "rank_avg_auc": float(ens_rank_auc),
        "gain_over_mean": float(ens_auc - np.mean(aucs)),
        "reaches_080": bool(max(ens_auc, ens_rank_auc) >= 0.80),
    }, indent=2))
    print(f"[ens] saved -> {ENS/'ensemble_analysis.json'}")


if __name__ == "__main__":
    main()

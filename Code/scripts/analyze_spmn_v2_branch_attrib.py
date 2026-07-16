"""Round-17 analysis: what does the frozen mechanism-evidence branch (bf) DO?

The branch is a zero-init additive linear readout fit on the frozen phase-1 base
logit. We decompose its per-pair contribution
    branch_delta = logit(post_bf_score) - logit(base_score)
and relate it to the raw shared-enzyme (PK) and shared-target (PD) counts and the
shared-support size. Goal: explain the round-15/16 puzzle (PK has a small branch
edge despite PD's slightly stronger univariate signal) by showing HOW the residual
branch uses PK vs PD, rather than just its ablation score.

Reads the final-config npz (run_spmn_v2_aware --save-predictions, which now also
stores base_score): aware1_and_seed{S}_mech_sdeg_dpool_topk_cw0.1_det_bf_test_s2_scores.npz
Aggregates over seeds 42/43/44.
"""
from __future__ import annotations

import glob

import numpy as np
from scipy.stats import pearsonr, spearmanr

from run_spmn_v2_aware import _load_frames, _load_support_cache, density
from run_spmn_v2_standalone import ROOT

RES = ROOT / "Code/runs/spmn_v2_standalone"
SEEDS = (42, 43, 44)
TAG = "aware1_and_seed{s}_mech_sdeg_dpool_topk_cw0.1_det_bf_test_s2_scores.npz"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _pair_key(a, b):
    return np.array([f"{x}|{y}" for x, y in zip(a, b)])


def main() -> None:
    d_all, enz_all, tgt_all, supp_all, y_all = [], [], [], [], []
    for s in SEEDS:
        hits = sorted(glob.glob(str(RES / TAG.format(s=s))))
        if not hits:
            print(f"[seed {s}] MISSING {TAG.format(s=s)}; skip"); continue
        z = np.load(hits[0], allow_pickle=True)
        if "base_score" not in z.files:
            print(f"[seed {s}] npz has no base_score (rerun with patched --save-predictions); skip")
            continue
        frames = _load_frames(s)
        data = _load_support_cache("and", s, 3, 64, 400, False)
        te = data["test_s2"]; n = len(te["y"])
        mech = density.build_mech_feature_matrix(te["rela"], te["relb"], te["offsets"], n)
        fr = frames["test_s2"]
        key_frame = _pair_key(fr["drug_a_id"].astype(str).to_numpy(),
                              fr["drug_b_id"].astype(str).to_numpy())
        kz = _pair_key(z["pair_a"].astype(str), z["pair_b"].astype(str))
        idx_c = {k: i for i, k in enumerate(key_frame)}
        ci = np.array([idx_c[k] for k in kz])              # cache row for each npz row
        delta = _logit(z["y_score"].astype(np.float64)) - _logit(z["base_score"].astype(np.float64))
        d_all.append(delta)
        enz_all.append(mech[ci, 0].astype(np.float64))
        tgt_all.append(mech[ci, 1].astype(np.float64))
        supp_all.append(np.diff(te["offsets"]).astype(np.float64)[ci])
        y_all.append(z["y_true"].astype(np.float64))

    if not d_all:
        print("no data"); return
    d = np.concatenate(d_all); enz = np.concatenate(enz_all)
    tgt = np.concatenate(tgt_all); supp = np.concatenate(supp_all); y = np.concatenate(y_all)
    print("=" * 64)
    print("ROUND 17 — branch logit-contribution attribution (pooled 3 seeds)")
    print("=" * 64)
    print(f"n pairs                    : {len(d)}")
    print(f"branch Δlogit  mean/std    : {d.mean():+.4f} / {d.std():.4f}")
    print(f"|Δlogit| mean              : {np.abs(d).mean():.4f}")
    print(f"frac pairs branch moved (|Δ|>0.01): {(np.abs(d)>0.01).mean():.4f}")
    print("\ncorrelation of branch Δlogit with raw mech counts:")
    print(f"  enzyme(PK): pearson {pearsonr(d, enz)[0]:+.4f}  spearman {spearmanr(d, enz)[0]:+.4f}")
    print(f"  target(PD): pearson {pearsonr(d, tgt)[0]:+.4f}  spearman {spearmanr(d, tgt)[0]:+.4f}")
    print(f"  support   : pearson {pearsonr(d, supp)[0]:+.4f}  spearman {spearmanr(d, supp)[0]:+.4f}")
    # signed: does the branch push toward the correct label? align Δ with (2y-1)
    aligned = d * (2 * y - 1)
    print(f"\nbranch Δ aligned with label (Δ·(2y-1)) mean: {aligned.mean():+.4f}  "
          f"(>0 means branch helps on average)")
    print(f"  frac pairs branch pushes correct direction: {(aligned>0).mean():.4f}")
    # conditioned means
    print("\nmean Δlogit conditioned on presence:")
    for nm, mask in [("PK>0", enz > 0), ("PK=0", enz == 0),
                     ("PD>0", tgt > 0), ("PD=0", tgt == 0),
                     ("PK>0&PD=0", (enz > 0) & (tgt == 0)),
                     ("PD>0&PK=0", (tgt > 0) & (enz == 0))]:
        if mask.sum():
            print(f"  {nm:10s} n={mask.sum():5d}  Δ={d[mask].mean():+.4f}  "
                  f"aligned={(d[mask]*(2*y[mask]-1)).mean():+.4f}")
    # by support bucket
    print("\nmean |Δlogit| by support bucket:")
    for lo, hi, nm in [(-0.5, 0.5, "supp=0"), (0.5, 3.5, "supp1-3"),
                       (3.5, 50, "supp4-50"), (50, 1e9, "supp>50")]:
        m = (supp > lo) & (supp <= hi)
        if m.sum():
            print(f"  {nm:9s} n={m.sum():5d}  |Δ|={np.abs(d[m]).mean():.4f}  "
                  f"aligned={(d[m]*(2*y[m]-1)).mean():+.4f}")


if __name__ == "__main__":
    main()

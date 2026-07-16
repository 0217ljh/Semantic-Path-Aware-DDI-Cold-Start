"""Round-16 analysis: PK/PD coverage + naked-vs-full error-flip on cold-start S2.

Two analyses, no training:
  Part A (coverage) — rule out a "PK wins only because it is more prevalent /
    higher-scale" artifact behind the round-15 PK-dominant ablation. Per test pair
    raw shared-enzyme (PK) and shared-target (PD) counts: fraction nonzero, count
    distribution, correlation, and univariate AUROC/AP of each raw count vs label.
  Part B (error-flip) — naked AND vs full pipeline per-pair on test: which pairs
    flip correct, bucketed by total shared-support count, and whether the rescued
    set is enriched for low-support / PK-positive pairs vs the harmed set.

Reads the saved prediction npz (run_spmn_v2_aware --save-predictions):
  naked: aware0_and_seed{S}_..._test_s2_scores.npz
  full : aware1_and_seed{S}_mech_sdeg_dpool_topk_bf_test_s2_scores.npz
Aligns the two by (drug_a_id, drug_b_id). Aggregates over seeds 42/43/44.
"""
from __future__ import annotations

import glob

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from run_spmn_v2_aware import _load_frames, _load_support_cache, density
from run_spmn_v2_standalone import ROOT

RES = ROOT / "Code/runs/spmn_v2_standalone"
SEEDS = (42, 43, 44)


def _find_npz(pattern: str) -> str | None:
    hits = sorted(glob.glob(str(RES / pattern)))
    return hits[0] if hits else None


def _pair_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([f"{x}|{y}" for x, y in zip(a, b)])


def main() -> None:
    cov_enz, cov_tgt, all_enz, all_tgt, all_y = [], [], [], [], []
    # error-flip accumulators
    fl_supp, fl_enz, fl_y, fl_naked_correct, fl_full_correct = [], [], [], [], []

    for s in SEEDS:
        frames = _load_frames(s)
        data = _load_support_cache("and", s, 3, 64, 400, False)
        te = data["test_s2"]
        n = len(te["y"])
        # raw mech counts (n,3): col0 enzyme(PK), col1 target(PD), col2 transporter
        mech = density.build_mech_feature_matrix(te["rela"], te["relb"],
                                                 te["offsets"], n)
        enz = mech[:, 0].astype(np.float64)
        tgt = mech[:, 1].astype(np.float64)
        supp = np.diff(te["offsets"]).astype(np.float64)   # shared-support count
        y = te["y"].astype(np.float64)
        all_enz.append(enz); all_tgt.append(tgt); all_y.append(y)
        cov_enz.append((enz > 0).mean()); cov_tgt.append((tgt > 0).mean())

        # align predictions by (a,b)
        fr = frames["test_s2"]
        key_frame = _pair_key(fr["drug_a_id"].astype(str).to_numpy(),
                              fr["drug_b_id"].astype(str).to_numpy())
        nk = _find_npz(f"aware0_and_seed{s}_*test_s2_scores.npz")
        fu = _find_npz(f"aware1_and_seed{s}_mech_sdeg_dpool_topk_bf_test_s2_scores.npz")
        if nk is None or fu is None:
            print(f"[seed {s}] MISSING npz naked={nk} full={fu}; skip flip")
            continue
        zn = np.load(nk, allow_pickle=True); zf = np.load(fu, allow_pickle=True)
        kn = _pair_key(zn["pair_a"].astype(str), zn["pair_b"].astype(str))
        kf = _pair_key(zf["pair_a"].astype(str), zf["pair_b"].astype(str))
        # map full + cache(=frame order) onto the naked key order
        idx_f = {k: i for i, k in enumerate(kf)}
        idx_c = {k: i for i, k in enumerate(key_frame)}
        ok = [k in idx_f and k in idx_c for k in kn]
        ok = np.array(ok)
        kn_ok = kn[ok]
        sn = zn["y_score"][ok].astype(np.float64)
        sf = np.array([zf["y_score"][idx_f[k]] for k in kn_ok], dtype=np.float64)
        yy = zn["y_true"][ok].astype(np.float64)
        ci = np.array([idx_c[k] for k in kn_ok])
        fl_supp.append(supp[ci]); fl_enz.append(enz[ci]); fl_y.append(yy)
        # per-pair "correct" via 0.5 threshold (label-agnostic ranking is AUROC;
        # for flips we need a per-pair decision, 0.5 on the sigmoid prob)
        fl_naked_correct.append(((sn >= 0.5) == (yy >= 0.5)).astype(np.float64))
        fl_full_correct.append(((sf >= 0.5) == (yy >= 0.5)).astype(np.float64))

    enz = np.concatenate(all_enz); tgt = np.concatenate(all_tgt); y = np.concatenate(all_y)
    print("=" * 64)
    print("PART A — PK vs PD raw-count coverage (test, pooled 3 seeds)")
    print("=" * 64)
    print(f"n test pairs (pooled)        : {len(y)}")
    print(f"frac with shared-enzyme(PK)>0: {np.mean(cov_enz):.4f}  (per-seed {[round(c,4) for c in cov_enz]})")
    print(f"frac with shared-target(PD)>0: {np.mean(cov_tgt):.4f}  (per-seed {[round(c,4) for c in cov_tgt]})")
    nzc = enz[enz > 0]; nzt = tgt[tgt > 0]
    print(f"enzyme count|>0  mean/median/max: {nzc.mean():.2f}/{np.median(nzc):.0f}/{nzc.max():.0f}")
    print(f"target count|>0  mean/median/max: {nzt.mean():.2f}/{np.median(nzt):.0f}/{nzt.max():.0f}")
    print(f"corr(enzyme,target) pearson  : {np.corrcoef(enz, tgt)[0,1]:.4f}")
    print(f"univariate AUROC  enzyme={roc_auc_score(y, enz):.4f}  target={roc_auc_score(y, tgt):.4f}")
    print(f"univariate AP     enzyme={average_precision_score(y, enz):.4f}  target={average_precision_score(y, tgt):.4f}")
    print(f"  (label prevalence {y.mean():.4f})")

    if fl_supp:
        supp = np.concatenate(fl_supp); enzf = np.concatenate(fl_enz)
        yf = np.concatenate(fl_y)
        nc = np.concatenate(fl_naked_correct); fc = np.concatenate(fl_full_correct)
        rescued = (nc == 0) & (fc == 1)
        harmed = (nc == 1) & (fc == 0)
        print("\n" + "=" * 64)
        print("PART B — naked AND vs full pipeline error-flip (pooled 3 seeds)")
        print("=" * 64)
        print(f"aligned pairs                : {len(yf)}")
        print(f"naked  accuracy@0.5          : {nc.mean():.4f}")
        print(f"full   accuracy@0.5          : {fc.mean():.4f}")
        print(f"rescued (wrong->right)       : {rescued.sum()}  ({rescued.mean()*100:.2f}%)")
        print(f"harmed  (right->wrong)       : {harmed.sum()}  ({harmed.mean()*100:.2f}%)")
        print(f"net                          : {rescued.sum()-harmed.sum():+d}")
        # bucket flips by shared-support count
        bins = [(-0.5, 0.5), (0.5, 1.5), (1.5, 3.5), (3.5, 1e9)]
        names = ["supp=0", "supp=1", "supp=2-3", "supp>=4"]
        print("\nsupport-count bucket | n | naked_acc | full_acc | rescued | harmed")
        for (lo, hi), nm in zip(bins, names):
            m = (supp > lo) & (supp <= hi)
            if m.sum() == 0:
                continue
            print(f"  {nm:9s} | {m.sum():5d} | {nc[m].mean():.4f} | {fc[m].mean():.4f} | "
                  f"{(rescued&m).sum():4d} | {(harmed&m).sum():4d}")
        # PK enrichment of rescued vs harmed vs overall
        print("\nPK(shared-enzyme>0) prevalence:")
        print(f"  overall  : {(enzf>0).mean():.4f}")
        print(f"  rescued  : {(enzf[rescued]>0).mean():.4f}  (n={rescued.sum()})")
        print(f"  harmed   : {(enzf[harmed]>0).mean():.4f}  (n={harmed.sum()})")
        print(f"  mean shared-support: rescued={supp[rescued].mean():.2f} "
              f"harmed={supp[harmed].mean():.2f} overall={supp.mean():.2f}")


if __name__ == "__main__":
    main()

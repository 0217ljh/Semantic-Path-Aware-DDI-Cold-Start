"""Does the adapter rescue exactly the ASYMMETRIC-heavy pairs? (story-defining)

Thesis: drug-readout GNNs (and bounded-depth path methods like NBFNet-L3) miss
ASYMMETRIC common-reachability — mediators near one drug, far from the other,
at longer total distance. The adapter explicitly captures it. If true, the pairs
the adapter gets right but NBFNet gets wrong should be disproportionately
asymmetric.

Joins three artifacts (all on the same seed42 S2 test set):
  - adapter per-pair scores (sum asym-on, from --save-predictions), index-aligned
    to the SUM support cache (both built from frames["test_s2"] in order),
  - NBFNet per-pair scores (its test_s2_scores.npz), joined by drug pair,
  - the SUM support cache (per-mediator d_a/d_b) -> per-pair asymmetry score.

Asymmetry score per pair = fraction of support mediators in LONG-ASYMMETRIC cells
(d_a != d_b AND d_a+d_b >= 4) — the cells AND(l=2) drops and NBFNet(L=3) can't
reach. Reports (1) mean asym score per E0 correctness group, (2) AUC per
asymmetry quartile for NBFNet vs adapter vs rank-blend. Read-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
CACHE = (ROOT / "Code/data/_cache/"
         "spmn_v2_supports_sum_seed42_lmax5_kpt128_nmax1200_cp0.npz")
ADP = (ROOT / "Code/runs/spmn_v2_standalone/"
       "sum_asym1_absdiff0_seed42_test_s2_scores.npz")
NBF = (ROOT / "Code/runs/2026-06-06_15-55-42__run_nbfnet_v1_71__"
       "nbfnet_v1_71_merged_seed42__seed42/test_s2_scores.npz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--adapter-npz", default=str(ADP))
    ap.add_argument("--nbfnet-npz", default=str(NBF))
    args = ap.parse_args()

    # adapter scores (index = cache test_s2 order)
    za = np.load(args.adapter_npz, allow_pickle=True)
    pa = za["pair_a"].astype(str); pb = za["pair_b"].astype(str)
    y = za["y_true"].astype(int); s_adp = za["y_score"].astype(float)
    n = len(y)

    # SUM cache: per-pair support d_a/d_b via offsets (same order as adapter npz)
    zc = np.load(args.cache)
    off = zc["test_s2__offsets"]; da = zc["test_s2__da"]; db = zc["test_s2__db"]
    yc = zc["test_s2__y"].astype(int)
    assert len(off) - 1 == n, f"cache n={len(off)-1} != adapter n={n}"
    assert (yc == y).all(), "cache/adapter label order mismatch — alignment broken"

    long_asym_frac = np.zeros(n); frac_asym = np.zeros(n); n_med = np.zeros(n, int)
    for i in range(n):
        s, e = off[i], off[i + 1]
        n_med[i] = e - s
        if e > s:
            dai = da[s:e].astype(int); dbi = db[s:e].astype(int)
            asym = dai != dbi
            frac_asym[i] = asym.mean()
            long_asym_frac[i] = (asym & (dai + dbi >= 4)).mean()

    # NBFNet scores joined by drug pair
    zn = np.load(args.nbfnet_npz, allow_pickle=True)
    nlut = {(str(a), str(b)): float(sc) for a, b, sc in
            zip(zn["pair_a"], zn["pair_b"], zn["y_score"])}
    s_nbf = np.full(n, np.nan)
    for i in range(n):
        v = nlut.get((pa[i], pb[i])) or nlut.get((pb[i], pa[i]))
        if v is not None:
            s_nbf[i] = v
    ok = ~np.isnan(s_nbf)
    print(f"[asym-rescue] pairs={n}, nbfnet-aligned={ok.sum()}")
    y, s_adp, s_nbf = y[ok], s_adp[ok], s_nbf[ok]
    long_asym_frac, frac_asym, n_med = long_asym_frac[ok], frac_asym[ok], n_med[ok]

    adp_ok = (s_adp > 0.5) == (y > 0.5)
    nbf_ok = (s_nbf > 0.5) == (y > 0.5)
    groups = {
        "both right": adp_ok & nbf_ok,
        "only nbfnet right": nbf_ok & ~adp_ok,
        "only ADAPTER right": adp_ok & ~nbf_ok,
        "neither": ~adp_ok & ~nbf_ok,
    }

    print("\n[1] mean asymmetry score per E0 correctness group:")
    print(f"  {'group':<20} {'n':>5} {'long-asym frac':>15} {'frac asym':>11} {'supp':>7}")
    for name, m in groups.items():
        if m.sum():
            print(f"  {name:<20} {int(m.sum()):>5} {long_asym_frac[m].mean():>15.3f} "
                  f"{frac_asym[m].mean():>11.3f} {n_med[m].mean():>7.0f}")
    overall = long_asym_frac.mean()
    print(f"  {'(overall)':<20} {len(y):>5} {overall:>15.3f} {frac_asym.mean():>11.3f} "
          f"{n_med.mean():>7.0f}")
    oa = groups["only ADAPTER right"]; on = groups["only nbfnet right"]
    print(f"\n  -> only-ADAPTER long-asym frac {long_asym_frac[oa].mean():.3f} vs "
          f"only-nbfnet {long_asym_frac[on].mean():.3f} vs overall {overall:.3f}")
    print("     (story holds if only-ADAPTER > only-nbfnet and > overall)")

    print("\n[2] AUC by asymmetry quartile (low->high long-asym frac):")
    q = np.quantile(long_asym_frac, [0.25, 0.5, 0.75])
    binid = np.digitize(long_asym_frac, q)
    print(f"  {'quartile':<10} {'n':>5} {'nbfnet':>8} {'adapter':>8} {'blend':>8}")
    ra = np.argsort(np.argsort(s_adp)) / (len(s_adp) - 1)
    rn = np.argsort(np.argsort(s_nbf)) / (len(s_nbf) - 1)
    blend = 0.5 * ra + 0.5 * rn
    for q_ in range(4):
        m = binid == q_
        if m.sum() > 20 and 0 < y[m].mean() < 1:
            print(f"  Q{q_+1:<8} {int(m.sum()):>5} {roc_auc_score(y[m], s_nbf[m]):>8.4f} "
                  f"{roc_auc_score(y[m], s_adp[m]):>8.4f} {roc_auc_score(y[m], blend[m]):>8.4f}")
    print("  (story holds if adapter/blend gain over nbfnet GROWS toward high-asym Q4)")


if __name__ == "__main__":
    main()

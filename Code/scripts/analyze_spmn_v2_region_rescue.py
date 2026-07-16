"""Where does the adapter actually beat NBFNet? Stratify by (d_a,d_b) REGION.

Tests two refined hypotheses (vs the refuted "long-asymmetric" one):
  - both-far / central: mediators with both d_a>=2 and d_b>=2 are (a) most diluted
    in a drug-readout GNN (far from BOTH drugs) and (b) invisible to NBFNet's
    drug-incident KG (reachable only via protein-protein / pathway edges NBFNet
    dropped). The symmetric centre (2,2) is the extreme case.
  - protein-specific: maybe the effect is carried by protein mediators in those
    cells.

For each region mask over (d_a,d_b) [optionally restricted to protein type], we
compute each pair's support fraction in that region, then report NBFNet / adapter
/ rank-blend AUC in the lowest vs highest quartile of that fraction. The story
for a region holds if (blend - nbfnet) GROWS from Q1 to Q4 — i.e. the adapter's
lift concentrates where that region dominates the support. Read-only.
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
PROTEIN_TYPE = 0  # KIND_ORDER[0] == "protein_gene"


def _region_masks(da, db, typ):
    prot = typ == PROTEIN_TYPE
    return {
        "near (1,1)":        (da == 1) & (db == 1),
        "one-arm=1":         ((da == 1) | (db == 1)) & ~((da == 1) & (db == 1)),
        "both-far (>=2,>=2)": (da >= 2) & (db >= 2),
        "sym-centre (2,2)":  (da == 2) & (db == 2),
        "asym-far (both>=2, da!=db)": (da >= 2) & (db >= 2) & (da != db),
        "protein both-far":  (da >= 2) & (db >= 2) & prot,
        "protein sym (2,2)": (da == 2) & (db == 2) & prot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--adapter-npz", default=str(ADP))
    ap.add_argument("--nbfnet-npz", default=str(NBF))
    args = ap.parse_args()

    za = np.load(args.adapter_npz, allow_pickle=True)
    pa = za["pair_a"].astype(str); pb = za["pair_b"].astype(str)
    y = za["y_true"].astype(int); s_adp = za["y_score"].astype(float)
    n = len(y)

    zc = np.load(args.cache)
    off = zc["test_s2__offsets"]; da = zc["test_s2__da"]; db = zc["test_s2__db"]
    typ = zc["test_s2__typ"]
    assert len(off) - 1 == n and (zc["test_s2__y"].astype(int) == y).all()

    region_names = list(_region_masks(np.array([1]), np.array([1]), np.array([0])).keys())
    frac = {r: np.zeros(n) for r in region_names}
    for i in range(n):
        s, e = off[i], off[i + 1]
        if e <= s:
            continue
        masks = _region_masks(da[s:e].astype(int), db[s:e].astype(int),
                              typ[s:e].astype(int))
        for r, m in masks.items():
            frac[r][i] = m.mean()

    zn = np.load(args.nbfnet_npz, allow_pickle=True)
    nlut = {(str(a), str(b)): float(sc) for a, b, sc in
            zip(zn["pair_a"], zn["pair_b"], zn["y_score"])}
    s_nbf = np.array([nlut.get((pa[i], pb[i]), nlut.get((pb[i], pa[i]), np.nan))
                      for i in range(n)])
    m = ~np.isnan(s_nbf)
    y, s_adp, s_nbf = y[m], s_adp[m], s_nbf[m]
    for r in region_names:
        frac[r] = frac[r][m]
    ra = np.argsort(np.argsort(s_adp)) / (len(s_adp) - 1)
    rn = np.argsort(np.argsort(s_nbf)) / (len(s_nbf) - 1)
    blend = 0.5 * ra + 0.5 * rn
    print(f"[region-rescue] pairs={len(y)}  (nbfnet AUC overall "
          f"{roc_auc_score(y, s_nbf):.4f}, adapter {roc_auc_score(y, s_adp):.4f}, "
          f"blend {roc_auc_score(y, blend):.4f})\n")

    def auc(score, mask):
        return roc_auc_score(y[mask], score[mask]) if (mask.sum() > 20 and
                                                       0 < y[mask].mean() < 1) else float("nan")

    print(f"{'region':<28} {'Q1 nbf/adp/bln':>22}   {'Q4 nbf/adp/bln':>22}   {'Δblend Q1→Q4':>12}")
    for r in region_names:
        f = frac[r]
        # low quartile vs high quartile of this region's fraction
        lo, hi = np.quantile(f, 0.25), np.quantile(f, 0.75)
        q1 = f <= lo; q4 = f >= hi
        n1 = auc(s_nbf, q1); a1 = auc(s_adp, q1); b1 = auc(blend, q1)
        n4 = auc(s_nbf, q4); a4 = auc(s_adp, q4); b4 = auc(blend, q4)
        d1, d4 = b1 - n1, b4 - n4
        print(f"{r:<28} {n1:.3f}/{a1:.3f}/{b1:.3f}   {n4:.3f}/{a4:.3f}/{b4:.3f}   "
              f"{d1:+.3f}→{d4:+.3f}")
    print("\n  story for a region holds if Δblend grows (more +) from Q1 to Q4,")
    print("  and ideally adapter AUC in Q4 is competitive (not collapsing).")


if __name__ == "__main__":
    main()

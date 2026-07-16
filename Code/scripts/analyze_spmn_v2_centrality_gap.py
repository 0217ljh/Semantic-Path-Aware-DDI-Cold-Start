"""Load-bearing figure for Claim ①: does the AND-adapter's advantage over EmerGNN
GROW with mediator chain-centrality?

Story (spmn_v2_theory_anchor Claim ①): a shared mediator becomes usable evidence once
each drug reaches it (radius max(d_a,d_b)); a propagation/flow GNN (EmerGNN) must
transport signal across the full d_a+d_b route, so CENTRAL mediators (balanced-deep,
extreme = (2,2): 2 hops from BOTH drugs) are the most diluted. AND pools position-
agnostically → equal status. Prediction: AND_AUROC − EmerGNN_AUROC grows as a pair's
support becomes more central.

This is the CLEAN head-to-head (adapter-alone vs EmerGNN-alone per centrality bin), vs
the earlier region_rescue proof-of-concept (weak SUM-asym adapter + NBFNet + blend).

Reads (read-only, no training):
  AND best per-pair : aware1_and_seed{S}_mech_sdeg_dpool_topk_bf_test_s2_scores.npz
  EmerGNN per-pair  : <emergnn aligned S2 run>/predictions_s2.parquet (drug_a_id,drug_b_id,label,pred)
  support cache     : _load_support_cache('and', S, 3, 64, 400, False) for per-pair (d_a,d_b)
AND npz row order == frames test_s2 order == support-cache order (1:1).
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_spmn_v2_aware import _load_frames, _load_support_cache
from run_spmn_v2_standalone import ROOT

AND_TAG = "aware1_and_seed{s}_mech_sdeg_dpool_topk_bf_test_s2_scores.npz"
EMER = ROOT / ("Code/runs/2026-05-18_14-43-23__run_baseline__emergnn_binary_"
               "seed42_aligned__seed42/predictions_s2.parquet")


def _canon(a, b):
    return (a, b) if a <= b else (b, a)


def _regions(da, db):
    """per-mediator boolean region membership (AND support: da,db in {1,2})."""
    return {
        "near(1,1)": (da == 1) & (db == 1),
        "one-arm=1": ((da == 1) ^ (db == 1)),
        "sym-centre(2,2)": (da == 2) & (db == 2),
        "both>=2": (da >= 2) & (db >= 2),
    }


def _auc(y, s, mask):
    if mask.sum() < 30 or len(np.unique(y[mask])) < 2:
        return float("nan")
    return roc_auc_score(y[mask], s[mask])


def main() -> None:
    s = 42
    z = np.load(glob.glob(str(ROOT / "Code/runs/spmn_v2_standalone" / AND_TAG.format(s=s)))[0],
                allow_pickle=True)
    a_pa = z["pair_a"].astype(str); a_pb = z["pair_b"].astype(str)
    y = z["y_true"].astype(int); s_and = z["y_score"].astype(float)
    n = len(y)

    em = pd.read_parquet(EMER)
    em_key = {_canon(str(r.drug_a_id), str(r.drug_b_id)): (float(r.pred), int(r.label))
              for r in em.itertuples()}
    s_emer = np.full(n, np.nan); ok = np.zeros(n, bool)
    for i in range(n):
        k = _canon(a_pa[i], a_pb[i])
        if k in em_key:
            s_emer[i] = em_key[k][0]; ok[i] = True
    print(f"[centrality-gap] seed{s}: AND pairs {n}, matched to EmerGNN {ok.sum()}")

    # per-pair centrality composition from support cache
    data = _load_support_cache("and", s, 3, 64, 400, False)
    te = data["test_s2"]; off = te["offsets"]; da = te["da"]; db = te["db"]
    frames = _load_frames(s); fr = frames["test_s2"]
    # sanity: cache order == npz order (both frames order)
    assert len(fr) == n, f"frame {len(fr)} != npz {n}"
    reg_names = list(_regions(np.array([1]), np.array([1])).keys())
    frac = {r: np.zeros(n) for r in reg_names}
    supp = np.zeros(n)
    for i in range(n):
        s0, e0 = int(off[i]), int(off[i + 1])
        supp[i] = e0 - s0
        if e0 <= s0:
            continue
        rr = _regions(da[s0:e0].astype(int), db[s0:e0].astype(int))
        for r in reg_names:
            frac[r][i] = rr[r].mean()

    m = ok & (supp > 0)
    y_m = y[m]; sa = s_and[m]; se = s_emer[m]
    print(f"  usable pairs (matched & supp>0): {m.sum()}")
    print(f"  overall AUROC: AND {_auc(y_m, sa, np.ones(m.sum(), bool)):.4f}  "
          f"EmerGNN {_auc(y_m, se, np.ones(m.sum(), bool)):.4f}  "
          f"gap {_auc(y_m, sa, np.ones(m.sum(),bool)) - _auc(y_m, se, np.ones(m.sum(),bool)):+.4f}")

    print("\n=== AUROC by CENTRALITY quartile (pairs binned by sym-centre(2,2) support fraction) ===")
    c = frac["sym-centre(2,2)"][m]
    # quartiles by centrality score (ties -> use rank)
    order = np.argsort(c, kind="stable")
    q = np.zeros(m.sum(), int); q[order] = (np.arange(m.sum()) * 4 // m.sum()).clip(0, 3)
    print("quartile |   n | mean (2,2)-frac | AND AUROC | EmerGNN AUROC |   gap")
    for qi in range(4):
        mm = q == qi
        print(f"   Q{qi+1}    | {mm.sum():4d} |     {c[mm].mean():.3f}      |  "
              f"{_auc(y_m, sa, mm):.4f}  |   {_auc(y_m, se, mm):.4f}    | "
              f"{_auc(y_m, sa, mm) - _auc(y_m, se, mm):+.4f}")

    print("\n=== AUROC by region: pairs with HIGH (top-tertile) region support fraction ===")
    print("region            |   n | AND AUROC | EmerGNN AUROC |   gap")
    for r in reg_names:
        f = frac[r][m]
        thr = np.quantile(f[f > 0], 0.66) if (f > 0).sum() > 30 else 1.0
        mm = f >= max(thr, 1e-9)
        if mm.sum() < 30:
            continue
        print(f"  {r:15s} | {mm.sum():4d} |  {_auc(y_m, sa, mm):.4f}  |   "
              f"{_auc(y_m, se, mm):.4f}    | {_auc(y_m, sa, mm) - _auc(y_m, se, mm):+.4f}")


if __name__ == "__main__":
    main()

"""E0 — complementarity gate: does the SUM asymmetric adapter add to NBFNet?

Loads the per-pair S2 test scores of (a) NBFNet and (b) the standalone adapter,
aligns them by drug pair, and asks whether the two carry COMPLEMENTARY signal —
the precondition for the fusion head to beat NBFNet alone. Three read-outs:

  1. Score correlation (Spearman). Low corr => the models rank pairs differently.
  2. Rank-normalised convex-combination AUC sweep w*nbfnet + (1-w)*adapter. If any
     interior w beats w=1 (pure NBFNet), a learned fusion has headroom. This is a
     NON-circular lower bound (fixed weights, no fitting on the test set).
  3. Error-set overlap at the 0.5 threshold — how many pairs the adapter gets
     right that NBFNet gets wrong (the complementary cell).

Read-only. No GPU, no training. If the convex sweep never beats pure NBFNet and
correlation is high, fusion is unlikely to help and we should stop / re-scope.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
NBF_DEFAULT = (ROOT / "Code/runs/2026-06-06_15-55-42__run_nbfnet_v1_71__"
               "nbfnet_v1_71_merged_seed42__seed42/test_s2_scores.npz")
ADP_DEFAULT = (ROOT / "Code/runs/spmn_v2_standalone/"
               "sum_asym1_absdiff0_seed42_test_s2_scores.npz")


def _load(path):
    z = np.load(path, allow_pickle=True)
    a = z["pair_a"].astype(str); b = z["pair_b"].astype(str)
    lut = {}
    for i in range(len(a)):
        lut[(a[i], b[i])] = (float(z["y_true"][i]), float(z["y_score"][i]))
    return lut


def _rank01(x):
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbfnet-npz", default=str(NBF_DEFAULT))
    ap.add_argument("--adapter-npz", default=str(ADP_DEFAULT))
    args = ap.parse_args()

    nbf = _load(args.nbfnet_npz)
    adp = _load(args.adapter_npz)
    print(f"[E0] nbfnet pairs={len(nbf)}  adapter pairs={len(adp)}")

    y, s_nbf, s_adp, n_rev, n_miss = [], [], [], 0, 0
    for key, (yt, sn) in nbf.items():
        hit = adp.get(key)
        if hit is None:
            hit = adp.get((key[1], key[0]))  # try reversed orientation
            if hit is not None:
                n_rev += 1
        if hit is None:
            n_miss += 1
            continue
        ya, sa = hit
        if int(ya) != int(yt):           # label disagreement => misalignment
            n_miss += 1
            continue
        y.append(yt); s_nbf.append(sn); s_adp.append(sa)
    y = np.array(y); s_nbf = np.array(s_nbf); s_adp = np.array(s_adp)
    print(f"[E0] aligned={len(y)} (reversed-orient={n_rev}, unmatched={n_miss})")
    if len(y) < 100:
        print("[E0] too few aligned pairs — check the two score files"); sys.exit(1)

    auc_nbf = roc_auc_score(y, s_nbf)
    auc_adp = roc_auc_score(y, s_adp)
    rho = spearmanr(s_nbf, s_adp).statistic
    print(f"\n[E0] sanity AUC: nbfnet={auc_nbf:.4f}  adapter={auc_adp:.4f}")
    print(f"[E0] Spearman(score_nbf, score_adp) = {rho:.3f}  "
          f"(low => complementary ranking)")

    # --- convex-combination AUC sweep (rank-normalised, non-circular) ---
    r_nbf, r_adp = _rank01(s_nbf), _rank01(s_adp)
    print("\n[E0] rank-normalised convex combo  w*nbfnet + (1-w)*adapter:")
    print("   w     AUC      vs pure-nbfnet")
    best_w, best_auc = 1.0, auc_nbf
    for w in np.linspace(0.0, 1.0, 11):
        auc_w = roc_auc_score(y, w * r_nbf + (1 - w) * r_adp)
        flag = "  <-- beats nbfnet" if auc_w > auc_nbf + 1e-9 else ""
        print(f"  {w:.1f}   {auc_w:.4f}   {auc_w - auc_nbf:+.4f}{flag}")
        if auc_w > best_auc:
            best_w, best_auc = w, auc_w
    verdict = ("HEADROOM: a fixed blend beats pure NBFNet -> fusion worth building"
               if best_auc > auc_nbf + 1e-9 else
               "NO headroom from a fixed blend -> fusion unlikely to help as-is")
    print(f"\n[E0] best blend w={best_w:.1f} AUC={best_auc:.4f} "
          f"(nbfnet alone {auc_nbf:.4f}, +{best_auc - auc_nbf:.4f}) => {verdict}")

    # --- error-set overlap at 0.5 ---
    ok_nbf = (s_nbf > 0.5) == (y > 0.5)
    ok_adp = (s_adp > 0.5) == (y > 0.5)
    both = int((ok_nbf & ok_adp).sum())
    only_nbf = int((ok_nbf & ~ok_adp).sum())
    only_adp = int((~ok_nbf & ok_adp).sum())
    neither = int((~ok_nbf & ~ok_adp).sum())
    print(f"\n[E0] correctness overlap @0.5 (n={len(y)}):")
    print(f"  both right        {both}")
    print(f"  only nbfnet right {only_nbf}")
    print(f"  only adapter right{only_adp:>5}   <-- adapter rescues NBFNet errors")
    print(f"  neither           {neither}")
    print(f"  adapter unique-correct rate = {only_adp / len(y):.3f}")


if __name__ == "__main__":
    main()

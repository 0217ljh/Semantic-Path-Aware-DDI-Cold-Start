"""Split an already-trained NBFNet's saved S2 predictions by PK vs PD mechanism.

Answers the real question (no ablation, DDI channel left in place exactly as the
model was trained): does the actual NBFNet, which already propagates over
seen-drug DDI edges, score pharmacodynamic (PD) positives worse than
pharmacokinetic (PK) positives on cold-start S2?

Reads a run's test_s2_scores.npz (pair_a, pair_b, y_true, y_score), maps each
positive pair to its DrugBank ddi_type -> PK/PD bucket, and reports per-bucket
AUROC (bucket positives vs the shared negatives) + score distributions.

Run (from project root, via WSL conda env project_1):
  python Code/scripts/analyze_nbfnet_s2_pkpd_split.py \
      --scores Code/runs/2026-06-06_00-14-08__run_nbfnet__nbfnet_merged_L3_dim16__seed42/test_s2_scores.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]  # -> Code/
DDI_POS_CSV = ROOT / "data/KG/drugbank/filtered/ddi_edges.csv"


def pkpd_bucket(t: str) -> str:
    t = str(t).lower()
    if any(k in t for k in ("metabolism", "excretion", "serum concentration",
                            "absorption", "protein binding", "clearance")):
        return "PK"
    if any(k in t for k in ("risk or severity", "activities", "efficacy",
                            "cns depression", "qtc", "hypertension",
                            "hypotensive", "sedative", "adverse effects")):
        return "PD"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, type=str)
    args = ap.parse_args()

    d = np.load(args.scores, allow_pickle=True)
    a = d["pair_a"].astype(str); b = d["pair_b"].astype(str)
    y = d["y_true"].astype(int); s = d["y_score"].astype(float)
    print(f"[load] {args.scores}  n={len(y)}  pos={int(y.sum())} neg={int((y==0).sum())}")
    print(f"[sanity] overall S2 AUROC on this file = {roc_auc_score(y, s):.4f}")

    # map positive pairs -> ddi_type via unordered key
    pos = pd.read_csv(DDI_POS_CSV)
    key2type = {}
    for x, yy, t in zip(pos["drug_a_id"].astype(str), pos["drug_b_id"].astype(str),
                        pos["ddi_type"].astype(str)):
        key2type[frozenset((x, yy))] = t

    bucket = np.array(["neg"] * len(y), dtype=object)
    n_unmapped = 0
    for i in range(len(y)):
        if y[i] == 1:
            t = key2type.get(frozenset((a[i], b[i])))
            if t is None:
                bucket[i] = "pos_unmapped"; n_unmapped += 1
            else:
                bucket[i] = pkpd_bucket(t)
    print(f"[map] positives mapped to ddi_type; unmapped={n_unmapped}")

    neg_mask = y == 0
    neg_scores = s[neg_mask]
    print(f"\n================= NBFNet S2: PK vs PD (DDI channel present, no ablation) =================")
    print(f"{'bucket':14s} {'n_pos':>7s} {'mean_score':>11s} {'AUROC vs neg':>13s}")
    print(f"{'NEG (shared)':14s} {int(neg_mask.sum()):7d} {neg_scores.mean():11.4f} {'-':>13s}")
    for bk in ["PK", "PD", "other", "pos_unmapped"]:
        m = bucket == bk
        if m.sum() == 0:
            continue
        pos_scores = s[m]
        yy = np.concatenate([np.ones(m.sum()), np.zeros(neg_mask.sum())])
        ss = np.concatenate([pos_scores, neg_scores])
        auc = roc_auc_score(yy, ss) if m.sum() > 0 else float("nan")
        print(f"{bk:14s} {int(m.sum()):7d} {pos_scores.mean():11.4f} {auc:13.4f}")


if __name__ == "__main__":
    sys.exit(main())

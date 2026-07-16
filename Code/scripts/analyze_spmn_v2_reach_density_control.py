"""Control for the [D] pos-neg gap: is the discriminative shared-reachability a
genuine MECHANISM signal, or a density CONFOUND (positive pairs simply pair a
sparse drug with a denser partner, so they overlap more for free)?

Follow-up to ``analyze_spmn_v2_reach_vs_density.py`` table [D], which showed the
pos-neg Adamic-Adar gap widening with reach (strongest in the sparse bin Q1).
That gap is suspect: we binned pairs by ``min(deg1_a, deg1_b)`` (the sparser
arm), but the OTHER arm's density is free to differ between positives and
negatives. If real DDIs preferentially attach a sparse drug to a well-annotated
hub-drug, positives overlap more without any extra mechanism specificity.

Three controls at the sane operating point (reach l=2; l=3 for contrast):

  (1) confound  -- within each density bin, mean min-arm / max-arm 1-hop degree
                   for positives vs negatives. A large max-arm gap (pos > neg)
                   IS the confound.
  (2) normalize -- replace raw |A∩B| with density-invariant set overlap:
                   Jaccard |A∩B|/|A∪B| and overlap coef |A∩B|/min(|A|,|B|).
                   If the pos-neg gap SURVIVES normalization, positives overlap
                   MORE THAN their density predicts -> genuine mechanism. If it
                   vanishes, [D] was just density.
  (3) specificity -- restrict the intersection to NON-hub mediators
                   (degree <= hub_deg). If the surviving gap lives in specific
                   mediators (not hubs), the signal is the kind a specificity-
                   weighting adapter can actually exploit.

Read-only. Reuses the same bounded-BFS reachability as the reach diagnostic.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.spmn_v1.retrieval import MergedKG  # noqa: E402

THREE_SEED_DIR = ROOT / "Code/data/coldddi_legacy/800drug_3seed"


def _drug_reach(kg: MergedKG, idx: int, l_max: int):
    nodes, dist = kg.neighborhood(idx, l_max)
    keep = (~kg.is_drug[nodes]) & (dist >= 1)
    nodes, dist = nodes[keep], dist[keep]
    deg1 = int((dist == 1).sum())
    order = np.argsort(nodes, kind="stable")
    return nodes[order], dist[order].astype(np.int64), deg1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--hub-deg", type=int, default=1000)
    args = ap.parse_args()
    L = args.l_max
    levels = [l for l in (2, 3) if l <= L]

    frame = pd.read_parquet(THREE_SEED_DIR / f"seed{args.seed}" / "test_s2.parquet")
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    y = frame["label"].to_numpy().astype(int)

    print("[control] loading merged KG ...", flush=True)
    kg = MergedKG.from_parquet()
    uniq = sorted(set(a_ids.tolist()) | set(b_ids.tolist()))
    reach: dict[str, tuple] = {}
    deg1: dict[str, int] = {}
    t0 = time.time()
    for did in uniq:
        idx = kg.id_to_idx.get(did)
        if idx is not None:
            reach[did] = _drug_reach(kg, idx, L)
            deg1[did] = reach[did][2]
    print(f"[control] BFS {len(reach)} drugs ({time.time()-t0:.1f}s)", flush=True)

    n = len(frame)
    min_deg = np.full(n, -1.0); max_deg = np.full(n, -1.0)
    # per-level metrics
    raw = {l: np.zeros(n) for l in levels}
    jac = {l: np.zeros(n) for l in levels}
    ovl = {l: np.zeros(n) for l in levels}
    spec = {l: np.zeros(n) for l in levels}           # non-hub intersection count
    spec_ovl = {l: np.zeros(n) for l in levels}       # non-hub overlap coef
    valid = np.zeros(n, dtype=bool)

    for i in range(n):
        ra, rb = reach.get(a_ids[i]), reach.get(b_ids[i])
        if ra is None or rb is None or a_ids[i] == b_ids[i]:
            continue
        na, da_all, d1a = ra
        nb, db_all, d1b = rb
        valid[i] = True
        min_deg[i] = min(d1a, d1b); max_deg[i] = max(d1a, d1b)
        common, ia, ib = np.intersect1d(na, nb, assume_unique=True,
                                        return_indices=True)
        if common.size == 0:
            continue
        da = da_all[ia]; db = db_all[ib]
        mx = np.maximum(da, db)
        deg = kg.degree[common]
        nothub = deg <= args.hub_deg
        for l in levels:
            sa = int((da_all <= l).sum()); sb = int((db_all <= l).sum())
            m = mx <= l
            inter = int(m.sum())
            union = sa + sb - inter
            mn = min(sa, sb)
            raw[l][i] = inter
            jac[l][i] = inter / union if union > 0 else 0.0
            ovl[l][i] = inter / mn if mn > 0 else 0.0
            si = int((m & nothub).sum())
            spec[l][i] = si
            spec_ovl[l][i] = si / mn if mn > 0 else 0.0

    v = valid
    dens = min_deg[v]; yv = y[v]
    qs = np.quantile(dens, np.linspace(0, 1, args.n_bins + 1))
    binid = np.clip(np.digitize(dens, qs[1:-1]), 0, args.n_bins - 1)
    mnv, mxv = min_deg[v], max_deg[v]

    print(f"\n[control] {v.sum()} valid pairs; density bin edges (min-arm deg1): "
          f"{np.round(qs,1).tolist()}")

    def grp(arr, b, lab):
        m = (binid == b) & (yv == lab)
        return arr[v][m].mean() if m.any() else float("nan")

    print("\n[1] CONFOUND CHECK -- mean 1-hop degree of min-arm / max-arm, "
          "pos vs neg per density bin")
    print(f"  {'bin':<8}  min-arm pos/neg     max-arm pos/neg   (max-arm gap pos-neg)")
    for b in range(args.n_bins):
        mp = grp(mnv, b, 1); mn_ = grp(mnv, b, 0)
        xp = grp(mxv, b, 1); xn = grp(mxv, b, 0)
        print(f"  Q{b+1:<6}  {mp:6.1f} / {mn_:6.1f}      {xp:7.1f} / {xn:7.1f}    "
              f"({xp-xn:+.1f})")
    print("  -> if max-arm pos >> neg (esp Q1), positives are denser-for-free = confound")

    for l in levels:
        print(f"\n[2] NORMALIZED OVERLAP @ reach l={l} -- pos / neg / gap "
              f"(does the [D] gap survive density normalization?)")
        print(f"  {'bin':<8}  {'raw |A∩B|':>20}   {'Jaccard':>20}   {'overlap-coef':>20}")
        for b in range(args.n_bins):
            rp, rn = grp(raw[l], b, 1), grp(raw[l], b, 0)
            jp, jn = grp(jac[l], b, 1), grp(jac[l], b, 0)
            op, on = grp(ovl[l], b, 1), grp(ovl[l], b, 0)
            print(f"  Q{b+1:<6}  {rp:7.1f}/{rn:7.1f}/{rp-rn:+6.1f}   "
                  f"{jp:.3f}/{jn:.3f}/{jp-jn:+.3f}   "
                  f"{op:.3f}/{on:.3f}/{op-on:+.3f}")

    l0 = levels[0]
    print(f"\n[3] SPECIFICITY @ reach l={l0} -- NON-hub (deg<={args.hub_deg}) "
          f"intersection: pos / neg / gap")
    print(f"  {'bin':<8}  {'non-hub count':>22}   {'non-hub overlap-coef':>24}")
    for b in range(args.n_bins):
        sp, sn = grp(spec[l0], b, 1), grp(spec[l0], b, 0)
        op, on = grp(spec_ovl[l0], b, 1), grp(spec_ovl[l0], b, 0)
        print(f"  Q{b+1:<6}  {sp:7.1f}/{sn:7.1f}/{sp-sn:+7.1f}   "
              f"{op:.3f}/{on:.3f}/{op-on:+.3f}")
    print("  -> signal is exploitable iff pos>neg SURVIVES in non-hub mediators "
          "(not carried only by hubs)")

    print("\nverdict: [D] is genuine mechanism (not density artifact) iff the "
          "Jaccard/overlap gap stays clearly positive AFTER normalization AND the "
          "non-hub gap stays positive -- esp. in sparse Q1.")


if __name__ == "__main__":
    main()

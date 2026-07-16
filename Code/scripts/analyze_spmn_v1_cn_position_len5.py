"""Common-neighbor POSITION breakdown by (d_a, d_b), corridor length fixed.

Clarifies the original "structural indicator" statistic. For each positive
test_s2 pair we build the corridor support and tabulate the (d_a, d_b) position
of every shared mediator, where d_a = d_sym(a, v), d_b = d_sym(v, b).

Budget mode (matches retrieval.build_pair_support):
  --mode sum  (default):  keep mediators with d_a + d_b <= l_max  (WALK LENGTH)
  --mode and          :   keep mediators with d_a <= l_max-1 AND d_b <= l_max-1

So "fixed length 5" == --mode sum --l-max 5  (d_a + d_b <= 5).
The earlier (1.6% / 93.2%) numbers were --mode and --l-max 3 (positions up to (2,2)).

Verified data only. Run:
  python Code/scripts/analyze_spmn_v1_cn_position_len5.py --mode sum --l-max 5 --n-pos 1919
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import MergedKG  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["sum", "and"], default="sum")
    ap.add_argument("--l-max", type=int, default=5)
    ap.add_argument("--n-pos", type=int, default=1919)
    args = ap.parse_args()

    support_and = args.mode == "and"
    depth = max(0, args.l_max - 1)
    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].reset_index(drop=True).iloc[:args.n_pos]
    kg = MergedKG.from_parquet()

    dist_cnt: Counter = Counter()        # (d_a, d_b) -> count of mediators
    n_pairs_used = 0
    n_pairs_empty = 0
    per_pair_support = []

    # RAW UNCAPPED typed common-reachability: intersect bounded neighbourhoods
    # directly (no per-type top-k / global cap, no degree-penalty selection).
    # This is the true common-neighbor structural distribution.
    for i in range(len(pos)):
        ai = kg.id_to_idx.get(str(pos.iloc[i, 0]))
        bi = kg.id_to_idx.get(str(pos.iloc[i, 1]))
        if ai is None or bi is None or ai == bi:
            continue
        nodes_a, dist_a = kg.neighborhood(ai, depth)
        nodes_b, dist_b = kg.neighborhood(bi, depth)
        common, ia, ib = np.intersect1d(nodes_a, nodes_b, return_indices=True)
        if common.size == 0:
            n_pairs_empty += 1
            continue
        da = dist_a[ia].astype(np.int64)
        db = dist_b[ib].astype(np.int64)
        keep = (da > 0) & (db > 0)                       # exclude the two drugs
        if not support_and:
            keep &= (da + db) <= args.l_max              # sum budget = walk length
        da = da[keep]; db = db[keep]
        if da.size == 0:
            n_pairs_empty += 1
            continue
        n_pairs_used += 1
        per_pair_support.append(int(da.size))
        for k in range(da.size):
            dist_cnt[(int(da[k]), int(db[k]))] += 1

    tot = sum(dist_cnt.values())
    print(f"=== CN position breakdown (test_s2 positives) ===")
    print(f"mode={args.mode}  l_max={args.l_max}  "
          f"({'d_a+d_b<=l_max' if not support_and else 'd_a<=l_max-1 AND d_b<=l_max-1'})")
    print(f"pairs used={n_pairs_used}  empty={n_pairs_empty}  "
          f"total mediators={tot}  mean support/pair={np.mean(per_pair_support):.1f}\n")

    print(f"-- (d_a, d_b) position distribution --")
    # sort by (d_a+d_b, d_a) for readability
    for (da, db) in sorted(dist_cnt, key=lambda p: (p[0] + p[1], p[0])):
        c = dist_cnt[(da, db)]
        print(f"  d_a={da} d_b={db}  (len={da+db}): {c:9d}  ({100*c/max(1,tot):5.2f}%)")

    print(f"\n-- by total walk length (d_a+d_b) --")
    len_cnt: Counter = Counter()
    for (da, db), c in dist_cnt.items():
        len_cnt[da + db] += c
    for L in sorted(len_cnt):
        print(f"  length {L}: {len_cnt[L]:9d}  ({100*len_cnt[L]/max(1,tot):5.2f}%)")

    direct = dist_cnt.get((1, 1), 0)
    print(f"\n>> (1,1) direct common neighbor = {100*direct/max(1,tot):.2f}%")
    print(f">> everything beyond direct (need >1 hop on some side) = "
          f"{100*(tot-direct)/max(1,tot):.2f}%")


if __name__ == "__main__":
    main()

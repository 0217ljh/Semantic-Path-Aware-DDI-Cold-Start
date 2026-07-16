"""Per-(d_a, d_b) mediator-count census over S2 cold-start pairs.

Quantifies, on real data, how the mediator set grows when we move the support
budget from the current AND(l=2) box to the SUM triangle at L=4 and L=5.

For every S2 test pair (a, b) we BFS each drug to ``depth`` hops on the
symmetrised merged KG (cached per drug), intersect the two neighbourhoods, and
bin every shared *mediator* (non-drug, non-endpoint node) by the integer tuple
``(d_a, d_b) = (dist(a, u), dist(b, u))``. We then aggregate, across pairs:

  * total mediator count per (d_a, d_b) cell (summed over pairs),
  * mean per pair,
  * fraction of pairs with >=1 mediator in that cell.

The report slices the grid into the AND(l=2) baseline cells and the SUM
increments (L=4 adds the sum-4 asymmetric ring, L=5 adds the sum-5 ring), so the
"how big is the SUM-over-AND delta" question gets a factual answer instead of a
geometric guess.

Read-only analysis. No GPU, no training. Distances are undirected hop counts
(graph convention: length = number of edges).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import MergedKG  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def _binary_frame(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    p = pos[["drug_a_id", "drug_b_id"]].copy()
    p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy()
    n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--depth", type=int, default=4,
                    help="max single-arm hop budget (need 4 to reach cell (1,4))")
    ap.add_argument("--max-pairs", type=int, default=4000,
                    help="subsample this many S2 test pairs (0 = all)")
    ap.add_argument("--subsample-seed", type=int, default=0)
    args = ap.parse_args()

    pkl = PKL_DIR / f"seed{args.seed}.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(pkl)
    print(f"[census] dataset: {pkl}", flush=True)
    ds = PairDataset.from_pkl(str(pkl))
    test = _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))
    print(f"[census] test_s2 pairs (pos+neg) = {len(test)}", flush=True)

    if args.max_pairs and len(test) > args.max_pairs:
        rng = np.random.default_rng(args.subsample_seed)
        idx = rng.choice(len(test), size=args.max_pairs, replace=False)
        test = test.iloc[np.sort(idx)].reset_index(drop=True)
        print(f"[census] subsampled to {len(test)} pairs "
              f"(subsample_seed={args.subsample_seed})", flush=True)

    kg = MergedKG.from_parquet()
    depth = args.depth

    a_ids = test["drug_a_id"].astype(str).to_numpy()
    b_ids = test["drug_b_id"].astype(str).to_numpy()

    # accumulators keyed by (d_a, d_b)
    total = defaultdict(int)          # summed mediator count over pairs
    pairs_hit = defaultdict(int)      # # pairs with >=1 mediator in cell
    n_used = 0
    n_oov = 0
    t0 = time.time()

    for i in range(len(test)):
        ai = kg.id_to_idx.get(a_ids[i])
        bi = kg.id_to_idx.get(b_ids[i])
        if ai is None or bi is None or ai == bi:
            n_oov += 1
            continue
        nodes_a, dist_a = kg.neighborhood(ai, depth)
        nodes_b, dist_b = kg.neighborhood(bi, depth)
        common, ia, ib = np.intersect1d(
            nodes_a, nodes_b, assume_unique=True, return_indices=True)
        if common.size == 0:
            n_used += 1
            continue
        da = dist_a[ia].astype(np.int64)
        db = dist_b[ib].astype(np.int64)
        keep = (~kg.is_drug[common]) & (common != ai) & (common != bi)
        da = da[keep]
        db = db[keep]
        if da.size == 0:
            n_used += 1
            continue
        # one bincount over flattened (d_a, d_b) cells
        flat = da * (depth + 1) + db
        cells, counts = np.unique(flat, return_counts=True)
        for c, cnt in zip(cells.tolist(), counts.tolist()):
            key = (c // (depth + 1), c % (depth + 1))
            total[key] += int(cnt)
            pairs_hit[key] += 1
        n_used += 1
        if (i + 1) % 1000 == 0:
            print(f"  [{i + 1}/{len(test)}] {time.time() - t0:.0f}s", flush=True)

    print(f"[census] used {n_used} pairs (oov/self {n_oov}) in "
          f"{time.time() - t0:.0f}s", flush=True)

    if n_used == 0:
        print("no usable pairs"); return

    # ---- grid print ----
    def mean(key):
        return total[key] / n_used

    print("\n=== mean mediators per pair, by (d_a, d_b) cell ===", flush=True)
    header = "d_a\\d_b " + "".join(f"{j:>10}" for j in range(1, depth + 1))
    print(header)
    for da_ in range(1, depth + 1):
        row = f"{da_:>5}  "
        for db_ in range(1, depth + 1):
            row += f"{mean((da_, db_)):>10.2f}"
        print(row)

    print("\n=== fraction of pairs with >=1 mediator in cell ===", flush=True)
    print(header)
    for da_ in range(1, depth + 1):
        row = f"{da_:>5}  "
        for db_ in range(1, depth + 1):
            row += f"{pairs_hit[(da_, db_)] / n_used:>10.3f}"
        print(row)

    # ---- region rollups ----
    and2 = [(a, b) for a in (1, 2) for b in (1, 2)]
    sum4_ring = [(1, 3), (3, 1), (2, 2)]                 # new cells reaching sum==4
    sum4_new = [(1, 3), (3, 1)]                          # (2,2) already in AND box
    sum5_ring = [(1, 4), (4, 1), (2, 3), (3, 2)]         # sum==5 cells

    def region_total(cells):
        return sum(total[c] for c in cells)

    def region_mean(cells):
        return region_total(cells) / n_used

    print("\n=== region rollup (mean mediators per pair) ===", flush=True)
    print(f"AND(l=2) box          {and2}")
    print(f"  total/pair        = {region_mean(and2):.2f}")
    print(f"SUM(L=4) new cells    {sum4_new}  (over AND box; (2,2) already in box)")
    print(f"  added/pair        = {region_mean(sum4_new):.2f}")
    print(f"  -> SUM(L=4)/pair  = {region_mean(and2 + sum4_new):.2f}  "
          f"(x{region_mean(and2 + sum4_new) / max(region_mean(and2), 1e-9):.2f} vs AND)")
    print(f"SUM(L=5) new cells    {sum5_ring}  (added on top of SUM L=4)")
    print(f"  added/pair        = {region_mean(sum5_ring):.2f}")
    print(f"  -> SUM(L=5)/pair  = {region_mean(and2 + sum4_new + sum5_ring):.2f}  "
          f"(x{region_mean(and2 + sum4_new + sum5_ring) / max(region_mean(and2), 1e-9):.2f} vs AND)")


if __name__ == "__main__":
    main()

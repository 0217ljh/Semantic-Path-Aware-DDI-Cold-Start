"""Phase-B: does the path/mediator distribution match the hyper-edge motivation?

Tests the LOCKED motivation empirically on positive DDI pairs (merged KG, AND
support):
  M1(i)  subtype spread: shared mediators distribute across relation buckets
         (target/enzyme/transporter/carrier/pathway/gene-reg/effect/...) — DDIs
         involve multiple mechanism CATEGORIES, not one.
  M1(ii) distance spread: shared mediators sit at varied (d_a,d_b) positions
         (1,1)/(1,2)/(2,2) — convergence happens beyond 1-hop, so position-
         agnostic pooling captures more than direct common neighbors.
  M2     cross-type: which type-pairs (tau,tau') co-occur on a-b corridors
         (off-diagonal mass = real cross-type structure).

Verified data only. Run:
  python Code/scripts/analyze_spmn_v1_path_distribution.py --seed 42 --l-max 3 --n-pos 1919
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
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

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, KIND_ORDER, REL_2HOP, REL_OTHER, build_pair_support,
)

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
REL_NAMES = ["target", "enzyme", "transporter", "carrier", "pathway",
             "gene-reg", "effect/SE", "indication", "contraindic", "other", "2hop"]


def _rel(kg: MergedKG, d: int, u: int, dist: int) -> int:
    if dist == 1:
        return kg.drug_rel.get(d, {}).get(u, REL_OTHER)
    return REL_2HOP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--n-pos", type=int, default=1919)
    args = ap.parse_args()

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].reset_index(drop=True)
    pos = pos.iloc[:args.n_pos]
    kg = MergedKG.from_parquet()

    rel_cnt = Counter()                       # relation bucket of shared mediators
    dist_cnt = Counter()                      # (d_a, d_b) of shared mediators
    type_cnt = np.zeros(N_TYPES, dtype=np.int64)
    copath = np.zeros((N_TYPES, N_TYPES), dtype=np.int64)
    n_subtypes_per_pair = []                  # how many distinct rel buckets per pair
    n_subtypes_excl_2hop = []                 # distinct SPECIFIC rel buckets (no 2hop)
    n_pairs_used = 0

    from my_code.models.spmn_v1.struct_features import compute_struct_features
    for i in range(len(pos)):
        ai = kg.id_to_idx.get(str(pos.iloc[i, 0]))
        bi = kg.id_to_idx.get(str(pos.iloc[i, 1]))
        if ai is None or bi is None or ai == bi:
            continue
        sup = build_pair_support(kg, ai, bi, l_max=args.l_max, support_and=True)
        if sup.n_support == 0:
            continue
        n_pairs_used += 1
        type_cnt += np.bincount(sup.type_id.astype(np.int64), minlength=N_TYPES)
        pair_rels = set()
        for k in range(sup.n_support):
            u = int(sup.global_idx[k]); da = int(sup.d_a[k]); db = int(sup.d_b[k])
            ra = _rel(kg, ai, u, da); rb = _rel(kg, bi, u, db)
            rel_cnt[ra] += 1; rel_cnt[rb] += 1
            pair_rels.add(ra); pair_rels.add(rb)
            dist_cnt[(da, db)] += 1
        n_subtypes_per_pair.append(len(pair_rels))
        n_subtypes_excl_2hop.append(len(pair_rels - {REL_2HOP, REL_OTHER}))
        feat = compute_struct_features(kg, sup, with_copath=True)
        copath += feat.s_tau_tau

    tot_med = sum(rel_cnt.values())
    print(f"=== Phase-B path distribution (test_s2 positives, used={n_pairs_used}) ===\n")

    print("-- M1(i-a) NODE-TYPE of shared mediators (the bulk carrier) --")
    tt = type_cnt.sum()
    for t in np.argsort(-type_cnt):
        if type_cnt[t] == 0:
            continue
        print(f"  {KIND_ORDER[t]:18s} {int(type_cnt[t]):9d}  ({100*type_cnt[t]/max(1,tt):4.1f}%)")

    print("\n-- M1(i-b) relation-bucket of shared mediators --")
    for b, c in rel_cnt.most_common():
        print(f"  {REL_NAMES[b]:14s} {c:8d}  ({100*c/max(1,tot_med):4.1f}%)")
    arr = np.array(n_subtypes_per_pair)
    arr_x = np.array(n_subtypes_excl_2hop)
    print(f"  >> mean distinct rel-buckets/pair (incl 2hop) = {arr.mean():.2f} ; "
          f">=3 = {100*(arr>=3).mean():.1f}%")
    print(f"  >> mean distinct SPECIFIC rel-buckets/pair (EXCL 2hop) = {arr_x.mean():.2f} ; "
          f"pairs with 0 specific = {100*(arr_x==0).mean():.1f}%")

    print("\n-- M1(ii) distance spread: (d_a,d_b) of shared mediators --")
    tot_d = sum(dist_cnt.values())
    for (da, db), c in sorted(dist_cnt.items()):
        print(f"  d_a={da} d_b={db}: {c:8d}  ({100*c/max(1,tot_d):4.1f}%)")
    direct = dist_cnt.get((1, 1), 0)
    print(f"  >> (1,1) direct common-neighbor = {100*direct/max(1,tot_d):.1f}% ; "
          f"rest (need >1 hop on a side) = {100*(tot_d-direct)/max(1,tot_d):.1f}%")

    print("\n-- M2 cross-type co-path: top (tau,tau') co-occurring on corridors --")
    flat = [(copath[i, j], i, j) for i in range(N_TYPES) for j in range(N_TYPES)]
    flat.sort(reverse=True)
    diag = sum(copath[i, i] for i in range(N_TYPES))
    off = copath.sum() - diag
    print(f"  total co-path = {copath.sum()} ; off-diagonal (cross-type) "
          f"= {off} ({100*off/max(1,copath.sum()):.1f}%)")
    for c, i, j in flat[:12]:
        if c == 0:
            break
        tag = "  (same-type)" if i == j else ""
        print(f"  {KIND_ORDER[i]:>14s} -> {KIND_ORDER[j]:<14s} {c:8d}{tag}")


if __name__ == "__main__":
    main()

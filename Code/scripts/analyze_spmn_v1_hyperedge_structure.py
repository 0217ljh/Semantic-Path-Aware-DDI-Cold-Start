"""Phase-B (corrected): mechanism-category split + WITHIN-hyper-edge position spread.

Two things the lead actually wants:
  (A) Split protein_gene mediators by MECHANISM CATEGORY (relation bucket:
      target / enzyme / transporter / carrier) — only defined for 1-hop
      drug->protein edges. Shows the target/enzyme/transporter breakdown.
  (B) WITHIN one (pair, type) hyper-edge, do members sit at DIFFERENT (d_a,d_b)
      positions (one closer to A, another closer to B)? This is what
      "position-agnostic pooling" actually means — NOT the global (2,2) share.

Run: python Code/scripts/analyze_spmn_v1_hyperedge_structure.py --seed 42 --n-pos 1919
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
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
    MergedKG, KIND_ORDER, REL_OTHER, build_pair_support,
)

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
REL_NAMES = ["target", "enzyme", "transporter", "carrier", "pathway",
             "gene-reg", "effect/SE", "indication", "contraindic", "other", "2hop"]
PG = KIND_ORDER.index("protein_gene")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pos", type=int, default=1919)
    args = ap.parse_args()

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].reset_index(drop=True).iloc[:args.n_pos]
    kg = MergedKG.from_parquet()

    # (A) protein_gene 1-hop split by mechanism category (rel of drug->protein)
    pg_relA = Counter()
    # (B) within-hyper-edge position spread
    he_n_positions = []          # distinct (d_a,d_b) positions per hyper-edge (>=2 members)
    he_both_sides = 0            # hyper-edges with a member closer-to-A AND one closer-to-B
    he_total = 0
    he_size = []

    for i in range(len(pos)):
        ai = kg.id_to_idx.get(str(pos.iloc[i, 0])); bi = kg.id_to_idx.get(str(pos.iloc[i, 1]))
        if ai is None or bi is None or ai == bi:
            continue
        sup = build_pair_support(kg, ai, bi, l_max=3, support_and=True)
        if sup.n_support == 0:
            continue
        by_type: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for k in range(sup.n_support):
            u = int(sup.global_idx[k]); da = int(sup.d_a[k]); db = int(sup.d_b[k]); t = int(sup.type_id[k])
            by_type[t].append((da, db))
            if t == PG and da == 1:                       # 1-hop drug_a -> protein
                pg_relA[kg.drug_rel.get(ai, {}).get(u, REL_OTHER)] += 1
            if t == PG and db == 1:                       # 1-hop drug_b -> protein
                pg_relA[kg.drug_rel.get(bi, {}).get(u, REL_OTHER)] += 1
        for t, members in by_type.items():
            if len(members) < 2:
                continue
            he_total += 1
            he_size.append(len(members))
            he_n_positions.append(len(set(members)))
            closer_a = any(da < db for da, db in members)
            closer_b = any(da > db for da, db in members)
            if closer_a and closer_b:
                he_both_sides += 1

    print(f"=== (A) protein_gene mediators split by MECHANISM CATEGORY "
          f"(1-hop drug->protein relation) ===")
    tot = sum(pg_relA.values())
    for b, c in pg_relA.most_common():
        print(f"  {REL_NAMES[b]:14s} {c:8d}  ({100*c/max(1,tot):4.1f}%)")
    print(f"  (these are the ~4% of protein_gene mediators that are 1-hop; the "
          f"2-hop bulk has no single relation)")

    print(f"\n=== (B) WITHIN-hyper-edge position spread "
          f"(hyper-edges with >=2 members, n={he_total}) ===")
    npos = np.array(he_n_positions); sz = np.array(he_size)
    print(f"  mean members/hyper-edge = {sz.mean():.1f}")
    print(f"  mean DISTINCT (d_a,d_b) positions per hyper-edge = {npos.mean():.2f}")
    print(f"  hyper-edges with members at >=2 distinct positions = {100*(npos>=2).mean():.1f}%")
    print(f"  hyper-edges with BOTH a closer-to-A and a closer-to-B member "
          f"(asymmetric) = {100*he_both_sides/max(1,he_total):.1f}%")


if __name__ == "__main__":
    main()

"""Probe the support-induced mediator<->mediator KG subgraph (option A edges).

For the fine-grained interaction layer we want, per pair, the real merged-KG
edges whose BOTH endpoints are mediators in the pair's SUM support. This script
builds a relation-aware adjacency from the merged-KG edge table (the CSR inside
MergedKG is binarised, so relation must be rebuilt), then on a sample of S2
pairs measures, per pair:

  * E = number of support-internal typed edges (and unique undirected pairs),
  * fraction of support mediators with >=1 internal edge,
  * the relation-type distribution of those edges (the edge_kind vocab),
  * the (type_u -> type_v) distribution (how much is protein-centred / K3-like),
  * max within-support node degree (hub-edge concentration).

This tells us whether option-A edges are tractable (E magnitude), what the
edge_kind vocabulary looks like, and whether the edges are mechanistic
(protein<->pathway/GO/disease) before we commit to caching them. Read-only.
"""
from __future__ import annotations

import argparse
import sys
import time
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
    DEFAULT_EDGES_PATH, KIND_ORDER, MergedKG, build_pair_support,
)

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def _build_rel_adjacency(kg: MergedKG):
    """Relation-aware symmetrised CSR: per node, neighbour ids + relation ids.

    Returns (indptr, nbr, rel_id, rel_vocab) where rel_vocab[id] = relation str.
    """
    edges = pd.read_parquet(DEFAULT_EDGES_PATH, columns=["src", "dst", "relation"])
    src = edges["src"].astype(str).map(kg.id_to_idx).to_numpy()
    dst = edges["dst"].astype(str).map(kg.id_to_idx).to_numpy()
    rel = edges["relation"].astype(str).to_numpy()
    valid = ~(pd.isna(src) | pd.isna(dst))
    src = src[valid].astype(np.int64); dst = dst[valid].astype(np.int64)
    rel = rel[valid]
    rel_vocab, rel_id = np.unique(rel, return_inverse=True)

    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    r = np.concatenate([rel_id, rel_id]).astype(np.int32)
    keep = u != v
    u, v, r = u[keep], v[keep], r[keep]
    order = np.argsort(u, kind="stable")
    u_s = u[order]; nbr = v[order].astype(np.int32); rel_s = r[order]
    n = kg.n_nodes
    indptr = np.searchsorted(u_s, np.arange(n + 1)).astype(np.int64)
    return indptr, nbr, rel_s, rel_vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pairs", type=int, default=150)
    ap.add_argument("--l-max", type=int, default=5)
    ap.add_argument("--k-per-type", type=int, default=128)
    ap.add_argument("--n-max", type=int, default=1200)
    args = ap.parse_args()

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    rng = np.random.default_rng(0)
    take = min(args.n_pairs, len(pos))
    sample = pos.iloc[np.sort(rng.choice(len(pos), take, replace=False))]

    print("[edges] loading KG + building relation-aware adjacency ...", flush=True)
    kg = MergedKG.from_parquet()
    t0 = time.time()
    indptr, nbr, rel_s, rel_vocab = _build_rel_adjacency(kg)
    print(f"[edges] adjacency built in {time.time()-t0:.0f}s "
          f"(nnz={len(nbr):,}, n_rel={len(rel_vocab)})", flush=True)

    mask = np.zeros(kg.n_nodes, dtype=bool)
    E_list, frac_list, maxdeg_list, used = [], [], [], 0
    rel_counter: Counter = Counter()
    typepair_counter: Counter = Counter()
    n_uniq_list = []

    for ai_id, bi_id in sample.itertuples(index=False):
        ai = kg.id_to_idx.get(str(ai_id)); bi = kg.id_to_idx.get(str(bi_id))
        if ai is None or bi is None or ai == bi:
            continue
        sup = build_pair_support(kg, ai, bi, l_max=args.l_max, support_and=False,
                                 k_per_type=args.k_per_type, n_max=args.n_max)
        if sup.n_support == 0:
            continue
        S = sup.global_idx.astype(np.int64)
        mask[:] = False; mask[S] = True
        deg_in = np.zeros(kg.n_nodes, dtype=np.int64)
        us, vs, rs = [], [], []
        for u in S.tolist():
            a, b = indptr[u], indptr[u + 1]
            nb = nbr[a:b]; rl = rel_s[a:b]
            sel = mask[nb] & (u < nb.astype(np.int64))
            if sel.any():
                vv = nb[sel].astype(np.int64); rr = rl[sel]
                us.append(np.full(vv.shape, u)); vs.append(vv); rs.append(rr)
        if us:
            uu = np.concatenate(us); vv = np.concatenate(vs); rr = np.concatenate(rs)
            E = len(uu)
            np.add.at(deg_in, uu, 1); np.add.at(deg_in, vv, 1)
            uniq = len({(int(a), int(b)) for a, b in zip(uu, vv)})
            tu = kg.type_id[uu]; tv = kg.type_id[vv]
            for a, b in zip(tu, tv):
                lo, hi = sorted((int(a), int(b)))
                typepair_counter[(KIND_ORDER[lo], KIND_ORDER[hi])] += 1
            for rid, c in zip(*np.unique(rr, return_counts=True)):
                rel_counter[rel_vocab[rid]] += int(c)
        else:
            E = uniq = 0
        E_list.append(E); n_uniq_list.append(uniq)
        frac_list.append(float((deg_in[S] > 0).mean()))
        maxdeg_list.append(int(deg_in[S].max()) if S.size else 0)
        used += 1

    E = np.array(E_list); U = np.array(n_uniq_list)
    print(f"\n[edges] sampled {used} pairs (l_max={args.l_max}, n_max={args.n_max})")
    print(f"  E (typed edges)/pair : mean={E.mean():.0f} median={np.median(E):.0f} "
          f"max={E.max()} min={E.min()}")
    print(f"  unique undirected /pair: mean={U.mean():.0f} median={np.median(U):.0f} max={U.max()}")
    print(f"  frac support with >=1 edge: mean={np.mean(frac_list):.3f}")
    print(f"  max within-support node degree: mean={np.mean(maxdeg_list):.0f} "
          f"max={max(maxdeg_list)}  (hub concentration)")

    print("\n[edges] top relation types (edge_kind vocab):")
    for r, c in rel_counter.most_common(15):
        print(f"  {r:<28} {c}")
    print(f"  (total distinct relations seen: {len(rel_counter)})")

    print("\n[edges] top (type_u, type_v) pairs:")
    for (tu, tv), c in typepair_counter.most_common(15):
        print(f"  {tu:<20} {tv:<20} {c}")


if __name__ == "__main__":
    main()

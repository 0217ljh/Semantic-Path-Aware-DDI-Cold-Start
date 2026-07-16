"""Extract + cache the support-induced mediator<->mediator KG edges (A-trim).

Augments an existing SUM support cache (built by run_spmn_v2_standalone) with the
per-pair edges needed by the fine-grained interaction layer. Reuses the cached
supports (the `med` arrays) so NO BFS is re-run — we only intersect each support
with a relation-aware adjacency.

A-trim filter (option A minus dense non-mechanistic noise, per the edge probe):
  * DROP any edge with an endpoint of type `anatomy` or `exposure`
    (protein<->anatomy alone was ~63% of raw edges and the main hub source);
  * DROP relation `het:CrC` (compound-compound resemblance);
  * keep everything else (protein-centred mechanism + high-level pathway/GO/
    disease/effect links).

Per-node edge cap: if a support mediator has > `--cap` in-support neighbours,
keep the `cap` lowest global-degree ones (specific mediators over hubs).

Output companion cache stores, per split, directed edges v->u (message flows into
u) in support-LOCAL indices, with per-pair offsets and the edge_kind vocab. Local
indices are converted to batch-global at gather time by adding the support offset.

Read-only w.r.t. the KG. No GPU.
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

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    DEFAULT_EDGES_PATH, KIND_ORDER, MergedKG,
)

CACHE_DIR = ROOT / "Code/data/_cache"
DROP_TYPES = {"anatomy", "exposure"}
DROP_RELATIONS = {"het:CrC"}


def _build_filtered_adjacency(kg: MergedKG):
    """Relation-aware symmetrised CSR with the A-trim filter applied."""
    edges = pd.read_parquet(DEFAULT_EDGES_PATH, columns=["src", "dst", "relation"])
    src = edges["src"].astype(str).map(kg.id_to_idx).to_numpy()
    dst = edges["dst"].astype(str).map(kg.id_to_idx).to_numpy()
    rel = edges["relation"].astype(str).to_numpy()
    valid = ~(pd.isna(src) | pd.isna(dst))
    src = src[valid].astype(np.int64); dst = dst[valid].astype(np.int64)
    rel = rel[valid]

    drop_tid = {KIND_ORDER.index(t) for t in DROP_TYPES if t in KIND_ORDER}
    keep_type = ~(np.isin(kg.type_id[src], list(drop_tid))
                  | np.isin(kg.type_id[dst], list(drop_tid)))
    keep_rel = ~np.isin(rel, list(DROP_RELATIONS))
    keep = keep_type & keep_rel
    src, dst, rel = src[keep], dst[keep], rel[keep]
    rel_vocab, rel_id = np.unique(rel, return_inverse=True)

    u = np.concatenate([src, dst]); v = np.concatenate([dst, src])
    r = np.concatenate([rel_id, rel_id]).astype(np.int32)
    nl = u != v
    u, v, r = u[nl], v[nl], r[nl]
    order = np.argsort(u, kind="stable")
    u_s = u[order]; nbr = v[order].astype(np.int32); rel_s = r[order]
    indptr = np.searchsorted(u_s, np.arange(kg.n_nodes + 1)).astype(np.int64)
    return indptr, nbr, rel_s, rel_vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=5)
    ap.add_argument("--k-per-type", type=int, default=128)
    ap.add_argument("--n-max", type=int, default=1200)
    ap.add_argument("--cap", type=int, default=32)
    args = ap.parse_args()

    sup_cache = CACHE_DIR / (f"spmn_v2_supports_sum_seed{args.seed}_lmax{args.l_max}"
                             f"_kpt{args.k_per_type}_nmax{args.n_max}_cp0.npz")
    if not sup_cache.is_file():
        raise FileNotFoundError(f"support cache missing: {sup_cache}")
    print(f"[edges] support cache: {sup_cache}", flush=True)
    z = np.load(sup_cache)
    splits = ("train", "val_s2", "test_s2")

    print("[edges] loading KG + building A-trim adjacency ...", flush=True)
    kg = MergedKG.from_parquet()
    t0 = time.time()
    indptr, nbr, rel_s, rel_vocab = _build_filtered_adjacency(kg)
    print(f"[edges] adjacency in {time.time()-t0:.0f}s (nnz={len(nbr):,}, "
          f"n_edge_kinds={len(rel_vocab)})", flush=True)

    out = {"edge_kind_vocab": rel_vocab, "cap": np.array(args.cap),
           "drop_types": np.array(sorted(DROP_TYPES)),
           "drop_relations": np.array(sorted(DROP_RELATIONS))}
    for name in splits:
        med = z[f"{name}__med"]; off = z[f"{name}__offsets"]
        ed, es, ek, eoff = _extract(med, off, indptr, nbr, rel_s, kg.degree,
                                    kg.n_nodes, args.cap, name)
        out[f"{name}__e_dst"] = ed
        out[f"{name}__e_src"] = es
        out[f"{name}__e_kind"] = ek
        out[f"{name}__e_off"] = eoff
        print(f"  [{name}] edges={len(ed):,}  pairs={len(off)-1}  "
              f"mean E/pair={len(ed)/max(len(off)-1,1):.0f}", flush=True)

    edge_cache = CACHE_DIR / (f"spmn_v2_edges_sum_seed{args.seed}_lmax{args.l_max}"
                              f"_kpt{args.k_per_type}_nmax{args.n_max}_Atrim_cap{args.cap}.npz")
    np.savez_compressed(edge_cache, **out)
    print(f"[edges] saved -> {edge_cache}", flush=True)


def _extract(med, off, indptr, nbr, rel_s, deg, n_nodes, cap, tag):
    e_dst, e_src, e_kind = [], [], []
    counts = np.zeros(len(off) - 1, dtype=np.int64)
    g2l = np.full(n_nodes, -1, dtype=np.int64)
    t0 = time.time(); n = len(off) - 1
    for p in range(n):
        s, e = off[p], off[p + 1]
        S = med[s:e]; ns = S.shape[0]
        c = 0
        if ns:
            g2l[S] = np.arange(ns)
            for li in range(ns):
                g = int(S[li]); a, b = indptr[g], indptr[g + 1]
                nb = nbr[a:b]
                loc = g2l[nb]; sel = loc >= 0
                if not sel.any():
                    continue
                vloc = loc[sel]; vrel = rel_s[a:b][sel]; vglob = nb[sel]
                if vloc.shape[0] > cap:
                    keep = np.argsort(deg[vglob])[:cap]
                    vloc = vloc[keep]; vrel = vrel[keep]
                k = vloc.shape[0]
                e_dst.append(np.full(k, li, dtype=np.int64))
                e_src.append(vloc); e_kind.append(vrel.astype(np.int64))
                c += k
            g2l[S] = -1
        counts[p] = c
        if (p + 1) % 5000 == 0 or p + 1 == n:
            el = time.time() - t0
            print(f"  [{tag}] {p+1}/{n} {el:.0f}s eta {(n-p-1)/max((p+1)/el,1e-9):.0f}s",
                  flush=True)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    e_off = np.zeros(n + 1, dtype=np.int64); np.cumsum(counts, out=e_off[1:])
    return cat(e_dst), cat(e_src), cat(e_kind), e_off


if __name__ == "__main__":
    main()

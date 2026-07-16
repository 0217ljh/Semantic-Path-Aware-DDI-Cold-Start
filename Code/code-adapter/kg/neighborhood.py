"""Stage 2 / atom A2 - per-seed L-hop neighborhood WITH shortest-hop distance.

`N_<=L(seed) \\ {seed}` in idx-space over the A1 UNDIRECTED (bidirectional)
adjacency, returning each reachable node AND its shortest-hop distance d(seed,m)
(the BFS depth at which it was first reached). The distance is required by A3,
whose two shared-mediator modes threshold the arm-length pair (d_a, d_b):
  AND(tau): d_a<=tau AND d_b<=tau      (per-arm cap; default tau=2)
  SUM(sigma): d_a + d_b <= sigma        (total-detour cap; default sigma=4)

Traverses through ALL node types (paths may pass through other drugs). Returns
the FULL reachable set unfiltered - `\\ D` and the pairwise op are A3's job.
Only undirected supported. Note: to serve SUM(sigma) a neighborhood must be
built with L >= sigma - 1 (an arm may be as long as sigma-1); AND(tau) needs L>=tau.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kg.store import KGStore

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache"
NBHD_SCHEMA = "nbhd_v2"


def neighborhood_of_idx(kg: KGStore, seed_idx: int, L: int = 2):
    """Return (reach_idx int32[], reach_dist int16[]) for N_<=L(seed)\\{seed},
    sorted by idx. dist[k] = shortest hop from seed to reach_idx[k]."""
    if not (0 <= seed_idx < kg.n_nodes):
        raise ValueError(f"seed_idx {seed_idx} out of range [0,{kg.n_nodes})")
    if L <= 0:
        return np.empty(0, np.int32), np.empty(0, np.int16)
    visited = np.zeros(kg.n_nodes, dtype=bool)
    visited[seed_idx] = True
    frontier = np.array([seed_idx], dtype=np.int64)
    idx_parts, dist_parts = [], []
    for depth in range(1, L + 1):
        if frontier.size == 0:
            break
        nb = np.unique(np.concatenate([kg.neighbors_idx(int(u)) for u in frontier]))
        new = nb[~visited[nb]]
        visited[new] = True
        idx_parts.append(new)
        dist_parts.append(np.full(len(new), depth, dtype=np.int16))
        frontier = new
    if not idx_parts:
        return np.empty(0, np.int32), np.empty(0, np.int16)
    ridx = np.concatenate(idx_parts).astype(np.int32)
    rdist = np.concatenate(dist_parts)
    order = np.argsort(ridx, kind="stable")            # deterministic: sort by idx
    return ridx[order], rdist[order]


class NeighborhoodCSR:
    """Batch L-hop neighborhoods (idx + dist) over sorted-unique seeds, CSR-packed."""

    def __init__(self, seed_idx, indptr, reach_idx, reach_dist, L, directed, meta):
        self.seed_idx = seed_idx          # int32[m] sorted unique
        self.indptr = indptr              # int64[m+1]
        self.reach_idx = reach_idx        # int32[nnz]
        self.reach_dist = reach_dist      # int16[nnz]
        self.L = L; self.directed = directed; self.meta = meta

    def _row(self, seed_idx: int):
        pos = int(np.searchsorted(self.seed_idx, seed_idx))
        if pos >= len(self.seed_idx) or self.seed_idx[pos] != seed_idx:
            raise KeyError(f"seed_idx {seed_idx} not in this neighborhood batch")
        return self.indptr[pos], self.indptr[pos + 1]

    def of_idx(self, seed_idx: int) -> np.ndarray:
        s, e = self._row(seed_idx); return self.reach_idx[s:e]

    def of_dist(self, seed_idx: int) -> np.ndarray:
        s, e = self._row(seed_idx); return self.reach_dist[s:e]

    def sizes(self) -> np.ndarray:
        return np.diff(self.indptr)


def _seed_hash(seeds: np.ndarray) -> str:
    return hashlib.sha1(seeds.tobytes()).hexdigest()[:10]


def build_neighborhoods_idx(kg: KGStore, seed_indices, L: int = 2, directed: bool = False,
                            rebuild: bool = False, log=print) -> NeighborhoodCSR:
    if directed:
        raise NotImplementedError("A2 supports undirected (bidirectional) reachability only")
    seeds = np.unique(np.asarray(list(seed_indices), dtype=np.int32))
    if seeds.size and (seeds.min() < 0 or seeds.max() >= kg.n_nodes):
        raise ValueError("seed index out of range")
    fp = kg.meta.get("fingerprint", "nofp")
    key = _seed_hash(seeds)
    cache_dir = _CACHE / f"nbhd__{kg.meta.get('kg_name','kg')}__{fp}__{NBHD_SCHEMA}__L{L}__{key}"
    if cache_dir.is_dir() and not rebuild:
        try:
            r = _load(cache_dir)
            log(f"[A2] cache HIT: {cache_dir}")
            return r
        except (ValueError, KeyError, OSError) as e:
            log(f"[A2] cache stale/unreadable ({e}); rebuilding")
    log(f"[A2] cache MISS -> {len(seeds)} seeds, L={L} ...")

    indptr = np.zeros(len(seeds) + 1, dtype=np.int64)
    idx_chunks, dist_chunks = [], []
    for i, s in enumerate(seeds):
        ri, rd = neighborhood_of_idx(kg, int(s), L)
        idx_chunks.append(ri); dist_chunks.append(rd); indptr[i + 1] = indptr[i] + len(ri)
    reach_idx = np.concatenate(idx_chunks).astype(np.int32) if idx_chunks else np.empty(0, np.int32)
    reach_dist = np.concatenate(dist_chunks).astype(np.int16) if dist_chunks else np.empty(0, np.int16)
    meta = {"schema": NBHD_SCHEMA, "kg_fingerprint": fp, "L": L, "directed": directed,
            "n_seeds": int(len(seeds)), "seed_hash": key,
            "size_median": int(np.median(np.diff(indptr))) if len(seeds) else 0,
            "size_max": int(np.diff(indptr).max()) if len(seeds) else 0}
    _save(cache_dir, seeds, indptr, reach_idx, reach_dist, meta)
    log(f"[A2] built + cached: {cache_dir}  (seeds={len(seeds)} "
        f"median|N|={meta['size_median']} max={meta['size_max']})")
    return NeighborhoodCSR(seeds, indptr, reach_idx, reach_dist, L, directed, meta)


def neighborhoods(kg: KGStore, drug_ids, L: int = 2, skip_missing: bool = True,
                  log=print) -> NeighborhoodCSR:
    """String-id convenience wrapper (outside the hot path)."""
    idxs, missing = [], []
    for d in drug_ids:
        i = kg.get_idx(str(d))
        (idxs.append(i) if i is not None else missing.append(str(d)))
    if missing:
        msg = f"[A2] {len(missing)} drug_ids absent from KG"
        (log(msg + " (skipped)") if skip_missing else
         (_ for _ in ()).throw(KeyError(msg + f": {missing[:5]}")))
    return build_neighborhoods_idx(kg, idxs, L=L, log=log)


def _save(cache_dir, seeds, indptr, reach_idx, reach_dist, meta):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / "arrays.npz", seed_idx=seeds, indptr=indptr,
             reach_idx=reach_idx, reach_dist=reach_dist)
    (cache_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _load(cache_dir: Path) -> NeighborhoodCSR:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != NBHD_SCHEMA:
        raise ValueError(f"nbhd cache schema mismatch at {cache_dir}; rebuild")
    a = np.load(cache_dir / "arrays.npz")
    return NeighborhoodCSR(a["seed_idx"], a["indptr"], a["reach_idx"], a["reach_dist"],
                           meta["L"], meta["directed"], meta)


__all__ = ["neighborhood_of_idx", "NeighborhoodCSR", "build_neighborhoods_idx",
           "neighborhoods", "NBHD_SCHEMA"]

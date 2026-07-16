"""Stage 2 / Part B / atom B5 - arm relation feature r_tilde, via structural
path-relation COEFFICIENTS (z_r-agnostic; encoder swaps don't rebuild).

For a drug u and a mediator m, r_tilde_a(m) = LayerNorm( num(u,m) / den(u,m) ) with
    num(u,m) = rho * (c1(u,m) @ z_r) + (rho^2 / 2) * (c2(u,m) @ z_r)
    den(u,m) = rho * n1(u,m) + rho^2 * n2(u,m)
  c1[r] = # directed len-1 paths u->m carrying canonical relation r
  c2[r] = # relation-r edge occurrences across directed len-2 paths u->x->m
          (each len-2 path contributes BOTH its r1 and r2 to c2)
  n1 = #len-1 paths (= c1.sum()), n2 = #len-2 paths (= c2.sum()/2)
den=0 (no valid semantic path after relation drop) -> caller uses r_tilde=0.

DIRECTED traversal (codex-faithful): every edge row is traversable forward; an edge
is traversable in REVERSE only if d_directed is False (a symmetric edge goes both
ways, a directed edge does not). Parallel/multi edges are counted separately
(multigraph). het:metaedge (RELATION_DROP) is excluded. This is the store's intended
"directed multigraph view for the relation/path atom", distinct from the undirected
reachability view used for mediator finding.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from kg import relations as rel
from kg.store import KGStore

_CACHE = Path(__file__).resolve().parents[1] / "kg" / "_cache"
TRAV_SCHEMA = "path_rel_v2"       # v2: drop self-loops from the traversal graph


def _zr_map(kg: KGStore) -> tuple[np.ndarray, list[str]]:
    """raw KG rel id (0..58) -> z_r row (0..46), or -1 if dropped."""
    keys = rel.canonical_relations()                     # 47 sorted canonical keys
    key2row = {k: i for i, k in enumerate(keys)}
    m = np.full(len(kg.rel_names), -1, dtype=np.int64)
    for rid, raw in enumerate(kg.rel_names):
        c = rel.canonical_relation(raw)
        if c is not None:
            m[rid] = key2row[c]
    return m, keys


def build_traversal_csr(kg: KGStore, rebuild: bool = False, log=print) -> dict:
    """Directed-aware relation-tagged traversal graph as OUT + IN CSR. Cached."""
    fp = kg.meta.get("fingerprint", "nofp")
    zr_map, keys = _zr_map(kg)
    vocab_fp = hashlib.sha1((TRAV_SCHEMA + "|" + "|".join(keys)).encode()).hexdigest()[:8]
    cache_dir = _CACHE / f"path_rel__{kg.meta.get('kg_name', 'kg')}__{fp}__{vocab_fp}"
    if cache_dir.is_dir() and not rebuild:
        a = np.load(cache_dir / "arrays.npz")
        log(f"[B5] traversal CSR cache HIT: {cache_dir}")
        return {"t_indptr": a["t_indptr"], "t_dst": a["t_dst"], "t_zr": a["t_zr"],
                "ti_indptr": a["ti_indptr"], "ti_src": a["ti_src"], "ti_zr": a["ti_zr"],
                "n_rel": len(keys)}
    log(f"[B5] building traversal CSR (fp={fp}) ...")

    n = kg.n_nodes
    # reconstruct src per directed edge row (d_indptr is source-indexed)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(kg.d_indptr))
    dst = kg.d_dst.astype(np.int64)
    zr = zr_map[kg.d_rel.astype(np.int64)]
    directed = kg.d_directed.astype(bool)
    keep = (zr >= 0) & (src != dst)                      # drop het:metaedge + self-loops
    fs, fd, fz = src[keep], dst[keep], zr[keep]           # forward steps (all kept edges)
    und = keep & (~directed)                              # symmetric edges: also reverse
    rs, rd, rz = dst[und], src[und], zr[und]
    osrc = np.concatenate([fs, rs]); odst = np.concatenate([fd, rd])
    ozr = np.concatenate([fz, rz]).astype(np.int16)

    t_indptr, t_dst, t_zr = _csr(osrc, odst, ozr, n)      # OUT: source-indexed
    ti_indptr, ti_src, ti_zr = _csr(odst, osrc, ozr, n)   # IN: dest-indexed (predecessors)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_dir / "arrays.npz", t_indptr=t_indptr, t_dst=t_dst, t_zr=t_zr,
             ti_indptr=ti_indptr, ti_src=ti_src, ti_zr=ti_zr)
    (cache_dir / "meta.json").write_text(json.dumps(
        {"schema": TRAV_SCHEMA, "kg_fp": fp, "n_rel": len(keys), "n_steps": int(len(osrc)),
         "n_forward": int(len(fs)), "n_reverse": int(len(rs))}), encoding="utf-8")
    log(f"[B5] built traversal CSR: {len(osrc)} steps "
        f"({len(fs)} fwd + {len(rs)} rev), {len(keys)} relations -> {cache_dir}")
    return {"t_indptr": t_indptr, "t_dst": t_dst, "t_zr": t_zr,
            "ti_indptr": ti_indptr, "ti_src": ti_src, "ti_zr": ti_zr, "n_rel": len(keys)}


def _csr(key: np.ndarray, val: np.ndarray, tag: np.ndarray, n: int):
    """Group (val, tag) by key into CSR (indptr[n+1], val_sorted, tag_sorted)."""
    order = np.argsort(key, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, key.astype(np.int64) + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, val[order].astype(np.int32), tag[order]


class PathRelationCoeffs:
    """Per-arm (drug u, mediators) relation-count coefficients c1/c2/n1/n2, using
    the directed traversal CSR. Out-edge groupings per drug are memoized (drugs
    repeat across pairs); node-sized scratch buffers are reused per drug."""

    def __init__(self, csr: dict, type_id: np.ndarray | None = None, n_types: int = 0):
        self.t_indptr = csr["t_indptr"]; self.t_dst = csr["t_dst"]; self.t_zr = csr["t_zr"]
        self.ti_indptr = csr["ti_indptr"]; self.ti_src = csr["ti_src"]; self.ti_zr = csr["ti_zr"]
        self.n_rel = int(csr["n_rel"])
        self.n_nodes = len(self.t_indptr) - 1
        self.type_id = type_id                       # per-node type (for typed meta-path atoms)
        self.n_types = int(n_types)
        self._out: dict[int, dict] = {}
        self._memo: dict[tuple, tuple] = {}          # (u,m) -> (c1[R],c2[R],n1,n2), frozen (ztext)
        self._amemo: dict[tuple, tuple] = {}         # (u,m) -> sparse atom counts, frozen (pathway)
        # node-sized scratch (reused per drug; drug id stamped to avoid clearing)
        self._in_local = np.full(self.n_nodes, -1, dtype=np.int64)
        self._stamp = np.zeros(self.n_nodes, dtype=np.int64)
        self._cur = 0

    def _out_group(self, u: int) -> dict:
        g = self._out.get(u)
        if g is not None:
            return g
        s, e = self.t_indptr[u], self.t_indptr[u + 1]
        dst = self.t_dst[s:e]; zr = self.t_zr[s:e].astype(np.int64)
        xs = np.unique(dst)                               # distinct out-neighbors x
        loc = {int(x): i for i, x in enumerate(xs)}
        k = np.zeros(len(xs), dtype=np.int64)             # #u->x edges
        h1 = np.zeros((len(xs), self.n_rel), dtype=np.int64)  # relation hist of u->x
        for d, r in zip(dst, zr):
            i = loc[int(d)]; k[i] += 1; h1[i, r] += 1
        g = {"xs": xs, "loc": loc, "k": k, "h1": h1,
             "m2k": {int(x): int(kk) for x, kk in zip(xs, k)},
             "m2h1": {int(x): h1[i] for i, x in enumerate(xs)}}
        self._out[u] = g
        return g

    def arm_coeffs(self, u: int, mediators: np.ndarray):
        """(u, mediators[k]) -> (c1 [k,R], c2 [k,R], n1 [k], n2 [k]). Coeffs are frozen
        (KG static) so each (u,m) is computed once and memoized across pairs/epochs."""
        R = self.n_rel
        k_med = len(mediators)
        c1 = np.zeros((k_med, R), dtype=np.float64)
        c2 = np.zeros((k_med, R), dtype=np.float64)
        n1 = np.zeros(k_med, dtype=np.float64)
        n2 = np.zeros(k_med, dtype=np.float64)
        if k_med == 0:
            return c1, c2, n1, n2
        memo = self._memo
        miss = [(i, int(m)) for i, m in enumerate(mediators) if (u, int(m)) not in memo]
        if miss:
            g = self._out_group(u)
            self._cur += 1                                # stamp out(u) membership in O(1)
            xs = g["xs"]
            self._stamp[xs] = self._cur
            self._in_local[xs] = np.arange(len(xs))
            kloc = g["k"]; h1loc = g["h1"]
            for _, m in miss:
                c1r = np.zeros(R, dtype=np.float32); c2r = np.zeros(R, dtype=np.float32)
                n1v = 0.0; n2v = 0.0
                if m in g["m2k"]:                         # len-1: direct u->m edges
                    c1r = g["m2h1"][m].astype(np.float32); n1v = float(g["m2k"][m])
                s, e = self.ti_indptr[m], self.ti_indptr[m + 1]
                xin = self.ti_src[s:e]; zin = self.ti_zr[s:e].astype(np.int64)
                sel = self._stamp[xin] == self._cur       # len-2: x->m with x in out(u)
                if sel.any():
                    xk = xin[sel]; rk = zin[sel]; li = self._in_local[xk]; kx = kloc[li]
                    c2r = c2r + h1loc[li].sum(0).astype(np.float32)   # r1 side
                    np.add.at(c2r, rk, kx.astype(np.float32))         # r2 side (weighted by k_x)
                    n2v = float(kx.sum())
                memo[(u, m)] = (c1r, c2r, n1v, n2v)
        for i, m in enumerate(mediators):
            c1r, c2r, n1v, n2v = memo[(u, int(m))]
            c1[i] = c1r; c2[i] = c2r; n1[i] = n1v; n2[i] = n2v
        return c1, c2, n1, n2

    def arm_atoms(self, u: int, mediators: np.ndarray) -> np.ndarray:
        """Typed meta-path atoms (Phase 1). Returns A [k, R, T+1] float32 counts:
          A[:, r, t]  = # len-2 paths u->x->m with drug-side relation r AND t(x)=t  (t in 0..T-1)
          A[:, r, T]  = # len-1 direct u->m edges with relation r                    (DIRECT column)
        Drops r2 (only the mechanism-defining drug-side relation r1 is kept). Sparse-memoized."""
        if self.type_id is None:
            raise RuntimeError("arm_atoms needs type_id; construct PathRelationCoeffs(csr, type_id, n_types)")
        R, T = self.n_rel, self.n_types
        T1 = T + 1
        k = len(mediators)
        A = np.zeros((k, R, T1), dtype=np.float32)
        if k == 0:
            return A
        am = self._amemo
        miss = [int(m) for m in mediators if (u, int(m)) not in am]
        if miss:
            g = self._out_group(u)
            self._cur += 1
            xs = g["xs"]
            self._stamp[xs] = self._cur
            self._in_local[xs] = np.arange(len(xs))
            h1loc = g["h1"]                          # [nx, R] u->x relation histograms
            for m in miss:
                a = np.zeros((R, T1), dtype=np.float32)
                if m in g["m2h1"]:                   # len-1 DIRECT: u->m relation histogram
                    a[:, T] = g["m2h1"][m]
                s, e = self.ti_indptr[m], self.ti_indptr[m + 1]
                xin = self.ti_src[s:e]
                sel = self._stamp[xin] == self._cur  # intermediates x in out(u)
                if sel.any():
                    ux, jx = np.unique(xin[sel], return_counts=True)  # unique x + #x->m edges
                    li = self._in_local[ux]
                    tx = self.type_id[ux].astype(np.int64)            # intermediate types
                    contrib = h1loc[li] * jx[:, None]                 # [nux, R] = h1_x * j_x
                    np.add.at(a, (slice(None), tx), contrib.T.astype(np.float32))
                am[(u, m)] = _sparsify(a)
        for i, m in enumerate(mediators):
            idx, val = am[(u, int(m))]
            A[i].reshape(-1)[idx] = val
        return A


def _sparsify(a: np.ndarray):
    nz = np.flatnonzero(a.reshape(-1))
    return nz.astype(np.int32), a.reshape(-1)[nz].astype(np.float32)


__all__ = ["build_traversal_csr", "PathRelationCoeffs", "TRAV_SCHEMA"]

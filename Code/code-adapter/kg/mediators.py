"""Stage 2 / atom A3 - shared mediator set M_uv (thin transform over A2).

A mediator m must be reachable from BOTH endpoints (both arms finite), i.e.
m in N_<=L(u) ∩ N_<=L(v). The two modes then threshold the arm-length pair
(d_a, d_b) = (hop u->m, hop v->m):

  AND(tau):   d_a <= tau AND d_b <= tau        per-arm cap (default tau=2)
  SUM(sigma): d_a + d_b <= sigma               total-detour cap (default sigma=4)

Then `\\ D` removes drug nodes (mediators are non-drug). Result stays idx-space
and carries per-mediator type tags (kg.type_id) + the arm distances d_a, d_b
(needed by Part B's relation term). No relation/path semantics here.

Both modes are implemented; AND is the current default. SUM(sigma) is correct
only when the A2 neighborhood was built with L >= sigma - 1 (an arm may reach
sigma-1); AND(tau) needs L >= tau. No per-pair cache (cheap sorted set ops over
the reusable per-drug A2 cache).
"""
from __future__ import annotations

import numpy as np

from kg.neighborhood import NeighborhoodCSR
from kg.store import KGStore


def shared_mediators_idx(kg: KGStore, nbhd: NeighborhoodCSR, u_idx: int, v_idx: int,
                         mode: str = "and", tau: int = 2, sigma: int = 4) -> dict:
    """M_uv for one pair. Returns {med_idx, type_id, d_a, d_b} (idx-space)."""
    if mode == "and":
        need = tau
    elif mode == "sum":
        need = sigma - 1                 # an arm may be as long as sigma-1
    else:
        raise ValueError(f"mode must be 'and' or 'sum', got {mode!r}")
    if nbhd.L < need:
        raise ValueError(f"neighborhood L={nbhd.L} too small for mode={mode} "
                         f"(need L>={need}); rebuild A2 with a larger L")
    iu, du = nbhd.of_idx(u_idx), nbhd.of_dist(u_idx)
    iv, dv = nbhd.of_idx(v_idx), nbhd.of_dist(v_idx)
    # both A2 rows are sorted-unique by idx -> intersect gives common mediators + arm dists
    common, cu, cv = np.intersect1d(iu, iv, assume_unique=True, return_indices=True)
    d_a = du[cu].astype(np.int32); d_b = dv[cv].astype(np.int32)
    if mode == "and":
        keep = (d_a <= tau) & (d_b <= tau)
    elif mode == "sum":
        keep = (d_a + d_b) <= sigma
    else:
        raise ValueError(f"mode must be 'and' or 'sum', got {mode!r}")
    m = common[keep].astype(np.int32); d_a = d_a[keep]; d_b = d_b[keep]
    nd = ~kg.drug_mask[m]                                       # \ D
    m, d_a, d_b = m[nd], d_a[nd], d_b[nd]
    return {"med_idx": m, "type_id": kg.type_id[m], "d_a": d_a, "d_b": d_b}


def shared_mediators_batch(kg: KGStore, nbhd: NeighborhoodCSR, pairs_idx,
                           mode: str = "and", tau: int = 2, sigma: int = 4) -> list[dict]:
    """M_uv for a list of (u_idx, v_idx) pairs."""
    return [shared_mediators_idx(kg, nbhd, int(u), int(v), mode, tau, sigma)
            for u, v in pairs_idx]


def shared_mediators(kg: KGStore, nbhd: NeighborhoodCSR, u_id: str, v_id: str,
                     mode: str = "and", tau: int = 2, sigma: int = 4) -> dict:
    """String-id convenience wrapper (outside the hot path)."""
    u, v = kg.get_idx(str(u_id)), kg.get_idx(str(v_id))
    if u is None or v is None:
        raise KeyError(f"drug not in KG: {u_id if u is None else v_id}")
    return shared_mediators_idx(kg, nbhd, u, v, mode, tau, sigma)


__all__ = ["shared_mediators_idx", "shared_mediators_batch", "shared_mediators"]

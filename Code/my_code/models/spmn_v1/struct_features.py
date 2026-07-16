"""Exact zero-parameter structural features (frozen-design step 3).

Two explicit pairwise mechanism-structure features, both computed by exact set
/ shortest-distance operations (no learned parameters) on the symmetrised
merged KG. These are the load-bearing "structural-variable preservation"
signal: standard node-wise message passing cannot in general preserve them
(see ``Notes/Log/algorithm_design_debate.md`` R3/R12), so we measure them
directly.

* ``s_tau^(l)(a,b)`` — per-type common-reachability count
  ``|{u : phi(u)=tau, d_sym(a,u)+d_sym(u,b) <= l}|``. Taken from
  :attr:`PairSupport.type_count`, which is EXACT (uncapped).

* ``s_{tau,tau'}^(l)(a,b)`` — directional cross-type co-path count
  ``|{(u,v): phi(u)=tau, phi(v)=tau', d(a,u)+d(u,v)+d(v,b) <= l, u != v}|``
  using bounded-walk semantics (R7): existence of a directed walk
  ``a~>u~>v~>b`` of length <= l is exactly ``d(a,u)+d(u,v)+d(v,b) <= l`` on the
  symmetrised graph. ``u`` carries the a-side budget, ``v`` the b-side budget,
  so the count is ordered (a==>b). Computed over the hub-capped support, so it
  is exact *given the retained support* (the documented truncation
  approximation, R7/R13-A7).

The co-path feature is gated by ``with_copath`` so Phase-1 can first test
whether ``s_tau`` alone carries signal before paying the O(n^2)-ish cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from my_code.models.spmn_v1.retrieval import MergedKG, N_TYPES, PairSupport


@dataclass
class StructFeatures:
    """Container for the explicit structural features of one pair."""

    s_tau: np.ndarray         # (N_TYPES,) int64 — exact per-type reachability
    s_tau_aa: np.ndarray      # (N_TYPES,) float — Adamic-Adar degree-weighted
    s_tau_tau: np.ndarray     # (N_TYPES, N_TYPES) int64 — directional co-path
    n_support: int            # capped support size (diagnostic)

    def as_vector(self) -> np.ndarray:
        """Flatten to a fixed-length float vector for a tabular head (XGBoost).

        Layout (length ``2*N_TYPES + N_TYPES**2 + 2``):
          ``[ s_tau | log1p(s_tau) | s_tau_tau.ravel() | n_support | n_active_types ]``
        log1p of the per-type counts gives the tree a less skewed split var;
        raw counts are kept too.
        """
        s_tau = self.s_tau.astype(np.float64)
        n_active = float((self.s_tau > 0).sum())
        return np.concatenate([
            s_tau,
            np.log1p(s_tau),
            self.s_tau_tau.astype(np.float64).ravel(),
            np.array([float(self.n_support), n_active], dtype=np.float64),
        ])

    @staticmethod
    def vector_names() -> list[str]:
        names = [f"s_tau[{t}]" for t in range(N_TYPES)]
        names += [f"log_s_tau[{t}]" for t in range(N_TYPES)]
        names += [f"s_copath[{i},{j}]" for i in range(N_TYPES) for j in range(N_TYPES)]
        names += ["n_support", "n_active_types"]
        return names

    @staticmethod
    def vector_length() -> int:
        return 2 * N_TYPES + N_TYPES * N_TYPES + 2


_TRI_I, _TRI_J = np.triu_indices(N_TYPES)


def symmetric_binary_vector(feat: "StructFeatures") -> np.ndarray:
    """Flatten to the symmetric feature layout used by the binary task.

    Layout (length ``2*N_TYPES + N_TYPES*(N_TYPES+1)//2 + 2``):
      ``[ s_tau | log1p(s_tau) | sym_copath_upper_tri | n_support | n_active ]``
    where ``sym_copath = s_tau_tau + s_tau_tau.T`` (the binary DDI label is
    order-invariant, so the directional co-path is symmetrised). This is the
    SAME layout as the Phase-1 probe, so features are comparable across phases.
    """
    s = feat.s_tau.astype(np.float64)
    aa = feat.s_tau_aa.astype(np.float64)
    sym = (feat.s_tau_tau + feat.s_tau_tau.T).astype(np.float64)
    return np.concatenate([
        s,
        np.log1p(s),
        aa,                       # Adamic-Adar degree-weighted per-type count
        sym[_TRI_I, _TRI_J],
        np.array([float(feat.n_support), float((feat.s_tau > 0).sum())],
                 dtype=np.float64),
    ])


def symmetric_binary_dim() -> int:
    return 3 * N_TYPES + N_TYPES * (N_TYPES + 1) // 2 + 2


def compute_struct_features(
    kg: MergedKG,
    support: PairSupport,
    *,
    with_copath: bool = True,
) -> StructFeatures:
    """Compute ``s_tau`` and (optionally) ``s_{tau,tau'}`` for one pair."""
    s_tau = support.type_count.astype(np.int64)
    s_tau_aa = support.type_aa.astype(np.float64)
    s_tau_tau = np.zeros((N_TYPES, N_TYPES), dtype=np.int64)

    if not with_copath or support.n_support == 0:
        return StructFeatures(s_tau=s_tau, s_tau_aa=s_tau_aa,
                              s_tau_tau=s_tau_tau, n_support=support.n_support)

    l_max = support.l_max
    glob = support.global_idx
    da = support.d_a.astype(np.int64)
    db = support.d_b.astype(np.int64)
    type_id = support.type_id.astype(np.int64)
    # Global node id -> local support position (-1 if outside support), so the
    # per-u inner loop is a vectorised gather instead of a Python loop over
    # (possibly hub-sized) BFS frontiers.
    local_of = np.full(kg.n_nodes, -1, dtype=np.int64)
    local_of[glob] = np.arange(support.n_support, dtype=np.int64)

    for iu in range(support.n_support):
        # Max u->v walk length still within budget: d(u,v) <= l_max - da[u] - db[v];
        # since db[v] >= 1 for every support node, depth bound uses db_min = 1.
        depth = l_max - int(da[iu]) - 1
        if depth < 1:
            continue
        reached, dist_u = kg.bounded_bfs(int(glob[iu]), depth)
        iv = local_of[reached]                       # local pos or -1
        keep = (dist_u > 0) & (iv >= 0)              # distinct, in-support
        if not keep.any():
            continue
        iv = iv[keep]
        duv = dist_u[keep].astype(np.int64)
        within = (int(da[iu]) + duv + db[iv]) <= l_max
        if not within.any():
            continue
        tv = type_id[iv[within]]
        # Accumulate ordered (tau_u -> tau_v) counts for this u.
        np.add.at(s_tau_tau[type_id[iu]], tv, 1)

    return StructFeatures(s_tau=s_tau, s_tau_aa=s_tau_aa, s_tau_tau=s_tau_tau,
                          n_support=support.n_support)


def compute_copath_pairs(
    kg: MergedKG, support: PairSupport, l_copath: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the co-path LOCAL-index pairs ``(iu, iv)`` (a-side u, b-side v),
    BOTH endpoints in ``support``, satisfying ``d(a,u)+d(u,v)+d(v,b) <=
    l_copath``, u != v. The walk budget ``l_copath`` may exceed the support's
    own ``l_max`` to admit MORE cross-type corridors for the molecular channel
    to weight (the endpoints stay in the support, so they carry bridge weights
    w(u), w(v)). Returned as index arrays so the molecular head can compute
    ``q_{τ,τ'} = Σ w_a(u)·w_b(v)`` differentiably.

    Returns ``(iu, iv)`` int64 arrays of equal length (possibly empty)."""
    if support.n_support == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    budget = support.l_max if l_copath is None else int(l_copath)
    glob = support.global_idx
    da = support.d_a.astype(np.int64)
    db = support.d_b.astype(np.int64)
    local_of = np.full(kg.n_nodes, -1, dtype=np.int64)
    local_of[glob] = np.arange(support.n_support, dtype=np.int64)

    iu_list: list[np.ndarray] = []
    iv_list: list[np.ndarray] = []
    for iu in range(support.n_support):
        depth = budget - int(da[iu]) - 1   # min db is 1, so d(u,v) <= budget-da-1
        if depth < 1:
            continue
        reached, dist_u = kg.bounded_bfs(int(glob[iu]), depth)
        iv = local_of[reached]
        keep = (dist_u > 0) & (iv >= 0)
        if not keep.any():
            continue
        iv = iv[keep]
        duv = dist_u[keep].astype(np.int64)
        within = (int(da[iu]) + duv + db[iv]) <= budget
        if not within.any():
            continue
        iv_ok = iv[within]
        iu_list.append(np.full(iv_ok.shape[0], iu, dtype=np.int64))
        iv_list.append(iv_ok)
    if not iu_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(iu_list), np.concatenate(iv_list)


__all__ = [
    "StructFeatures",
    "compute_struct_features",
    "compute_copath_pairs",
    "symmetric_binary_vector",
    "symmetric_binary_dim",
]

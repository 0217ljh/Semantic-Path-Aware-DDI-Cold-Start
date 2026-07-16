"""Symmetrized merged-KG loading + bounded-distance corridor support.

Frozen-design steps 1-2 (``Notes/Log/algorithm_design_debate.md`` R7/R8):

  1. Build the symmetrized merged KG ``G_sym``: every triple is treated as
     bidirectional for *reachability* (raw triple direction is a curation
     artifact, R8). The relation label and the original ``directed`` flag are
     retained for later relation-direction features but do NOT affect
     distances.
  2. For a query pair ``(a, b)`` build a compact corridor support
     ``G^(a,b)`` = nodes ``v`` with ``d_sym(a,v) + d_sym(v,b) <= l_max``,
     then control hubs with a degree-penalised per-type top-K and a global
     node cap ``n_max``.

Distances are exact shortest-hop distances on the *unweighted symmetrized*
graph; bounded BFS keeps each query cheap. Node ``kind`` is canonicalised to
the same 11-group + ``other`` taxonomy as PMP v1.6
(``my_code/models/pmp_v1/precompute_pmp_cache.py``) so the coarse level is
directly comparable to v1.6. The table is mirrored here (not imported) to keep
this package self-contained.

Schema verified 2026-06-22 against
``Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet`` (cols
``id, kind, name, source_kg``; 178,029 rows) and
``edges__...__mask1.parquet`` (cols ``src, src_kind, dst, dst_kind, relation,
source_kg, directed``; 7,099,528 rows; DDI edges already masked).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# --------------------------------------------------------------------------
# Type taxonomy — mirrors PMP v1.6 (11 KG kind groups + "other") for
# direct comparability of the coarse level. Do NOT import from pmp_v1: keep
# this package independently inspectable (a constant table, not shared logic).
# --------------------------------------------------------------------------
KIND_GROUPS: dict[str, list[str]] = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "pathway": ["Pathway", "pathway"],
    "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["Disease", "disease"],
    "anatomy": ["Anatomy", "anatomy"],
    "compound": ["Compound"],
    "biological_process": ["Biological Process", "biological_process"],
    "molecular_function": ["Molecular Function", "molecular_function"],
    "cellular_component": ["Cellular Component", "cellular_component"],
    "pharmacologic_class": ["Pharmacologic Class"],
    "exposure": ["exposure"],
}
KIND_ORDER: tuple[str, ...] = tuple(KIND_GROUPS.keys()) + ("other",)
N_TYPES: int = len(KIND_ORDER)
OTHER_TYPE_ID: int = N_TYPES - 1
_KIND_TO_GROUP: dict[str, str] = {
    raw: grp for grp, raws in KIND_GROUPS.items() for raw in raws
}

#: Node kinds treated as drugs (never mediators). Matches v1.6's
#: ``non-drug = kind not in {"Drug", "drug"}``.
DRUG_KINDS: frozenset[str] = frozenset({"Drug", "drug"})

# --------------------------------------------------------------------------
# Drug->mediator RELATION buckets (the v1.6 rel(a->m) attribute). The relation
# carries mechanism specificity (db:target vs db:enzyme vs het:CdG "down-
# regulates gene" vs het:CcSE "causes side effect"). Validated to add +2.2pt
# KG-only. Bucket 9 = other, 10 = REL_2HOP (mediator not 1-hop from this drug).
# --------------------------------------------------------------------------
REL_BUCKET: dict[str, int] = {
    "db:target": 0, "prime:drug_protein": 0,
    "db:enzyme": 1, "db:transporter": 2, "db:carrier": 3, "db:pathway": 4,
    "het:CdG": 5, "het:CuG": 5, "het:CbG": 5,            # gene up/down/bind
    "het:CcSE": 6, "prime:drug_effect": 6,               # side effect / effect
    "prime:indication": 7, "prime:off-label use": 7,
    "prime:contraindication": 8,
}
REL_OTHER: int = 9
REL_2HOP: int = 10
N_REL_BUCKETS: int = 11

DEFAULT_NODES_PATH = (
    "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
)
DEFAULT_EDGES_PATH = (
    "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
)


@dataclass
class PairSupport:
    """Compact corridor support for one query pair ``(a, b)``.

    Arrays are aligned and indexed by *local* support position
    ``0 .. n_support-1``. ``global_idx`` maps each local position back to the
    KG node integer id.
    """

    a_idx: int
    b_idx: int
    global_idx: np.ndarray   # (n_support,) int64 — KG node ids (after hub cap)
    d_a: np.ndarray          # (n_support,) int16 — d_sym(a, v)
    d_b: np.ndarray          # (n_support,) int16 — d_sym(b, v)
    type_id: np.ndarray      # (n_support,) int16 — canonical type id in [0, N_TYPES)
    l_max: int
    #: Exact UNCAPPED per-type common-reachability count s_tau^(l) (N_TYPES,).
    #: This is the coarse structural feature and is exact (the hub cap only
    #: limits the downstream per-node set, not this count).
    type_count: np.ndarray
    #: Exact UNCAPPED Adamic-Adar-style degree-weighted per-type count
    #: (N_TYPES,): sum over common-reachable mediators u of 1/log(deg(u)).
    #: Down-weights ubiquitous hubs (low specificity) vs rare shared mediators
    #: (high specificity) — the failure analysis showed raw counts are swamped
    #: by high-degree protein/gene hubs.
    type_aa: np.ndarray

    @property
    def n_support(self) -> int:
        return int(self.global_idx.shape[0])


class MergedKG:
    """Symmetrized merged KG with bounded-BFS shortest-hop distances.

    Construct via :meth:`from_parquet`. Holds an undirected CSR adjacency
    (every triple added in both directions, de-duplicated) plus a per-node
    canonical type id and degree. Relation labels are *not* stored here; they
    are consumed by :mod:`struct_features` directly from the edge table when
    relation-direction features are enabled.
    """

    def __init__(
        self,
        *,
        node_ids: list[str],
        indptr: np.ndarray,
        indices: np.ndarray,
        type_id: np.ndarray,
        is_drug: np.ndarray,
        drug_rel: dict[int, dict[int, int]] | None = None,
    ) -> None:
        self.node_ids: list[str] = node_ids
        self.id_to_idx: dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
        self.indptr: np.ndarray = indptr          # (n_nodes + 1,) int64 — CSR row ptr
        self.indices: np.ndarray = indices         # (nnz,) int32 — undirected, de-duplicated
        self.type_id: np.ndarray = type_id         # (n_nodes,) int16
        self.is_drug: np.ndarray = is_drug         # (n_nodes,) bool
        self.degree: np.ndarray = np.diff(indptr).astype(np.int64)
        # {drug_idx: {neighbour_idx: rel_bucket}} for 1-hop drug-incident edges.
        self.drug_rel: dict[int, dict[int, int]] = drug_rel or {}
        # Bounded-neighbourhood cache: (node_idx, depth) -> (nodes, dists).
        self._nbhd_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    # -- construction -------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        nodes_path: str | Path = DEFAULT_NODES_PATH,
        edges_path: str | Path = DEFAULT_EDGES_PATH,
    ) -> "MergedKG":
        nodes = pd.read_parquet(nodes_path, columns=["id", "kind"])
        edges = pd.read_parquet(edges_path, columns=["src", "dst", "relation"])

        if nodes["id"].isna().any():
            raise ValueError("merged-KG nodes parquet has null id(s)")
        node_ids: list[str] = nodes["id"].astype(str).tolist()
        n_nodes = len(node_ids)
        if len(set(node_ids)) != n_nodes:
            raise ValueError("merged-KG node ids are not unique")
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        kinds = nodes["kind"].astype(str).to_numpy()
        type_id = np.fromiter(
            (KIND_ORDER.index(_KIND_TO_GROUP.get(k, "other")) for k in kinds),
            dtype=np.int16,
            count=n_nodes,
        )
        is_drug = np.fromiter(
            (k in DRUG_KINDS for k in kinds), dtype=bool, count=n_nodes
        )

        # Map endpoints to integer ids; drop any edge whose endpoint is not a
        # known node (defensive — merged KG is self-consistent, but masked
        # edge tables can reference pruned ids).
        src = edges["src"].astype(str).map(id_to_idx).to_numpy()
        dst = edges["dst"].astype(str).map(id_to_idx).to_numpy()
        valid = ~(pd.isna(src) | pd.isna(dst))
        rel = edges["relation"].astype(str).map(
            lambda r: REL_BUCKET.get(r, REL_OTHER)).to_numpy()[valid]
        src = src[valid].astype(np.int64)
        dst = dst[valid].astype(np.int64)

        # Build {drug: {neighbour: rel_bucket}} from 1-hop drug-incident edges
        # (both directions, since the stored triple direction is arbitrary).
        drug_rel: dict[int, dict[int, int]] = {}
        s_is_drug = is_drug[src]
        d_is_drug = is_drug[dst]
        for s, d, b in zip(src[s_is_drug], dst[s_is_drug], rel[s_is_drug]):
            drug_rel.setdefault(int(s), {}).setdefault(int(d), int(b))
        for s, d, b in zip(dst[d_is_drug], src[d_is_drug], rel[d_is_drug]):
            drug_rel.setdefault(int(s), {}).setdefault(int(d), int(b))

        # Symmetrise (every triple bidirectional) and drop self-loops; the CSR
        # build via scipy de-duplicates parallel edges so `degree` is the true
        # number of distinct undirected neighbours (the hub penalty depends on
        # this being correct).
        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        loop = u != v
        u = u[loop]
        v = v[loop]
        adj = sp.coo_matrix(
            (np.ones(u.shape[0], dtype=np.int8), (u, v)),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        adj.sum_duplicates()
        adj.data[:] = 1  # collapse multiplicities to a binary adjacency

        return cls(
            node_ids=node_ids,
            indptr=adj.indptr.astype(np.int64),
            indices=adj.indices.astype(np.int32),
            type_id=type_id,
            is_drug=is_drug,
            drug_rel=drug_rel,
        )

    def support_relations(
        self, support: "PairSupport",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-mediator relation buckets (rel_a, rel_b) aligned with
        ``support.global_idx``: bucket of the a->m / b->m edge if m is 1-hop
        from that drug, else ``REL_2HOP``."""
        n = support.n_support
        rel_a = np.full(n, REL_2HOP, dtype=np.int64)
        rel_b = np.full(n, REL_2HOP, dtype=np.int64)
        ra = self.drug_rel.get(support.a_idx, {})
        rb = self.drug_rel.get(support.b_idx, {})
        for i, m in enumerate(support.global_idx.tolist()):
            if m in ra:
                rel_a[i] = ra[m]
            if m in rb:
                rel_b[i] = rb[m]
        return rel_a, rel_b

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    # -- bounded BFS --------------------------------------------------------

    def bounded_bfs(self, source: int, max_hops: int) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised bounded BFS on the symmetrised CSR.

        Returns ``(nodes, dists)`` int arrays for every node within
        ``max_hops`` of ``source`` (including ``source`` at distance 0). The
        per-hop neighbour gather is fully vectorised so hub frontiers (degree
        up to ~3.5e4 here) stay cheap.
        """
        n = self.n_nodes
        dist = np.full(n, -1, dtype=np.int16)
        dist[source] = 0
        reached = [np.array([source], dtype=np.int64)]
        frontier = np.array([source], dtype=np.int64)
        indptr = self.indptr
        indices = self.indices
        for hop in range(1, max_hops + 1):
            if frontier.size == 0:
                break
            starts = indptr[frontier]
            counts = indptr[frontier + 1] - starts
            total = int(counts.sum())
            if total == 0:
                break
            # Gather all neighbours of the frontier in one vectorised shot:
            # positions = repeat(starts, counts) + (arange(total) - block offsets).
            block_start = np.repeat(np.cumsum(counts) - counts, counts)
            pos = np.repeat(starts, counts) + (np.arange(total) - block_start)
            neigh = indices[pos]
            neigh = np.unique(neigh)
            fresh = neigh[dist[neigh] == -1]
            if fresh.size == 0:
                break
            dist[fresh] = hop
            reached.append(fresh)
            frontier = fresh
        nodes = np.concatenate(reached)
        return nodes, dist[nodes].astype(np.int16)

    def neighborhood(self, source: int, depth: int) -> tuple[np.ndarray, np.ndarray]:
        """Memoised :meth:`bounded_bfs`. Drugs recur across many pairs, so
        caching the per-drug bounded neighbourhood turns per-pair work into a
        cheap intersection."""
        key = (source, depth)
        cached = self._nbhd_cache.get(key)
        if cached is None:
            cached = self.bounded_bfs(source, depth)
            self._nbhd_cache[key] = cached
        return cached

    def marginal_affinity(self, source: int, depth: int) -> np.ndarray:
        """Per-drug MARGINAL type-incidence affinity (N_TYPES,).

        ``affinity[tau] = log1p( #non-drug type-tau nodes within `depth` hops )``
        — the v1.6 routing signal, here a first-class marginal of the hyper-edge
        incidence (gates the joint pair pooling, and is the pre-training target).
        Mediator-only (drugs excluded), so it is a pure mechanism profile.
        """
        nodes, _ = self.neighborhood(source, depth)
        t = self.type_id[nodes]
        nd = ~self.is_drug[nodes]
        cnt = np.bincount(t[nd].astype(np.int64), minlength=N_TYPES)
        return np.log1p(cnt.astype(np.float32))


def build_pair_support(
    kg: MergedKG,
    a_idx: int,
    b_idx: int,
    *,
    l_max: int = 3,
    k_per_type: int = 64,
    n_max: int = 400,
    deg_penalty: float = 1.0,
    support_and: bool = False,
) -> PairSupport:
    """Build the compact corridor support ``G^(a,b)`` (frozen step 2).

    A node ``v`` is a corridor candidate iff
    ``d_sym(a,v) + d_sym(v,b) <= l_max``. Drugs ``a``/``b`` themselves and any
    other drug nodes are excluded (mediators only). Hub control, applied in
    order:

      1. distance budget (the candidate filter above),
      2. per-type top-``k_per_type`` by corridor score
         ``r(v) = -(d_a + d_b) - deg_penalty * log(1 + degree(v))``
         (degree-*penalised* so genuine close hubs survive while ubiquitous
         far hubs are dropped),
      3. a global cap ``n_max`` by the same score.

    Returns a :class:`PairSupport`. May be empty (``n_support == 0``) — callers
    must handle the fragment-only fallback case.

    Implementation: a non-drug mediator ``v != a, b`` has ``d_a(v) >= 1`` and
    ``d_b(v) >= 1``, so the budget ``d_a + d_b <= l_max`` forces each side
    ``<= l_max - 1``. We therefore only need bounded neighbourhoods to depth
    ``l_max - 1`` (cached per drug), intersected on shared nodes. Distances are
    computed on the full symmetrised graph, so a corridor may still transit a
    third drug; only the *returned mediator set* excludes drug nodes (drug
    identity is never a mechanism type).
    """
    if a_idx == b_idx:
        raise ValueError(f"self-pair query (a_idx == b_idx == {a_idx})")

    depth = max(0, l_max - 1)
    nodes_a, dist_a = kg.neighborhood(a_idx, depth)
    nodes_b, dist_b = kg.neighborhood(b_idx, depth)

    # Shared reachable nodes (corridor candidates) via sorted intersection.
    common, ia, ib = np.intersect1d(
        nodes_a, nodes_b, assume_unique=True, return_indices=True
    )
    if common.size == 0:
        return _empty_support(a_idx, b_idx, l_max)

    da = dist_a[ia].astype(np.int64)
    db = dist_b[ib].astype(np.int64)
    # "sum" mode: corridor budget d_a + d_b <= l_max.
    # "and" mode (support_and): keep every shared node within depth = l_max-1 of
    # BOTH drugs, i.e. {d_a <= l_max-1 AND d_b <= l_max-1}. At l_max=3 this is
    # exactly v1.6's N_<=2(a) ∩ N_<=2(b) — it ALSO keeps the (2,2) mediators the
    # sum budget drops.
    budget_ok = np.ones(da.shape, dtype=bool) if support_and else (da + db) <= l_max
    not_drug = ~kg.is_drug[common]
    not_endpoint = (common != a_idx) & (common != b_idx)
    sel = budget_ok & not_drug & not_endpoint
    if not sel.any():
        return _empty_support(a_idx, b_idx, l_max)

    cand_arr = common[sel]
    da_arr = da[sel].astype(np.int16)
    db_arr = db[sel].astype(np.int16)
    type_arr = kg.type_id[cand_arr]
    deg_arr = kg.degree[cand_arr]
    # Exact uncapped s_tau^(l): count of common-reachable mediators per type.
    type_count = np.bincount(type_arr.astype(np.int64), minlength=N_TYPES)
    # Adamic-Adar weight 1/log(deg); clip deg>=2 so log>0 (deg-1 nodes -> weight
    # capped, deg-0 impossible here since u is reachable from both endpoints).
    aa_w = 1.0 / np.log(np.clip(deg_arr.astype(np.float64), 2.0, None))
    type_aa = np.bincount(type_arr.astype(np.int64), weights=aa_w,
                          minlength=N_TYPES)
    score = -(da_arr.astype(np.float64) + db_arr.astype(np.float64)) \
        - deg_penalty * np.log1p(deg_arr.astype(np.float64))

    # Deterministic ranking key: higher score first, ties broken by node id.
    def _topk(pos: np.ndarray, k: int) -> np.ndarray:
        if pos.size <= k:
            return pos
        order = np.lexsort((cand_arr[pos], -score[pos]))  # primary -score, tie node id
        return pos[order[:k]]

    # (2) per-type top-k.
    keep_mask = np.zeros(cand_arr.shape[0], dtype=bool)
    for t in range(N_TYPES):
        t_pos = np.where(type_arr == t)[0]
        if t_pos.size:
            keep_mask[_topk(t_pos, k_per_type)] = True
    kept = np.where(keep_mask)[0]

    # (3) global cap by the same score.
    kept = _topk(kept, n_max)
    kept.sort()  # stable, readable order in the returned support

    return PairSupport(
        a_idx=a_idx,
        b_idx=b_idx,
        global_idx=cand_arr[kept],
        d_a=da_arr[kept],
        d_b=db_arr[kept],
        type_id=type_arr[kept],
        l_max=l_max,
        type_count=type_count,
        type_aa=type_aa,
    )


def _empty_support(a_idx: int, b_idx: int, l_max: int) -> PairSupport:
    return PairSupport(
        a_idx=a_idx,
        b_idx=b_idx,
        global_idx=np.empty(0, dtype=np.int64),
        d_a=np.empty(0, dtype=np.int16),
        d_b=np.empty(0, dtype=np.int16),
        type_id=np.empty(0, dtype=np.int16),
        l_max=l_max,
        type_count=np.zeros(N_TYPES, dtype=np.int64),
        type_aa=np.zeros(N_TYPES, dtype=np.float64),
    )


__all__ = [
    "KIND_GROUPS",
    "KIND_ORDER",
    "N_TYPES",
    "OTHER_TYPE_ID",
    "DRUG_KINDS",
    "MergedKG",
    "PairSupport",
    "build_pair_support",
    "DEFAULT_NODES_PATH",
    "DEFAULT_EDGES_PATH",
]

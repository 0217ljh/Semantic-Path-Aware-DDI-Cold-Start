"""Junction-finder: per drug pair, identify "meeting node" candidates.

Junction = an intermediate node likely to lie on a mechanistic causal path
between the two drugs. Five variants (per first_step_plan.md §4.6):

  J1     : shared 1-hop neighbors (structural, no hyperparameter)
  J3-PK  : J1 restricted to `kind ∈ {Gene/Protein, Pathway}` (molecular layer)
  J3-PD  : J1 restricted to `kind ∈ {SideEffect, Disease, Anatomy, Phenotype}`
  J3-both: J3-PK ∪ J3-PD
  J7     : shared 2-hop neighbors (deeper junction variant)
  Jr     : random-set with same cardinality as J1 (capacity control, R6 in plan)

Input: pre-built adjacency dict {node_id -> set(neighbor_ids)} + entity-id -> int map.
Output: per-pair junction sets as int-id lists, padded to a fixed K (default 32).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


def find_shared_neighbors(
    drug_a: int, drug_b: int,
    adj_int: dict[int, set[int]],
    kind_filter: set[str] | None = None,
    id2kind: dict[int, str] | None = None,
) -> set[int]:
    """Return |N1(a) ∩ N1(b)|, optionally restricted to `kind_filter`."""
    na = adj_int.get(drug_a, set())
    nb = adj_int.get(drug_b, set())
    inter = na & nb
    if kind_filter is None or id2kind is None:
        return inter
    return {n for n in inter if id2kind.get(n, "") in kind_filter}


def find_2hop_intersection(
    drug_a: int, drug_b: int,
    adj_int: dict[int, set[int]],
) -> set[int]:
    """|N2(a) ∩ N2(b)| via 2-hop set expansion. Excludes a/b themselves."""
    na2 = set()
    for nbr in adj_int.get(drug_a, set()):
        na2 |= adj_int.get(nbr, set())
    nb2 = set()
    for nbr in adj_int.get(drug_b, set()):
        nb2 |= adj_int.get(nbr, set())
    return (na2 & nb2) - {drug_a, drug_b}


LAYER_PK = {"Gene/Protein", "Pathway"}
LAYER_PD = {"SideEffect", "Disease", "Anatomy", "Phenotype"}


def build_junction_table(
    pair_indices: list[tuple[int, int]],
    adj_int: dict[int, set[int]],
    id2kind: dict[int, str],
    junction_type: str = "J1",
    max_junctions: int = 32,
    rng_seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (B, K) int matrix of junction node IDs per pair.

    Returns:
        junction_ids: shape (len(pair_indices), max_junctions), padded with -1
        junction_mask: shape (len(pair_indices), max_junctions), 1 = valid, 0 = pad

    Padding semantics: -1 means "no junction at this slot"; mask=0.
    """
    rng = np.random.default_rng(rng_seed)
    B = len(pair_indices)
    junction_ids = np.full((B, max_junctions), -1, dtype=np.int64)
    junction_mask = np.zeros((B, max_junctions), dtype=np.float32)

    for i, (a, b) in enumerate(pair_indices):
        if junction_type == "J1":
            cand = find_shared_neighbors(a, b, adj_int)
        elif junction_type == "J3-PK":
            cand = find_shared_neighbors(a, b, adj_int, LAYER_PK, id2kind)
        elif junction_type == "J3-PD":
            cand = find_shared_neighbors(a, b, adj_int, LAYER_PD, id2kind)
        elif junction_type == "J3-both":
            cand = find_shared_neighbors(a, b, adj_int, LAYER_PK | LAYER_PD, id2kind)
        elif junction_type == "J7":
            cand = find_2hop_intersection(a, b, adj_int)
        elif junction_type == "Jr":
            j1 = find_shared_neighbors(a, b, adj_int)
            n_target = len(j1)
            cand = set(rng.choice(list(adj_int.keys()), size=min(n_target, max_junctions), replace=False))
        else:
            raise ValueError(f"Unknown junction_type={junction_type!r}")

        cand_list = sorted(cand)[:max_junctions]
        if len(cand_list) > 0:
            junction_ids[i, :len(cand_list)] = cand_list
            junction_mask[i, :len(cand_list)] = 1.0

    if verbose:
        n_with_junction = (junction_mask.sum(axis=1) > 0).sum()
        print(f"[junction] type={junction_type} K={max_junctions} "
              f"coverage={n_with_junction:,}/{B:,} ({100*n_with_junction/B:.1f}%)")
    return junction_ids, junction_mask


def build_adj_int(
    edges_df,
    entity2id: dict[str, int],
) -> dict[int, set[int]]:
    """Convert merged-KG edges DataFrame to {int_id: set(int_id)} adjacency."""
    adj = defaultdict(set)
    src = edges_df["src"].astype(str)
    dst = edges_df["dst"].astype(str)
    for s, d in zip(src, dst):
        si = entity2id.get(s)
        di = entity2id.get(d)
        if si is None or di is None:
            continue
        adj[si].add(di)
        adj[di].add(si)
    return adj

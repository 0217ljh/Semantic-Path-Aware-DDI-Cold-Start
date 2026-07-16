"""Find structural gaps in PK/PD subgraphs for VME injection.

A "gap" = a pair (a, c) where:
  - a and c are both in the subgraph
  - distance(a, c) ≥ 2 in this subgraph
  - no intermediate node exists in this subgraph

We restrict candidate gaps to (a, c) ON shortest paths between train+test
drug pairs (otherwise enumeration is combinatorial).

Output: list of (a_id, c_id, layer) tuples for VME generation.
Hard cap: 20K gaps per layer (PK + PD), per first_step_plan.md §4.7
'gap pruning' specification.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))


def _build_adj_set(edges_df: pd.DataFrame) -> dict[str, set[str]]:
    adj = defaultdict(set)
    for s, d in zip(edges_df["src"].astype(str), edges_df["dst"].astype(str)):
        adj[s].add(d)
        adj[d].add(s)
    return adj


def find_gaps_for_pairs(
    drug_pairs: list[tuple[str, str]],
    subgraph_edges_df: pd.DataFrame,
    layer_name: str,
    max_gaps: int = 20_000,
    max_paths_per_pair: int = 3,
) -> list[tuple[str, str, str]]:
    """Find candidate VME gaps along shortest paths between drug pairs.

    For each pair (s, t), do BFS up to 3 hops; for each path found, identify
    'gap positions' where a hypothetical intermediate would be useful.

    Heuristic gap criterion: if two adjacent nodes in the path are themselves
    distance ≥2 apart in the FULL subgraph (i.e. the path uses a long-range
    edge), the gap between them is a VME candidate.

    For computational tractability, in this skeleton we use a SIMPLER
    criterion: for each pair (s, t) at distance ≥3, the canonical pair
    (a=closest neighbor of s, c=closest neighbor of t) is the VME gap.
    """
    adj = _build_adj_set(subgraph_edges_df)
    gap_counter: Counter[tuple[str, str]] = Counter()

    for s, t in drug_pairs:
        if s not in adj or t not in adj:
            continue
        # BFS from s, looking for t
        parents = {s: None}
        frontier = {s}
        found_at = None
        for hop in range(1, 4):
            next_frontier = set()
            for n in frontier:
                for nb in adj[n]:
                    if nb not in parents:
                        parents[nb] = n
                        next_frontier.add(nb)
            if t in next_frontier:
                found_at = hop
                break
            frontier = next_frontier
            if not frontier:
                break

        if found_at is None or found_at < 2:
            continue  # no path or direct neighbor

        # Reconstruct path from t back to s
        path = [t]
        cur = t
        while parents.get(cur) is not None:
            cur = parents[cur]
            path.append(cur)
        path.reverse()
        # Gap candidates: each (path[i-1], path[i+1]) where i is an internal node
        # Use this as proxy: between adjacent internal positions
        for i in range(1, len(path) - 1):
            a, c = path[i - 1], path[i + 1]
            # canonical ordering for dedup
            key = tuple(sorted([a, c]))
            gap_counter[key] += 1

    # Take top-K most frequent gaps
    top = gap_counter.most_common(max_gaps)
    return [(a, c, layer_name) for (a, c), _cnt in top]


def enumerate_all_gaps(
    drug_pairs: list[tuple[str, str]],
    pk_edges: pd.DataFrame,
    pd_edges: pd.DataFrame,
    max_gaps_per_layer: int = 20_000,
) -> list[tuple[str, str, str]]:
    """Enumerate gaps for both PK and PD subgraphs. Cap per layer."""
    pk_gaps = find_gaps_for_pairs(drug_pairs, pk_edges, "PK", max_gaps_per_layer)
    pd_gaps = find_gaps_for_pairs(drug_pairs, pd_edges, "PD", max_gaps_per_layer)
    print(f"[vme_gap] PK gaps: {len(pk_gaps):,}; PD gaps: {len(pd_gaps):,}")
    return pk_gaps + pd_gaps

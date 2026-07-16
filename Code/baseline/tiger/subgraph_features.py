"""Per-drug BKG subgraph builder for TIGER.

Supports all 3 paper extractors (paper Sec "Biomedical Knowledge Graph
Channel" / Su et al. AAAI 2024):

  - ``"randomWalk"`` (default) — DeepWalk; original Round-1 implementation
  - ``"khop-subtree"``         — k-hop BFS with per-hop fan-out cap
  - ``"probability"``          — personalized PageRank-weighted sampling

Ports the ``subtreeExtractor`` / ``probExtractor`` / ``rwExtractor`` branches of
``Code-Released/baseline/TIGER/data_process.py:generate_node_subgraphs``
but produces a ``torch_geometric.data.Data`` object per drug directly
(no JSON caching) and consumes the BKG dict from
:func:`baseline.tiger.kg_builder.build_bkg_from_merged_parquet`.

Restored 2026-05-18 (codex round-1 fix #3): the previous version only
supported ``rwExtractor`` which violated CLAUDE.md §"Baseline 规范" §4
("不允许借口'新任务'省略 paper 算法核心").

Each drug's :class:`Data` object carries:

* ``x``           — ``LongTensor`` of global BKG node indices in the
                    drug's subgraph (fed to the node-type
                    :class:`Embedding`).
* ``edge_index``  — ``[2, n_directed_sub_edges]`` relabeled subgraph
                    edges (drug-local index space).
* ``id``          — boolean mask over ``subset`` indicating the center
                    node (the drug itself).
* ``rel_index``   — relation id per direct edge in the subgraph.
* ``sp_edge_index`` — shortest-path edges within the subgraph.
* ``sp_value``    — sp lengths.
* ``sp_edge_rel`` — packed sp relation: for length-1, the underlying
                    KG relation id; for length>1, ``length + num_rel``
                    (matches upstream's offset trick).

Concurrency: we run subgraph extraction sequentially. The cost is
``O(n_drugs)`` deepwalks + ``O(n_drugs)`` ``subgraph()`` calls; for
2k drugs this is ~10-30 seconds.
"""

from __future__ import annotations

import warnings
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, subgraph

import random

from baseline.tiger.random_walk import deepwalk_unique, seed_walks


# ---------------------------------------------------------------------
# Node-set selectors (one per paper extractor)
# ---------------------------------------------------------------------

def _select_nodes_random_walk(
    g: nx.Graph, start: int, *, num_walks: int, walk_length: int
) -> list[int]:
    """DeepWalk-based node selection. Always includes ``start``.

    Paper Sec "DeepWalk-based Extractor" (TIGER-DW). Returns the unique
    set of nodes visited by ``num_walks`` walks of length ``walk_length``
    starting from ``start``.
    """
    subset = deepwalk_unique(
        g, start_node=start, num_walks=num_walks, walk_length=walk_length
    )
    if start not in subset:
        subset.append(start)
    return subset


def _select_nodes_khop_subtree(
    g: nx.Graph, start: int, *, khop: int, fanout: int
) -> list[int]:
    """k-hop BFS with per-hop fan-out cap. Always includes ``start``.

    Paper Sec "k-subtree-based Extractor" (TIGER-KS). Visits up to
    ``fanout`` neighbours per node per hop (random sample without
    replacement when degree > fanout), recursing ``khop`` hops. Returns
    the union of visited nodes.

    Note: paper Sec "Experimental Settings" specifies ``k = 2`` and
    fanout ``= 4`` for TIGER-KS on DrugBank.
    """
    visited: set[int] = {start}
    frontier: list[int] = [start]
    for _ in range(khop):
        next_frontier: list[int] = []
        for node in frontier:
            if node not in g:
                continue
            neighbours = list(g.neighbors(node))
            if len(neighbours) > fanout:
                neighbours = random.sample(neighbours, fanout)
            for nbr in neighbours:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.append(nbr)
        frontier = next_frontier
    return sorted(visited)


def _select_nodes_probability(
    g: nx.Graph, start: int, *, fixed_num: int
) -> list[int]:
    """Personalized-PageRank-weighted sampling of ``fixed_num`` nodes.

    Paper Sec "Probability-based Extractor" (TIGER-P). Replaces the
    upstream ``probExtractor`` which calls ``google_matrix`` (a dense
    N×N float64 matrix) — that path OOMs on our 391k-node merged BKG
    (~1.11 TiB allocation). We use NetworkX's sparse iterative
    :func:`networkx.pagerank` with a personalization vector concentrated
    on ``start``, then sample ``fixed_num`` nodes by the resulting
    distribution (without replacement). ``start`` is always included.
    """
    # Build personalization vector concentrated on start node.
    if start not in g:
        return [start]
    pers = {start: 1.0}
    try:
        pr_map = nx.pagerank(g, personalization=pers, max_iter=200, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        # Fall back to uniform sampling among neighbours if PageRank
        # fails to converge (rare on connected biomedical KGs).
        pr_map = {n: 1.0 / max(len(g.nodes()), 1) for n in g.nodes()}

    nodes = list(pr_map.keys())
    probs = np.array([pr_map[n] for n in nodes], dtype=np.float64)
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return [start]
    probs /= probs_sum

    k = min(fixed_num, len(nodes))
    sampled_idx = np.random.choice(len(nodes), size=k, replace=False, p=probs)
    subset_set: set[int] = {int(nodes[i]) for i in sampled_idx}
    subset_set.add(int(start))
    return sorted(subset_set)


def _undirected_edge_tensor(
    edge_list: list[list[int]], rel_list: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build undirected edge_index + rel by concat-with-reverse.

    Matches upstream ``data_process.generate_node_subgraphs`` lines
    273-275: ``undirected_edge_index = cat((edge_index, reverse), 1)``
    and ``undirected_rel_index = cat((rel, rel), 0)``.
    """
    if not edge_list:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.long),
        )
    e = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # [2, n]
    r = torch.tensor(rel_list, dtype=torch.long)
    rev = torch.stack((e[1], e[0]), dim=0)
    e_undir = torch.cat((e, rev), dim=1)
    r_undir = torch.cat((r, r), dim=0)
    return e_undir, r_undir


def _shortest_path_pairs(edge_index_np: np.ndarray) -> np.ndarray:
    """All-pairs SP lengths on a (small) directed graph. Returns
    ``[[i, j, length], ...]`` sorted, matching upstream's
    ``calculate_shortest_path``."""
    g = nx.DiGraph()
    g.add_edges_from(edge_index_np.tolist())
    rows = []
    for node_i, node_ij in nx.all_pairs_shortest_path_length(g):
        for node_j, length in node_ij.items():
            rows.append([node_i, node_j, length])
    rows.sort()
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def build_drug_subgraphs(
    bkg: dict[str, Any],
    *,
    extractor: str = "randomWalk",
    num_walks: int = 1,
    walk_length: int = 32,
    khop: int = 2,
    khop_fanout: int = 4,
    fixed_num: int = 32,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[dict[int, Data], int, int]:
    """Build one PyG ``Data`` subgraph per drug via DeepWalk extraction.

    Args:
        bkg: dict returned by
            :func:`baseline.tiger.kg_builder.build_bkg_from_merged_parquet`.
        num_walks: number of DeepWalk passes per drug.
        walk_length: walk length per pass.
        seed: RNG seed for reproducibility.
        verbose: print summary.

    Returns:
        ``(subgraphs, max_degree, max_sp_rel)``:
          - ``subgraphs[drug_idx] -> Data`` (drug_idx is the BKG node index)
          - ``max_degree``: max in-subgraph degree across all drugs (used
            to size :class:`NodeFeatures.degree_encoder`).
          - ``max_sp_rel``: max value of ``sp_edge_rel`` across all
            drugs (used to size ``num_relations_graph``).
    """
    # Seed BOTH ``random`` (used by deepwalk + khop fanout sampling)
    # AND ``numpy.random`` (used by probability extractor's
    # ``np.random.choice``). codex round-3 found that without seeding
    # numpy, TIGER-P sampling was non-reproducible.
    seed_walks(seed)
    np.random.seed(seed)
    n_drugs = bkg["n_drugs"]
    num_rel = bkg["num_rel"]
    edge_list = bkg["edge_list"]
    rel_list = bkg["rel_list"]

    # Build undirected edges + matching rel ids
    edge_index_undir, rel_index_undir = _undirected_edge_tensor(edge_list, rel_list)

    # Build nx.Graph for DeepWalk (undirected, ignores multi-edges)
    g = nx.Graph()
    if edge_index_undir.numel() > 0:
        g.add_edges_from(edge_index_undir.t().tolist())

    # Ensure all drugs are in the graph even if isolated (self-loop
    # added in kg_builder should already cover this, but be safe)
    for d in range(n_drugs):
        if d not in g:
            g.add_node(d)

    subgraphs: dict[int, Data] = {}
    max_degree = 0
    max_sp_rel = 0

    with warnings.catch_warnings():
        # Silence torch_geometric "scatter" UserWarnings during degree()
        warnings.filterwarnings("ignore", category=UserWarning)

        for d_idx in range(n_drugs):
            # 1. Per-extractor node-set selection (paper 3 variants).
            if extractor == "randomWalk":
                subset_nodes = _select_nodes_random_walk(
                    g, d_idx, num_walks=num_walks, walk_length=walk_length
                )
            elif extractor == "khop-subtree":
                subset_nodes = _select_nodes_khop_subtree(
                    g, d_idx, khop=khop, fanout=khop_fanout
                )
            elif extractor == "probability":
                subset_nodes = _select_nodes_probability(
                    g, d_idx, fixed_num=fixed_num
                )
            else:
                raise ValueError(
                    f"unknown extractor={extractor!r}; expected "
                    "'randomWalk' / 'khop-subtree' / 'probability'"
                )
            if d_idx not in subset_nodes:
                subset_nodes.append(d_idx)  # always include center
            # Deterministic ordering
            subset_nodes_sorted = sorted(subset_nodes)
            subset_t = torch.tensor(subset_nodes_sorted, dtype=torch.long)

            # 2. Induced subgraph (relabel to 0..len(subset)-1)
            sub_e, sub_r = subgraph(
                subset_t,
                edge_index_undir,
                rel_index_undir,
                relabel_nodes=True,
                num_nodes=bkg["n_total_nodes"],
            )

            # 3. Center-node mask in subset-local space
            center_local = subset_nodes_sorted.index(d_idx)
            mapping_id = torch.zeros(len(subset_nodes_sorted), dtype=torch.bool)
            mapping_id[center_local] = True

            # 4. Subgraph degree (for max_degree statistic)
            if sub_e.size(1) > 0:
                _, col = sub_e
                deg = degree(col, num_nodes=len(subset_nodes_sorted), dtype=torch.long)
                d_max = int(deg.max().item())
                max_degree = max(max_degree, d_max)
            else:
                # isolated subgraph (single self-loop only) — make sure
                # at least a self-edge exists so downstream attention has
                # something to work with
                sub_e = torch.tensor([[center_local], [center_local]], dtype=torch.long)
                sub_r = torch.tensor([0], dtype=torch.long)

            # 5. Shortest paths within the subgraph (relabeled space).
            #    Use directed graph for sp; length-1 edges inherit the
            #    underlying rel_id, length>1 edges get length + num_rel.
            sub_e_np = sub_e.t().numpy()
            sp_arr = _shortest_path_pairs(sub_e_np)
            if sp_arr.size == 0:
                sp_edge_index = torch.tensor(
                    [[center_local], [center_local]], dtype=torch.long
                )
                sp_value = torch.tensor([1], dtype=torch.long)
                sp_edge_rel = torch.tensor([0], dtype=torch.long)
            else:
                sp_edge_index = torch.tensor(sp_arr[:, :2].T, dtype=torch.long)
                sp_value = torch.tensor(sp_arr[:, 2], dtype=torch.long)
                # Build sp_rel: for length-1, look up matching edge's rel;
                # for length>1, use length + num_rel
                sp_rel_np = sp_arr[:, 2].copy()
                direct_mask = sp_arr[:, 2] == 1
                if direct_mask.any():
                    # Build (src,dst) -> rel lookup from sub_e + sub_r
                    edge_lookup: dict[tuple[int, int], int] = {}
                    for k in range(sub_e.size(1)):
                        s = int(sub_e[0, k].item())
                        t = int(sub_e[1, k].item())
                        edge_lookup[(s, t)] = int(sub_r[k].item())
                    direct_pairs = sp_arr[direct_mask, :2]
                    direct_positions = np.where(direct_mask)[0]
                    for pos, (s, t) in zip(direct_positions, direct_pairs):
                        sp_rel_np[pos] = edge_lookup.get(
                            (int(s), int(t)), 0
                        )
                sp_rel_np[~direct_mask] += num_rel
                sp_edge_rel = torch.tensor(sp_rel_np, dtype=torch.long)
                max_sp_rel = max(max_sp_rel, int(sp_rel_np.max()))

            data = Data(
                x=subset_t,  # LongTensor of global BKG node indices
                edge_index=sub_e,
                id=mapping_id,
                rel_index=sub_r,
                sp_edge_index=sp_edge_index,
                sp_value=sp_value.float(),
                sp_edge_rel=sp_edge_rel,
            )
            subgraphs[d_idx] = data

    if verbose:
        print(
            f"[tiger-subgraph] built {len(subgraphs)} drug subgraphs; "
            f"max_subgraph_degree={max_degree}, max_sp_rel={max_sp_rel}"
        )

    return subgraphs, max_degree, max_sp_rel

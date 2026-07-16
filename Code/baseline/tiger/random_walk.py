"""DeepWalk random-walk helper for TIGER BKG subgraphs.

Port of ``Code-Released/baseline/TIGER/randomWalk/{walker.py,node2vec.py}``
restricted to the ``dw=True`` (DeepWalk) branch — TIGER's
``rwExtractor`` only ever uses DeepWalk (``Node2vec(..., dw=True)``),
so the alias-edge / Node2Vec machinery is dead code in the upstream
pipeline.

Given a ``networkx.Graph`` and a start node, ``deepwalk_unique`` runs
``num_walks`` random walks of length ``walk_length`` from the start
node and returns the union of visited nodes (the start node is always
included). This matches ``BasicWalker.simulate_walks`` in upstream,
where each walk is extended (not appended) into a flat node list and
then de-duplicated via ``set``.
"""

from __future__ import annotations

import random
from typing import Iterable

import networkx as nx


def _single_walk(graph: nx.Graph, start_node: int, walk_length: int) -> list[int]:
    """One DeepWalk-style random walk from ``start_node``."""
    walk: list[int] = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        nbrs = list(graph.neighbors(cur))
        if not nbrs:
            break
        walk.append(random.choice(nbrs))
    return walk


def deepwalk_unique(
    graph: nx.Graph,
    start_node: int,
    num_walks: int = 1,
    walk_length: int = 32,
) -> list[int]:
    """Return the de-duplicated node set visited by ``num_walks`` DeepWalk
    walks of length ``walk_length`` starting at ``start_node``.

    Matches upstream ``BasicWalker.simulate_walks`` semantics: a single
    flat list of visited nodes, then ``list(set(...))`` to drop dups.
    Guarantees ``start_node`` is present in the output even if the
    graph is empty or the start node has no neighbors (walk is just
    ``[start_node]`` in that case).
    """
    visited: list[int] = []
    for _ in range(num_walks):
        visited.extend(_single_walk(graph, start_node, walk_length))
    return list(set(visited))


def seed_walks(seed: int | None) -> None:
    """Set ``random`` seed for reproducible BKG subgraph sampling."""
    if seed is not None:
        random.seed(seed)


def deepwalk_unique_many(
    graph: nx.Graph,
    start_nodes: Iterable[int],
    num_walks: int = 1,
    walk_length: int = 32,
) -> dict[int, list[int]]:
    """Convenience: run :func:`deepwalk_unique` for each start node."""
    return {
        int(s): deepwalk_unique(graph, int(s), num_walks=num_walks, walk_length=walk_length)
        for s in start_nodes
    }

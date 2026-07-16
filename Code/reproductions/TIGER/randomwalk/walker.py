"""Random-walk subgraph sampler (DeepWalk + Node2Vec variants).

Direct port of upstream ``randomWalk/walker.py``. The TIGER paper uses
:class:`BasicWalker` (DeepWalk) — :class:`Walker` (Node2Vec) is kept for
fidelity even though the paper does not use it.

NOTE upstream behaviour preserved: :class:`BasicWalker.simulate_walks`
returns ``list(set(walks))`` — i.e. the random-walk path is deduplicated and
order is destroyed. This is what makes the "DeepWalk-based extractor"
described in the paper actually a *node set* sampler, not a sequence walker.
The fixed_size requested via ``length`` may not be reached if the graph is
small (set may have <length distinct nodes).
"""
from __future__ import annotations

import random
from typing import List

import numpy as np


def deepwalk_walk_wrapper(class_instance, walk_length, start_node):
    class_instance.deepwalk_walk(walk_length, start_node)


class BasicWalker:
    def __init__(self, G, start_nodes, workers: int) -> None:
        self.G = G
        self.workers = workers
        self.start_nodes = start_nodes

    def deepwalk_walk(self, walk_length: int, start_node) -> List:
        """Simulate a random walk starting from start node."""
        G = self.G
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def simulate_walks(self, num_walks: int, walk_length: int) -> List:
        """Repeatedly simulate random walks from each node.

        WARNING: returns ``list(set(walks))`` — duplicates removed, sequence
        order lost. This is upstream's actual behaviour.
        """
        walks: List = []
        for _walk_iter in range(num_walks):
            for node in self.start_nodes:
                walks.extend(self.deepwalk_walk(walk_length=walk_length, start_node=node))
        return list(set(walks))


class Walker:
    """Node2Vec walker (alias-table sampling). Not used in paper experiments;
    kept for fidelity to upstream."""

    def __init__(self, G, p: float, q: float, workers: int) -> None:
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    nxt = cur_nbrs[alias_draw(alias_edges[pos][0], alias_edges[pos][1])]
                    walk.append(nxt)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(f"{walk_iter + 1} / {num_walks}")
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q
        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller, larger = [], []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        (smaller if q[kk] < 1.0 else larger).append(kk)

    while smaller and larger:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        (smaller if q[large] < 1.0 else larger).append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    return J[kk]

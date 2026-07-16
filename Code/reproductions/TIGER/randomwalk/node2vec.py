"""Node2Vec / DeepWalk wrapper. Mirrors upstream ``randomWalk/node2vec.py``.

Removed upstream's dead ``gensim.models.Word2Vec`` import (unused). Set
``dw=True`` to get DeepWalk (used by TIGER paper); ``dw=False`` selects
Node2Vec (alias-table sampling, NOT used by paper experiments).
"""
from __future__ import annotations

from typing import List

from . import walker


class Node2vec:
    def __init__(
        self,
        start_nodes: List,
        graph,
        path_length: int,
        num_paths: int,
        p: float = 1.0,
        q: float = 1.0,
        dw: bool = False,
        **kwargs,
    ) -> None:
        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:
            self.walker = walker.BasicWalker(graph, start_nodes, workers=kwargs["workers"])
        else:
            self.walker = walker.Walker(graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()
        self.walks = self.walker.simulate_walks(num_walks=num_paths, walk_length=path_length)

    def get_walks(self) -> List:
        return self.walks

"""DeepWalk / Node2Vec subgraph sampler.

Mirrors upstream ``randomWalk/`` directory (renamed to lowercase to satisfy
PEP 8). Only :class:`Node2vec` with ``dw=True`` (DeepWalk) is used by the
data_process.py subgraph extractors in paper experiments.
"""
from __future__ import annotations

from .node2vec import Node2vec

__all__ = ["Node2vec"]

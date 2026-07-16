"""TIGER model subpackage (paper-faithful reproduction).

Mirrors upstream ``Blair1213/TIGER/model/`` directory.
"""
from __future__ import annotations

from .graph_transformer import (
    GraphTransformer,
    GraphTransformerEncode,
    MultiheadAttention,
    SpatialEncoding,
)
from .tiger import TIGER, Discriminator, NodeFeatures, init_params

__all__ = [
    "GraphTransformer",
    "GraphTransformerEncode",
    "MultiheadAttention",
    "SpatialEncoding",
    "TIGER",
    "Discriminator",
    "NodeFeatures",
    "init_params",
]

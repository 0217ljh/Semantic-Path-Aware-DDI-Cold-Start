"""Two thin output stages over the shared ``StructuralVariableCore``.

``StandaloneHead`` scores ``z_adapter`` directly — used to evaluate the adapter
in isolation (asymmetric vs non-asymmetric hyper-edge). ``FusionHead`` concatenates
``z_adapter`` with a backbone pair representation ``z_backbone`` (e.g. NBFNet's
``cat[h_sym, q]``) and scores the union — the pluggable-adapter setting.

Both reuse the same ``core``, so they differ only in the final MLP. The
StandaloneHead scorer has the same shape as ``SPMNRelHead.head_mlp`` when the
core runs in its reproduce-old mode (use_asym=False, no absdiff embed).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .batch import SupportBatch
from .core import StructuralVariableCore


class StandaloneHead(nn.Module):
    def __init__(self, core: StructuralVariableCore, hidden: int = 128,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.core = core
        self.scorer = nn.Sequential(
            nn.Linear(core.out_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: SupportBatch) -> torch.Tensor:
        return self.scorer(self.core(batch)).squeeze(-1)


class FusionHead(nn.Module):
    def __init__(self, core: StructuralVariableCore, z_backbone_dim: int,
                 hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.core = core
        self.z_backbone_dim = int(z_backbone_dim)
        self.fusion = nn.Sequential(
            nn.Linear(core.out_dim + int(z_backbone_dim), hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, batch: SupportBatch, z_backbone: torch.Tensor) -> torch.Tensor:
        z = self.core(batch)
        return self.fusion(torch.cat([z, z_backbone], dim=-1)).squeeze(-1)


__all__ = ["StandaloneHead", "FusionHead"]

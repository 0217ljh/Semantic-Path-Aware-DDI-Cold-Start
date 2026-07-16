"""Decomposed-scorer heads for the aware adapter (Step 1+).

``DecomposedStandaloneHead`` = ``StandaloneHead``'s mediator scorer over
``z_adapter`` PLUS a zero-init linear promiscuity/regime branch ``b(v_ab)`` on the
per-pair density features (``spmn_v2.density``):

    s(a, b) = scorer(core(batch)) + b(v_ab)

``b`` is zero-initialised, so the head STARTS identical to the naked-AND
``StandaloneHead`` and only learns a marginal one-arm correction — the decomposed
scorer that absorbs the known drug-promiscuity signal without letting the mediator
pool relearn it (codex Step 1). ``use_branch=False`` reproduces ``StandaloneHead``
exactly (the Step 0 reference, runnable through this same head).

These are new heads; the original ``StandaloneHead`` / ``FusionHead`` are unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .batch import SupportBatch
from .core import StructuralVariableCore


class DecomposedStandaloneHead(nn.Module):
    def __init__(
        self, core: StructuralVariableCore, n_pair_feats: int = 6,
        hidden: int = 128, dropout: float = 0.2, *, use_branch: bool = True,
    ) -> None:
        super().__init__()
        self.core = core
        self.use_branch = bool(use_branch)
        self.scorer = nn.Sequential(
            nn.Linear(core.out_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        if self.use_branch:
            self.branch = nn.Linear(int(n_pair_feats), 1)
            nn.init.zeros_(self.branch.weight)
            nn.init.zeros_(self.branch.bias)

    def forward(self, batch: SupportBatch,
                pair_feats: torch.Tensor | None = None) -> torch.Tensor:
        s = self.scorer(self.core(batch)).squeeze(-1)
        if self.use_branch:
            if pair_feats is None:
                raise ValueError(
                    "DecomposedStandaloneHead(use_branch=True) requires pair_feats")
            s = s + self.branch(pair_feats).squeeze(-1)
        return s


class DecomposedFusionHead(nn.Module):
    """``FusionHead`` + the same zero-init promiscuity branch — the grafted-onto-
    baseline mode (concat ``z_adapter`` with a backbone pair vector, plus b(v_ab))."""

    def __init__(
        self, core: StructuralVariableCore, z_backbone_dim: int,
        n_pair_feats: int = 6, hidden: int = 128, dropout: float = 0.2,
        *, use_branch: bool = True,
    ) -> None:
        super().__init__()
        self.core = core
        self.z_backbone_dim = int(z_backbone_dim)
        self.use_branch = bool(use_branch)
        self.fusion = nn.Sequential(
            nn.Linear(core.out_dim + int(z_backbone_dim), hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        if self.use_branch:
            self.branch = nn.Linear(int(n_pair_feats), 1)
            nn.init.zeros_(self.branch.weight)
            nn.init.zeros_(self.branch.bias)

    def forward(self, batch: SupportBatch, z_backbone: torch.Tensor,
                pair_feats: torch.Tensor | None = None) -> torch.Tensor:
        z = self.core(batch)
        s = self.fusion(torch.cat([z, z_backbone], dim=-1)).squeeze(-1)
        if self.use_branch:
            if pair_feats is None:
                raise ValueError(
                    "DecomposedFusionHead(use_branch=True) requires pair_feats")
            s = s + self.branch(pair_feats).squeeze(-1)
        return s


__all__ = ["DecomposedStandaloneHead", "DecomposedFusionHead"]

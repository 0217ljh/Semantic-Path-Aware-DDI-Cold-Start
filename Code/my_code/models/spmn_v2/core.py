"""StructuralVariableCore — the shared adapter body that produces ``z_adapter``.

Migrated from ``spmn_v1.rel_head.SPMNRelHead`` (relation-enriched per-(pair,type)
attention pool + W_T type routing + explicit struct features), refactored to
emit the pair-level vector *before* any scoring MLP so two heads can share it.

Two switches add the new structural ingredients:

  * ``use_asym``: split each type-tau pool into a symmetric (d_a == d_b) and an
    asymmetric (d_a != d_b) channel. Pool shape becomes ``(B, K, 2, d)``; the
    W_T routing stays ``K x K`` and is applied to both channels, which keeps the
    two channels distinct all the way to ``z_adapter``.
  * ``use_absdiff_embed``: append a small embedding keyed by ``|d_a - d_b|`` to
    each mediator message.

Invariant: with ``use_asym=False`` and ``use_absdiff_embed=False`` the produced
``z_adapter`` equals ``SPMNRelHead``'s pre-head vector
``cat([c_tilde.reshape(B, K*d), struct_feats])`` exactly (same submodule names,
same ops, R=1 collapses the channel axis). Verified in
``Code/scripts/verify_spmn_v2_core_equiv.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from .batch import SupportBatch


class StructuralVariableCore(nn.Module):
    def __init__(
        self,
        n_entities: int,
        n_types: int,
        n_rel_buckets: int,
        struct_dim: int,
        d: int = 32,
        type_dim: int = 8,
        rel_dim: int = 8,
        hidden: int = 128,
        dropout: float = 0.2,
        *,
        use_entity_embed: bool = True,
        use_asym: bool = False,
        use_absdiff_embed: bool = False,
        n_absdiff_buckets: int = 4,
        absdiff_dim: int = 4,
        use_dist_attn: bool = False,
        max_dist: int = 4,
    ) -> None:
        super().__init__()
        self.n_types = int(n_types)
        self.d = int(d)
        self.use_entity_embed = bool(use_entity_embed)
        self.use_asym = bool(use_asym)
        self.use_absdiff_embed = bool(use_absdiff_embed)
        self.n_absdiff_buckets = int(n_absdiff_buckets)
        #: number of within-type channels: 2 (sym / asym) when stratifying.
        self.n_chan = 2 if self.use_asym else 1

        # Mediator-message embeddings. entity_embed is a free per-mediator-entity
        # lookup (the main memorisation capacity). Dropping it (use_entity_embed
        # =False) makes the message PURELY STRUCTURAL — type + relation (+dist) —
        # i.e. fully inductive, no per-entity identity to overfit (NBFNet-style).
        if self.use_entity_embed:
            self.entity_embed = nn.Embedding(int(n_entities), self.d)
            nn.init.xavier_uniform_(self.entity_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, int(type_dim))
        nn.init.xavier_uniform_(self.type_embed.weight)
        self.rel_embed = nn.Embedding(int(n_rel_buckets), int(rel_dim))
        nn.init.xavier_uniform_(self.rel_embed.weight)

        phi_in = (self.d if self.use_entity_embed else 0) + int(type_dim) + 2 * int(rel_dim)
        if self.use_absdiff_embed:
            self.absdiff_embed = nn.Embedding(self.n_absdiff_buckets, int(absdiff_dim))
            nn.init.xavier_uniform_(self.absdiff_embed.weight)
            phi_in += int(absdiff_dim)
        self.phi_mlp = nn.Sequential(
            nn.Linear(phi_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.d),
        )
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)
        self.W_T = nn.Parameter(torch.empty(self.n_types, self.n_types))
        nn.init.xavier_uniform_(self.W_T)

        # Distance-conditioned attention: a learned additive bias over the
        # (d_a, d_b) grid added to the within-tau attention logits — a soft,
        # learned receptive field over the distance plane. Zero-initialised, so
        # even with use_dist_attn=True the model STARTS identical to the base
        # head and only learns to up/down-weight distance cells from there.
        self.use_dist_attn = bool(use_dist_attn)
        self.max_dist = int(max_dist)
        if self.use_dist_attn:
            self.dist_bias = nn.Parameter(
                torch.zeros(self.max_dist + 1, self.max_dist + 1))

        #: dimension of the produced z_adapter (read by the heads).
        self.out_dim = self.n_types * self.n_chan * self.d + int(struct_dim)

    def forward(self, batch: SupportBatch) -> torch.Tensor:
        B, K, d, R = int(batch.n_pairs), self.n_types, self.d, self.n_chan
        device = batch.device
        if batch.med_id.numel() > 0:
            feats = []
            if self.use_entity_embed:
                feats.append(self.entity_embed(batch.med_id))
            feats += [
                self.type_embed(batch.type_idx),
                self.rel_embed(batch.rel_a),
                self.rel_embed(batch.rel_b),
            ]
            if self.use_absdiff_embed:
                ad = (batch.d_a - batch.d_b).abs().clamp_(max=self.n_absdiff_buckets - 1)
                feats.append(self.absdiff_embed(ad))
            m = self.phi_mlp(torch.cat(feats, dim=-1))

            if R == 2:
                asym_bit = (batch.d_a != batch.d_b).long()
            else:
                asym_bit = torch.zeros_like(batch.type_idx)
            group = batch.pair_idx * (K * R) + batch.type_idx * R + asym_bit

            scores = self.within_attn_w(torch.tanh(self.within_attn_W(m))).squeeze(-1)
            if self.use_dist_attn:
                da_i = batch.d_a.clamp(0, self.max_dist)
                db_i = batch.d_b.clamp(0, self.max_dist)
                scores = scores + self.dist_bias[da_i, db_i]
            alpha = pyg_softmax(scores, group, num_nodes=B * K * R)
            c = torch.zeros(B * K * R, d, device=device)
            c.scatter_add_(0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * m)
        else:
            c = torch.zeros(B * K * R, d, device=device)

        c = c.view(B, K, R, d)
        transition = F.softmax(self.W_T, dim=-1)
        c_tilde = torch.einsum("kj,bjrd->bkrd", transition, c)
        return torch.cat([c_tilde.reshape(B, K * R * d), batch.struct_feats], dim=-1)


__all__ = ["StructuralVariableCore"]

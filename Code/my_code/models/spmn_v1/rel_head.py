"""SPMN v1 — relation-enriched coarse head (the clean Phase-2 design).

Per the empirical component verdict (Notes/Log/algorithm_design_debate.md):
the v1.6 pieces that ADD signal are the AND-intersection support, attention
pooling, and RELATION features (+2.2pt KG-only). Affinity routing and a/b
separation were shown REDUNDANT on the correct support, so they are dropped.

This head = the simple per-(pair,type) attention pool, but the mediator
representation is relation-enriched:

  phi(m) = MLP([ entity_embed[m] || type_embed[phi(m)] ||
                 rel_embed[rel(a->m)] || rel_embed[rel(m->b)] ])

then K x K routing, then fuse with the explicit structural features. No drug-id
embeddings; non-drug entity embeddings only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


class SPMNRelHead(nn.Module):
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
    ) -> None:
        super().__init__()
        self.n_types = int(n_types)
        self.d = int(d)

        self.entity_embed = nn.Embedding(int(n_entities), self.d)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, int(type_dim))
        nn.init.xavier_uniform_(self.type_embed.weight)
        self.rel_embed = nn.Embedding(int(n_rel_buckets), int(rel_dim))
        nn.init.xavier_uniform_(self.rel_embed.weight)

        phi_in = self.d + int(type_dim) + 2 * int(rel_dim)
        self.phi_mlp = nn.Sequential(
            nn.Linear(phi_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, self.d),
        )
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)
        self.W_T = nn.Parameter(torch.empty(self.n_types, self.n_types))
        nn.init.xavier_uniform_(self.W_T)
        self.head_mlp = nn.Sequential(
            nn.Linear(self.n_types * self.d + int(struct_dim), hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, med_id, pair_idx, type_idx, rel_a, rel_b,
                struct_feats, n_pairs):
        B, K, d = int(n_pairs), self.n_types, self.d
        device = struct_feats.device
        if med_id.numel() > 0:
            phi = self.phi_mlp(torch.cat([
                self.entity_embed(med_id),
                self.type_embed(type_idx),
                self.rel_embed(rel_a),
                self.rel_embed(rel_b),
            ], dim=-1))
            group = pair_idx * K + type_idx
            scores = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
            alpha = pyg_softmax(scores, group, num_nodes=B * K)
            c = torch.zeros(B * K, d, device=device)
            c.scatter_add_(0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * phi)
        else:
            c = torch.zeros(B * K, d, device=device)
        c = c.view(B, K, d)
        transition = F.softmax(self.W_T, dim=-1)
        c_tilde = torch.einsum("kj,bjd->bkd", transition, c)
        z = torch.cat([c_tilde.reshape(B, K * d), struct_feats], dim=-1)
        return self.head_mlp(z).squeeze(-1)


__all__ = ["SPMNRelHead"]

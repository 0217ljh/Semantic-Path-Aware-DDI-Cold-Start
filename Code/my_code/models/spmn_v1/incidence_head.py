"""SPMN v1 — unified hyper-edge incidence head (subsumes v1.6).

Expresses v1.6's machinery as facets of one attributed higher-order incidence
between the drug pair and its typed mediators (see
``Notes/Log/algorithm_design_debate.md``, "v1.6 into hyper-edge"):

  * within-(pair,type) attention pool over hyper-edge members  -> c[B,K,d]
    (v1.6 Step 2 / the joint incidence pooling).
  * MARGINAL per-drug affinity gates the joint channels (v1.6 Step 1): the
    drug's own per-type incidence ``affinity[d,k]`` drives a cluster-pair
    attention, giving directional channel weights attn_a / attn_b.
  * a/b-SEPARATED directional readout z_a / z_b (v1.6 Step 3).
  * explicit structural variables (s_tau, co-path, AA) concatenated — the
    hyper-edge generalisation beyond v1.6's free K x K transition.

Relation-type member attributes (rel(a->m), rel(m->b)) are a further facet to
be added next; this version covers affinity routing + a/b separation (the
v1.6 routing form not yet tested). No drug-id embeddings; mediator (non-drug)
entity embeddings only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


class SPMNIncidenceHead(nn.Module):
    def __init__(
        self,
        n_entities: int,
        n_types: int,
        struct_dim: int,
        d: int = 32,
        type_dim: int = 8,
        hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_entities = int(n_entities)
        self.n_types = int(n_types)
        self.struct_dim = int(struct_dim)
        self.d = int(d)
        self.type_dim = int(type_dim)
        self.hidden = int(hidden)

        self.entity_embed = nn.Embedding(self.n_entities, self.d)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, self.type_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        self.phi_mlp = nn.Sequential(
            nn.Linear(self.d + self.type_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.d),
        )
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)

        # Cluster-pair transition for the marginal-affinity routing (v1.6 W_T).
        self.W_T = nn.Parameter(torch.empty(self.n_types, self.n_types))
        nn.init.xavier_uniform_(self.W_T)

        # Head over [z_a (d) || z_b (d) || struct_feats].
        self.head_mlp = nn.Sequential(
            nn.Linear(2 * self.d + self.struct_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    def _pool(self, med_id, pair_idx, type_idx, B):
        """Within-(pair,type) attention pool -> c (B, K, d)."""
        K, d = self.n_types, self.d
        device = self.W_T.device
        if med_id.numel() == 0:
            return torch.zeros(B, K, d, device=device)
        phi = self.phi_mlp(
            torch.cat([self.entity_embed(med_id), self.type_embed(type_idx)], -1)
        )
        group = pair_idx * K + type_idx
        scores = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
        alpha = pyg_softmax(scores, group, num_nodes=B * K)
        c = torch.zeros(B * K, d, device=device)
        c.scatter_add_(0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * phi)
        return c.view(B, K, d)

    def _route(self, affinity_a, affinity_b):
        """v1.6 Step 1: cluster-pair attention marginals -> attn_a, attn_b (B,K)."""
        B, K = affinity_a.shape
        log_p_a = F.log_softmax(affinity_a, dim=-1)        # (B,K)
        log_p_b = F.log_softmax(affinity_b, dim=-1)
        log_T = F.log_softmax(self.W_T, dim=-1)            # (K,K)
        base = log_p_a.unsqueeze(2) + log_T.unsqueeze(0) + log_p_b.unsqueeze(1)
        alpha = F.softmax(base.view(B, K * K), dim=-1).view(B, K, K)
        attn_a = alpha.sum(dim=2)                          # (B,K) row marginal
        attn_b = alpha.sum(dim=1)                          # (B,K) col marginal
        return attn_a, attn_b

    def forward(self, med_id, pair_idx, type_idx, affinity_a, affinity_b,
                struct_feats, n_pairs):
        B = int(n_pairs)
        c = self._pool(med_id, pair_idx, type_idx, B)      # (B,K,d)
        attn_a, attn_b = self._route(affinity_a, affinity_b)
        z_a = torch.einsum("bk,bkd->bd", attn_a, c)        # (B,d)
        z_b = torch.einsum("bk,bkd->bd", attn_b, c)
        z = torch.cat([z_a, z_b, struct_feats], dim=-1)
        return self.head_mlp(z).squeeze(-1)


__all__ = ["SPMNIncidenceHead"]

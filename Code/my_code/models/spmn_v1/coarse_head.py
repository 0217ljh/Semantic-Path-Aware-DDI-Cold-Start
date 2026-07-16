"""SPMN v1 — Phase 2 neural coarse head (KG-only, binary).

Adds the first learnable parameters on top of the Phase-1 explicit structural
features: a learned attention pool over each type-tau hyper-edge's members
(the typed common-reachable mediators ``A_tau^(l)(a,b)``), routing across the K
type channels, then fusion with the exact structural-variable features
``s_tau`` / ``s_{tau,tau'}``.

Design choices (frozen-design + this session's decisions):
  * NO EmerGNN propagation backbone.
  * NO absolute positional encoding (lead, 2026-06-22): mediator representation
    is ``MLP([entity_embed, type_embed])`` only; distances are used solely as
    the reachability gate during support construction (in :mod:`retrieval`),
    never as a per-node feature here.
  * v1.6 is an (approximate) representable subset: the pooling + routing + MLP
    path mirrors v1.6's within-cluster pool + W_T routing; if the explicit
    features add nothing the head can ignore them, and vice-versa.
  * Binary / symmetric for now: pooling is over the (symmetric) support
    membership; the explicit co-path feature is symmetrised upstream. Role-aware
    (directional) variant is deferred to the multi-label flagship.

Per-entity (non-drug) embeddings are inductive-safe: drugs carry no id
embedding (R7) — only mediator KG entities, which are shared/seen.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax


class SPMNCoarseHead(nn.Module):
    """Per-type attention pool over hyper-edge members + routing + struct fusion.

    Forward consumes a *flattened* batch of mediators (variable count per pair)
    plus the per-pair explicit structural-feature vector.
    """

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

        # Mediator (KG entity) + type embeddings. Indexed by GLOBAL KG node id;
        # only non-drug ids are ever gathered (drugs never enter the support).
        self.entity_embed = nn.Embedding(self.n_entities, self.d)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, self.type_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)

        # Mediator representation phi(m) = MLP([entity_embed, type_embed]).
        self.phi_mlp = nn.Sequential(
            nn.Linear(self.d + self.type_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.d),
        )
        # Within-(pair,type) attention pool (v1.6-style).
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)

        # Routing across the K type channels: c_tilde = softmax(W_T) @ c.
        self.W_T = nn.Parameter(torch.empty(self.n_types, self.n_types))
        nn.init.xavier_uniform_(self.W_T)

        # Final head over [routed channels (K*d) || explicit struct feats].
        self.head_mlp = nn.Sequential(
            nn.Linear(self.n_types * self.d + self.struct_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    def forward(
        self,
        med_id: torch.Tensor,      # (E,) long — global entity id per mediator slot
        pair_idx: torch.Tensor,    # (E,) long — batch-local pair index in [0, B)
        type_idx: torch.Tensor,    # (E,) long — canonical type id in [0, K)
        struct_feats: torch.Tensor,  # (B, struct_dim) float
        n_pairs: int,
    ) -> torch.Tensor:
        """Return (B,) binary logits."""
        B = int(n_pairs)
        K = self.n_types
        d = self.d
        device = struct_feats.device

        if med_id.numel() > 0:
            # phi(m) for every mediator slot.
            phi = self.phi_mlp(
                torch.cat([self.entity_embed(med_id), self.type_embed(type_idx)],
                          dim=-1)
            )  # (E, d)
            # Group = pair * K + type, so each (pair, type) hyper-edge is one
            # softmax group. Pool with attention.
            group = pair_idx * K + type_idx                    # (E,)
            scores = self.within_attn_w(
                torch.tanh(self.within_attn_W(phi))
            ).squeeze(-1)                                       # (E,)
            alpha = pyg_softmax(scores, group, num_nodes=B * K)  # (E,)
            c = torch.zeros(B * K, d, device=device)
            c.scatter_add_(
                0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * phi,
            )
        else:
            c = torch.zeros(B * K, d, device=device)

        c = c.view(B, K, d)                                    # (B, K, d)
        # Routing across type channels.
        transition = F.softmax(self.W_T, dim=-1)               # (K, K)
        c_tilde = torch.einsum("kj,bjd->bkd", transition, c)   # (B, K, d)

        z = torch.cat([c_tilde.reshape(B, K * d), struct_feats], dim=-1)
        return self.head_mlp(z).squeeze(-1)                    # (B,)


__all__ = ["SPMNCoarseHead"]

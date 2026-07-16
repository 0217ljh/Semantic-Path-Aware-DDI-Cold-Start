"""PMP v1.4 — Layer-2-only (per-cluster within pool) scoring module.

Created 2026-06-03.

This is the within-pool-only ablation of v1.2:
  - Within-cluster pool kept (intersection mediators, type / rel context in phi)
  - Cluster path entirely DROPPED (no W_T, no cluster_attn, no H_a/H_b)
  - mlp_score input is z_within (d dims) ONLY

EmerGNN backbone still exists in trainer but does not enter score (same bypass
as v1.1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMPv1_4Module(nn.Module):
    """v1.4 scoring module: within-pool only -> MLP_score.

    All learnable parameters that participate in the score live in this
    module: mediator_embed, type_embed, rel_embed, phi_mlp, within_attn_W/w,
    mlp_score.
    """

    def __init__(
        self,
        n_mediators: int,
        n_types: int,
        n_rels_plus_special: int,
        n_clusters: int,
        d: int = 32,
        type_dim: int | None = None,
        rel_dim: int | None = None,
        hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_mediators = int(n_mediators)
        self.n_types = int(n_types)
        self.n_rels_plus_special = int(n_rels_plus_special)
        self.n_clusters = int(n_clusters)
        self.d = int(d)
        self.type_dim = int(type_dim) if type_dim is not None else max(self.d // 4, 8)
        self.rel_dim = int(rel_dim) if rel_dim is not None else max(self.d // 4, 8)
        self.hidden = int(hidden)
        self.dropout_p = float(dropout)

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        self.mediator_embed = nn.Embedding(self.n_mediators, self.d)
        nn.init.xavier_uniform_(self.mediator_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, self.type_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)
        self.rel_embed = nn.Embedding(self.n_rels_plus_special, self.rel_dim)
        nn.init.xavier_uniform_(self.rel_embed.weight)

        # ------------------------------------------------------------------
        # Within-cluster pool components (same as v1.1 / v1.2)
        # ------------------------------------------------------------------
        phi_in_dim = self.d + self.type_dim + 2 * self.rel_dim
        self.phi_mlp = nn.Sequential(
            nn.Linear(phi_in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.d),
        )
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)

        # Final score MLP — input is z_within only (d dims).
        self.mlp_score = nn.Sequential(
            nn.Linear(self.d, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    def within_cluster_pool_forward(
        self,
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
        p_a: torch.Tensor,
        p_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute z_within = sum_k (p_a[k] * p_b[k]) * z_within^k.

        Identical to v1.1 / v1.2's within_cluster_pool_forward. p_a / p_b
        come from softmax(affinity_*) computed by `forward()` below.
        """
        device = p_a.device
        B = p_a.size(0)
        weights = p_a * p_b

        z_within = torch.zeros(B, self.d, device=device)

        K = self.n_clusters
        if len(med_idx_per_cluster) != K:
            raise ValueError(
                f"Expected len(med_idx_per_cluster) == n_clusters={K}, "
                f"got {len(med_idx_per_cluster)}"
            )

        for k in range(K):
            mask_k = mask_per_cluster[k]
            any_valid = mask_k.any(dim=1)
            if not any_valid.any():
                continue

            med_idx_k = med_idx_per_cluster[k]
            rel_a_k = rel_a_per_cluster[k]
            rel_b_k = rel_b_per_cluster[k]
            type_id_k = int(type_id_per_cluster[k])

            h_m = self.mediator_embed(med_idx_k)
            type_tensor = torch.full(
                (B, med_idx_k.size(1)), type_id_k,
                dtype=torch.long, device=device,
            )
            e_type = self.type_embed(type_tensor)
            e_ram = self.rel_embed(rel_a_k)
            e_rmb = self.rel_embed(rel_b_k)
            phi_in = torch.cat([h_m, e_type, e_ram, e_rmb], dim=-1)
            phi = self.phi_mlp(phi_in)

            s = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
            neg_inf = torch.finfo(s.dtype).min
            s = s.masked_fill(~mask_k, neg_inf)
            s_safe = torch.where(any_valid.unsqueeze(1), s, torch.zeros_like(s))
            alpha_within = F.softmax(s_safe, dim=1)
            alpha_within = alpha_within * mask_k.float()
            z_k = (alpha_within.unsqueeze(-1) * phi).sum(dim=1)

            weight_k = weights[:, k].unsqueeze(-1)
            z_within = z_within + weight_k * z_k

        return z_within

    def forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v1.4 score.

        Returns:
          score:     (B,) scalar logit per pair.
          z_within:  (B, d) within-pool representation (diagnostic).
        """
        # p_a, p_b used only as joint cluster relevance weights inside the
        # within-pool. No cluster path forward, no T, no W_T.
        p_a = F.softmax(affinity_a, dim=-1)
        p_b = F.softmax(affinity_b, dim=-1)

        z_within = self.within_cluster_pool_forward(
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_per_cluster, type_id_per_cluster, p_a, p_b,
        )
        score = self.mlp_score(z_within).squeeze(-1)
        return score, z_within


__all__ = ["PMPv1_4Module"]

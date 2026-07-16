"""PMP v1.3 — Layer-1-only (pair-conditional cluster path) scoring module.

Created 2026-06-03.

This is the cluster-path-only ablation of v1.2:
  - Pair-conditional H_a / H_b cluster repr (same as v1.2)
  - Cluster path through W_T softmax transition (same as v1.2)
  - Relation context r_a / r_b in psi (same as v1.2)
  - Within-cluster pool entirely DROPPED
  - mlp_score input is z_cluster_pair (hidden dims) ONLY

EmerGNN backbone still exists in trainer but does not enter score (same bypass
as v1.1 / v1.2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMPv1_3Module(nn.Module):
    """v1.3 scoring module: cluster path only -> MLP_score.

    All learnable parameters that participate in the score live in this
    module: mediator_embed, type_embed, rel_embed, W_T, cluster_attn_W/w,
    mlp_pair_cluster, mlp_score.
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

        # Learnable cluster transition T = softmax(W_T)
        self.W_T = nn.Parameter(torch.empty(self.n_clusters, self.n_clusters))
        nn.init.xavier_uniform_(self.W_T)

        # Pair-conditional cluster representation attention
        self.cluster_attn_W = nn.Linear(self.d, self.d)
        self.cluster_attn_w = nn.Linear(self.d, 1, bias=False)

        # Cluster path readout
        h_path_dim = 4 * self.d + 2 * self.rel_dim
        self.mlp_pair_cluster = nn.Sequential(
            nn.Linear(h_path_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.hidden),
        )

        # Final score MLP — input is z_cluster_pair only (no concat).
        self.mlp_score = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    def cluster_transition(self) -> torch.Tensor:
        return F.softmax(self.W_T, dim=-1)

    def _pair_conditional_cluster_repr(
        self,
        med_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute pair-conditional cluster repr H, (B, K, d).

        For each cluster k, AttnPool mediator_embed over the drug's
        cluster-k mediators. Empty cluster -> zero row.
        """
        K = self.n_clusters
        if len(med_per_cluster) != K or len(mask_per_cluster) != K:
            raise ValueError(
                f"Expected len(med_per_cluster) == len(mask_per_cluster) == "
                f"n_clusters={K}, got {len(med_per_cluster)} / "
                f"{len(mask_per_cluster)}"
            )
        B = med_per_cluster[0].size(0)
        device = med_per_cluster[0].device
        H = torch.zeros(B, K, self.d, device=device)

        for k in range(K):
            mask_k = mask_per_cluster[k]
            any_valid = mask_k.any(dim=1)
            if not any_valid.any():
                continue
            med_k = med_per_cluster[k]
            h_m = self.mediator_embed(med_k)

            s = self.cluster_attn_w(torch.tanh(self.cluster_attn_W(h_m))).squeeze(-1)
            neg_inf = torch.finfo(s.dtype).min
            s = s.masked_fill(~mask_k, neg_inf)
            s_safe = torch.where(any_valid.unsqueeze(1), s, torch.zeros_like(s))
            alpha = F.softmax(s_safe, dim=1)
            alpha = alpha * mask_k.float()
            h_k = (alpha.unsqueeze(-1) * h_m).sum(dim=1)
            h_k = torch.where(any_valid.unsqueeze(1), h_k, torch.zeros_like(h_k))
            H[:, k, :] = h_k

        return H

    def cluster_path_forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        rel_dist_a: torch.Tensor,
        rel_dist_b: torch.Tensor,
        med_a_per_cluster: list[torch.Tensor],
        mask_a_per_cluster: list[torch.Tensor],
        med_b_per_cluster: list[torch.Tensor],
        mask_b_per_cluster: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run pair-conditional cluster path. Returns z_cluster_pair (B, hidden)."""
        B = affinity_a.size(0)
        K = self.n_clusters
        d = self.d

        p_a = F.softmax(affinity_a, dim=-1)
        p_b = F.softmax(affinity_b, dim=-1)

        H_a = self._pair_conditional_cluster_repr(med_a_per_cluster, mask_a_per_cluster)
        H_b = self._pair_conditional_cluster_repr(med_b_per_cluster, mask_b_per_cluster)

        T = self.cluster_transition()
        S = torch.einsum("bi,ij,bj->bij", p_a, T, p_b)
        S_sum = S.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        alpha = S / S_sum

        e_i = H_a.unsqueeze(2).expand(B, K, K, d)
        e_j = H_b.unsqueeze(1).expand(B, K, K, d)

        r_a = rel_dist_a @ self.rel_embed.weight
        r_b = rel_dist_b @ self.rel_embed.weight
        r_a_to_i = r_a.unsqueeze(2).expand(B, K, K, self.rel_dim)
        r_j_to_b = r_b.unsqueeze(1).expand(B, K, K, self.rel_dim)

        psi = torch.cat(
            [e_i, e_j, (e_i - e_j).abs(), e_i * e_j, r_a_to_i, r_j_to_b],
            dim=-1,
        )

        h_path = (alpha.unsqueeze(-1) * psi).sum(dim=(1, 2))
        z_cluster_pair = self.mlp_pair_cluster(h_path)
        return z_cluster_pair

    def forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        rel_dist_a: torch.Tensor,
        rel_dist_b: torch.Tensor,
        med_a_per_cluster: list[torch.Tensor],
        mask_a_per_cluster: list[torch.Tensor],
        med_b_per_cluster: list[torch.Tensor],
        mask_b_per_cluster: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute v1.3 score.

        Returns:
          score:           (B,) scalar logit per pair.
          z_cluster_pair:  (B, hidden) cluster-path representation (diagnostic).
        """
        z_cluster_pair = self.cluster_path_forward(
            affinity_a, affinity_b, rel_dist_a, rel_dist_b,
            med_a_per_cluster, mask_a_per_cluster,
            med_b_per_cluster, mask_b_per_cluster,
        )
        score = self.mlp_score(z_cluster_pair).squeeze(-1)
        return score, z_cluster_pair


__all__ = ["PMPv1_3Module"]

"""PMP v1.1 — full cluster-path + per-cluster within-pool module (rewrite 2026-06-02).

Implements the locked v1.1 architecture per user spec:

  1. cluster_embed = nn.Embedding(12, d), init = mean({mediator_embed[m] for m in
     members(k)}) per cluster; trainable.
  2. T = softmax(W_T) with W_T an (n_clusters, n_clusters) learnable matrix
     (Xavier init).
  3. Per-(drug, cluster) relation context r_d_to_k computed at forward time as
     a weighted sum of rel_embed under the precomputed drug_cluster_rel_dist.
  4. Cluster path (explicit pairwise):
     p_a = softmax(c_a), p_b = softmax(c_b)
     S_ij = p_a[i] * T[i, j] * p_b[j]
     alpha_ij = S_ij / sum_{u,v} S_uv
     psi(i, j, a, b) = concat([
         cluster_embed[i], cluster_embed[j],
         |cluster_embed[i] - cluster_embed[j]|,
         cluster_embed[i] * cluster_embed[j],
         r_a_to_i, r_j_to_b,
     ])
     h_path = sum_{i, j} alpha_ij * psi(i, j, a, b)
     z_cluster_pair = MLP_pair_cluster(h_path)
  5. Within-cluster pool (per-cluster, weighted by joint cluster relevance):
     for k in 0..n_clusters - 1:
       M_k(a, b) = members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)
       z_within^k = AttnPool({phi(m) for m in M_k}) or 0 if empty
     z_within = sum_k (p_a[k] * p_b[k]) * z_within^k
  6. Final score (replaces EmerGNN backbone entirely):
     z_pair = concat([z_cluster_pair, z_within])
     score = MLP_score(z_pair)   # scalar
     loss = BCE(sigmoid(score), y)

EmerGNN backbone is built by the trainer for shuffle_train(mode='S2') protocol
inheritance but its forward output is NOT used in score; v1 PMP residual fusion
is also bypassed. The two unused param groups waste optimizer slots but are
left in place to minimize surgery on the parent training scaffold.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PMPv1_1Module(nn.Module):
    """Full v1.1 scoring module. Owns ALL learnable parameters that participate
    in the score function (mediator_embed, type_embed, rel_embed, cluster_embed,
    W_T, phi, attention, MLP_pair_cluster, MLP_score).

    Forward signature accepts precomputed per-(drug, cluster) affinity and
    relation distribution, plus per-cluster batched mediator tensors gathered
    by the trainer.
    """

    def __init__(
        self,
        n_mediators: int,
        n_types: int,
        n_rels_plus_special: int,
        n_clusters: int,
        d: int = 64,
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

        # ------------------------------------------------------------------
        # Embeddings
        # ------------------------------------------------------------------
        # Mediator embedding: per-mediator d-dim learnable vector.
        self.mediator_embed = nn.Embedding(self.n_mediators, self.d)
        nn.init.xavier_uniform_(self.mediator_embed.weight)
        # Type embedding (12 types incl "other").
        self.type_embed = nn.Embedding(self.n_types, self.type_dim)
        # Relation embedding (n_rels + 1 for rel_2hop_id).
        self.rel_embed = nn.Embedding(self.n_rels_plus_special, self.rel_dim)
        # Cluster embedding (12 clusters). Init from member mediator mean via
        # init_cluster_embed_from_members() after the trainer sets up the
        # cluster member mapping. Initially zero — explicit init required.
        self.cluster_embed = nn.Embedding(self.n_clusters, self.d)
        nn.init.zeros_(self.cluster_embed.weight)

        # ------------------------------------------------------------------
        # Learnable cluster transition T = softmax(W_T)
        # ------------------------------------------------------------------
        self.W_T = nn.Parameter(torch.empty(self.n_clusters, self.n_clusters))
        nn.init.xavier_uniform_(self.W_T)

        # ------------------------------------------------------------------
        # phi for within-cluster pool: per-mediator d-dim embedding
        # phi(m) = MLP([mediator_embed[m], type_embed[k], rel_embed[a->m], rel_embed[m->b]])
        # ------------------------------------------------------------------
        phi_in_dim = self.d + self.type_dim + 2 * self.rel_dim
        self.phi_mlp = nn.Sequential(
            nn.Linear(phi_in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.d),
        )
        # Attention for within-cluster pool.
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)

        # ------------------------------------------------------------------
        # Cluster path readout
        # h_path = sum_{ij} alpha_ij * psi(i, j, a, b)
        # psi has shape 4*d + 2*rel_dim
        # ------------------------------------------------------------------
        h_path_dim = 4 * self.d + 2 * self.rel_dim
        self.mlp_pair_cluster = nn.Sequential(
            nn.Linear(h_path_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.hidden),
        )

        # ------------------------------------------------------------------
        # Final score MLP. Input: concat[z_cluster_pair (hidden), z_within (d)]
        # ------------------------------------------------------------------
        score_in_dim = self.hidden + self.d
        self.mlp_score = nn.Sequential(
            nn.Linear(score_in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    # ------------------------------------------------------------------
    # cluster_embed initialization from member mediator means
    # ------------------------------------------------------------------

    def init_cluster_embed_from_members(
        self, cluster_member_pmp_ids: dict[int, list[int]]
    ) -> None:
        """Initialize cluster_embed[k] to the mean of member mediator_embeds.

        Args:
          cluster_member_pmp_ids: dict[k -> list of mediator PMP indices].
            Clusters with no members in the PMP vocab are left at zero.

        Must be called AFTER mediator_embed has been Xavier-initialized
        (handled in __init__).
        """
        with torch.no_grad():
            weight = self.cluster_embed.weight  # (n_clusters, d)
            device = weight.device
            for k in range(self.n_clusters):
                ids = cluster_member_pmp_ids.get(k, [])
                if not ids:
                    continue
                ids_t = torch.tensor(ids, dtype=torch.long, device=device)
                member_embeds = self.mediator_embed(ids_t)  # (n_members, d)
                weight[k] = member_embeds.mean(dim=0)

    # ------------------------------------------------------------------
    # T = softmax(W_T)
    # ------------------------------------------------------------------

    def cluster_transition(self) -> torch.Tensor:
        """Return row-normalized T = softmax(W_T), shape (n_clusters, n_clusters)."""
        return F.softmax(self.W_T, dim=-1)

    # ------------------------------------------------------------------
    # Cluster path forward
    # ------------------------------------------------------------------

    def cluster_path_forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        rel_dist_a: torch.Tensor,
        rel_dist_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the explicit pairwise cluster path.

        Args:
          affinity_a, affinity_b: (B, K) log1p typed mediator counts per pair.
          rel_dist_a, rel_dist_b: (B, K, n_rels_plus_special) relation
            distribution per (drug, cluster).

        Returns:
          z_cluster_pair: (B, hidden)
          p_a, p_b: (B, K) softmax distributions over clusters (used by
            within-cluster pool for joint relevance weighting).
        """
        B = affinity_a.size(0)
        K = self.n_clusters
        d = self.d

        # Cluster distributions
        p_a = F.softmax(affinity_a, dim=-1)  # (B, K)
        p_b = F.softmax(affinity_b, dim=-1)  # (B, K)

        # Learnable transition
        T = self.cluster_transition()  # (K, K)

        # S_ij = p_a[i] * T[i, j] * p_b[j]    -> (B, K, K)
        S = torch.einsum("bi,ij,bj->bij", p_a, T, p_b)
        S_sum = S.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)  # (B, 1, 1)
        alpha = S / S_sum  # (B, K, K)

        # Cluster embeddings broadcast to (B, K, K, d).
        e = self.cluster_embed.weight  # (K, d)
        e_i = e.unsqueeze(0).unsqueeze(2).expand(B, K, K, d)  # i-dim from rows
        e_j = e.unsqueeze(0).unsqueeze(1).expand(B, K, K, d)  # j-dim from cols

        # Relation context per (drug, cluster). rel_dist: (B, K, R), embed (R, rel_dim)
        # r_a: (B, K, rel_dim) where r_a[b, k] = E_r[rel_embed | dist[b, k]]
        r_a = rel_dist_a @ self.rel_embed.weight  # (B, K, rel_dim)
        r_b = rel_dist_b @ self.rel_embed.weight  # (B, K, rel_dim)
        # r_a_to_i shapes to (B, K, K, rel_dim) by broadcasting along j.
        r_a_to_i = r_a.unsqueeze(2).expand(B, K, K, self.rel_dim)
        r_j_to_b = r_b.unsqueeze(1).expand(B, K, K, self.rel_dim)

        # psi = concat([e_i, e_j, |e_i - e_j|, e_i * e_j, r_a_to_i, r_j_to_b])
        psi = torch.cat(
            [e_i, e_j, (e_i - e_j).abs(), e_i * e_j, r_a_to_i, r_j_to_b],
            dim=-1,
        )  # (B, K, K, 4*d + 2*rel_dim)

        # h_path = sum_{ij} alpha_ij * psi
        h_path = (alpha.unsqueeze(-1) * psi).sum(dim=(1, 2))  # (B, 4*d + 2*rel_dim)

        z_cluster_pair = self.mlp_pair_cluster(h_path)  # (B, hidden)
        return z_cluster_pair, p_a, p_b

    # ------------------------------------------------------------------
    # Within-cluster pool forward
    # ------------------------------------------------------------------

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

        Args:
          med_idx_per_cluster[k]: (B, M_max_k) long, mediator PMP indices.
          rel_a_per_cluster[k]:   (B, M_max_k) long, rel_id for (a, m).
          rel_b_per_cluster[k]:   (B, M_max_k) long, rel_id for (m, b).
          mask_per_cluster[k]:    (B, M_max_k) bool, True for real mediator.
          type_id_per_cluster[k]: scalar int, the cluster's type id.
          p_a, p_b:               (B, K) cluster distributions from cluster path.

        Returns:
          z_within: (B, d)
        """
        device = p_a.device
        B = p_a.size(0)
        weights = p_a * p_b  # (B, K)

        z_within = torch.zeros(B, self.d, device=device)

        for k in range(self.n_clusters):
            mask_k = mask_per_cluster[k]  # (B, M_max_k)
            any_valid = mask_k.any(dim=1)  # (B,)
            if not any_valid.any():
                continue  # entire batch has no mediators in this cluster

            med_idx_k = med_idx_per_cluster[k]  # (B, M_max_k)
            rel_a_k = rel_a_per_cluster[k]
            rel_b_k = rel_b_per_cluster[k]
            type_id_k = int(type_id_per_cluster[k])

            # phi(m) for each mediator slot. type_id is constant within cluster k.
            h_m = self.mediator_embed(med_idx_k)  # (B, M_max_k, d)
            type_tensor = torch.full(
                (B, med_idx_k.size(1)), type_id_k,
                dtype=torch.long, device=device,
            )
            e_type = self.type_embed(type_tensor)  # (B, M_max_k, type_dim)
            e_ram = self.rel_embed(rel_a_k)  # (B, M_max_k, rel_dim)
            e_rmb = self.rel_embed(rel_b_k)  # (B, M_max_k, rel_dim)
            phi_in = torch.cat([h_m, e_type, e_ram, e_rmb], dim=-1)
            phi = self.phi_mlp(phi_in)  # (B, M_max_k, d)

            # Attention pool
            s = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
            neg_inf = torch.finfo(s.dtype).min
            s = s.masked_fill(~mask_k, neg_inf)
            # Safe softmax for all-padded rows
            s_safe = torch.where(any_valid.unsqueeze(1), s, torch.zeros_like(s))
            alpha_within = F.softmax(s_safe, dim=1)
            alpha_within = alpha_within * mask_k.float()
            z_k = (alpha_within.unsqueeze(-1) * phi).sum(dim=1)  # (B, d)

            # Weight by joint cluster relevance p_a[k] * p_b[k]
            weight_k = weights[:, k].unsqueeze(-1)  # (B, 1)
            z_within = z_within + weight_k * z_k

        return z_within

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        rel_dist_a: torch.Tensor,
        rel_dist_b: torch.Tensor,
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute v1.1 score.

        Returns:
          score:           (B,) scalar logit per pair.
          z_cluster_pair:  (B, hidden) cluster-path representation (diagnostic).
          z_within:        (B, d) within-cluster pool representation (diagnostic).
        """
        z_cluster_pair, p_a, p_b = self.cluster_path_forward(
            affinity_a, affinity_b, rel_dist_a, rel_dist_b,
        )
        z_within = self.within_cluster_pool_forward(
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_per_cluster, type_id_per_cluster, p_a, p_b,
        )
        z_pair = torch.cat([z_cluster_pair, z_within], dim=-1)  # (B, hidden + d)
        score = self.mlp_score(z_pair).squeeze(-1)  # (B,)
        return score, z_cluster_pair, z_within


__all__ = ["PMPv1_1Module"]

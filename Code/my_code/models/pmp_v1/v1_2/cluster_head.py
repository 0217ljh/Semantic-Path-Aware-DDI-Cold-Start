"""PMP v1.2 — pair-conditional cluster-path + per-cluster within-pool module.

Created 2026-06-03.

Implements the locked v1.2 architecture per user spec. The single architectural
change vs v1.1: the cluster path's per-cluster representation e_i is no longer
the fixed shared cluster_embed[i] (initialized from member mean). It is now the
pair-conditional H_a[:, i, :] = AttnPool over mediator_embed[m] for
m in members(cluster_i) ∩ N(a). Symmetrically for e_j and drug b.

Cache schema unchanged from v1.1 (the v1.1 cluster cache v2 is reused).

The within-cluster pool, the cluster transition T = softmax(W_T), the relation
context r_a / r_b, the cluster-path readout MLP, and the final score MLP are
all UNCHANGED from v1.1.

Forward signature additions (over v1.1):
  med_a_per_cluster, mask_a_per_cluster: per-drug-a per-cluster mediator
    tensors (list of K (B, M_max_a_k) tensors).
  med_b_per_cluster, mask_b_per_cluster: same for drug b.

Empty pools (drug has no mediators in cluster k) → 0 vector for H[:, k, :].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMPv1_2Module(nn.Module):
    """v1.2 scoring module: pair-conditional cluster path + within-pool + score MLP.

    All learnable parameters that participate in the score live in this module
    (mediator_embed, type_embed, rel_embed, W_T, cluster_attn, phi, within_attn,
    MLP_pair_cluster, MLP_score). EmerGNN backbone params are built by the
    trainer but their forward output does not enter score.
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
        # Embeddings (cluster_embed REMOVED in v1.2 — replaced by pair-conditional H)
        # ------------------------------------------------------------------
        self.mediator_embed = nn.Embedding(self.n_mediators, self.d)
        nn.init.xavier_uniform_(self.mediator_embed.weight)
        self.type_embed = nn.Embedding(self.n_types, self.type_dim)
        nn.init.xavier_uniform_(self.type_embed.weight)
        self.rel_embed = nn.Embedding(self.n_rels_plus_special, self.rel_dim)
        nn.init.xavier_uniform_(self.rel_embed.weight)

        # ------------------------------------------------------------------
        # Learnable cluster transition T = softmax(W_T)
        # ------------------------------------------------------------------
        self.W_T = nn.Parameter(torch.empty(self.n_clusters, self.n_clusters))
        nn.init.xavier_uniform_(self.W_T)

        # ------------------------------------------------------------------
        # Pair-conditional cluster representation attention (NEW in v1.2)
        # H[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(d)})
        # ------------------------------------------------------------------
        self.cluster_attn_W = nn.Linear(self.d, self.d)
        self.cluster_attn_w = nn.Linear(self.d, 1, bias=False)

        # ------------------------------------------------------------------
        # Within-cluster pool (UNCHANGED from v1.1)
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

        # ------------------------------------------------------------------
        # Cluster path readout (UNCHANGED shape from v1.1; only e_i source changed)
        # psi has 4*d + 2*rel_dim dims
        # ------------------------------------------------------------------
        h_path_dim = 4 * self.d + 2 * self.rel_dim
        self.mlp_pair_cluster = nn.Sequential(
            nn.Linear(h_path_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.hidden),
        )

        # ------------------------------------------------------------------
        # Final score MLP (UNCHANGED from v1.1)
        # ------------------------------------------------------------------
        score_in_dim = self.hidden + self.d
        self.mlp_score = nn.Sequential(
            nn.Linear(score_in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    # ------------------------------------------------------------------
    # T = softmax(W_T)
    # ------------------------------------------------------------------

    def cluster_transition(self) -> torch.Tensor:
        """Return row-normalized T = softmax(W_T), shape (n_clusters, n_clusters)."""
        return F.softmax(self.W_T, dim=-1)

    # ------------------------------------------------------------------
    # Pair-conditional cluster representation
    # ------------------------------------------------------------------

    def _pair_conditional_cluster_repr(
        self,
        med_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
    ) -> torch.Tensor:
        """For one drug (a or b), compute H of shape (B, K, d).

        H[:, k, :] = AttnPool({mediator_embed[m] : m in members(k) ∩ N(drug)}).
        Empty M_k(drug) → row is zeros (preserves "no mediator in this cluster"
        as a defined-but-non-informative signal).

        Args:
          med_per_cluster: list of K (B, M_max_k) long tensors, mediator
            PMP indices (0 in padding slots).
          mask_per_cluster: list of K (B, M_max_k) bool tensors.

        Returns:
          H: (B, K, d).
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
            mask_k = mask_per_cluster[k]  # (B, M_max_k)
            any_valid = mask_k.any(dim=1)  # (B,)
            if not any_valid.any():
                continue  # entire batch has no mediator of this cluster

            med_k = med_per_cluster[k]  # (B, M_max_k)
            h_m = self.mediator_embed(med_k)  # (B, M_max_k, d)

            # Attention scores
            s = self.cluster_attn_w(torch.tanh(self.cluster_attn_W(h_m))).squeeze(-1)
            neg_inf = torch.finfo(s.dtype).min
            s = s.masked_fill(~mask_k, neg_inf)
            # Safe softmax for all-padded rows (replace -inf row with 0 row).
            s_safe = torch.where(any_valid.unsqueeze(1), s, torch.zeros_like(s))
            alpha = F.softmax(s_safe, dim=1)
            alpha = alpha * mask_k.float()
            h_k = (alpha.unsqueeze(-1) * h_m).sum(dim=1)  # (B, d)

            # Zero out drugs with no mediator in this cluster.
            h_k = torch.where(any_valid.unsqueeze(1), h_k, torch.zeros_like(h_k))
            H[:, k, :] = h_k

        return H

    # ------------------------------------------------------------------
    # Cluster path forward
    # ------------------------------------------------------------------

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pair-conditional cluster path.

        Args:
          affinity_a, affinity_b: (B, K) log1p typed mediator counts per drug.
          rel_dist_a, rel_dist_b: (B, K, n_rels_plus_special) relation dist.
          med_a_per_cluster[k], mask_a_per_cluster[k]: drug-a's per-cluster
            mediator tensors (B, M_max_a_k).
          med_b_per_cluster[k], mask_b_per_cluster[k]: drug-b's same.

        Returns:
          z_cluster_pair: (B, hidden)
          p_a, p_b: (B, K) cluster distributions (used by within-pool).
        """
        B = affinity_a.size(0)
        K = self.n_clusters
        d = self.d

        # Cluster distributions (same shape & formula as v1.1).
        p_a = F.softmax(affinity_a, dim=-1)  # (B, K)
        p_b = F.softmax(affinity_b, dim=-1)  # (B, K)

        # Pair-conditional cluster representations (NEW vs v1.1).
        H_a = self._pair_conditional_cluster_repr(med_a_per_cluster, mask_a_per_cluster)
        H_b = self._pair_conditional_cluster_repr(med_b_per_cluster, mask_b_per_cluster)
        # H_a, H_b: (B, K, d)

        # Learnable transition T = softmax(W_T) (UNCHANGED).
        T = self.cluster_transition()  # (K, K)

        # S_ij = p_a[i] * T[i, j] * p_b[j] (UNCHANGED).
        S = torch.einsum("bi,ij,bj->bij", p_a, T, p_b)
        S_sum = S.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
        alpha = S / S_sum  # (B, K, K)

        # e_i, e_j broadcast from H_a, H_b (CHANGED vs v1.1).
        # H_a is "drug-a's per-cluster picture". Slot i (row) of psi uses H_a[i].
        # H_b is "drug-b's per-cluster picture". Slot j (col) of psi uses H_b[j].
        e_i = H_a.unsqueeze(2).expand(B, K, K, d)  # (B, K, K, d), varies along K-rows
        e_j = H_b.unsqueeze(1).expand(B, K, K, d)  # (B, K, K, d), varies along K-cols

        # Relation context (UNCHANGED from v1.1).
        r_a = rel_dist_a @ self.rel_embed.weight  # (B, K, rel_dim)
        r_b = rel_dist_b @ self.rel_embed.weight
        r_a_to_i = r_a.unsqueeze(2).expand(B, K, K, self.rel_dim)
        r_j_to_b = r_b.unsqueeze(1).expand(B, K, K, self.rel_dim)

        # psi (UNCHANGED structure; e_i / e_j source replaced with pair-conditional H).
        psi = torch.cat(
            [e_i, e_j, (e_i - e_j).abs(), e_i * e_j, r_a_to_i, r_j_to_b],
            dim=-1,
        )  # (B, K, K, 4*d + 2*rel_dim)

        # h_path = sum_{ij} alpha_ij * psi (UNCHANGED).
        h_path = (alpha.unsqueeze(-1) * psi).sum(dim=(1, 2))  # (B, 4*d + 2*rel_dim)

        z_cluster_pair = self.mlp_pair_cluster(h_path)  # (B, hidden)
        return z_cluster_pair, p_a, p_b

    # ------------------------------------------------------------------
    # Within-cluster pool forward (UNCHANGED from v1.1)
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

        Identical to v1.1's within_cluster_pool_forward.
        """
        device = p_a.device
        B = p_a.size(0)
        weights = p_a * p_b  # (B, K)

        z_within = torch.zeros(B, self.d, device=device)

        for k in range(self.n_clusters):
            mask_k = mask_per_cluster[k]  # (B, M_max_k)
            any_valid = mask_k.any(dim=1)  # (B,)
            if not any_valid.any():
                continue

            med_idx_k = med_idx_per_cluster[k]
            rel_a_k = rel_a_per_cluster[k]
            rel_b_k = rel_b_per_cluster[k]
            type_id_k = int(type_id_per_cluster[k])

            h_m = self.mediator_embed(med_idx_k)  # (B, M_max_k, d)
            type_tensor = torch.full(
                (B, med_idx_k.size(1)), type_id_k,
                dtype=torch.long, device=device,
            )
            e_type = self.type_embed(type_tensor)  # (B, M_max_k, type_dim)
            e_ram = self.rel_embed(rel_a_k)
            e_rmb = self.rel_embed(rel_b_k)
            phi_in = torch.cat([h_m, e_type, e_ram, e_rmb], dim=-1)
            phi = self.phi_mlp(phi_in)  # (B, M_max_k, d)

            s = self.within_attn_w(torch.tanh(self.within_attn_W(phi))).squeeze(-1)
            neg_inf = torch.finfo(s.dtype).min
            s = s.masked_fill(~mask_k, neg_inf)
            s_safe = torch.where(any_valid.unsqueeze(1), s, torch.zeros_like(s))
            alpha_within = F.softmax(s_safe, dim=1)
            alpha_within = alpha_within * mask_k.float()
            z_k = (alpha_within.unsqueeze(-1) * phi).sum(dim=1)  # (B, d)

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
        med_a_per_cluster: list[torch.Tensor],
        mask_a_per_cluster: list[torch.Tensor],
        med_b_per_cluster: list[torch.Tensor],
        mask_b_per_cluster: list[torch.Tensor],
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute v1.2 score.

        Returns:
          score:           (B,) scalar logit per pair.
          z_cluster_pair:  (B, hidden) cluster-path representation (diagnostic).
          z_within:        (B, d) within-cluster pool representation (diagnostic).
        """
        z_cluster_pair, p_a, p_b = self.cluster_path_forward(
            affinity_a, affinity_b, rel_dist_a, rel_dist_b,
            med_a_per_cluster, mask_a_per_cluster,
            med_b_per_cluster, mask_b_per_cluster,
        )
        z_within = self.within_cluster_pool_forward(
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_per_cluster, type_id_per_cluster, p_a, p_b,
        )
        z_pair = torch.cat([z_cluster_pair, z_within], dim=-1)  # (B, hidden + d)
        score = self.mlp_score(z_pair).squeeze(-1)  # (B,)
        return score, z_cluster_pair, z_within


__all__ = ["PMPv1_2Module"]

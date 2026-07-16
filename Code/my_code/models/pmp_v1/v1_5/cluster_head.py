"""PMP v1.5 — sequential cluster-attention -> within-pool scoring module
with prior + content residual attention (Codex Option B).

Created 2026-06-03.

Step 1 (cluster-cluster attention) uses pair-conditional H_a^k, H_b^k as a
CONTENT RESIDUAL on top of the affinity prior:

    log alpha_{ij} ∝ log p_a[i] + log T[i, j] + log p_b[j]
                     + lambda_content * g(H_a^i, H_b^j)

`g` is a small MLP (3d -> 16 -> 1) with the last layer zero-initialized.
`lambda_content` is a learnable scalar, init 0.0. Together this guarantees
v1.5 BEGINS its training trajectory at the pure-affinity attention (Option A)
and learns whether to incorporate content-conditional routing.

Step 2 (within-pool) is unchanged in structure from v1.1 / v1.2 / v1.4 but is
reweighted by the Step 1 row / col attention marginals (NOT by
p_a[k] * p_b[k]).

mediator_embed gradient path
----------------------------
mediator_embed feeds two locations:
  1. Step 1 H_a^k / H_b^k (via cluster_attn pool) -> alpha attention LOGITS
     (multiplied by lambda_content, init 0)
  2. Step 2 phi(m) (within-pool) -> z_within^k -> z_evidence

Because (1) only contributes via a SCALAR multiplier (alpha re-weights the
within-pool stack), the gradient through (1) does not create a sibling
readout branch. It modulates the same forward path as (2). This avoids the
v1.2 parallel-branch gradient competition pathology while still letting
mediator content influence cluster-pair routing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PMPv1_5Module(nn.Module):
    """v1.5 scoring module: content-aware cluster-cluster attention -> within-pool -> MLP_score.

    All learnable params that participate in the score:
      mediator_embed, type_embed, rel_embed,
      W_T,
      cluster_attn_W/w (pair-conditional pool for H_a/H_b),
      g_mlp + lambda_content (content residual in alpha),
      phi_mlp, within_attn_W/w (within-pool),
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
        content_hidden: int = 16,
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
        self.content_hidden = int(content_hidden)
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
        # Learnable cluster transition T = softmax(W_T)
        # ------------------------------------------------------------------
        self.W_T = nn.Parameter(torch.empty(self.n_clusters, self.n_clusters))
        nn.init.xavier_uniform_(self.W_T)

        # ------------------------------------------------------------------
        # Pair-conditional cluster representation attention
        # H_a/H_b = AttnPool(mediator_embed) over per-drug per-cluster mediators.
        # ------------------------------------------------------------------
        self.cluster_attn_W = nn.Linear(self.d, self.d)
        self.cluster_attn_w = nn.Linear(self.d, 1, bias=False)

        # ------------------------------------------------------------------
        # Content residual MLP for alpha
        # g_mlp([H_a^i, H_b^j, H_a^i ⊙ H_b^j]) -> scalar logit
        # Last linear layer is zero-initialized so the residual contributes
        # zero at init (model behaves as pure affinity attention initially).
        # ------------------------------------------------------------------
        g_in_dim = 3 * self.d
        self.g_mlp = nn.Sequential(
            nn.Linear(g_in_dim, self.content_hidden),
            nn.ReLU(),
            nn.Linear(self.content_hidden, 1),
        )
        # Zero-init the LAST Linear so g(H_a, H_b) == 0 at init.
        with torch.no_grad():
            last_linear = self.g_mlp[-1]
            assert isinstance(last_linear, nn.Linear)
            last_linear.weight.zero_()
            last_linear.bias.zero_()

        # Learnable scalar multiplier on the content residual.
        #
        # 2026-06-04 REVERTED to zeros(1) (the original "dead init"
        # configuration that empirically gave the winning v1.5 A result:
        # AUC=0.7764 / AUPRC=0.7911 / lambda_content_final=0.0).
        #
        # WHY THIS IS LEFT AT ZERO ON PURPOSE:
        # - With BOTH g_mlp[-1] zero-init AND lambda_content=0, content_logit
        #   ≡ 0 always; gradient never flows into g_mlp[-1] or lambda_content.
        #   The content residual machinery is effectively dead code.
        # - This means alpha = softmax of pure affinity prior
        #   (log p_a + log T + log p_b) for the entire training. mediator_embed
        #   only receives gradient through phi → within-pool → score, NOT
        #   through cluster_attn → H_a/H_b → g_mlp → alpha. Single gradient
        #   path = no gradient competition.
        # - Empirically, activating the content residual (lambda init = 1.0
        #   reproducibility experiment 2026-06-04) DEGRADED test AUC from
        #   0.7764 to 0.7547. The cluster_attn path's gradient pressure on
        #   mediator_embed reintroduces the v1.2 parallel-branch competition
        #   pathology that v1.5 was designed to avoid.
        # - See _reviews/2026-06-04__dead_init_fix.md for the full analysis.
        #
        # We keep the H_a/H_b / g_mlp / lambda_content machinery in the module
        # for now (as documented dead code) so v1.5 stays one git-edit away
        # from the stop-grad / detached-content experiments (v1.5 B-iso).
        self.lambda_content = nn.Parameter(torch.zeros(1))

        # ------------------------------------------------------------------
        # Within-cluster pool components (same as v1.1 / v1.2 / v1.4)
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
        # Final score MLP — input is concat[z_evidence_a, z_evidence_b].
        # ------------------------------------------------------------------
        self.mlp_score = nn.Sequential(
            nn.Linear(2 * self.d, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
        )

    # ------------------------------------------------------------------
    # T = softmax(W_T) and log T = log_softmax(W_T) for numerical stability
    # ------------------------------------------------------------------

    def cluster_transition(self) -> torch.Tensor:
        return F.softmax(self.W_T, dim=-1)

    def cluster_log_transition(self) -> torch.Tensor:
        return F.log_softmax(self.W_T, dim=-1)

    # ------------------------------------------------------------------
    # Pair-conditional cluster representation
    # ------------------------------------------------------------------

    def _pair_conditional_cluster_repr(
        self,
        med_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
    ) -> torch.Tensor:
        """For one drug, compute H of shape (B, K, d).

        H[:, k, :] = AttnPool({mediator_embed[m] : m in members(k) ∩ N(drug)}).
        Empty M_k(drug) -> row is zeros.
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

    # ------------------------------------------------------------------
    # Step 1: prior + content-residual cluster-cluster attention
    # ------------------------------------------------------------------

    def cluster_attention(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        H_a: torch.Tensor,
        H_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute alpha (B, K, K) and its row / col marginals.

        Logit = log_p_a[i] + log_T[i, j] + log_p_b[j]
              + lambda_content * g_mlp([H_a^i, H_b^j, H_a^i ⊙ H_b^j])

        Softmax is taken over the joint (i, j) axes via flatten/reshape so
        sum_{i,j} alpha == 1 per batch sample.

        Args:
          affinity_a, affinity_b: (B, K) log1p typed mediator counts.
          H_a, H_b: (B, K, d) pair-conditional cluster reprs.

        Returns:
          alpha:  (B, K, K) attention.
          attn_a: (B, K) row marginal.
          attn_b: (B, K) col marginal.
        """
        B = affinity_a.size(0)
        K = self.n_clusters
        d = self.d

        # Affinity prior (log probability factorization).
        log_p_a = F.log_softmax(affinity_a, dim=-1)  # (B, K)
        log_p_b = F.log_softmax(affinity_b, dim=-1)  # (B, K)
        log_T = self.cluster_log_transition()  # (K, K)
        # base[b, i, j] = log_p_a[b, i] + log_T[i, j] + log_p_b[b, j]
        base_logit = (
            log_p_a.unsqueeze(2)  # (B, K, 1)
            + log_T.unsqueeze(0)  # (1, K, K)
            + log_p_b.unsqueeze(1)  # (B, 1, K)
        )  # (B, K, K)

        # Content residual.
        # content_input[b, i, j, :] = concat[H_a[b, i, :], H_b[b, j, :], H_a[b, i, :] * H_b[b, j, :]]
        H_a_i = H_a.unsqueeze(2).expand(B, K, K, d)  # varies along row index
        H_b_j = H_b.unsqueeze(1).expand(B, K, K, d)  # varies along col index
        content_input = torch.cat([H_a_i, H_b_j, H_a_i * H_b_j], dim=-1)  # (B, K, K, 3d)
        content_logit = self.g_mlp(content_input).squeeze(-1)  # (B, K, K)

        # Total logit.
        total_logit = base_logit + self.lambda_content * content_logit  # (B, K, K)

        # Softmax over joint (i, j).
        flat = total_logit.view(B, K * K)
        alpha_flat = F.softmax(flat, dim=-1)
        alpha = alpha_flat.view(B, K, K)

        attn_a = alpha.sum(dim=2)  # (B, K) row marginal (sum over j)
        attn_b = alpha.sum(dim=1)  # (B, K) col marginal (sum over i)
        return alpha, attn_a, attn_b

    # ------------------------------------------------------------------
    # Step 2: within-cluster pool, stacked per cluster k
    # ------------------------------------------------------------------

    def within_cluster_pool_perk(
        self,
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
    ) -> torch.Tensor:
        """Compute z_within^k for each cluster k, stacked as (B, K, d).

        z_within^k = AttnPool(phi(m) for m in members(k) ∩ N(a) ∩ N(b)).
        Empty M_k yields zeros for that row.
        """
        K = self.n_clusters
        if len(med_idx_per_cluster) != K:
            raise ValueError(
                f"Expected len(med_idx_per_cluster) == n_clusters={K}, "
                f"got {len(med_idx_per_cluster)}"
            )
        B = med_idx_per_cluster[0].size(0)
        device = med_idx_per_cluster[0].device
        z_stack = torch.zeros(B, K, self.d, device=device)

        for k in range(K):
            mask_k = mask_per_cluster[k]
            any_valid = mask_k.any(dim=1)
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
            e_type = self.type_embed(type_tensor)
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
            z_k = (alpha_within.unsqueeze(-1) * phi).sum(dim=1)
            z_k = torch.where(any_valid.unsqueeze(1), z_k, torch.zeros_like(z_k))
            z_stack[:, k, :] = z_k

        return z_stack

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        med_a_per_cluster: list[torch.Tensor],
        mask_a_per_cluster: list[torch.Tensor],
        med_b_per_cluster: list[torch.Tensor],
        mask_b_per_cluster: list[torch.Tensor],
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute v1.5 score.

        Returns:
          score:        (B,) scalar logit per pair.
          z_evidence:   (B, 2*d) concat of a-marginal / b-marginal evidence (diagnostic).
          alpha:        (B, K, K) Step 1 cluster-pair attention (diagnostic).
          attn_a:       (B, K) Step 1 row marginal (diagnostic).
          attn_b:       (B, K) Step 1 col marginal (diagnostic).
        """
        # Pair-conditional cluster reprs (used only by Step 1 content residual).
        H_a = self._pair_conditional_cluster_repr(med_a_per_cluster, mask_a_per_cluster)
        H_b = self._pair_conditional_cluster_repr(med_b_per_cluster, mask_b_per_cluster)

        # Step 1: prior + content residual attention.
        alpha, attn_a, attn_b = self.cluster_attention(
            affinity_a, affinity_b, H_a, H_b,
        )

        # Step 2: per-cluster within-pool (intersection mediators).
        z_stack = self.within_cluster_pool_perk(
            med_idx_per_cluster, rel_a_per_cluster, rel_b_per_cluster,
            mask_per_cluster, type_id_per_cluster,
        )

        # Attention-marginal weighted sums.
        z_evidence_a = torch.einsum("bk,bkd->bd", attn_a, z_stack)  # (B, d)
        z_evidence_b = torch.einsum("bk,bkd->bd", attn_b, z_stack)  # (B, d)
        z_evidence = torch.cat([z_evidence_a, z_evidence_b], dim=-1)  # (B, 2d)

        score = self.mlp_score(z_evidence).squeeze(-1)  # (B,)
        return score, z_evidence, alpha, attn_a, attn_b


__all__ = ["PMPv1_5Module"]

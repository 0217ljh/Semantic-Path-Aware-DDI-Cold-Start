"""PMP v1.6 — PyG MessagePassing equivalent of v1.5 A.

Created 2026-06-04.

Strict mathematical equivalence claim
-------------------------------------
For the SAME input (affinity_a, affinity_b, intersection mediator tensors,
masks, type_id_per_cluster) and the SAME learnable parameter values
(mediator_embed, type_embed, rel_embed, W_T, phi_mlp, within_attn_W,
within_attn_w, mlp_score), the forward output of `PMPv1_6_Module` equals
the forward output of `PMPv1_5Module` (v1.5 A) within numerical precision
(~1e-5 max abs diff).

The equivalence is verified:
  - statically by codex (review of the message / aggregate functions vs
    v1.5 A's within_cluster_pool_perk)
  - numerically by a smoke test that constructs both modules with identical
    learnable params and compares forward outputs on a synthetic batch.

Architecture
------------
Same as v1.5 A:

  Step 1 (analytical): cluster-pair attention marginals via
                       log_p_a + log_T + log_p_b -> softmax -> marginalize.
  Step 2 (PyG MP):     bipartite mediator -> cluster message passing with
                       content-based attention aggregation.
  Step 3 (analytical): cluster pooling weighted by Step 1's attention marginals.
  Step 4 (analytical): MLP_score on concat[a-marginal evidence, b-marginal evidence].

The only structural difference vs v1.5 A is in Step 2: v1.5 A loops over K
clusters, gathering per-cluster padded mediator tensors and pooling each
independently. v1.6 flattens all (pair, cluster, mediator) triples into
a single bipartite edge set and runs one PyG MessagePassing call.
Both produce identical per-cluster z_within^k.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class WithinClusterMP(MessagePassing):
    """PyG MessagePassing for the within-pool of v1.5 A.

    Source nodes: mediator "edge slots" (one per (pair, cluster, mediator) triple).
    Target nodes: cluster nodes (12 per pair, 12 * B in batched layout).

    Per edge:
      x_source       = mediator_embed[m]                          (d,)
      edge_attr      = concat[type_embed[k], rel_embed[a->m], rel_embed[m->b]]
                       (type_dim + 2*rel_dim,)
      message(edge)  = phi_mlp(concat[x_source, edge_attr])       (d,)
                       = MLP([med, type[k], rel(a->m), rel(m->b)])
      score(edge)    = within_attn_w(tanh(within_attn_W(message))) (1,)

    Aggregate at target cluster c (= K*pair + cluster_k):
      alpha_edge     = softmax over edges incident to c of score(edge)
      h_c            = sum_{edge -> c} alpha_edge * message(edge)

    This is bit-for-bit equivalent to v1.5 A's within_cluster_pool_perk for
    each (pair, cluster_k) pair (i.e. each target cluster c in this bipartite
    graph).
    """

    def __init__(
        self,
        d: int,
        type_dim: int,
        rel_dim: int,
        hidden: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(aggr=None, flow="source_to_target")
        self.d = int(d)
        self.type_dim = int(type_dim)
        self.rel_dim = int(rel_dim)
        self.hidden = int(hidden)

        phi_in_dim = self.d + self.type_dim + 2 * self.rel_dim
        # Use the SAME architecture as v1.5 A's phi_mlp (Linear -> ReLU ->
        # Dropout -> Linear). Layer instances are exposed at attributes named
        # `phi_mlp`, `within_attn_W`, `within_attn_w` so equivalence init can
        # be done at the trainer level by copying state_dict slices.
        self.phi_mlp = nn.Sequential(
            nn.Linear(phi_in_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.d),
        )
        self.within_attn_W = nn.Linear(self.d, self.d)
        self.within_attn_w = nn.Linear(self.d, 1, bias=False)

    def forward(
        self,
        mediator_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """Run bipartite mediator -> cluster MP and return (num_clusters, d).

        Args:
          mediator_h:    (E, d) per-edge mediator embedding (each edge has its
                         own source "slot"; mediators are NOT deduplicated).
          edge_index:    (2, E) bipartite edges. Row 0 is source index in
                         [0, E), row 1 is target cluster index in
                         [0, num_clusters).
          edge_attr:     (E, type_dim + 2*rel_dim) per-edge attribute concat.
          num_clusters:  total number of target cluster nodes = 12 * B.

        Returns:
          cluster_h:     (num_clusters, d). Cluster nodes with no incoming
                         edges receive zero.
        """
        if mediator_h.size(0) == 0:
            return torch.zeros(num_clusters, self.d, device=mediator_h.device)
        # Bipartite: x = (x_src, x_dst). x_dst can be None when we just want
        # to aggregate INTO the destination side and have no target features.
        return self.propagate(
            edge_index,
            x=(mediator_h, None),
            edge_attr=edge_attr,
            size=(mediator_h.size(0), num_clusters),
        )

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # x_j: (E, d) source-side gathered features (one row per edge).
        # edge_attr: (E, type_dim + 2*rel_dim).
        # phi(m) = phi_mlp(concat[mediator_embed[m], type_embed[k], rel(a->m), rel(m->b)])
        phi_in = torch.cat([x_j, edge_attr], dim=-1)
        return self.phi_mlp(phi_in)

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None = None,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        # inputs: (E, d) messages.
        # index:  (E,) target cluster index per edge.
        # dim_size: total number of target cluster nodes (12 * B).
        E = inputs.size(0)
        if E == 0:
            assert dim_size is not None
            return torch.zeros(dim_size, self.d, device=inputs.device)
        # Per-edge attention score (same as v1.5 A's within attention).
        scores = self.within_attn_w(torch.tanh(self.within_attn_W(inputs))).squeeze(-1)  # (E,)
        # Per-destination softmax. This is the EXACT v1.5 A within-pool
        # softmax: edges incident to the same (pair, cluster) target form one
        # group; softmax is taken within each group.
        alpha = pyg_softmax(scores, index, num_nodes=dim_size)  # (E,)
        # Weighted sum at each target.
        weighted = alpha.unsqueeze(-1) * inputs  # (E, d)
        out = torch.zeros(
            dim_size if dim_size is not None else int(index.max().item()) + 1,
            self.d,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        # scatter_add_ for the weighted sum.
        out.scatter_add_(
            0,
            index.unsqueeze(-1).expand_as(weighted),
            weighted,
        )
        return out


class PMPv1_6_Module(nn.Module):
    """PyG-based equivalent of PMPv1_5Module (v1.5 A).

    The cluster-pair attention (Step 1) stays analytical because it is a
    closed-form 3-step random walk; expressing it as PyG MP would obscure
    rather than clarify. The within-pool (Step 2) is reframed as PyG
    MessagePassing on a bipartite mediator -> cluster graph. The cluster
    pooling (Step 3) and score (Step 4) are analytical, identical to v1.5 A.

    All learnable parameters live in this module (or in WithinClusterMP):
      mediator_embed, type_embed, rel_embed, W_T,
      within_pool_mp.phi_mlp, within_pool_mp.within_attn_W,
      within_pool_mp.within_attn_w, mlp_score.

    Dead-code components from v1.5 A (cluster_attn_W/w, g_mlp,
    lambda_content) are NOT included — they don't affect output anyway.
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
        # Embeddings (same shapes, same init as v1.5 A)
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
        # PyG MessagePassing for within-pool
        # ------------------------------------------------------------------
        self.within_pool_mp = WithinClusterMP(
            d=self.d,
            type_dim=self.type_dim,
            rel_dim=self.rel_dim,
            hidden=self.hidden,
            dropout=dropout,
        )

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
    # log T for numerical stability
    # ------------------------------------------------------------------

    def cluster_log_transition(self) -> torch.Tensor:
        return F.log_softmax(self.W_T, dim=-1)

    # ------------------------------------------------------------------
    # Step 1: analytical cluster attention marginals (same as v1.5 A)
    # ------------------------------------------------------------------

    def cluster_attention_marginals(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (alpha, attn_a, attn_b).

        alpha[b, i, j] = softmax over (i, j) of
                        (log_p_a[b, i] + log_T[i, j] + log_p_b[b, j])
        attn_a[b, k]   = sum_j alpha[b, k, j]
        attn_b[b, k]   = sum_i alpha[b, i, k]
        """
        B = affinity_a.size(0)
        K = self.n_clusters

        log_p_a = F.log_softmax(affinity_a, dim=-1)  # (B, K)
        log_p_b = F.log_softmax(affinity_b, dim=-1)
        log_T = self.cluster_log_transition()  # (K, K)
        base_logit = (
            log_p_a.unsqueeze(2)
            + log_T.unsqueeze(0)
            + log_p_b.unsqueeze(1)
        )  # (B, K, K)
        alpha_flat = F.softmax(base_logit.view(B, K * K), dim=-1)
        alpha = alpha_flat.view(B, K, K)
        attn_a = alpha.sum(dim=2)  # (B, K)
        attn_b = alpha.sum(dim=1)  # (B, K)
        return alpha, attn_a, attn_b

    # ------------------------------------------------------------------
    # Step 2: build bipartite edge tensors from padded per-cluster
    # intersection mediator tensors, then run PyG MessagePassing.
    # ------------------------------------------------------------------

    def _build_bipartite_edges(
        self,
        med_idx_per_cluster: list[torch.Tensor],
        rel_a_per_cluster: list[torch.Tensor],
        rel_b_per_cluster: list[torch.Tensor],
        mask_per_cluster: list[torch.Tensor],
        type_id_per_cluster: list[int],
        B: int,
        K: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Flatten the padded per-cluster intersection tensors into bipartite
        edge tensors for PyG MessagePassing.

        Layout convention: target cluster global index = K * b + k for pair
        b and cluster k, so that reshape via `view(B, K, d)` correctly
        unpacks the cluster-pooled output.

        Returns:
          mediator_h:   (E, d) per-edge mediator embedding (one row per edge,
                        from `self.mediator_embed[m]`).
          edge_index:   (2, E) bipartite [source_slot, cluster_global_idx].
                        Sources are unique (no mediator deduplication), so
                        source_slot = 0..E-1.
          edge_attr:    (E, type_dim + 2*rel_dim) concat
                        [type_embed[k], rel_embed[a->m], rel_embed[m->b]].
          num_clusters: K * B.
        """
        device = med_idx_per_cluster[0].device
        all_med_ids = []
        all_cluster_global = []
        all_type_ids = []
        all_rel_a = []
        all_rel_b = []

        for k in range(K):
            mask_k = mask_per_cluster[k]  # (B, M_max_k)
            med_k = med_idx_per_cluster[k]
            rel_a_k = rel_a_per_cluster[k]
            rel_b_k = rel_b_per_cluster[k]
            type_id_k = int(type_id_per_cluster[k])

            # valid_b: pair index in batch, valid_j: slot index in M_max_k
            valid_b, valid_j = torch.where(mask_k)  # both (E_k,)
            if valid_b.numel() == 0:
                continue
            m_ids = med_k[valid_b, valid_j]
            ra_ids = rel_a_k[valid_b, valid_j]
            rb_ids = rel_b_k[valid_b, valid_j]
            cluster_global = K * valid_b + k
            type_ids = torch.full(
                (valid_b.numel(),), type_id_k,
                dtype=torch.long, device=device,
            )
            all_med_ids.append(m_ids)
            all_cluster_global.append(cluster_global)
            all_type_ids.append(type_ids)
            all_rel_a.append(ra_ids)
            all_rel_b.append(rb_ids)

        num_clusters = K * B
        if not all_med_ids:
            mediator_h = torch.zeros(0, self.d, device=device)
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.zeros(
                0, self.type_dim + 2 * self.rel_dim, device=device,
            )
            return mediator_h, edge_index, edge_attr, num_clusters

        flat_med = torch.cat(all_med_ids)
        flat_cluster = torch.cat(all_cluster_global)
        flat_type = torch.cat(all_type_ids)
        flat_rel_a = torch.cat(all_rel_a)
        flat_rel_b = torch.cat(all_rel_b)
        E = flat_med.size(0)

        # Per-edge source feature: mediator_embed[m]
        mediator_h = self.mediator_embed(flat_med)  # (E, d)
        # Per-edge attribute: concat[type_embed[k], rel_a, rel_b]
        type_e = self.type_embed(flat_type)
        rel_a_e = self.rel_embed(flat_rel_a)
        rel_b_e = self.rel_embed(flat_rel_b)
        edge_attr = torch.cat([type_e, rel_a_e, rel_b_e], dim=-1)
        # edge_index: source_slot = 0..E-1, target = cluster_global
        source_slot = torch.arange(E, device=device)
        edge_index = torch.stack([source_slot, flat_cluster], dim=0)  # (2, E)
        return mediator_h, edge_index, edge_attr, num_clusters

    # ------------------------------------------------------------------
    # Full forward
    # ------------------------------------------------------------------

    def forward(
        self,
        affinity_a: torch.Tensor,
        affinity_b: torch.Tensor,
        # The following per-drug per-cluster slots are accepted for API
        # compatibility with v1.5 trainer's `_combined_logit` but NOT used
        # here (v1.5 A's H_a/H_b machinery is dead code under lambda = 0).
        med_a_per_cluster: list[torch.Tensor] | None = None,
        mask_a_per_cluster: list[torch.Tensor] | None = None,
        med_b_per_cluster: list[torch.Tensor] | None = None,
        mask_b_per_cluster: list[torch.Tensor] | None = None,
        # The intersection tensors are the actual Step 2 inputs.
        med_idx_per_cluster: list[torch.Tensor] | None = None,
        rel_a_per_cluster: list[torch.Tensor] | None = None,
        rel_b_per_cluster: list[torch.Tensor] | None = None,
        mask_per_cluster: list[torch.Tensor] | None = None,
        type_id_per_cluster: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute v1.6 score.

        Returns the same 5-tuple as PMPv1_5Module:
          score:        (B,) scalar logit.
          z_evidence:   (B, 2*d) concat of a-marginal / b-marginal evidence.
          alpha:        (B, K, K) cluster-pair attention.
          attn_a:       (B, K) row marginal.
          attn_b:       (B, K) col marginal.
        """
        if med_idx_per_cluster is None or mask_per_cluster is None:
            raise ValueError(
                "v1.6 forward requires intersection mediator tensors. "
                "Got None for med_idx_per_cluster or mask_per_cluster."
            )

        B = affinity_a.size(0)
        K = self.n_clusters
        d = self.d

        # Step 1: analytical cluster attention marginals.
        alpha, attn_a, attn_b = self.cluster_attention_marginals(
            affinity_a, affinity_b,
        )

        # Step 2: bipartite edge construction + PyG MessagePassing.
        mediator_h, edge_index, edge_attr, num_clusters = (
            self._build_bipartite_edges(
                med_idx_per_cluster,
                rel_a_per_cluster,  # type: ignore[arg-type]
                rel_b_per_cluster,  # type: ignore[arg-type]
                mask_per_cluster,
                type_id_per_cluster,  # type: ignore[arg-type]
                B, K,
            )
        )
        z_flat = self.within_pool_mp(
            mediator_h, edge_index, edge_attr, num_clusters,
        )  # (K * B, d)
        # Layout: cluster_global = K * b + k. View(B, K, d) unpacks.
        z_stack = z_flat.view(B, K, d)

        # Step 3: attention-marginal weighted aggregation (same as v1.5 A).
        z_evidence_a = torch.einsum("bk,bkd->bd", attn_a, z_stack)  # (B, d)
        z_evidence_b = torch.einsum("bk,bkd->bd", attn_b, z_stack)
        z_evidence = torch.cat([z_evidence_a, z_evidence_b], dim=-1)  # (B, 2d)

        # Step 4: score.
        score = self.mlp_score(z_evidence).squeeze(-1)
        return score, z_evidence, alpha, attn_a, attn_b


__all__ = ["PMPv1_6_Module", "WithinClusterMP"]

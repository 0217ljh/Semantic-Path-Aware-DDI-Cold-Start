"""PMP v1.2 — pair-conditional cluster path module (created 2026-06-03).

Architectural delta from v1.1
-----------------------------
- Cluster path's e_i (was the fixed shared cluster_embed[i] initialized from
  mean of member mediator embeddings) is REPLACED with pair-conditional
  h_a^i = AttnPool(mediator_embed[m] for m in members(i) ∩ N(a)).
  Symmetrically, e_j becomes h_b^j = AttnPool(... ∩ N(b)).
- cluster_embed nn.Embedding REMOVED from the v1.2 module (no longer used).
- W_T (learnable cluster transition, T = softmax(W_T)) UNCHANGED.
- Per-cluster within-pool UNCHANGED (still intersection-based).
- EmerGNN backbone bypass UNCHANGED (score is from cluster path + within only).
- Cache schema UNCHANGED — v1.2 reuses v1.1 cluster cache v2.

Locked v1.2 algorithm
---------------------
::

    Embeddings:
      mediator_embed  (n_mediators, d)         Xavier init
      type_embed      (n_types, type_dim)
      rel_embed       (n_rels + 1, rel_dim)
      W_T             (n_clusters, n_clusters) Xavier init; T = softmax(W_T)
      cluster_attn_W, cluster_attn_w           pair-conditional cluster pool
      phi_mlp, within_attn_W, within_attn_w    within-cluster pool (same as v1.1)
      mlp_pair_cluster, mlp_score              readouts (same shape as v1.1)

    Pair-conditional cluster repr (per cluster k, per drug):
      H_a[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(a)})
      H_b[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(b)})
      Empty M_k(a) → H_a[:, k, :] = 0.

    Cluster path (explicit pairwise, single transition):
      p_a = softmax(c_a),  p_b = softmax(c_b)
      S_ij    = p_a[i] * T[i, j] * p_b[j]
      alpha_ij = S_ij / sum_uv S_uv
      e_i      = H_a[:, i, :]     # pair-conditional, replaces cluster_embed[i]
      e_j      = H_b[:, j, :]     # pair-conditional, replaces cluster_embed[j]
      r_a      = rel_dist_a @ rel_embed.weight   # (B, K, rel_dim)
      r_b      = rel_dist_b @ rel_embed.weight
      psi(i,j,a,b) = concat([e_i, e_j, |e_i - e_j|, e_i * e_j,
                             r_a[:, i, :], r_b[:, j, :]])
      h_path = sum_{ij} alpha_ij * psi
      z_cluster_pair = MLP_pair_cluster(h_path)

    Per-cluster within-pool (UNCHANGED from v1.1):
      for k in 0..n_clusters - 1:
        M_k = members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)
        z_within^k = AttnPool({phi(m) for m in M_k}) or 0
      z_within = sum_k (p_a[k] * p_b[k]) * z_within^k

    Score:
      z_pair = concat([z_cluster_pair, z_within])
      score  = MLP_score(z_pair)
      loss   = BCE(sigmoid(score), y)

Components
----------
- `cluster_head.PMPv1_2Module`
- `pmp_v1_2_trainer._PerModeEmerGNN_PMP_v1_2`

This package is a new directory. v1_1/ is untouched per the
"不得删除或重构已有函数/类/文件" rule.
"""

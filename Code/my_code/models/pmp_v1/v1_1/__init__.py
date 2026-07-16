"""PMP v1.1 (rewrite 2026-06-02) — explicit pairwise cluster path + per-cluster
within-pool + final MLP score (replaces EmerGNN backbone).

LOCKED Algorithm 1 v1.1:

  Embeddings:
    mediator_embed (n_mediators, d), Xavier init.
    cluster_embed (12, d), init = mean({mediator_embed[m] for m in members(k)})
                              per cluster, trainable afterward.
    type_embed (n_types, type_dim).
    rel_embed (n_rels_plus_special, rel_dim).
    W_T (12, 12), Xavier init; T = softmax(W_T) at forward.

  Cluster path (explicit pairwise):
    p_a = softmax(c_a),   p_b = softmax(c_b)
    r_a = sum_r drug_cluster_rel_dist[a, :, r] * rel_embed[r]    (B, K, rel_dim)
    r_b similarly
    S_ij    = p_a[i] * T[i, j] * p_b[j]
    alpha_ij = S_ij / sum_{u,v} S_uv
    psi(i, j, a, b) = concat([
        cluster_embed[i], cluster_embed[j],
        |cluster_embed[i] - cluster_embed[j]|,
        cluster_embed[i] * cluster_embed[j],
        r_a[:, i, :], r_b[:, j, :]
    ])
    h_path = sum_{i, j} alpha_ij * psi(i, j, a, b)
    z_cluster_pair = MLP_pair_cluster(h_path)

  Per-cluster within-pool:
    for k in 0..n_clusters - 1:
      M_k(a, b) = members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)
      z_within^k = AttnPool({phi(m) for m in M_k(a, b)})  or 0 if empty
      phi(m) = MLP([mediator_embed[m], type_embed[k], rel_embed[a -> m],
                     rel_embed[m -> b]])
    z_within = sum_k (p_a[k] * p_b[k]) * z_within^k

  Score (replaces EmerGNN backbone):
    z_pair = concat([z_cluster_pair, z_within])
    score = MLP_score(z_pair)
    loss = BCE(sigmoid(score), y)
    training protocol: shuffle_train(mode='S2') inherited from EmerGNN.

Canonical imports:
    from my_code.models.pmp_v1.v1_1.pmp_v1_1_trainer import (
        _PerModeEmerGNN_PMP_v1_1, DEFAULT_CLUSTER_CACHE,
    )
    from my_code.models.pmp_v1.v1_1.cluster_head import PMPv1_1Module
    from my_code.models.pmp_v1.v1_1.precompute_cluster_cache import (
        build_cluster_cache, ensure_cluster_cache,
    )
"""
from __future__ import annotations

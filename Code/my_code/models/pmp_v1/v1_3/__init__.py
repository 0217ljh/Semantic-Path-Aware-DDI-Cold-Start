"""PMP v1.3 — Layer-1-only (pair-conditional cluster path), created 2026-06-03.

Diagnostic / ablation variant of v1.2. Drops the within-cluster pool entirely
so the score depends only on Layer 1 (the pair-conditional cluster path).
Used to measure Layer 1's standalone cold-start AUC when trained in isolation
(vs the joint v1.2 cluster_only diagnostic which trains both branches together
and zeros within at inference).

Locked v1.3 algorithm
---------------------
Cache reuse: PMP cache v1, v1.1 cluster cache v2. Per-drug per-cluster mediator
map (cluster_max_mediators_per_drug cap) built at load (same as v1.2).

Embeddings (within-pool components REMOVED):
  mediator_embed (n_mediators, d)         Xavier
  type_embed     (n_types, type_dim)
  rel_embed      (n_rels+1, rel_dim)
  W_T            (n_clusters, n_clusters) Xavier; T = softmax(W_T)
  cluster_attn_W, cluster_attn_w          pair-conditional pool

Pair-conditional cluster repr (per cluster k, per drug):
  H_a[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(a)})
  H_b[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(b)})
  Empty M_k(a) -> H_a[:, k, :] = 0.

Cluster path:
  p_a = softmax(c_a), p_b = softmax(c_b)
  T = softmax(W_T)
  S_ij     = p_a[i] * T[i, j] * p_b[j]
  alpha_ij = S_ij / sum_uv S_uv
  e_i      = H_a[:, i, :]
  e_j      = H_b[:, j, :]
  psi(i,j,a,b) = concat([e_i, e_j, |e_i - e_j|, e_i * e_j,
                         r_a[:, i, :], r_b[:, j, :]])
  h_path   = sum_{ij} alpha_ij * psi
  z_cluster_pair = MLP_pair_cluster(h_path)            (B, hidden)

Score:
  score = MLP_score(z_cluster_pair)                    # input dim = hidden
  loss  = BCE(sigmoid(score), y)

Components REMOVED vs v1.2:
  - phi_mlp                          (within-pool mediator encoder)
  - within_attn_W, within_attn_w
  - within_cluster_pool_forward()
  - intersection (M_k^{a∩b}) tensor construction in trainer
  - z_within concat into mlp_score input  (mlp_score takes z_cluster_pair only)

EmerGNN backbone bypass: UNCHANGED from v1.1 / v1.2.

This package is a new directory. v1_2/, v1_1/ are untouched.
"""

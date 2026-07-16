"""PMP v1.4 — Layer-2-only (per-cluster within pool), created 2026-06-03.

Diagnostic / ablation variant of v1.2. Drops the cluster path entirely so the
score depends only on Layer 2 (the per-cluster within-pool over intersection
mediators). Used to measure Layer 2's standalone cold-start AUC when trained
in isolation (vs the joint v1.2 within_only diagnostic which trains both
branches together and zeros cluster path at inference).

Locked v1.4 algorithm
---------------------
Cache reuse: PMP cache v1, v1.1 cluster cache v2. Per-drug per-cluster
mediator map is NOT built (not needed; v1.4 inherits v1.1 trainer, not v1.2).

Embeddings (cluster-path components REMOVED):
  mediator_embed (n_mediators, d)         Xavier
  type_embed     (n_types, type_dim)
  rel_embed      (n_rels+1, rel_dim)
  phi_mlp, within_attn_W, within_attn_w   within-pool (same as v1.1)

NOT present (vs v1.2):
  W_T                                     (no cluster transition)
  cluster_attn_W, cluster_attn_w
  cluster_embed                           (v1.1 had this, removed in v1.4 too)
  mlp_pair_cluster

Per-cluster within-pool:
  p_a = softmax(c_a), p_b = softmax(c_b)   # used only as joint relevance weight
  for k in 0..n_clusters - 1:
    M_k = members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)
    z_within^k = AttnPool({phi(m) for m in M_k}) or 0
      phi(m) = MLP([mediator_embed[m], type_embed[k],
                    rel_embed[a->m], rel_embed[m->b]])
  z_within = sum_k (p_a[k] * p_b[k]) * z_within^k       (B, d)

Score:
  score = MLP_score(z_within)                    # input dim = d
  loss  = BCE(sigmoid(score), y)

Components REMOVED vs v1.2:
  - W_T
  - cluster_attn_W, cluster_attn_w
  - _pair_conditional_cluster_repr (H_a, H_b)
  - cluster_path_forward
  - psi construction and h_path readout
  - mlp_pair_cluster
  - per-drug per-cluster mediator tensor construction in trainer
  - z_cluster_pair concat into mlp_score input  (mlp_score takes z_within only)

EmerGNN backbone bypass: UNCHANGED from v1.1.

This package is a new directory. v1_1/, v1_2/, v1_3/ are untouched.
"""

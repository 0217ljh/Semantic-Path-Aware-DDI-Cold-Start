"""PMP v1.6 — PyG-form re-implementation of v1.5 A (created 2026-06-04).

Goal: REWRITE v1.5 A as a torch_geometric MessagePassing-based GNN, with the
strict equivalence claim that for ANY input the forward output equals v1.5 A
within numerical precision (~1e-5 max abs diff when both modules start from
identical learnable params).

This package is NOT an architectural evolution. It is a sanity-check
re-implementation. The value is:

1. Validate the GNN-paradigm framing of v1.5 A (sanity check).
2. Provide a PyG-derived base class so future GNN-specific add-ons
   (R-GCN typed message kernels, DiffPool, HGT-style heterogeneous
   attention, cross-pair co-attention, etc.) can extend incrementally.
3. Paper positioning: enable the claim "PMP-v1.6 = GNN-form
   formulation of the v1.5 A analytical pipeline."

Architecture (matches v1.5 A bit-for-bit at math level)
-------------------------------------------------------
Cluster-pair attention (Step 1) is kept ANALYTICAL because it is already a
clean closed-form expression of a 3-step random walk on the cluster-cluster
graph; wrapping it in a MessagePassing layer would add noise, not value:

  log_p_a = log_softmax(c_a)
  log_p_b = log_softmax(c_b)
  log_T   = log_softmax(W_T, dim=-1)             # T is row-stochastic
  alpha   = softmax over (i, j) of
            (log_p_a[i] + log_T[i, j] + log_p_b[j])
  attn_a^k = sum_j alpha[k, j]
  attn_b^k = sum_i alpha[i, k]

Within-cluster pool (Step 2) is reframed as PyG MessagePassing on a
pair-specific bipartite graph:

  Sources: mediator "edge slots" (one per (pair, cluster, mediator-in-intersection))
  Targets: cluster nodes (12 per pair, 12 * B in batched layout)

  Edge attributes per edge (m, k, pair_b) for m in M_k^{a∩b} of pair b:
    edge_attr = concat[type_embed[k], rel_embed[a -> m], rel_embed[m -> b]]
  Source feature per edge:
    x_source = mediator_embed[m]

  message(edge) = phi_mlp(concat[x_source, edge_attr])
                = MLP([mediator_embed[m], type_embed[k],
                       rel_embed[a -> m], rel_embed[m -> b]])
                  -- identical to v1.5 A's phi(m)
  aggregate at target cluster c_k (per pair):
    score(m)  = within_attn_w(tanh(within_attn_W(message(m))))
    alpha_m   = softmax over edges incident to c_k of score(m)
    h_{c_k}   = sum_m alpha_m * message(m)
                  -- identical to v1.5 A's z_within^k

Cluster pooling (Step 3) is analytical:
  z_evidence_a = sum_k attn_a^k * h_{c_k}
  z_evidence_b = sum_k attn_b^k * h_{c_k}
  z_pair       = concat[z_evidence_a, z_evidence_b]
  score        = MLP_score(z_pair)

Equivalence claim
-----------------
For ANY input (affinity, intersection mediator tensors, mask) and ANY shared
initialization of learnable params, the forward output is IDENTICAL to v1.5 A
within numerical precision (~1e-5 max abs diff). The equivalence is verified
both statically (codex review of the message / aggregate functions) and
numerically (smoke test in the trainer's _reviews/).

Components
----------
- `cluster_head.PMPv1_6_Module` — top-level scoring module
- `cluster_head.WithinClusterMP`     — PyG MessagePassing layer
- `pmp_v1_6_trainer._PerModeEmerGNN_PMP_v1_6` — trainer

What's deliberately NOT included (vs v1.5 A)
--------------------------------------------
v1.5 A has dead-code components (cluster_attn_W, cluster_attn_w, g_mlp,
lambda_content) kept for future stop-grad experiments. v1.6 does NOT
include these dead components since they don't affect the output anyway.
This means v1.6 has slightly fewer parameters than v1.5 A (~2.6K fewer),
but identical OUTPUT for any input. Strict numerical equivalence still holds
when comparing on the components that are present in both.

What's deliberately the SAME (vs v1.5 A)
----------------------------------------
- mediator_embed (n_mediators, d), Xavier init
- type_embed (n_types, type_dim), Xavier init
- rel_embed (n_rels_plus_special, rel_dim), Xavier init
- W_T (n_clusters, n_clusters), Xavier init
- phi_mlp: (d + type_dim + 2*rel_dim) -> hidden -> d (lives inside WithinClusterMP)
- within_attn_W, within_attn_w (live inside WithinClusterMP)
- mlp_score: (2*d) -> hidden -> 1
- Cache reuse: PMP cache v1, v1.1 cluster cache v2 (no rebuild).
- EmerGNN backbone bypass: UNCHANGED from v1.5 A.

Inheritance / trainer
---------------------
v1.6 trainer inherits v1.5 trainer (which inherits v1.2). All cache
construction (per-drug per-cluster + intersection) is reused. Only the
`_build_aux_head` and `_combined_logit` are overridden to build / call the
GNN module. The per-drug per-cluster tensors are still built (inherited from
v1.5's inheritance chain) but they are NOT passed to the GNN forward — they
are dead-input slots, mirroring v1.5 A's dead-code components.

This package is a new directory. v1_1/, v1_2/, v1_3/, v1_4/, v1_5/ are
untouched.
"""

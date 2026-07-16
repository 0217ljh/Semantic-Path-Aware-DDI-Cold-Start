"""PMP v1.5 — sequential cluster-attention -> within-pool, with prior + content
residual attention (created 2026-06-03).

Architectural redesign of v1.2 to fix joint-training gradient competition.

Motivation
----------
v1.2 had Layer 1 (cluster path readout) and Layer 2 (within pool) in PARALLEL
with concat-then-MLP fusion. Ablations:
  v1.3 (L1 only) = 0.7414       v1.2 L1 diagnostic = 0.7016
  v1.4 (L2 only) = 0.7703       v1.2 L2 diagnostic = 0.7344
  v1.2 joint     = 0.7745
Each layer trained alone significantly OUTPERFORMED its joint-train
diagnostic (+3.6 pt for L1, +3.6 pt for L2). The naive concat fusion was
destroying ~3 pt of per-branch quality and gaining only ~0.4 pt over L2
alone — clear evidence of gradient competition over shared mediator_embed.

v1.5 redesign (sequential, Codex Option B)
-------------------------------------------
Layer 1 collapses to cluster-cluster attention ONLY (no separate
z_cluster_pair readout). The pair-conditional cluster representations
H_a^k, H_b^k ARE computed and feed into the attention LOGITS as a
**content residual on top of an affinity prior**:

    log alpha_{ij} ∝ log p_a[i] + log T[i, j] + log p_b[j]
                     + lambda * g(H_a^i, H_b^j)

`g` is a small 2-layer MLP (3d -> 16 -> 1). `lambda` is a learnable scalar.
Both g's last layer and lambda are zero-initialized so the model BEGINS as
pure affinity routing (equivalent to Option A) and learns to incorporate
content-conditional routing over training.

Layer 2 (within-pool) is unchanged in structure from v1.4 but is reweighted
by the Step 1 attention marginals (NOT by p_a[k] * p_b[k] as in v1.2/v1.4).

Locked v1.5 algorithm
---------------------
::

    Embeddings:
      mediator_embed  (n_mediators, d)         Xavier
      type_embed      (n_types, type_dim)
      rel_embed       (n_rels + 1, rel_dim)
      W_T             (n_clusters, n_clusters) Xavier; T = softmax(W_T)
      cluster_attn_W (d, d), cluster_attn_w (d, 1)    pair-conditional pool (same as v1.2)
      g_mlp (3d -> 16 -> 1)                          content residual; last layer ZERO-INIT
      lambda_content (scalar nn.Parameter)            INIT 0.0; learnable scalar
      phi_mlp, within_attn_W, within_attn_w           within-pool (same as v1.1/v1.2/v1.4)
      mlp_score (2*d -> hidden -> 1)                  concat[a-margin, b-margin] of z_within

    Step 1. content-aware cluster-cluster attention
      H_a[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(a)})  # pair-conditional
      H_b[:, k, :] = AttnPool({mediator_embed[m] for m in members(k) ∩ N(b)})

      log_p_a    = log_softmax(c_a)
      log_p_b    = log_softmax(c_b)
      log_T      = log_softmax(W_T, dim=-1)
      base_logit[i, j] = log_p_a[i] + log_T[i, j] + log_p_b[j]

      content_input[i, j] = concat[H_a^i, H_b^j, H_a^i ⊙ H_b^j]
      content_logit[i, j] = g_mlp(content_input)

      logit_ij = base_logit + lambda_content * content_logit
      alpha_ij = softmax over (i, j) of logit_ij                  # (B, K, K)

      attn_a^k = sum_j alpha_{k, j}                               # (B, K) row marginal
      attn_b^k = sum_i alpha_{i, k}                               # (B, K) col marginal

    Step 2. attention-marginal weighted within-pool
      for k in 0..n_clusters - 1:
        M_k = members(cluster_k) ∩ N_<=2(a) ∩ N_<=2(b)
        z_within^k = AttnPool({phi(m) for m in M_k}) or 0
          phi(m) = MLP([mediator_embed[m], type_embed[k],
                        rel_embed[a->m], rel_embed[m->b]])
      z_evidence_a = sum_k attn_a^k * z_within^k                  # (B, d)
      z_evidence_b = sum_k attn_b^k * z_within^k                  # (B, d)
      z_evidence   = concat[z_evidence_a, z_evidence_b]           # (B, 2*d)

    Score:
      score = MLP_score(z_evidence)
      loss  = BCE(sigmoid(score), y)

Components vs v1.2
------------------
KEPT (same structure):
  W_T, cluster_attn_W/w (now used in Step 1 attention logit, not psi)
  phi_mlp, within_attn_W/w (Step 2 within-pool)

NEW (vs v1.2 / v1.4):
  g_mlp + lambda_content (content residual in alpha)

REMOVED (vs v1.2):
  psi(i, j) construction
  h_path readout
  mlp_pair_cluster
  z_cluster_pair branch (no parallel branch)
  rel_dist_a, rel_dist_b at the forward (no r_a / r_b in psi, since psi gone)

Inheritance / cache
-------------------
v1.5 trainer inherits v1.2 trainer (needs per-drug per-cluster mediator map
for H_a^k, H_b^k). v1.1 cluster cache schema v2 is reused as-is.
EmerGNN backbone bypass: UNCHANGED.

This package is a new directory. v1_1/, v1_2/, v1_3/, v1_4/ are untouched.
"""

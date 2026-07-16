"""PMP v1 — Pair-Mediator Pooling for cold-start DDI prediction.

Self-contained v1 module. Future versions live as sibling folders
(pmp_v2/, pmp_v3/, ...) so each version is independently inspectable.

Algorithm 1 (locked pseudocode):
  1. M = N_<=2(a) intersect N_<=2(b)   (typed common neighbors, non-drug)
  2. if M empty: z = 0
  3. else: phi(m) = MLP([h_m ; e_type(m) ; e_rel(a,m) ; e_rel(m,b)])
          alpha = softmax(w^T tanh(W phi(m)))
          z = sum_m alpha_m * phi_m
  4. l_base = EmerGNN_logit(a, b)          (unchanged from MNAH parent)
  5. l_pmp  = MLP_score(z)
  6. y_hat  = sigmoid(l_base + softplus(raw_beta) * l_pmp)

Minimal change from MNAH (verified 0.7670 S2): swap MNAH's 22-d scalar count
head for Layer 2 attention pool over typed mediator embeddings. EmerGNN
backbone + shuffle_train(mode='S2') + residual fusion all unchanged.

Canonical import path:
    from my_code.models.pmp_v1.pmp_trainer import _PerModeEmerGNN_PMP, PMPHead, DEFAULT_PMP_CACHE
    from my_code.models.pmp_v1.precompute_pmp_cache import build_pmp_cache
"""
from __future__ import annotations

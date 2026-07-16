"""Bipartite invariance core — the ACTUAL general-graph bipartition (FGCN-DKS-style,
Wang et al. ICLR 2026), NOT the earlier scalar-gate anchor.

Idea (Notes/Ideas/idea_stable_predictive_vs_spurious.md): keep the FULL shared support
in BOTH branches; decompose each mediator message h_i relative to the pair prototype
mu_p into a COMMON/redundant part h_i^|| (projection) and a DISTINCT/marginal part
h_i^perp (residual). Two within-type attention pools:
    z_C = pool(h_i^||)                 (common / over-general channel)
    z_D = pool(m_i * h_i^perp)         (distinct / mechanism channel), m_i = redundancy gate
Both are routed by W_T and flattened. The core returns cat[z_C, z_D, struct_feats] so the
existing DecomposedStandaloneHead scorer acts as the joint readout f_J (nothing is removed
from prediction -> unlike the hard prune). The per-branch pooled reps are cached on the
module (`_pooled_C`, `_pooled_D`) so the runner can add the invariance-training criteria:
  - GRL label-nulling on z_C   (make the common channel label-UNINFORMATIVE: I(z_C;y)~0)
  - cross-view invariance      (z_C, z_D stable across the OOD support-degradation views)
  - orthogonality separation   (cos^2(z_C, z_D) -> 0)
This mirrors FGCN-DKS's mask separation (Eq 1-4) + invariance loss (Eq 5-8) but refines
"invariant vs variant" into "stable-predictive vs stable-redundant" for cold-start DDI.

Redundancy descriptors for the gate m_i (structural-only, NO endpoint identity -> no leak):
  r_sig     = within-pair (type,rel_a,rel_b) signature duplication
  r_rep_loo = cos(h_i, mu_{p,-i})   leave-one-out prototype cosine (high=generic)

Subclasses StructuralVariableCore, duplicates its forward with the bipartite additions,
leaving the parent UNCHANGED (same convention as AwareStructuralCore).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from .batch import SupportBatch
from .core import StructuralVariableCore


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class LabelAdversary(nn.Module):
    """Small classifier q_psi(y | z_C) used with a gradient-reversal layer so the
    encoder is pushed to make z_C label-uninformative (DANN-style)."""

    def __init__(self, in_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z_c: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(z_c, lambd)).squeeze(-1)


class BipartiteAwareCore(StructuralVariableCore):
    def __init__(
        self, *args,
        gate_eps: float = 0.05,
        gate_hidden: int = 16,
        n_rel_buckets: int = 11,
        **kwargs,
    ) -> None:
        super().__init__(*args, n_rel_buckets=n_rel_buckets, **kwargs)
        self.gate_eps = float(gate_eps)
        self.n_rel_buckets = int(n_rel_buckets)
        # pair-prototype attention vector u (for mu_p)
        self.proto_u = nn.Linear(self.d, 1, bias=False)
        # redundancy gate m_i = eps + (1-eps) sigmoid(mlp([r_sig, r_rep_loo]))
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
        #: doubled pooled block (z_C ++ z_D) + struct.
        self.out_dim = 2 * self.n_types * self.n_chan * self.d + int(
            self.out_dim - self.n_types * self.n_chan * self.d)
        self._pooled_C: torch.Tensor | None = None
        self._pooled_D: torch.Tensor | None = None
        self._last_m: torch.Tensor | None = None

    def forward(self, batch: SupportBatch) -> torch.Tensor:
        B, K, d, R = int(batch.n_pairs), self.n_types, self.d, self.n_chan
        device = batch.device
        struct = batch.struct_feats
        if batch.med_id.numel() == 0:
            zc = torch.zeros(B, K * R * d, device=device)
            zd = torch.zeros(B, K * R * d, device=device)
            self._pooled_C, self._pooled_D, self._last_m = zc, zd, None
            return torch.cat([zc, zd, struct], dim=-1)

        feats = []
        if self.use_entity_embed:
            feats.append(self.entity_embed(batch.med_id))
        feats += [self.type_embed(batch.type_idx),
                  self.rel_embed(batch.rel_a), self.rel_embed(batch.rel_b)]
        if self.use_absdiff_embed:
            ad = (batch.d_a - batch.d_b).abs().clamp_(max=self.n_absdiff_buckets - 1)
            feats.append(self.absdiff_embed(ad))
        h = self.phi_mlp(torch.cat(feats, dim=-1))               # (M, d) per-mediator h_i

        pair = batch.pair_idx
        # pair prototype mu_p = sum_i softmax_pair(u.h_i) h_i
        pscore = self.proto_u(h).squeeze(-1)                     # (M,)
        palpha = pyg_softmax(pscore, pair, num_nodes=B)
        mu = torch.zeros(B, d, device=device)
        mu.scatter_add_(0, pair.unsqueeze(-1).expand(-1, d), palpha.unsqueeze(-1) * h)
        mu_i = mu[pair]                                          # (M, d)
        # parallel (common) + perpendicular (distinct) decomposition
        denom = (mu_i * mu_i).sum(-1, keepdim=True) + 1e-6
        coef = (h * mu_i).sum(-1, keepdim=True) / denom
        h_par = coef * mu_i                                     # (M, d) common
        h_perp = h - h_par                                     # (M, d) distinct

        # redundancy descriptors (structural, no identity)
        nrb = self.n_rel_buckets
        ones = torch.ones_like(pair, dtype=torch.float32)
        size = torch.zeros(B, device=device).scatter_add_(0, pair, ones)
        size_i = size[pair]
        sig = (batch.type_idx.long() * nrb + batch.rel_a.long()) * nrb + batch.rel_b.long()
        key = pair.long() * (self.n_types * nrb * nrb) + sig
        uniq, inv, cnt = torch.unique(key, return_counts=True, return_inverse=True)
        r_sig = (cnt[inv].float() - 1.0) / (size_i - 1.0).clamp(min=1.0)
        sum_h = torch.zeros(B, d, device=device)
        sum_h.scatter_add_(0, pair.unsqueeze(-1).expand(-1, d), h)
        mu_loo = (sum_h[pair] - h) / (size_i - 1.0).clamp(min=1.0).unsqueeze(-1)
        r_rep = F.cosine_similarity(h, mu_loo, dim=-1, eps=1e-6)
        single = size_i <= 1.0
        r_sig = torch.where(single, torch.zeros_like(r_sig), r_sig)
        r_rep = torch.where(single, torch.zeros_like(r_rep), r_rep)
        m_gate = self.gate_eps + (1.0 - self.gate_eps) * torch.sigmoid(
            self.gate_mlp(torch.stack([r_sig, r_rep], dim=-1)).squeeze(-1))  # (M,)
        self._last_m = m_gate

        # within-(pair,type,chan) attention groups (as parent)
        if R == 2:
            asym_bit = (batch.d_a != batch.d_b).long()
        else:
            asym_bit = torch.zeros_like(batch.type_idx)
        group = pair * (K * R) + batch.type_idx * R + asym_bit
        s = self.within_attn_w(torch.tanh(self.within_attn_W(h))).squeeze(-1)   # (M,)
        if self.use_dist_attn:
            da_i = batch.d_a.clamp(0, self.max_dist); db_i = batch.d_b.clamp(0, self.max_dist)
            s = s + self.dist_bias[da_i, db_i]
        aC = pyg_softmax(s, group, num_nodes=B * K * R)
        aD = pyg_softmax(s + torch.log(m_gate), group, num_nodes=B * K * R)
        cC = torch.zeros(B * K * R, d, device=device)
        cD = torch.zeros(B * K * R, d, device=device)
        gexp = group.unsqueeze(-1).expand(-1, d)
        cC.scatter_add_(0, gexp, aC.unsqueeze(-1) * h_par)
        cD.scatter_add_(0, gexp, aD.unsqueeze(-1) * (m_gate.unsqueeze(-1) * h_perp))

        transition = F.softmax(self.W_T, dim=-1)
        cC = torch.einsum("kj,bjrd->bkrd", transition, cC.view(B, K, R, d)).reshape(B, K * R * d)
        cD = torch.einsum("kj,bjrd->bkrd", transition, cD.view(B, K, R, d)).reshape(B, K * R * d)
        self._pooled_C, self._pooled_D = cC, cD
        return torch.cat([cC, cD, struct], dim=-1)


__all__ = ["BipartiteAwareCore", "LabelAdversary", "grad_reverse"]

"""Stage 4 / DDI-LoRA adapter - M_A (assignment) and M_B (prototypes) submodules
plus the orchestrating adapter. Paper Method: tex 337-443.

M_A and M_B are separate submodules (different params, inits, roles; independent
RQ2 ablation toggles; separate RQ3 param-space exposure):
  * MediatorAssignment (M_A side): holds {U_t, W_A, W_B}; COMPUTES the block-sparse
    coefficients beta_uvm (never materialized as a dense N x a matrix).
  * TypedPrototypes (M_B side): holds the trainable prototype bank B_t (= M_B) and
    per-type residual scale gamma_t; produces the per-mediator residual gamma*beta*B_t.

STATUS: step-1 contract. assign/proto forwards are implemented; the DDILoRAAdapter
read-out (per-(pair,type) attention pooling -> z_uv -> MLP+Head) is stubbed for step 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import AdapterBatch
from .config import AdapterConfig


class MediatorAssignment(nn.Module):
    """M_A side. beta_uvm = softmax(node_m + relation_uvm) over the type's lam prototypes.
      node_m       = U_{t(m)} z_m                       (U: [T, lam, d_z])
      relation_uvm = W_A r_tilde_a(m) + W_B r_tilde_b(m)  (W_A, W_B: [lam, d_z], W_A != W_B)
    U_t is warm-started from k-means centroids of z_m within type (step 5); here it is
    randomly initialized. W_A/W_B use standard init (distinct -> encode directionality)."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.U = nn.Parameter(torch.empty(cfg.n_types, cfg.lam, cfg.d_z))
        self.W_A = nn.Parameter(torch.empty(cfg.lam, cfg.d_z))
        self.W_B = nn.Parameter(torch.empty(cfg.lam, cfg.d_z))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.U, std=0.02)          # replaced by k-means warm-start (step 5)
        nn.init.xavier_uniform_(self.W_A)
        nn.init.xavier_uniform_(self.W_B)          # distinct draw -> W_A != W_B
        with torch.no_grad():                      # keep the relation arm from swamping U z_m
            self.W_A.mul_(self.cfg.rel_init_scale)
            self.W_B.mul_(self.cfg.rel_init_scale)

    @torch.no_grad()
    def warmstart_from_centroids(self, centroids: torch.Tensor, tau: float) -> None:
        """(U_t^0)_k = c_{t,k} / tau: set U from per-type k-means centroids of z_m
        (paper warm-start). centroids: [T, lam, d_z]; overwrites the random init."""
        if centroids.shape != self.U.shape:
            raise ValueError(f"centroids {tuple(centroids.shape)} != U {tuple(self.U.shape)}")
        self.U.data.copy_(centroids.to(self.U) / float(tau))

    def forward(self, z_m: torch.Tensor, r_a: torch.Tensor, r_b: torch.Tensor,
                med_type: torch.Tensor) -> torch.Tensor:
        """(z_m,r_a,r_b: [M, d_z], med_type: [M]) -> beta: [M, lam] (softmax over lam).
        r_a/r_b are the ARM features (typed meta-path p_a, or z_r-text r_tilde, or zeros)."""
        rel = torch.einsum("ld,md->ml", self.W_A, r_a) + torch.einsum("ld,md->ml", self.W_B, r_b)
        if self.cfg.use_node:                                      # node term (ablatable)
            rel = rel + torch.einsum("mld,md->ml", self.U[med_type], z_m)
        return F.softmax(rel, dim=-1)                              # [M, lam]


class PathwayFeature(nn.Module):
    """Typed meta-path arm feature p_a(m) from (drug-side relation r1, intermediate type
    t(x)) atom counts (Phase 1; codex-locked). Replaces the frozen-z_r r_tilde.
      p_a(m) = LayerNorm( sum_{r,t} w[m,r,t] E[r,t] / sum_{r,t} w[m,r,t] )
      w = A * rho_col          (rho_col: rho^2 for the len-2 type cols, rho for DIRECT)
    E [R, T+1, d_z] = the LEARNED low-capacity atom table; it IS the per-(relation, type)
    contribution (relation is the row index), so no separate relation scalar is used - a
    scalar gate would be unidentifiable (absorbable into E scale, washed by LayerNorm on
    single-atom mediators; codex). The PK/PD mechanism map is read from E's routing
    W_A E[r,t] (adapter_state), not from a scalar. den=0 -> p=0."""

    def __init__(self, n_rel: int, n_types: int, d_z: int, rho: float):
        super().__init__()
        self.n_rel = n_rel; self.n_types = n_types; self.d_z = d_z
        self.E = nn.Parameter(torch.empty(n_rel, n_types + 1, d_z))
        nn.init.normal_(self.E, std=0.1)
        col = torch.full((n_types + 1,), float(rho) ** 2)              # len-2 type columns
        col[n_types] = float(rho)                                      # DIRECT (len-1) column
        self.register_buffer("rho_col", col)

    def forward(self, A: torch.Tensor) -> torch.Tensor:               # A [M, R, T+1] -> [M, d_z]
        w = A * self.rho_col[None, None, :]                          # [M, R, T+1]
        num = torch.einsum("mrt,rtd->md", w, self.E)                  # [M, d_z]
        tot = w.sum(dim=(1, 2))                                       # [M]
        p = F.layer_norm(num / tot.clamp_min(1e-9).unsqueeze(-1), (self.d_z,))
        return torch.where((tot > 0).unsqueeze(-1), p, torch.zeros_like(p))


class TypedPrototypes(nn.Module):
    """M_B side. The trainable prototype bank B_t (= M_B, [T, lam, d]) + per-type residual
    scale gamma_t. residual(m) = gamma_{t(m)} * (beta_uvm @ B_{t(m)}) in R^d.
    B random init N(0, 1/lam); gamma init eps (small -> adapter starts near H_base)."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.B = nn.Parameter(torch.empty(cfg.n_types, cfg.lam, cfg.d))   # = M_B
        self.gamma = nn.Parameter(torch.full((cfg.n_types,), float(cfg.eps)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.B, std=(1.0 / self.cfg.lam) ** 0.5)         # N(0, 1/lam)
        with torch.no_grad():
            self.gamma.fill_(float(self.cfg.eps))

    def residual(self, beta: torch.Tensor, med_type: torch.Tensor) -> torch.Tensor:
        """(beta: [M, lam], med_type: [M]) -> residual: [M, d] = gamma_t * (beta @ B_t)."""
        B_t = self.B[med_type]                                   # [M, lam, d]
        r = torch.einsum("ml,mld->md", beta, B_t)               # [M, d]
        return self.gamma[med_type].unsqueeze(-1) * r           # [M, d]


class DDILoRAAdapter(nn.Module):
    """Orchestrates M_A + M_B + read-out. forward(AdapterBatch) -> (s_uv, beta).
    n_uvm = h^base_m + gamma_t beta_uvm B_t; per-(pair,type) attention pool -> g_t;
    z_uv = [g_1||..||g_T] in R^{T d}; s_uv = Head(MLP(z_uv)). beta returned for the
    entropy regularizer."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.assign = MediatorAssignment(cfg)      # M_A
        self.proto = TypedPrototypes(cfg)          # M_B
        self.attn = nn.Linear(cfg.d, 1)            # per-mediator attention logit a_uvm
        self.aa_gate = nn.Parameter(torch.ones(()))  # learnable gate on the Adamic-Adar prior
        #                     (init 1 -> starts AA-warm-started; training can decay it to 0
        #                      so the learned attention takes over -> a warm-start, not a fixed bias)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_types * cfg.d, cfg.mlp_hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout))
        self.head = nn.Linear(cfg.mlp_hidden, cfg.n_out)

    def mediator_residual(self, batch: AdapterBatch):
        """Shared M_A->M_B path. Returns (n_uvm [M, d], beta [M, lam]).
        M_B ablation (cfg.use_mb=False): drop the typed-prototype residual -> h_base only."""
        beta = self.assign(batch.z_m, batch.r_a, batch.r_b, batch.med_type)   # [M, lam]
        n = batch.h_base
        if self.cfg.use_mb:
            n = n + self.proto.residual(beta, batch.med_type)                 # [M, d]
        return n, beta

    def pool(self, n: torch.Tensor, batch: AdapterBatch) -> torch.Tensor:
        """Per-(pair,type) attention pooling -> z_uv [B, T*d].
        g_t = sum_m w_uvm n_uvm with w = softmax(a_uvm) within each (pair, type) group
        (group id = pair_idx*T + t(m)). Missing (pair,type) groups stay all-zero.
        a_uvm = learned attention logit (+ optional Adamic-Adar prior batch.aa_weight)."""
        B, T, d = batch.n_pairs, self.cfg.n_types, self.cfg.d
        n_groups = B * T
        if n.shape[0] == 0:                                   # M == 0: empty support
            return n.new_zeros(B, T * d)
        gid = batch.pair_idx * T + batch.med_type             # [M] in [0, B*T)
        a = self.attn(n).squeeze(-1)                          # [M] attention logit
        if batch.aa_weight is not None:
            a = a + self.aa_gate * batch.aa_weight            # AA warm-start prior (gated)
        gmax = n.new_full((n_groups,), float("-inf")).scatter_reduce(
            0, gid, a, reduce="amax", include_self=True)      # per-group max (stability)
        e = (a - gmax[gid]).exp()                             # [M]
        gsum = n.new_zeros(n_groups).index_add(0, gid, e)     # [B*T]
        w = e / gsum[gid].clamp_min(1e-12)                    # [M] softmax weights
        g = n.new_zeros(n_groups, d).index_add(0, gid, w.unsqueeze(-1) * n)   # [B*T, d]
        return g.reshape(B, T * d)                            # z_uv

    def forward(self, batch: AdapterBatch):
        """AdapterBatch -> (s_uv: [n_pairs, n_out], beta: [M, lam]).
        n_uvm -> per-(pair,type) pool -> z_uv=[g_1||..||g_T] -> Head(MLP(z_uv)).
        beta is returned for the entropy regularizer (computed in adapter.losses)."""
        n, beta = self.mediator_residual(batch)               # [M, d], [M, lam]
        z_uv = self.pool(n, batch)                            # [B, T*d]
        s_uv = self.head(self.mlp(z_uv))                      # [B, n_out]
        return s_uv, beta

    def pair_repr(self, batch: AdapterBatch):
        """The pooled adapter pair representation z_uv [B, T*d] (BEFORE MLP+Head).
        Under a ZeroHBaseProvider (h_base=0) this is the PURE M_A.M_B semantic signal
        the backbone composer projects + adds as a correction. Returns (z_uv, beta)."""
        n, beta = self.mediator_residual(batch)               # [M, d], [M, lam]
        return self.pool(n, batch), beta                      # [B, T*d], [M, lam]

    @property
    def z_dim(self) -> int:
        """Width of z_uv (= T * d)."""
        return self.cfg.n_types * self.cfg.d

    @torch.no_grad()
    def adapter_state(self) -> dict:
        """RQ2/RQ3 param-space hooks: the learned M_B prototypes + M_A maps."""
        return {"M_B": self.proto.B.detach().cpu(), "gamma": self.proto.gamma.detach().cpu(),
                "U": self.assign.U.detach().cpu(),
                "W_A": self.assign.W_A.detach().cpu(), "W_B": self.assign.W_B.detach().cpu()}


__all__ = ["MediatorAssignment", "TypedPrototypes", "DDILoRAAdapter"]

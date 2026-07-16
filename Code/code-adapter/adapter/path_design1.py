"""Path Design 1 - factorized node-family x route-variant adapter (in M_A.M_B, unsupervised).

The DDI mechanism = HOW the two drugs JOINTLY reach a shared mediator, not just WHAT they
share. So the assignment factorizes (codex-locked, round 3):
    alpha_{m,k} = softmax_k( <U_{t,k}, z_m> / tau )         node semantics -> mediator FAMILY k
    c_{m,k,h}   = LSE( <P r_a, Q r_b>, <P r_b, Q r_a> )/sqrt(q)   symmetric bilinear route detector
    pi_{m,k,h}  = softmax_h( eta * c_{m,k,h} )              joint route motif -> mechanism VARIANT h
    beta_{m,k,h}= alpha_{m,k} * pi_{m,k,h}
    n_m = h^base_m + gamma_t * sum_{k,h} beta_{m,k,h} B_{t,k,h}   variant chooses M_B CONTENT
node picks the family (low-capacity, cannot resolve within-family mechanism); the route
CONJUNCTION picks the mechanism variant, which changes what the mediator contributes. This
"division of labor" is what lets path escape the additive-absorption trap. Fully unsupervised
and inside M_A.M_B (no parallel branch). Mechanism (PK/PD) alignment is read POST-HOC.

The equivalent "path" is the connecting typed meta-path  u -[route_a]-> m <-[route_b]- v ;
the bilinear conjunction assembles the two half-arms into one pair-connecting path whose
variant is its mechanism. r_a/r_b are top-2 sparse mixtures over 611 route tokens
(47 direct relations + 47*12 (relation, intermediate-type) indirect tokens).

Ablations (prove node AND path each help, and TOGETHER > either alone):
  node_on=True,  path_on=True   -> full (alpha x pi)
  node_on=True,  path_on=False  -> node-only  (pi uniform 1/H)
  node_on=False, path_on=True   -> path-only  (alpha uniform 1/K)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import AdapterBatch


class FactorizedAdapter(nn.Module):
    def __init__(self, n_types: int, d_z: int, d: int, n_out: int, n_route_tokens: int,
                 K: int = 3, H: int = 2, q: int = 8, d_r: int = 32, tau: float = 1.0,
                 eta: float = 0.1, mlp_hidden: int = 256, dropout: float = 0.2,
                 gamma_init: float = 1.0, node_on: bool = True, path_on: bool = True):
        super().__init__()
        self.T, self.d_z, self.d, self.K, self.H, self.q, self.d_r = n_types, d_z, d, K, H, q, d_r
        self.tau, self.eta = float(tau), float(eta)
        self.node_on, self.path_on = bool(node_on), bool(path_on)
        # M_A side: node family map U, route bilinear detectors P/Q, route-token table
        self.U = nn.Parameter(torch.empty(n_types, K, d_z))            # node -> family
        self.E_tok = nn.Parameter(torch.empty(n_route_tokens, d_r))    # route token embeddings
        self.P = nn.Parameter(torch.empty(n_types, K, H, q, d_r))
        self.Q = nn.Parameter(torch.empty(n_types, K, H, q, d_r))
        # M_B side: factorized typed prototypes (family x variant) + residual scale
        self.B = nn.Parameter(torch.empty(n_types, K, H, d))
        self.gamma = nn.Parameter(torch.full((n_types,), float(gamma_init)))
        # read-out (per-(pair,type) attention pool -> MLP -> head), same as the flat adapter
        self.attn = nn.Linear(d, 1)
        self.aa_gate = nn.Parameter(torch.ones(()))
        self.mlp = nn.Sequential(nn.Linear(n_types * d, mlp_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(mlp_hidden, n_out)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.U, std=0.02)                 # replaced by k-means warm-start
        nn.init.normal_(self.E_tok, std=1.0)              # r = LayerNorm(sum) rescales anyway
        # P/Q so that <P r_a, Q r_b>/sqrt(q) ~ O(1) (r is LayerNorm'd -> elems ~N(0,1)):
        # need P elems ~ N(0, 1/d_r). NOT downscaled - route picks variant h on a SEPARATE
        # axis from node's family k, so it cannot swamp the node term; too-small init instead
        # leaves the route dormant (c~0 -> pi uniform -> no gradient).
        std_pq = (1.0 / self.d_r) ** 0.5
        nn.init.normal_(self.P, std=std_pq); nn.init.normal_(self.Q, std=std_pq)
        for t in range(self.T):                           # B: diverse across variants within (t,k)
            for k in range(self.K):
                nn.init.orthogonal_(self.B[t, k]) if self.H <= self.d else nn.init.normal_(self.B[t, k], std=(1.0 / self.H) ** 0.5)

    @torch.no_grad()
    def warmstart_U(self, centroids_K: torch.Tensor, tau: float) -> None:
        """centroids_K: [T, K, d_z] per-type k-means (K clusters) -> U = c / tau."""
        if centroids_K.shape != self.U.shape:
            raise ValueError(f"centroids {tuple(centroids_K.shape)} != U {tuple(self.U.shape)}")
        self.U.data.copy_(centroids_K.to(self.U) / float(tau))

    def _route(self, tok_idx: torch.Tensor, tok_wt: torch.Tensor) -> torch.Tensor:
        """(tok_idx [M,2] long, tok_wt [M,2]) -> r [M, d_r] = LayerNorm(sum wt * E_tok[idx])."""
        emb = self.E_tok[tok_idx]                         # [M, 2, d_r]
        r = (emb * tok_wt.unsqueeze(-1)).sum(1)           # [M, d_r]
        return F.layer_norm(r, (self.d_r,))

    def assign(self, z_m, med_type, r_a, r_b):
        """-> (beta [M,K,H], alpha [M,K], pi [M,K,H])."""
        M = z_m.shape[0]
        # alpha: node family
        node_logit = torch.einsum("mkd,md->mk", self.U[med_type], z_m) / self.tau   # [M,K]
        alpha = F.softmax(node_logit, dim=-1) if self.node_on else node_logit.new_full((M, self.K), 1.0 / self.K)
        # c: symmetric bilinear route conjunction detector
        P = self.P[med_type]; Q = self.Q[med_type]        # [M,K,H,q,d_r]
        Pa = torch.einsum("mkhqd,md->mkhq", P, r_a); Qb = torch.einsum("mkhqd,md->mkhq", Q, r_b)
        Pb = torch.einsum("mkhqd,md->mkhq", P, r_b); Qa = torch.einsum("mkhqd,md->mkhq", Q, r_a)
        s = 1.0 / math.sqrt(self.q)
        c = torch.logsumexp(torch.stack([(Pa * Qb).sum(-1) * s, (Pb * Qa).sum(-1) * s], 0), 0)  # [M,K,H]
        pi = F.softmax(self.eta * c, dim=-1) if self.path_on else c.new_full((M, self.K, self.H), 1.0 / self.H)
        return alpha.unsqueeze(-1) * pi, alpha, pi

    def forward(self, batch: AdapterBatch):
        B, T, d = batch.n_pairs, self.T, self.d
        if batch.n_mediators == 0:
            z = batch.pair_idx.new_zeros(B, T * d, dtype=torch.float32)
            return self.head(self.mlp(z)), {}
        r_a = self._route(batch.tok_a_idx, batch.tok_a_wt)
        r_b = self._route(batch.tok_b_idx, batch.tok_b_wt)
        beta, alpha, pi = self.assign(batch.z_m, batch.med_type, r_a, r_b)   # [M,K,H]
        Bt = self.B[batch.med_type]                                          # [M,K,H,d]
        resid = self.gamma[batch.med_type].unsqueeze(-1) * torch.einsum("mkh,mkhd->md", beta, Bt)
        n = batch.h_base + resid                                            # [M,d]
        z_uv = self._pool(n, batch)
        s_uv = self.head(self.mlp(z_uv))
        aux = {"alpha": alpha, "pi": pi, "beta": beta,
               "reg": self._reg(beta)}                                      # load-balance + diversity
        return s_uv, aux

    def _pool(self, n, batch):
        B, T, d = batch.n_pairs, self.T, self.d
        gid = batch.pair_idx * T + batch.med_type
        ng = B * T
        a = self.attn(n).squeeze(-1)
        if batch.aa_weight is not None:
            a = a + self.aa_gate * batch.aa_weight
        gmax = n.new_full((ng,), float("-inf")).scatter_reduce(0, gid, a, reduce="amax", include_self=True)
        e = (a - gmax[gid]).exp()
        gsum = n.new_zeros(ng).index_add(0, gid, e)
        w = e / gsum[gid].clamp_min(1e-12)
        g = n.new_zeros(ng, d).index_add(0, gid, w.unsqueeze(-1) * n)
        return g.reshape(B, T * d)

    def _reg(self, beta):
        # load-balance on EFFECTIVE usage (codex): within each family k, variants h should be
        # used ~1/H, weighted by the family's actual usage so dead families don't count.
        u = beta.mean(0)                                  # [K,H] effective usage
        a = u.sum(-1, keepdim=True).clamp_min(1e-9)       # [K,1] family usage
        lb = (a.squeeze(-1) * ((u / a - 1.0 / self.H) ** 2).sum(-1)).sum()
        # diversity: variants within (t,k) should be distinct directions in M_B
        Bn = F.normalize(self.B, dim=-1)                  # [T,K,H,d]
        if self.H >= 2:
            cos = torch.einsum("tkhd,tkgd->tkhg", Bn, Bn)
            eye = torch.eye(self.H, device=Bn.device)
            div = (cos * (1 - eye)).abs().mean()          # push variants apart
        else:
            div = Bn.new_zeros(())
        return lb + div

    @torch.no_grad()
    def mediator_states(self, batch: AdapterBatch):
        """Analysis hook (no training path change): per-mediator (n_m adapted, h_base) and
        (beta, alpha, pi). n_m = h_base + gamma*sum beta B is M_plug's mediator rep; h_base
        is M_bb's. Mirrors forward() exactly without pooling."""
        if batch.n_mediators == 0:
            z = batch.pair_idx.new_zeros(0, self.d)
            return {"n_m": z, "h_base": z}
        r_a = self._route(batch.tok_a_idx, batch.tok_a_wt)
        r_b = self._route(batch.tok_b_idx, batch.tok_b_wt)
        beta, alpha, pi = self.assign(batch.z_m, batch.med_type, r_a, r_b)
        Bt = self.B[batch.med_type]
        resid = self.gamma[batch.med_type].unsqueeze(-1) * torch.einsum("mkh,mkhd->md", beta, Bt)
        return {"n_m": batch.h_base + resid, "h_base": batch.h_base,
                "beta": beta, "alpha": alpha, "pi": pi}

    @torch.no_grad()
    def mechanism_state(self) -> dict:
        """Post-hoc emergent-alignment hooks: prototype content + route detectors."""
        return {"B": self.B.detach().cpu(), "U": self.U.detach().cpu(),
                "P": self.P.detach().cpu(), "Q": self.Q.detach().cpu(),
                "E_tok": self.E_tok.detach().cpu()}

    # -- AdapterBackboneComposer (design R) seam -----------------------------
    # The composer treats the adapter as a z_uv factory: it calls pair_repr() to get the
    # pooled PURE M_A.M_B pair rep (h_base=0 under ZeroHBaseProvider), projects it with W/LN,
    # and adds it as a low-rank correction to the backbone pair rep. beta is returned 3-D
    # [M,K,H] so the composer can apply aux_reg (pd1 has NO flat beta-entropy; its regularizer
    # is the load-balance + B-diversity term, applied by the composer via aux_reg).
    @property
    def z_dim(self) -> int:
        """Width of the pooled pair rep z_uv (= T * d)."""
        return self.T * self.d

    def pair_repr(self, batch: AdapterBatch):
        """-> (z_uv [B, T*d], beta [M,K,H]). Mirrors forward() up to _pool (no mlp/head).
        Under the composer's ZeroHBaseProvider h_base=0, so z_uv is the pure M_A.M_B signal."""
        B, T, d = batch.n_pairs, self.T, self.d
        if batch.n_mediators == 0:                            # empty-support guard (codex)
            z = batch.pair_idx.new_zeros(B, T * d, dtype=torch.float32)
            beta = batch.pair_idx.new_zeros(0, self.K, self.H, dtype=torch.float32)
            return z, beta
        r_a = self._route(batch.tok_a_idx, batch.tok_a_wt)
        r_b = self._route(batch.tok_b_idx, batch.tok_b_wt)
        beta, _alpha, _pi = self.assign(batch.z_m, batch.med_type, r_a, r_b)
        Bt = self.B[batch.med_type]
        resid = self.gamma[batch.med_type].unsqueeze(-1) * torch.einsum("mkh,mkhd->md", beta, Bt)
        n = batch.h_base + resid
        return self._pool(n, batch), beta

    def aux_reg(self, beta: torch.Tensor) -> torch.Tensor:
        """Public seam: the pd1 load-balance + B-diversity regularizer (the aux['reg'] term
        forward() emits). The composer adds reg_weight * aux_reg(beta) to its loss."""
        if beta.numel() == 0:                                 # empty minibatch -> no assignment
            return beta.new_zeros(())
        return self._reg(beta)

    @torch.no_grad()
    def adapter_state(self) -> dict:
        """RQ2/RQ3 param-space hooks (composer.adapter_state calls this). Parity with the flat
        adapter (M_B/U) plus the pd1 route detectors (P/Q/E_tok), matching mechanism_state."""
        return {"M_B": self.B.detach().cpu(), "gamma": self.gamma.detach().cpu(),
                "U": self.U.detach().cpu(), "P": self.P.detach().cpu(),
                "Q": self.Q.detach().cpu(), "E_tok": self.E_tok.detach().cpu()}


__all__ = ["FactorizedAdapter"]

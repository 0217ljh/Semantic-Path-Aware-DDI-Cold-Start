"""GatedAwareStructuralCore — StructuralVariableCore + a learned, structural-only
mediator distinctiveness gate (anchor / control for the stable-predictive vs
stable-spurious idea, see Notes/Ideas/idea_stable_predictive_vs_spurious.md).

ANCHOR scope (codex plan step 1): a single pooled readout (no parallel/residual
split, no GRL adversary yet). A soft gate ``m_i in (eps, 1)`` is injected as an
additive ``+ log(m_i)`` bias into the within-type attention logits BEFORE the
softmax, so it competes with the other mediators of the same (pair, type) group.
This subclasses ``StructuralVariableCore`` and duplicates its forward body verbatim
except for the one gate-injection line, honouring the no-modify-existing rule
(same pattern as ``AwareStructuralCore``).

The gate input is structural-ONLY (no drug identity, no labels -> no leakage). The
decisive sub-ablation is the gate INPUT set:
  * ``"degree"``  : m_i = f(standardised log1p(deg_i))             -- learned hub down-weight
  * ``"redund"``  : m_i = f(r_sig, r_rep_loo)                       -- redundancy only
  * ``"full"``    : m_i = f(standardised log1p(deg_i), r_sig, r_rep_loo)
If ``degree`` ~= ``full`` the gate has reduced to the corridor ``-log1p(deg)`` term;
if ``redund``/``full`` win, there is path-redundancy signal beyond degree.

Redundancy descriptors (path-set properties, NOT a per-node scalar):
  * ``r_sig``     = within-pair signature duplication: fraction of co-mediators in
    the SAME pair support sharing this mediator's (type, rel_a, rel_b) signature.
  * ``r_rep_loo`` = leave-one-out prototype cosine cos(h_i, mu_{p,-i}); high = the
    mediator is well explained by the rest of the bundle (generic), low = it carries
    a distinctive marginal direction.

Zero-init-ish invariant: the gate's final linear is zero-weight + positive bias so
that at start ``m_i ~= 1`` (``log m_i ~= 0``) and the core ~reproduces its parent.
This lets a run FINE-TUNE from a locked support-degradation checkpoint with the gate
near-off, then learn the gating from there.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from .batch import SupportBatch
from .core import StructuralVariableCore

#: gate-input modes (the decisive degree-vs-redundancy ablation).
GATE_INPUTS: tuple[str, ...] = ("degree", "redund", "full")


class GatedAwareStructuralCore(StructuralVariableCore):
    def __init__(
        self, *args,
        gate_input: str = "full",
        gate_hidden: int = 16,
        gate_eps: float = 0.05,
        deg_log: torch.Tensor | None = None,
        deg_mu: float = 0.0,
        deg_sigma: float = 1.0,
        n_rel_buckets: int = 11,
        **kwargs,
    ) -> None:
        super().__init__(*args, n_rel_buckets=n_rel_buckets, **kwargs)
        if gate_input not in GATE_INPUTS:
            raise ValueError(f"gate_input must be one of {GATE_INPUTS}, got {gate_input!r}")
        self.gate_input = str(gate_input)
        self.gate_eps = float(gate_eps)
        self.n_rel_buckets = int(n_rel_buckets)
        if deg_log is None:
            raise ValueError("GatedAwareStructuralCore requires deg_log (per-node log1p degree)")
        self.register_buffer("deg_log", torch.as_tensor(deg_log, dtype=torch.float32))
        self.register_buffer("deg_mu", torch.tensor(float(deg_mu)))
        self.register_buffer("deg_sigma", torch.tensor(float(deg_sigma) + 1e-6))
        n_in = {"degree": 1, "redund": 2, "full": 3}[self.gate_input]
        self.gate_mlp = nn.Sequential(
            nn.Linear(n_in, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        # start near-off: zero last layer, positive bias -> sigma(+3)~=0.95 -> m~=0.95.
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.constant_(self.gate_mlp[-1].bias, 3.0)
        #: cache of the last forward's per-mediator gate (for budget reg / analysis).
        self._last_m: torch.Tensor | None = None

    def _gate_features(self, batch: SupportBatch, m_msg: torch.Tensor) -> torch.Tensor:
        """Assemble the structural-only gate input (M, n_in). m_msg = mediator messages
        h_i (used for the leave-one-out representation-redundancy cosine)."""
        cols = []
        z_deg = (self.deg_log[batch.med_id] - self.deg_mu) / self.deg_sigma   # (M,)
        if self.gate_input in ("degree", "full"):
            cols.append(z_deg.unsqueeze(-1))
        if self.gate_input in ("redund", "full"):
            pair = batch.pair_idx
            n_p = int(batch.n_pairs)
            # per-pair support size
            ones = torch.ones_like(pair, dtype=torch.float32)
            size = torch.zeros(n_p, device=pair.device).scatter_add_(0, pair, ones)  # (n_pairs,)
            size_i = size[pair]                                                       # (M,)
            # r_sig: within-pair (type, rel_a, rel_b) signature duplication
            nrb = self.n_rel_buckets
            sig = (batch.type_idx.long() * nrb + batch.rel_a.long()) * nrb + batch.rel_b.long()
            key = pair.long() * (self.n_types * nrb * nrb) + sig
            uniq, inv, cnt = torch.unique(key, return_counts=True, return_inverse=True)
            dup_i = cnt[inv].float()                                  # #mediators sharing sig (incl self)
            r_sig = (dup_i - 1.0) / (size_i - 1.0).clamp(min=1.0)     # exclude self; (M,)
            # r_rep_loo: cos(h_i, mu_{p,-i}); leave-one-out pair-prototype cosine
            d = m_msg.shape[-1]
            sum_h = torch.zeros(n_p, d, device=m_msg.device)
            sum_h.scatter_add_(0, pair.unsqueeze(-1).expand(-1, d), m_msg)
            mu_loo = (sum_h[pair] - m_msg) / (size_i - 1.0).clamp(min=1.0).unsqueeze(-1)
            r_rep = F.cosine_similarity(m_msg, mu_loo, dim=-1, eps=1e-6)             # (M,)
            # singletons (size==1) have no co-mediators -> define as maximally distinct
            single = size_i <= 1.0
            r_sig = torch.where(single, torch.zeros_like(r_sig), r_sig)
            r_rep = torch.where(single, torch.zeros_like(r_rep), r_rep)
            cols.append(r_sig.unsqueeze(-1))
            cols.append(r_rep.unsqueeze(-1))
        return torch.cat(cols, dim=-1)

    def gate_budget_reg(self, target: float) -> torch.Tensor | float:
        """(mean_i m_i - target)^2 from the last forward (codex budget term)."""
        if self._last_m is None or self._last_m.numel() == 0:
            return 0.0
        return (self._last_m.mean() - float(target)) ** 2

    def forward(self, batch: SupportBatch) -> torch.Tensor:
        B, K, d, R = int(batch.n_pairs), self.n_types, self.d, self.n_chan
        device = batch.device
        if batch.med_id.numel() > 0:
            feats = []
            if self.use_entity_embed:
                feats.append(self.entity_embed(batch.med_id))
            feats += [
                self.type_embed(batch.type_idx),
                self.rel_embed(batch.rel_a),
                self.rel_embed(batch.rel_b),
            ]
            if self.use_absdiff_embed:
                ad = (batch.d_a - batch.d_b).abs().clamp_(max=self.n_absdiff_buckets - 1)
                feats.append(self.absdiff_embed(ad))
            m = self.phi_mlp(torch.cat(feats, dim=-1))

            if R == 2:
                asym_bit = (batch.d_a != batch.d_b).long()
            else:
                asym_bit = torch.zeros_like(batch.type_idx)
            group = batch.pair_idx * (K * R) + batch.type_idx * R + asym_bit

            scores = self.within_attn_w(torch.tanh(self.within_attn_W(m))).squeeze(-1)
            if self.use_dist_attn:
                da_i = batch.d_a.clamp(0, self.max_dist)
                db_i = batch.d_b.clamp(0, self.max_dist)
                scores = scores + self.dist_bias[da_i, db_i]
            # --- the only new behaviour: learned distinctiveness gate -------------
            gi = self._gate_features(batch, m)
            m_gate = self.gate_eps + (1.0 - self.gate_eps) * torch.sigmoid(
                self.gate_mlp(gi).squeeze(-1))                       # (M,) in (eps, 1)
            self._last_m = m_gate
            scores = scores + torch.log(m_gate)
            # ----------------------------------------------------------------------
            alpha = pyg_softmax(scores, group, num_nodes=B * K * R)
            c = torch.zeros(B * K * R, d, device=device)
            c.scatter_add_(0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * m)
        else:
            c = torch.zeros(B * K * R, d, device=device)
            self._last_m = None

        c = c.view(B, K, R, d)
        transition = F.softmax(self.W_T, dim=-1)
        c_tilde = torch.einsum("kj,bjrd->bkrd", transition, c)
        return torch.cat([c_tilde.reshape(B, K * R * d), batch.struct_feats], dim=-1)


__all__ = ["GatedAwareStructuralCore", "GATE_INPUTS"]

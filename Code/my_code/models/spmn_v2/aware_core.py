"""Aware core — StructuralVariableCore + a monotone degree (hub-suppression) gate.

Subclasses ``StructuralVariableCore`` and overrides ``forward`` to subtract a
per-mediator degree penalty from the within-type attention logits, BEFORE the
softmax pool:

    scores_m  <-  scores_m  -  g_tau * phi_tau(z_m)        z_m = standardised log1p(deg_m)
    phi_tau(z) = sum_j softplus(c_{tau,j}) * relu(z - knot_j)     (monotone increasing)

This is the Step-2 / Step-3 hub gate (codex). It is the natural extension of the
existing ``use_dist_attn`` hook (a learned additive bias on the same attention
logits), so the base ``StructuralVariableCore`` is left UNCHANGED; the only new
behaviour lives here.

Zero-init invariant: ``g_tau`` (the per-type gate magnitude) is zero-initialised, so
at start ``gate = 0`` EXACTLY and the core reproduces its parent (i.e. Step 2 starts
identical to Step 1 / naked AND). ``softplus(c)`` keeps the shape monotone in degree;
``g_tau`` learns how hard to suppress, per type. ``deg_gate_mode``:
  * ``"global"`` (Step 2): ``g_tau`` is a free per-type scalar — a pair-independent
    degree penalty (the control for Step 3).
  * ``"pair"`` (Step 3, reserved): ``g_tau`` is conditioned on a per-pair regime
    vector; not implemented here (added in the Step 3 file).

The forward body mirrors the parent's exactly except for the gate injection; it is
duplicated (not refactored into the parent) to honour the no-modify-existing rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as pyg_softmax

from .batch import SupportBatch
from .core import StructuralVariableCore

#: hinge knot locations on the standardised log1p(degree) axis (codex Step 2).
DEG_KNOTS: tuple[float, ...] = (-0.5, 0.0, 0.75, 1.5)


class AwareStructuralCore(StructuralVariableCore):
    def __init__(
        self, *args,
        use_deg_gate: bool = False,
        deg_gate_mode: str = "global",
        deg_log: torch.Tensor | None = None,
        deg_mu: float = 0.0,
        deg_sigma: float = 1.0,
        n_q_feats: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.use_deg_gate = bool(use_deg_gate)
        self.deg_gate_mode = str(deg_gate_mode)
        if self.use_deg_gate:
            if deg_log is None:
                raise ValueError("use_deg_gate=True requires deg_log (per-node log1p degree)")
            if self.deg_gate_mode not in ("global", "pair"):
                raise ValueError(
                    f"deg_gate_mode must be 'global' (Step 2) or 'pair' (Step 3), "
                    f"got {self.deg_gate_mode!r}")
            self.register_buffer("deg_log",
                                 torch.as_tensor(deg_log, dtype=torch.float32))
            self.register_buffer("deg_mu", torch.tensor(float(deg_mu)))
            self.register_buffer("deg_sigma", torch.tensor(float(deg_sigma) + 1e-6))
            self.register_buffer("deg_knots", torch.tensor(DEG_KNOTS))
            n_knots = self.deg_knots.numel()
            # gate_c: per-type hinge coefficients (softplus -> monotone in degree).
            # gate_g: per-type gate magnitude, zero-init -> exact zero gate at start.
            self.gate_c = nn.Parameter(torch.zeros(self.n_types, n_knots))
            self.gate_g = nn.Parameter(torch.zeros(self.n_types))
            if self.deg_gate_mode == "pair":
                if n_q_feats is None:
                    raise ValueError("deg_gate_mode='pair' (Step 3) requires n_q_feats")
                self.n_q_feats = int(n_q_feats)
                # per-type q-weights (softplus -> monotone in the pair-regime q_ab).
                # The overall gate is still EXACT zero at start via gate_g=0, so
                # Step 3 starts identical to base+branch (gate off).
                self.pair_v = nn.Parameter(torch.zeros(self.n_types, self.n_q_feats))

    def _deg_gate(self, med_id: torch.Tensor, type_idx: torch.Tensor,
                  pair_idx: torch.Tensor | None = None,
                  pair_q: torch.Tensor | None = None) -> torch.Tensor:
        """Per-mediator hub penalty, upper-clamped to 4.

        global (Step 2): gate = g_tau * phi_tau(z).
        pair (Step 3):   gate = g_tau * lambda_tau(q_ab) * phi_tau(z), where
            lambda_tau(q) = softplus( sum_i softplus(v_{tau,i}) * q_i )  (monotone in q).
        ``g_tau`` is kept >= 0 by projection after each optimizer step (see runner),
        so gate >= 0 always (phi, lambda >= 0); we only bound the TOP here. Upper-
        clamping avoids the dead-gate failure of a two-sided clamp."""
        z = (self.deg_log[med_id] - self.deg_mu) / self.deg_sigma          # (M,)
        hinge = torch.relu(z.unsqueeze(-1) - self.deg_knots)               # (M, n_knots)
        coeff = F.softplus(self.gate_c[type_idx])                          # (M, n_knots)
        phi = (coeff * hinge).sum(-1)                                      # (M,)
        gate = self.gate_g[type_idx] * phi                                 # (M,)
        if self.deg_gate_mode == "pair":
            if pair_q is None or pair_idx is None:
                raise ValueError("deg_gate_mode='pair' needs pair_idx + batch.pair_q")
            q_m = pair_q[pair_idx]                                         # (M, n_q)
            w = F.softplus(self.pair_v[type_idx])                          # (M, n_q) >= 0
            pair_factor = F.softplus((w * q_m).sum(-1))                    # (M,) >= 0
            gate = gate * pair_factor
        return gate.clamp(max=4.0)

    def gate_reg(self) -> torch.Tensor | float:
        """L2 on gate params + second-difference smoothness on the hinge coeffs."""
        if not self.use_deg_gate:
            return 0.0
        reg = (self.gate_g ** 2).sum() + (self.gate_c ** 2).sum()
        if self.gate_c.shape[1] >= 3:
            d2 = self.gate_c[:, 2:] - 2 * self.gate_c[:, 1:-1] + self.gate_c[:, :-2]
            reg = reg + (d2 ** 2).sum()
        if self.deg_gate_mode == "pair":
            reg = reg + (self.pair_v ** 2).sum()
        return reg

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
            if self.use_deg_gate:                       # <-- the only new behaviour
                scores = scores - self._deg_gate(
                    batch.med_id, batch.type_idx, batch.pair_idx, batch.pair_q)
            alpha = pyg_softmax(scores, group, num_nodes=B * K * R)
            c = torch.zeros(B * K * R, d, device=device)
            c.scatter_add_(0, group.unsqueeze(-1).expand(-1, d), alpha.unsqueeze(-1) * m)
        else:
            c = torch.zeros(B * K * R, d, device=device)

        c = c.view(B, K, R, d)
        transition = F.softmax(self.W_T, dim=-1)
        c_tilde = torch.einsum("kj,bjrd->bkrd", transition, c)
        return torch.cat([c_tilde.reshape(B, K * R * d), batch.struct_feats], dim=-1)


__all__ = ["AwareStructuralCore", "DEG_KNOTS"]

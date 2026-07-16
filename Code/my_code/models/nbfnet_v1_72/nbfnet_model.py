"""NBFNet v1.72 model — dual-source SIGNED-field interference (additive over v1.7).

New vs v1.7 (all additive; v1.7 model untouched):
  - SIGNED fields: per-layer activation switchable {relu (non-negative, =v1.7) ,
    signed (RMSNorm, can be negative)}. Signed fields are what let the downstream
    Hadamard cross-term be destructive (interference); the trainer owns the readout.
  - ACTION-signed edges: a per-edge sign multiplier s_e ∈ {+1,-1} (agonist/up=+1,
    antagonist/inhibitor/down=-1, neutral=+1) is multiplied into the DistMult
    message. Inverse edges inherit the SAME sign (handled by the trainer/builder).
  - Shared endpoint RMSNorm (`endpoint_norm`) used by the trainer to normalize the
    two endpoint readouts BEFORE the cross-term, so u⊙v is a stable agreement term.

This module only provides the SIGNED batched encoder. Combination (additive vs
Hadamard), sign threading, and diagnostics live in the v1.72 trainer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_code.models.nbfnet_v1_7.nbfnet_model import NBFNetDDI, NBFLayer


class NBFLayerV172(NBFLayer):
    """NBFLayer + per-edge action sign + switchable signed/relu activation.

    Reuses W_r / b_r / compute_w_q / aggregator from NBFLayer unchanged.
    """

    def forward_batched_signed(
        self,
        h: torch.Tensor,                # (S, n_nodes, d)
        h0: torch.Tensor,               # (S, n_nodes, d)
        edge_src: torch.Tensor,         # (E,)
        edge_dst: torch.Tensor,         # (E,)
        edge_rel: torch.Tensor,         # (E,)
        edge_sign: torch.Tensor,        # (E,) action sign per edge (+1/-1)
        q: torch.Tensor,                # (d,)
        n_nodes: int,
        signed: bool,
        rms: nn.Module,                 # this layer's RMSNorm (used iff signed)
    ) -> torch.Tensor:
        """Batched signed BF layer. msg = h_x ⊙ w_q(r) · s_e ; aggregate ; activation.

        Activation: RMSNorm (signed, can be negative) if `signed` else ReLU
        (non-negative, identical to v1.7). Returns (S, n_nodes, d).
        """
        w_q = self.compute_w_q(q)                       # (n_rel, d)
        w_e = w_q[edge_rel]                              # (E, d)
        msg = h[:, edge_src] * w_e                       # (S, E, d)
        msg = msg * edge_sign.view(1, -1, 1)             # action sign (broadcast over S, d)

        self_idx = torch.arange(n_nodes, device=h.device)
        msg_with_boundary = torch.cat([msg, h0], dim=1)               # (S, E+n, d)
        target_with_boundary = torch.cat([edge_dst, self_idx], dim=0)  # (E+n,)

        agg = self.aggregator.forward_batched(
            msg_with_boundary, target_with_boundary, n_nodes
        )                                                # (S, n_nodes, d)

        if signed:
            return rms(agg)            # signed activation (RMSNorm controls scale)
        return F.relu(agg)             # non-negative (= v1.7 behavior)


class NBFNetV172(NBFNetDDI):
    """NBFNetDDI + signed batched encoder with action-signed edges (additive).

    The single-source / non-signed v1.7 paths are inherited untouched.
    """

    def __init__(self, *args, signed_act: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.signed_act = bool(signed_act)
        # Rebuild layers as V172 (same shapes/params, adds the signed forward).
        self.layers = nn.ModuleList(
            [NBFLayerV172(self.d, self.n_rel, aggregator=None) for _ in range(self.L)]
        )
        # Per-layer RMSNorm for the signed activation.
        self.layer_rms = nn.ModuleList([nn.RMSNorm(self.d) for _ in range(self.L)])
        # SHARED endpoint norm for the two readout branches (necessary addition #2).
        # AFFINE-FREE (no learnable gamma): a learnable per-dim weight could go
        # negative during training and flip signs, which would break the
        # `activation="relu"` 2×2 control (it must stay non-negative). Without
        # gamma, RMSNorm = x / rms preserves the sign of its input: relu fields
        # (>=0) stay >=0, signed fields keep their signs. (codex review fix.)
        self.endpoint_norm = nn.RMSNorm(self.d, elementwise_affine=False)

    def bellman_ford_batched_signed(
        self,
        sources: torch.Tensor,          # (S,)
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        edge_sign: torch.Tensor,        # (E,) aligned with the edge tensors
    ) -> torch.Tensor:
        """L-layer signed batched BF rooted at S sources, with action-signed edges."""
        S = int(sources.size(0))
        h0 = torch.zeros(S, self.n_nodes, self.d, device=self.query.device)
        h0[torch.arange(S, device=sources.device), sources] = self.query
        h = h0
        for li, layer in enumerate(self.layers):
            h = layer.forward_batched_signed(
                h, h0, edge_src, edge_dst, edge_rel, edge_sign,
                self.query, self.n_nodes, self.signed_act, self.layer_rms[li],
            )
        return h

    def encode_from_sources_signed(
        self,
        sources: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        edge_sign: torch.Tensor,
        already_augmented: bool = False,
    ) -> torch.Tensor:
        """Signed batched encoder. If not already augmented, augment inverse edges
        AND duplicate the sign (inverse inherits the SAME sign — necessary #1)."""
        if not already_augmented:
            edge_src, edge_dst, edge_rel = self.augment_inverse_edges(
                edge_src, edge_dst, edge_rel
            )
            edge_sign = torch.cat([edge_sign, edge_sign], dim=0)
        sources = torch.as_tensor(sources, device=self.query.device, dtype=torch.long)
        return self.bellman_ford_batched_signed(
            sources, edge_src, edge_dst, edge_rel, edge_sign
        )

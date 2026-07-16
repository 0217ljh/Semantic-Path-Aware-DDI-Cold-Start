"""EmerGNN with meet-in-middle (MIM) readout.

Modifies the readout step of EmerGNN/EmerGNN_TAG. After bidirectional
flow propagation finishes, instead of just reading the terminal node's
hidden state, we ALSO read the hidden states at "junction" nodes
(structural meeting points between source and target) and aggregate
them into the final score.

Score head shape changes from `Linear(4*n_dim, 1)` (E mode) to
`Linear(6*n_dim, 1)` (E mode + junction aggregation):
  [head_emb, tail_emb, head_hid, tail_hid, junc_sum_uv, junc_sum_vu] -> logit
where junc_sum_uv = sum over junction nodes c of hid_uv[c, b, :].

Junction sets are EXTERNAL to the model (precomputed by junction_finder)
to keep this module focused. Per-batch junction tensor passed to forward().
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

# Reuse Screen 1's EmerGNN_TAG so TAG init + MIM readout can compose
from my_code.models.screen1_tag_init.emergnn_with_init import EmerGNN_TAG  # noqa: E402


class EmerGNN_MIM(EmerGNN_TAG):
    """EmerGNN with junction-aware readout.

    Forward signature changes: forward(head, tail, edge_src, edge_dst,
    edge_rel, junctions, junction_mask) where junctions is (B, K) long
    tensor of junction node IDs (-1 = pad) and junction_mask is (B, K)
    float mask (1=valid, 0=pad).

    If junctions is None or junction_mask sums to zero per batch, behaves
    identically to EmerGNN_TAG (terminal-only readout).
    """

    def __init__(
        self,
        n_ent: int,
        n_base_rel: int,
        n_dim: int = 64,
        length: int = 3,
        external_init: Optional[np.ndarray | torch.Tensor] = None,
        freeze_init: bool = False,
        feat: str = "X",
        morgan_features: Optional[np.ndarray] = None,
        morgan_feat_dim: int = 1024,
        use_mim: bool = True,
    ) -> None:
        # Build base TAG (feat='X' with external init)
        super().__init__(
            n_ent=n_ent,
            n_base_rel=n_base_rel,
            n_dim=n_dim,
            length=length,
            external_init=external_init,
            freeze_init=freeze_init,
            feat=feat,
            morgan_features=morgan_features,
            morgan_feat_dim=morgan_feat_dim,
        )
        self.use_mim = bool(use_mim)
        if self.use_mim:
            # Override score head: 6 * n_dim instead of 4 * n_dim
            self.Wr = nn.Linear(6 * n_dim, 1)
            nn.init.xavier_uniform_(self.Wr.weight)

    def forward(
        self,
        head: torch.Tensor,
        tail: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        junctions: torch.Tensor | None = None,
        junction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with optional junction-aware readout.

        Shapes:
          head, tail        : (B,)
          junctions         : (B, K), int (-1 = pad)
          junction_mask     : (B, K), float (1=valid, 0=pad)
        """
        if not self.use_mim:
            return super().forward(head, tail, edge_src, edge_dst, edge_rel)

        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        hid_uv = self._propagate(head, head_embed, ht_embed, edge_src, edge_dst, edge_rel)
        B = head.size(0)
        device = head.device
        b_arange = torch.arange(B, device=device)
        tail_hid = hid_uv[tail, b_arange]

        hid_vu = self._propagate(tail, tail_embed, ht_embed, edge_src, edge_dst, edge_rel)
        head_hid = hid_vu[head, b_arange]

        # Junction aggregation
        if junctions is None or junction_mask is None:
            junc_uv = torch.zeros(B, self.n_dim, device=device)
            junc_vu = torch.zeros(B, self.n_dim, device=device)
        else:
            # Clip pad values (-1) to safe index 0; mask will zero them out
            junc_safe = junctions.clamp_min(0)  # (B, K)
            # Gather hidden states at junction nodes for each batch item
            #   uv direction: hid_uv[c, b, :] for each (b, c) pair
            #   shape (B, K, n_dim)
            mask = junction_mask.unsqueeze(-1)  # (B, K, 1)
            # Build sparse-ish gather: for each (b, k) pull hid_uv[junc[b,k], b, :]
            # Vectorized via advanced indexing:
            K = junctions.size(1)
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, K)  # (B, K)
            junc_uv_all = hid_uv[junc_safe, b_idx]   # (B, K, n_dim)
            junc_vu_all = hid_vu[junc_safe, b_idx]   # (B, K, n_dim)
            # Mean over valid junctions (avoiding /0)
            denom = mask.sum(dim=1, keepdim=False).clamp_min(1.0)  # (B, 1)
            junc_uv = (junc_uv_all * mask).sum(dim=1) / denom
            junc_vu = (junc_vu_all * mask).sum(dim=1) / denom

        embed = torch.cat([head_embed, tail_embed, head_hid, tail_hid, junc_uv, junc_vu], dim=-1)
        logits = self.Wr(embed).squeeze(-1)
        return logits


__all__ = ["EmerGNN_MIM"]

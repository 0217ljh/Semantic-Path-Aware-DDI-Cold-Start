"""D2 — EmerGNN subclass that injects a per-batch meeting-mediator mask
as a per-layer attention bonus.

Round 4 D2 (per Notes/Log/d2_meet_mask_design.md §3.3; CP-1 PASS_WITH_NITS at
_reviews/2026-06-01__d2_design__round1.md).

Mechanism (per design §3.3 and §1 C1):

    bonus[v, b, :] = alpha_meet[l] * meet_mask[v, b] * new_hiddens[v, b, :]

added at every layer l AFTER `self.act(self.linear[l](new_hiddens))`
(parent EmerGNN.model.py:178) and BEFORE the hidden state is rolled into the
next layer. alpha_meet is a per-layer learnable scalar; init=0 so D2 starts
mathematically equivalent to parent EmerGNN at training step 0.

File-independence: subclasses baseline.emergnn.model.EmerGNN; does NOT modify
the parent. The override mirrors parent _propagate line-by-line (verified
against baseline/emergnn/model.py:116-180 on 2026-06-01) with the single bonus
addition.

Key citations (parent code lines verified 2026-06-01):
- baseline/emergnn/model.py:59  -> self.all_rel = 2 * n_base_rel + 1
- baseline/emergnn/model.py:78  -> per-layer self.rel_kg[l] embedding table
- baseline/emergnn/model.py:84  -> per-layer self.attn_relation[l]
- baseline/emergnn/model.py:116-180 -> _propagate body
- baseline/emergnn/model.py:178 -> self.act(self.linear[l](new_hiddens))
- baseline/emergnn/model.py:182-210 -> forward
- baseline/emergnn/model.py:213-239 -> _chunk_compute_and_scatter (free function)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _cp

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from baseline.emergnn.model import EmerGNN, _chunk_compute_and_scatter  # noqa: E402


class EmerGNNWithMeetMask(EmerGNN):
    """EmerGNN + per-layer meeting-mediator mask bonus.

    Args extending parent:
      alpha_meet_init: initial per-layer alpha_meet (default 0.0 → mathematically
          equivalent to parent EmerGNN at step 0)
      freeze_alpha_meet: if True, alpha_meet is frozen at init value AND
          requires_grad=False. Used by --d2-freeze-alpha-meet K3 control.
    """

    def __init__(
        self,
        *args,
        alpha_meet_init: float = 0.0,
        freeze_alpha_meet: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha_meet = nn.Parameter(
            torch.full((self.L,), float(alpha_meet_init), dtype=torch.float32)
        )
        if freeze_alpha_meet:
            # K3 mathematical pass-through: zero-add at every layer, no gradient.
            with torch.no_grad():
                self.alpha_meet.fill_(float(alpha_meet_init))
            self.alpha_meet.requires_grad_(False)
        self._freeze_alpha_meet = bool(freeze_alpha_meet)

    # ------------------------------------------------------------------
    # _propagate override: mirrors parent line-by-line + bonus injection
    # ------------------------------------------------------------------

    def _propagate(
        self,
        source_idx: torch.Tensor,           # (B,)
        source_embed: torch.Tensor,         # (B, n_dim)
        ht_embed: torch.Tensor,             # (B, 2*n_dim)
        edge_src: torch.Tensor,             # (E,)
        edge_dst: torch.Tensor,             # (E,)
        edge_rel: torch.Tensor,             # (E,)
        meet_mask: torch.Tensor | None = None,  # (n_ent, B) float {0,1} or None
    ) -> torch.Tensor:
        """Mirror parent EmerGNN._propagate (model.py:116-180) + D2 bonus.

        If meet_mask is None, the bonus path is skipped entirely (--d2-disable
        path; mathematically identical to parent).
        """
        B = source_idx.size(0)
        device = source_embed.device

        # Initialize hidden (parent line 132-135)
        hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
        batch_arange = torch.arange(B, device=device)
        hiddens[source_idx, batch_arange] = source_embed

        for l in range(self.L):
            # (a) Attention over relation slots (parent 138-145)
            rel_w = self.attn_relation[l](
                F.relu(self.relation_linear[l](ht_embed))
            )
            rel_w = torch.sigmoid(rel_w)
            rel_emb = self.rel_kg[l].weight
            rel_per_batch = rel_w.unsqueeze(-1) * rel_emb.unsqueeze(0)
            rel_t = rel_per_batch.transpose(0, 1)

            # (b) Chunked message passing (parent 156-177)
            E = edge_src.size(0)
            new_hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
            need_grad = self.training and hiddens.requires_grad
            use_checkpoint = need_grad and self._use_checkpoint
            for start in range(0, E, self._chunk_size):
                stop = min(start + self._chunk_size, E)
                es = edge_src[start:stop]
                er = edge_rel[start:stop]
                ed = edge_dst[start:stop]
                if use_checkpoint:
                    contrib = _cp.checkpoint(
                        _chunk_compute_and_scatter,
                        hiddens, rel_t, es, er, ed,
                        self.n_ent,
                        use_reentrant=False,
                    )
                else:
                    contrib = _chunk_compute_and_scatter(
                        hiddens, rel_t, es, er, ed, self.n_ent
                    )
                new_hiddens = new_hiddens + contrib
                del contrib

            # (c) Post-message activation (parent 178)
            new_hiddens = self.act(self.linear[l](new_hiddens))

            # (d) D2 BONUS: alpha_meet[l] * mask * new_hiddens (added after activation).
            # When meet_mask is None or alpha_meet is frozen at 0, the bonus is
            # identically zero — preserves parity with parent EmerGNN.
            if meet_mask is not None:
                # meet_mask: (n_ent, B); unsqueeze to (n_ent, B, 1) for broadcasting.
                bonus = self.alpha_meet[l] * meet_mask.unsqueeze(-1) * new_hiddens
                new_hiddens = new_hiddens + bonus

            hiddens = new_hiddens
        return hiddens

    # ------------------------------------------------------------------
    # forward override: threads meet_mask through both _propagate calls
    # ------------------------------------------------------------------

    def forward(
        self,
        head: torch.Tensor,          # (B,)
        tail: torch.Tensor,          # (B,)
        edge_src: torch.Tensor,      # (E,)
        edge_dst: torch.Tensor,      # (E,)
        edge_rel: torch.Tensor,      # (E,)
        meet_mask: torch.Tensor | None = None,  # (n_ent, B) float {0,1}
    ) -> torch.Tensor:
        """Mirror parent EmerGNN.forward (model.py:182-210) with meet_mask
        threaded into both u→v and v→u propagation calls."""
        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        hid_uv = self._propagate(
            head, head_embed, ht_embed, edge_src, edge_dst, edge_rel,
            meet_mask=meet_mask,
        )
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)
        tail_hid = hid_uv[tail, b_arange]

        hid_vu = self._propagate(
            tail, tail_embed, ht_embed, edge_src, edge_dst, edge_rel,
            meet_mask=meet_mask,
        )
        head_hid = hid_vu[head, b_arange]

        if self.feat == "E":
            embed = torch.cat([head_embed, tail_embed, head_hid, tail_hid], dim=-1)
        else:
            embed = torch.cat([head_hid, tail_hid], dim=-1)
        logits = self.Wr(embed).squeeze(-1)
        return logits


__all__ = ["EmerGNNWithMeetMask"]

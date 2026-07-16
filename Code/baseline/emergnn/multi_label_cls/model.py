"""EmerGNN multilabel model (200 side-effect head) — ported from the TWOSIDES
reproduction (LARS-research/EmerGNN, TWOSIDES/models.py).

Paper-faithful port of the TWOSIDES multilabel EmerGNN. Two deviations from the
literal upstream, both forced by this environment and both preserving the math:

  * ``torchdrug.layers.functional.generalized_rspmm`` is unavailable in
    ``project_1``. We replace it with the same chunked ``index_select`` +
    ``scatter_add`` message-passing primitive already used (and codex-verified
    equivalent, max abs diff 2.98e-08) by the binary/multiclass port in
    ``baseline/emergnn/model.py`` — see ``_chunk_compute_and_scatter``. The
    per-layer relation-gated aggregation ``out[t] += (attn[b,r] * rel_emb[r]) *
    hiddens[h]`` is identical to upstream ``relation_input = relation_weight *
    rel_embed`` fed to ``generalized_rspmm(KG, relation_input, hiddens,
    sum='add', mul='mul')`` (TWOSIDES/models.py:67-70).

  * The score head ``Wr`` outputs ``eval_rel`` logits (=200 side effects) exactly
    like upstream (``self.Wr = nn.Linear(2*args.n_dim, eval_rel)`` for feat='M',
    TWOSIDES/models.py:27). This is the essential difference from the binary port
    (Linear(..., 1)) and the multiclass port (Linear(..., n_classes) + softmax):
    multilabel keeps a FIXED 200-output sigmoid head (BCE), no per-fold re-vocab.

Faithful to upstream ``enc_ht`` structure: bidirectional (u->v and v->u) L-layer
message passing; feat='M' concatenates ``[head_hid, tail_hid]`` -> Wr(2*n_dim ->
200); feat='E' concatenates ``[head_embed, head_hid, tail_hid, tail_embed]`` ->
Wr(4*n_dim -> 200). We follow the paper default feat='M' for TWOSIDES S1/S2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _cp

try:
    from torch_scatter import scatter_add as _scatter_add
    _HAS_TORCH_SCATTER = True
except ImportError:  # pragma: no cover
    _HAS_TORCH_SCATTER = False


class EmerGNN_ML(nn.Module):
    """EmerGNN with a fixed multilabel head (eval_rel side-effect logits).

    Relation-slot convention matches the ported loader (see ``_core_twoside``):
      * ``all_rel_slots`` = 2*all_rel - eval_rel + 1 (forward + reverse KG
        relations + a shared self-loop slot), identical to upstream
        ``load_graph`` (TWOSIDES/load_data.py:159).
      * DDI fact edges use relation slot 0 (both directions) per upstream.
    """

    def __init__(
        self,
        n_ent: int,
        all_rel: int,          # = max(relation2id)+1 (523 for TWOSIDES)
        eval_rel: int,         # = number of side-effect labels (200)
        n_dim: int = 64,
        length: int = 3,
        feat: str = "M",       # 'M' Morgan / 'E' learned
        morgan_features: Optional[np.ndarray] = None,
        morgan_feat_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_ent = n_ent
        self.all_rel = all_rel
        self.eval_rel = eval_rel
        self.n_dim = n_dim
        self.L = length
        self.feat = feat
        # upstream: all_rel = 2*args.all_rel - args.eval_rel + 1 (models.py:17)
        self.all_rel_slots = 2 * all_rel - eval_rel + 1

        if feat == "E":
            self.ent_kg = nn.Embedding(n_ent, n_dim)
            self.Went = None
            self.Wr = nn.Linear(4 * n_dim, eval_rel)
        elif feat == "M":
            if morgan_features is None:
                raise ValueError("feat='M' requires morgan_features (n_ent, 1024).")
            assert morgan_features.shape[0] == n_ent
            assert morgan_features.shape[1] == morgan_feat_dim
            self.register_buffer(
                "ent_feat", torch.from_numpy(morgan_features.astype(np.float32))
            )
            self.Went = nn.Linear(morgan_feat_dim, n_dim)
            self.Wr = nn.Linear(2 * n_dim, eval_rel)
        else:
            raise ValueError(f"Unknown feat={feat!r}; expected 'M' or 'E'.")

        self.rel_kg = nn.ModuleList(
            [nn.Embedding(self.all_rel_slots, n_dim) for _ in range(self.L)]
        )
        self.linear = nn.ModuleList(
            [nn.Linear(n_dim, n_dim) for _ in range(self.L)]
        )
        self.act = nn.ReLU()
        # upstream bottleneck attention: Linear(2*n_dim, 5) -> Linear(5, all_rel_slots)
        self.relation_linear = nn.ModuleList(
            [nn.Linear(2 * n_dim, 5) for _ in range(self.L)]
        )
        self.attn_relation = nn.ModuleList(
            [nn.Linear(5, self.all_rel_slots) for _ in range(self.L)]
        )

        self._chunk_size = 100_000
        self._use_checkpoint = True
        self._init_weights()

    def set_chunk_size(self, chunk_size: int) -> None:
        self._chunk_size = int(chunk_size)

    def set_use_checkpoint(self, use_checkpoint: bool) -> None:
        self._use_checkpoint = bool(use_checkpoint)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.data.ndim > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p.data)

    def _entity_embed(self, idx: torch.Tensor) -> torch.Tensor:
        if self.feat == "E":
            return self.ent_kg(idx)
        return self.Went(self.ent_feat.index_select(0, idx))

    def _propagate(
        self,
        source_idx: torch.Tensor,     # (B,)
        source_embed: torch.Tensor,   # (B, n_dim)
        ht_embed: torch.Tensor,       # (B, 2*n_dim)
        edge_src: torch.Tensor,       # (E,)
        edge_dst: torch.Tensor,       # (E,)
        edge_rel: torch.Tensor,       # (E,)
    ) -> torch.Tensor:
        B = source_idx.size(0)
        device = source_embed.device
        hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
        batch_arange = torch.arange(B, device=device)
        hiddens[source_idx, batch_arange] = source_embed

        for l in range(self.L):
            rel_w = self.attn_relation[l](F.relu(self.relation_linear[l](ht_embed)))
            rel_w = torch.sigmoid(rel_w)                       # (B, all_rel_slots)
            rel_emb = self.rel_kg[l].weight                    # (all_rel_slots, n_dim)
            rel_per_batch = rel_w.unsqueeze(-1) * rel_emb.unsqueeze(0)
            rel_t = rel_per_batch.transpose(0, 1)              # (all_rel_slots, B, n_dim)

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
                        hiddens, rel_t, es, er, ed, self.n_ent,
                        use_reentrant=False,
                    )
                else:
                    contrib = _chunk_compute_and_scatter(
                        hiddens, rel_t, es, er, ed, self.n_ent
                    )
                new_hiddens = new_hiddens + contrib
                del contrib
            new_hiddens = self.act(self.linear[l](new_hiddens))
            hiddens = new_hiddens
        return hiddens

    def forward(
        self,
        head: torch.Tensor,      # (B,)
        tail: torch.Tensor,      # (B,)
        edge_src: torch.Tensor,  # (E,)
        edge_dst: torch.Tensor,  # (E,)
        edge_rel: torch.Tensor,  # (E,)
    ) -> torch.Tensor:
        """Return logits (B, eval_rel)."""
        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        # u -> v: source = head, read tail
        hid_uv = self._propagate(head, head_embed, ht_embed, edge_src, edge_dst, edge_rel)
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)
        tail_hid = hid_uv[tail, b_arange]

        # v -> u: source = tail, read head
        hid_vu = self._propagate(tail, tail_embed, ht_embed, edge_src, edge_dst, edge_rel)
        head_hid = hid_vu[head, b_arange]

        if self.feat == "E":
            embed = torch.cat([head_embed, head_hid, tail_hid, tail_embed], dim=-1)
        else:
            embed = torch.cat([head_hid, tail_hid], dim=-1)
        return self.Wr(embed)          # (B, eval_rel)


def _chunk_compute_and_scatter(
    hiddens: torch.Tensor,          # (n_ent, B, n_dim)
    rel_t: torch.Tensor,            # (all_rel_slots, B, n_dim)
    edge_src_chunk: torch.Tensor,   # (c,)
    edge_rel_chunk: torch.Tensor,   # (c,)
    edge_dst_chunk: torch.Tensor,   # (c,)
    n_ent: int,
) -> torch.Tensor:
    """ONE-chunk fused msg + scatter_add: out[t] += rel_t[r] * hiddens[h]."""
    h_feat = hiddens.index_select(0, edge_src_chunk)
    r_feat = rel_t.index_select(0, edge_rel_chunk)
    msg = h_feat * r_feat
    if _HAS_TORCH_SCATTER:
        return _scatter_add(msg, edge_dst_chunk, dim=0, dim_size=n_ent)
    out = torch.zeros(
        n_ent, hiddens.size(1), hiddens.size(2),
        device=hiddens.device, dtype=hiddens.dtype,
    )
    out.index_add_(0, edge_dst_chunk, msg)
    return out


__all__ = ["EmerGNN_ML"]

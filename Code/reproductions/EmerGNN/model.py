"""
EmerGNN (pure PyTorch) — no torchdrug / torch_scatter.

Replaces the CUDA kernel `torchdrug.layers.functional.generalized_rspmm` and
`torch_scatter.scatter_add` with portable PyTorch ops:

    * `generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')`
       -> for each edge (h, t, r):  out[t] += relation_input[r] * hiddens[h]
       implemented via per-chunk `index_select` + immediate `index_add_`
       directly into `new_hiddens`. NEVER materializes a full (E, B, n_dim)
       intermediate — chunking caps peak msg memory at chunk_size * B * D.
       Training mode wraps each chunk in `torch.utils.checkpoint.checkpoint`
       so backward recomputes the msg instead of storing it across chunks.

    * `scatter_add` -> `torch.zeros(...).index_add_(0, index, src)`.

Faithful to Zhang et al. 2023 for architecture: attention-weighted relation
aggregation with L=3 bidirectional message-passing layers; the score head
is a Linear(2*n_dim -> 1) producing a single-logit for binary DDI
(the original code predicts over eval_rel relations; we adapt by switching
the final `Wr` to output 1 for binary ColdDDI).
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
except ImportError:
    _HAS_TORCH_SCATTER = False


class EmerGNN(nn.Module):
    """Flow-based bidirectional message-passing DDI predictor."""

    def __init__(
        self,
        n_ent: int,
        n_base_rel: int,
        n_dim: int = 64,
        length: int = 3,
        feat: str = "M",  # 'M' for Morgan, 'E' for learned embedding
        morgan_features: Optional[np.ndarray] = None,
        morgan_feat_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.n_ent = n_ent
        self.n_base_rel = n_base_rel
        self.n_dim = n_dim
        self.L = length
        self.feat = feat
        self.all_rel = 2 * n_base_rel + 1  # forward + reverse + self-loop

        if feat == "E":
            self.ent_kg = nn.Embedding(n_ent, n_dim)
            self.Went = None
            self.Wr = nn.Linear(4 * n_dim, 1)  # binary logit
        elif feat == "M":
            if morgan_features is None:
                raise ValueError("feat='M' requires morgan_features (n_ent, 1024) array.")
            assert morgan_features.shape[0] == n_ent
            assert morgan_features.shape[1] == morgan_feat_dim
            # Store as a non-trainable buffer
            self.register_buffer("ent_feat", torch.from_numpy(morgan_features.astype(np.float32)))
            self.Went = nn.Linear(morgan_feat_dim, n_dim)
            self.Wr = nn.Linear(2 * n_dim, 1)  # binary logit
        else:
            raise ValueError(f"Unknown feat={feat!r}; expected 'M' or 'E'.")

        # Per-layer relation embedding tables and linear transforms
        self.rel_kg = nn.ModuleList([nn.Embedding(self.all_rel, n_dim) for _ in range(self.L)])
        self.linear = nn.ModuleList([nn.Linear(n_dim, n_dim) for _ in range(self.L)])
        self.act = nn.ReLU()

        # Attention over relation slots (bottlenecked through a 5-D layer per paper)
        self.relation_linear = nn.ModuleList([nn.Linear(2 * n_dim, 5) for _ in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, self.all_rel) for _ in range(self.L)])

        # Edge chunk size for chunked message passing. Smaller = less peak memory,
        # more chunk loop overhead. Default 100k → peak per chunk for B=32, D=64:
        # 100k * 32 * 64 * 4 = 820 MB. Override via set_chunk_size() if needed.
        self._chunk_size = 100_000
        # Whether to wrap each chunk's compute in torch.utils.checkpoint during
        # training (~2x slower forward+backward, much less memory). For larger
        # GPUs with enough VRAM, set False to speed up.
        self._use_checkpoint = True

        self._init_weights()

    def set_chunk_size(self, chunk_size: int) -> None:
        """Override the message-passing edge chunk size."""
        self._chunk_size = int(chunk_size)

    def set_use_checkpoint(self, use_checkpoint: bool) -> None:
        """Enable / disable gradient checkpointing on chunk message passing."""
        self._use_checkpoint = bool(use_checkpoint)

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.data.ndim > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p.data)

    def _entity_embed(self, idx: torch.Tensor) -> torch.Tensor:
        if self.feat == "E":
            return self.ent_kg(idx)
        else:
            return self.Went(self.ent_feat.index_select(0, idx))

    def _propagate(
        self,
        source_idx: torch.Tensor,           # (B,)
        source_embed: torch.Tensor,         # (B, n_dim)
        ht_embed: torch.Tensor,             # (B, 2*n_dim)
        edge_src: torch.Tensor,             # (E,) long, device=same
        edge_dst: torch.Tensor,             # (E,)
        edge_rel: torch.Tensor,             # (E,)
    ) -> torch.Tensor:
        """Run L layers of message passing from `source_idx` outwards.

        Returns: hidden tensor of shape (n_ent, B, n_dim) after L steps.
        """
        B = source_idx.size(0)
        device = source_embed.device

        # Initialize hidden: only source entities have the source embedding, rest are zero
        hiddens = torch.zeros(self.n_ent, B, self.n_dim, device=device)
        batch_arange = torch.arange(B, device=device)
        hiddens[source_idx, batch_arange] = source_embed

        for l in range(self.L):
            # (a) Attention over relation slots, conditioned on (head, tail) query
            rel_w = self.attn_relation[l](F.relu(self.relation_linear[l](ht_embed)))  # (B, all_rel)
            rel_w = torch.sigmoid(rel_w)                                              # (B, all_rel)
            rel_emb = self.rel_kg[l].weight                                           # (all_rel, n_dim)

            # Per-(batch,relation) gated relation embedding: (B, all_rel, n_dim)
            rel_per_batch = rel_w.unsqueeze(-1) * rel_emb.unsqueeze(0)
            rel_t = rel_per_batch.transpose(0, 1)  # (all_rel, B, n_dim)

            # (b) Message passing: out[t, b, :] += rel_per_batch[b, r, :] * hiddens[h, b, :]
            #     CHUNKED compute + scatter wrapped in gradient checkpoint.
            #     Each chunk_fn(...) returns a (n_ent, B, n_dim) contribution
            #     that we OUT-OF-PLACE accumulate into new_hiddens. The
            #     entire compute+scatter sits inside one checkpoint call so
            #     backward can recompute the chunk's intermediates instead
            #     of storing the (chunk*B*D) msg tensors across the loop.
            #     For E=3.6M with chunk_size=100k, this caps peak per chunk
            #     at ~3 * (chunk*B*D + n_ent*B*D) * 4 bytes ~ 2.5 GB.
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
            new_hiddens = self.act(self.linear[l](new_hiddens))
            hiddens = new_hiddens
        return hiddens

    def forward(
        self,
        head: torch.Tensor,          # (B,)
        tail: torch.Tensor,          # (B,)
        edge_src: torch.Tensor,      # (E,)
        edge_dst: torch.Tensor,      # (E,)
        edge_rel: torch.Tensor,      # (E,)
    ) -> torch.Tensor:
        """Return logits (B,)."""
        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        # u -> v propagation: source = head, read tail
        hid_uv = self._propagate(head, head_embed, ht_embed, edge_src, edge_dst, edge_rel)
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)
        tail_hid = hid_uv[tail, b_arange]           # (B, n_dim)

        # v -> u propagation: source = tail, read head
        hid_vu = self._propagate(tail, tail_embed, ht_embed, edge_src, edge_dst, edge_rel)
        head_hid = hid_vu[head, b_arange]           # (B, n_dim)

        if self.feat == "E":
            embed = torch.cat([head_embed, tail_embed, head_hid, tail_hid], dim=-1)
        else:
            embed = torch.cat([head_hid, tail_hid], dim=-1)
        logits = self.Wr(embed).squeeze(-1)
        return logits


def _chunk_compute_and_scatter(
    hiddens: torch.Tensor,        # (n_ent, B, n_dim)
    rel_t: torch.Tensor,          # (all_rel, B, n_dim)  (already transposed)
    edge_src_chunk: torch.Tensor, # (c,)
    edge_rel_chunk: torch.Tensor, # (c,)
    edge_dst_chunk: torch.Tensor, # (c,)
    n_ent: int,
) -> torch.Tensor:
    """ONE-chunk fused: compute msg + scatter_add into (n_ent, B, n_dim).

    Inlined for use under ``torch.utils.checkpoint``: returns the full
    chunk contribution as a (n_ent, B, n_dim) tensor; the msg and
    h_feat / r_feat intermediates are NOT held by autograd between
    chunks because checkpoint recomputes them during backward.

    Uses ``torch_scatter.scatter_add`` if installed (cleaner backward),
    else falls back to ``torch.zeros + index_add_`` (pure PyTorch).
    """
    h_feat = hiddens.index_select(0, edge_src_chunk)  # (c, B, D)
    r_feat = rel_t.index_select(0, edge_rel_chunk)    # (c, B, D)
    msg = h_feat * r_feat                              # (c, B, D)
    if _HAS_TORCH_SCATTER:
        return _scatter_add(msg, edge_dst_chunk, dim=0, dim_size=n_ent)
    # Pure PyTorch fallback (works but slightly slower)
    out = torch.zeros(n_ent, hiddens.size(1), hiddens.size(2), device=hiddens.device, dtype=hiddens.dtype)
    out.index_add_(0, edge_dst_chunk, msg)
    return out

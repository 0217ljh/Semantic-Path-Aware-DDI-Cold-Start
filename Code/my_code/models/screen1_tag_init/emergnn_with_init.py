"""EmerGNN with external init tensor (Screen 1 variant injection point).

Adds a new `feat='X'` (eXternal init) mode to the EmerGNN backbone:
  - X mode: nn.Embedding(n_ent, n_dim) initialized from a caller-provided
    init tensor; trainable by default so SGD can adapt; can be frozen via
    `freeze_init=True` for the strictest "external semantic only" experiment.

The architectural skeleton (relation attention, chunked message passing,
bidirectional flow, score head) is identical to the upstream EmerGNN — we
only swap how `_entity_embed` produces the (B, n_dim) source vector at the
start of each propagation.

This file does NOT modify any code under Code/baseline/emergnn/. It imports
the upstream class and subclasses it.

Per Codex review #1 WARN #12: this is the integration point that lets a
config dump prove "only init changes, backbone frozen".
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

# Import the upstream pure-PyTorch EmerGNN
from baseline.emergnn.model import EmerGNN as _UpstreamEmerGNN


class EmerGNN_TAG(_UpstreamEmerGNN):
    """EmerGNN with feat='X' external-init mode.

    Same forward / propagation as upstream; differs only at __init__ and
    `_entity_embed`. Score head follows the upstream 'E' mode shape
    (Linear(4*n_dim, 1)) so source embeddings carry into the final score.
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
        # Legacy passthroughs kept so a config blob with these can still build
        morgan_features: Optional[np.ndarray] = None,
        morgan_feat_dim: int = 1024,
    ) -> None:
        if feat != "X":
            # Forward non-X requests to the upstream class transparently.
            super().__init__(
                n_ent=n_ent,
                n_base_rel=n_base_rel,
                n_dim=n_dim,
                length=length,
                feat=feat,
                morgan_features=morgan_features,
                morgan_feat_dim=morgan_feat_dim,
            )
            return

        # ---- X mode init: skip the upstream __init__'s feat-branch and
        # ---- replicate only what's needed, then add our external embedding.
        nn.Module.__init__(self)
        self.n_ent = n_ent
        self.n_base_rel = n_base_rel
        self.n_dim = n_dim
        self.L = length
        self.feat = "X"
        self.all_rel = 2 * n_base_rel + 1

        if external_init is None:
            raise ValueError("feat='X' requires external_init Tensor[n_ent, n_dim]")
        if isinstance(external_init, np.ndarray):
            external_init = torch.from_numpy(external_init.astype(np.float32))
        external_init = external_init.float()
        assert external_init.shape == (n_ent, n_dim), (
            f"external_init shape {tuple(external_init.shape)} != ({n_ent}, {n_dim})"
        )

        self.ent_kg = nn.Embedding(n_ent, n_dim)
        with torch.no_grad():
            self.ent_kg.weight.copy_(external_init)
        if freeze_init:
            self.ent_kg.weight.requires_grad = False
        self.freeze_init = bool(freeze_init)

        # Score head matches upstream 'E' shape (head_emb + tail_emb + head_hid + tail_hid)
        self.Wr = nn.Linear(4 * n_dim, 1)
        # Required by parent code paths even though we don't use Went:
        self.Went = None

        # Per-layer relation embedding tables and linear transforms
        self.rel_kg = nn.ModuleList([nn.Embedding(self.all_rel, n_dim) for _ in range(self.L)])
        self.linear = nn.ModuleList([nn.Linear(n_dim, n_dim) for _ in range(self.L)])
        self.act = nn.ReLU()

        # Attention over relation slots (bottlenecked through 5-D layer per paper)
        self.relation_linear = nn.ModuleList([nn.Linear(2 * n_dim, 5) for _ in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, self.all_rel) for _ in range(self.L)])

        # Match upstream chunking defaults
        self._chunk_size = 100_000
        self._use_checkpoint = True

        # Xavier-init everything except ent_kg (which we already set)
        for n, p in self.named_parameters():
            if n == "ent_kg.weight":
                continue
            if p.data.ndim > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p.data)

    def _entity_embed(self, idx: torch.Tensor) -> torch.Tensor:
        if self.feat == "X":
            return self.ent_kg(idx)
        return super()._entity_embed(idx)

    def forward(
        self,
        head: torch.Tensor,
        tail: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: X mode uses 'E'-style concat (4*n_dim)."""
        if self.feat != "X":
            return super().forward(head, tail, edge_src, edge_dst, edge_rel)

        head_embed = self._entity_embed(head)
        tail_embed = self._entity_embed(tail)
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)

        hid_uv = self._propagate(head, head_embed, ht_embed, edge_src, edge_dst, edge_rel)
        B = head.size(0)
        b_arange = torch.arange(B, device=head.device)
        tail_hid = hid_uv[tail, b_arange]

        hid_vu = self._propagate(tail, tail_embed, ht_embed, edge_src, edge_dst, edge_rel)
        head_hid = hid_vu[head, b_arange]

        # X mode: same concat shape as 'E' (4 * n_dim) -> Wr(4*n_dim, 1)
        embed = torch.cat([head_embed, tail_embed, head_hid, tail_hid], dim=-1)
        logits = self.Wr(embed).squeeze(-1)
        return logits


__all__ = ["EmerGNN_TAG"]

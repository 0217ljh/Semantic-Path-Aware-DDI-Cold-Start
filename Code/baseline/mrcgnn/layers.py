"""MRCGNN sub-modules — Discriminator, AvgReadout, MLP head.

Ported + adapted from the upstream MRCGNN core
``Paper/Reference/Original-Code/MRCGNN/codes for MRCGNN/layer.py:28-80``
(AAAI-2023 "Multi-relational Contrastive Learning Graph Neural Network for
Drug-drug Interaction Event Prediction").

File-independence (CLAUDE.md §Baseline): this is a COPY+adapt, NOT an import of
the upstream tree. The three pieces are faithful re-implementations:

* ``Discriminator`` — the DGI-style bilinear contrastive scorer. Upstream hard-codes
  ``nn.Bilinear(32, 32, 1)`` (``layer.py:31``) because the second RGCN layer emits
  ``hidden2 = 32`` features. We keep the mechanism identical but take ``n_h = hidden2``
  as a constructor arg so the width is no longer a magic constant (correction #4).
* ``AvgReadout`` — mean graph-summary readout (``layer.py:58-67``), verbatim.
* ``MLPHead`` — the 7-entry ``nn.ModuleList`` MLP pair classifier (``layer.py:100-107``,
  applied via ``MRCGNN.MLP`` ``layer.py:125-129``). Upstream fixes the input width to
  ``448`` and output to ``65``; we recompute the input width from the fusion arithmetic
  and take the output width ``K_global`` as an arg (corrections #4, #5). See the width
  derivation in :func:`mlp_input_width`.

No device pinning here (correction #4): modules follow their parent's ``.to(device)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp_input_width(hidden1: int, hidden2: int, skip_dim: int) -> int:
    """Width of the concatenated pair vector fed to the MLP head.

    Faithful to upstream ``layer.py:176-187``:
      * layer-attention fusion per drug = ``cat(attt0 * x1_o, attt1 * x2_o)``
        -> ``hidden1 + hidden2`` features   (``layer.py:176``)
      * molecular skip concat per drug     -> ``+ skip_dim``  (``layer.py:184-185``)
        so each drug vector = ``hidden1 + hidden2 + skip_dim``
      * the pair concatenates the two drug vectors  (``layer.py:187``)
        so the MLP input = ``2 * (hidden1 + hidden2 + skip_dim)``

    With the paper defaults ``hidden1=64, hidden2=32, skip_dim=128`` this yields
    ``2 * (64 + 32 + 128) = 448`` — matching upstream ``nn.Linear(448, 256)``
    (``layer.py:100``).
    """
    return 2 * (hidden1 + hidden2 + skip_dim)


class Discriminator(nn.Module):
    """DGI-style bilinear discriminator (upstream ``layer.py:28-55``).

    Scores a node embedding against a global graph summary ``c``. ``n_h`` is the
    per-node embedding width (upstream ``hidden2 = 32``; now a constructor arg).
    """

    def __init__(self, n_h: int) -> None:
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self._weights_init(m)

    @staticmethod
    def _weights_init(m: nn.Module) -> None:
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c: torch.Tensor, h_pl: torch.Tensor, h_mi: torch.Tensor,
                s_bias1: torch.Tensor | None = None,
                s_bias2: torch.Tensor | None = None) -> torch.Tensor:
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)
        return logits


class AvgReadout(nn.Module):
    """Mean graph-summary readout (upstream ``layer.py:58-67``), verbatim."""

    def forward(self, seq: torch.Tensor, msk: torch.Tensor | None = None) -> torch.Tensor:
        if msk is None:
            return torch.mean(seq, 0)
        msk = torch.unsqueeze(msk, -1)
        return torch.sum(seq * msk, 0) / torch.sum(msk)


class MLPHead(nn.Module):
    """7-entry MLP pair classifier (upstream ``layer.py:100-107`` + ``MLP`` at
    ``layer.py:125-129``).

    Upstream layout: ``Linear(448, 256) -> ELU -> Dropout(0.1) -> Linear(256, 128)
    -> ELU -> Dropout(0.1) -> Linear(128, K)``. We keep that structure exactly;
    only the input width (from :func:`mlp_input_width`) and the output width
    (``K_global``) are parameterized (corrections #4, #5). The forward walks all 7
    entries, mirroring upstream ``for i in range(layer): vectors = self.mlp[i](...)``.
    """

    def __init__(self, in_width: int, n_classes: int) -> None:
        super().__init__()
        self.mlp = nn.ModuleList([
            nn.Linear(in_width, 256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_classes),
        ])

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        for layer in self.mlp:
            vectors = layer(vectors)
        return vectors


__all__ = ["Discriminator", "AvgReadout", "MLPHead", "mlp_input_width"]

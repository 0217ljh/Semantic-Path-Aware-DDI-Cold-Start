"""TrimNet molecular encoder — offline supervised pretrain for MRCGNN drug features.

Ported + adapted from upstream
``Paper/Reference/Original-Code/MRCGNN/codes for MRCGNN/trimnet/models.py:46-192``
(the ``MultiHeadTripletAttention`` triplet-attention message passing, the ``Block``
GRU-recurrent conv, and the ``TrimNet`` set2set encoder + pair classifier).
File-independence (CLAUDE.md §Baseline): COPY+adapt, no import of
``Paper/Reference/Original-Code/`` or ``reproductions/``.

Used by the OFFLINE builder only (``_data/necessary/build_trimnet_features.py``).
TrimNet is a SUPERVISED offline pretrain: it is trained as a DDI-event classifier on
the leaf's TRAIN pairs, then per-drug embeddings are dumped from the FINAL training
state via :meth:`TrimNet.get_weight` (upstream ``trimnet/train.py:179`` +
``models.py:173-192``) — NOT a best-val checkpoint.

Deviations from upstream (correction #4 + parameterization):
  * ``n_drugs`` replaces the hard-coded ``572`` in ``get_weight`` (upstream
    ``models.py:185`` ``repr_h.view((572,-1))``); the view is derived from the
    actual node count, and the embeddings are returned (not written to a fixed
    ``.npy`` path inside the module — the builder owns I/O).
  * device-agnostic: no ``os.environ["CUDA_VISIBLE_DEVICES"]`` / unconditional
    ``.cuda()`` (upstream ``models.py:31``); modules follow ``.to(device)``.
  * classifier output width ``n_classes`` parameterized (upstream fixes 65 via
    ``lin1``/``mlp`` ``models.py:124,140``). Unused-by-default heads (``lin1``,
    ``RESCAL``) dropped — the pair MLP head is what upstream ``forward`` uses
    (``models.py:168``).

Faithful hyperparameters (upstream ``trimnet/train.py:250`` ``TrimNet(55, 10,
hidden_dim=64, depth=3, heads=4, dropout=0.2, outdim=1)``): in_dim 55, edge_in_dim 10,
hidden_dim 64, depth 3, heads 4, dropout 0.2. Set2Set processing_steps 3
(``models.py:132``). The classifier MLP is the 7-entry ``256->256->128->K`` stack
(``models.py:134-141``); its 256 input = Set2Set doubles hidden_dim (64*2=128) per
drug, concatenated over the pair (128*2=256) (``models.py:156-168``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, Parameter
from torch.nn.functional import leaky_relu
from torch.nn.init import kaiming_uniform_, zeros_
from torch_geometric.nn import LayerNorm, Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax


class MultiHeadTripletAttention(MessagePassing):
    """Triplet (head, edge, tail) multi-head attention message passing.

    Verbatim from upstream ``models.py:46-95`` (only formatting adapted).
    """

    def __init__(self, node_channels: int, edge_channels: int, heads: int = 3,
                 negative_slope: float = 0.2, **kwargs) -> None:
        super().__init__(aggr="add", node_dim=0, **kwargs)
        self.node_channels = node_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.weight_node = Parameter(torch.Tensor(node_channels, heads * node_channels))
        self.weight_edge = Parameter(torch.Tensor(edge_channels, heads * node_channels))
        self.weight_triplet_att = Parameter(torch.Tensor(1, heads, 3 * node_channels))
        self.weight_scale = Parameter(torch.Tensor(heads * node_channels, node_channels))
        self.bias = Parameter(torch.Tensor(node_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight_node)
        kaiming_uniform_(self.weight_edge)
        kaiming_uniform_(self.weight_triplet_att)
        kaiming_uniform_(self.weight_scale)
        zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        x = torch.matmul(x, self.weight_node)
        edge_attr = torch.matmul(edge_attr, self.weight_edge)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_index_i, edge_attr, size_i):
        x_j = x_j.view(-1, self.heads, self.node_channels)
        x_i = x_i.view(-1, self.heads, self.node_channels)
        e_ij = edge_attr.view(-1, self.heads, self.node_channels)

        triplet = torch.cat([x_i, e_ij, x_j], dim=-1)
        alpha = (triplet * self.weight_triplet_att).sum(dim=-1)
        alpha = leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, ptr=None, num_nodes=size_i)
        alpha = alpha.view(-1, self.heads, 1)
        return alpha * e_ij * x_j

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.node_channels)
        aggr_out = torch.matmul(aggr_out, self.weight_scale)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def extra_repr(self) -> str:
        return "{node_channels}, {node_channels}, heads={heads}".format(**self.__dict__)


class Block(torch.nn.Module):
    """GRU-recurrent triplet-attention block (upstream ``models.py:98-112``)."""

    def __init__(self, dim: int, edge_dim: int, heads: int = 4, time_step: int = 3) -> None:
        super().__init__()
        self.time_step = time_step
        self.conv = MultiHeadTripletAttention(dim, edge_dim, heads)
        self.gru = GRU(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        for _ in range(self.time_step):
            m = F.celu(self.conv.forward(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.ln(x.squeeze(0))
        return x


class TrimNet(torch.nn.Module):
    """TrimNet encoder + pair DDI-event classifier (upstream ``models.py:115-192``).

    ``forward((h_data, t_data, rels))`` -> (pair logits (B, n_classes), rels) for the
    supervised pretrain (upstream ``models.py:148-170``). ``get_weight`` runs the
    encoder over a batch of ALL drugs and returns per-drug embeddings from the FINAL
    state (upstream ``models.py:173-192``).
    """

    def __init__(self, in_dim: int, edge_in_dim: int, hidden_dim: int = 64,
                 depth: int = 3, heads: int = 4, dropout: float = 0.1,
                 n_classes: int = 65) -> None:
        super().__init__()
        self.depth = depth
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.initial_norm = LayerNorm(in_dim)                 # upstream LayerNorm(55)
        self.lin0 = Linear(in_dim, hidden_dim)                # upstream Linear(55, hidden)
        self.convs = nn.ModuleList([
            Block(hidden_dim, edge_in_dim, heads) for _ in range(depth)
        ])
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        # pair classifier: Set2Set doubles hidden_dim per drug (2*hidden),
        # concatenated over the pair (4*hidden). upstream fixes 256 for hidden=64
        # (models.py:134-141). Parameterized output width -> n_classes.
        pair_in = 4 * hidden_dim
        self.mlp = nn.ModuleList([
            nn.Linear(pair_in, 256),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_classes),
        ])

    def _mlp(self, vectors, layer: int):
        for i in range(layer):
            vectors = self.mlp[i](vectors)
        return vectors

    def _encode(self, data):
        """Per-drug graph -> (n_drugs_in_batch, 2*hidden) embedding (upstream body of
        ``forward`` / ``get_weight``, ``models.py:150-156,177-181``)."""
        data.x = self.initial_norm(data.x, data.batch)
        x = F.celu(self.lin0(data.x))
        for conv in self.convs:
            x = x + F.dropout(conv(x, data.edge_index, data.edge_attr),
                              p=self.dropout, training=self.training)
        x = self.set2set(x, data.batch)
        return x

    def forward(self, triples):
        """Supervised-pretrain forward (upstream ``models.py:148-170``)."""
        h_data, t_data, rels = triples
        x = self._encode(h_data)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self._encode(t_data)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        xall = torch.cat((x, x1), dim=1)
        scores = self._mlp(xall, 7)
        return scores, rels

    def get_weight(self, drugs_batch, n_drugs: int) -> torch.Tensor:
        """Per-drug embeddings from the FINAL training state (upstream
        ``models.py:173-192``). ``n_drugs`` replaces the hard-coded 572 in the view.
        Returns a (n_drugs, 2*hidden_dim) tensor (upstream 128-d for hidden 64)."""
        repr_h = self._encode(drugs_batch)
        repr_h = repr_h.view((n_drugs, -1))
        return repr_h


__all__ = ["MultiHeadTripletAttention", "Block", "TrimNet"]

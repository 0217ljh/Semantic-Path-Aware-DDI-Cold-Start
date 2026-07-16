"""KnowDDI's ``model/GraphSAGE.py`` — global-graph generic node embedding.

Faithful port of KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/model/GraphSAGE.py``.

Divergence-point 2 from SumGNN (per the port brief): GraphSAGE runs ONCE on the
GLOBAL graph (``Classifier_model.forward`` gathers per-subgraph node reprs by
``ndata['idx']``), producing a layer-concatenated ``ndata['repr']``. This is NOT
SumGNN's per-subgraph R-GCN.

DGL-2.x note (codex thread 019f25df, point 6): ``dgl.nn.pytorch.conv.SAGEConv``
still accepts ``edge_weight`` on DGL 2.x — its documented forward signature is
``forward(graph, feat, edge_weight=None)`` in 2.0/2.1/2.4 (verified vs DGL docs,
NOT a compat shim). So ``layer(g, h, edge_weight=edge_weight)`` ports as-is.

Parametrization (locked decision): ``params.num_nodes`` sizes ``pre_embed`` and is
set by the wrapper from the ACTUAL entity count of the built graph (NOT the
hardcoded 35000 in the original train.py:65 / GraphSAGE.py:19).

INDEPENDENT copy (CLAUDE.md file-independence); no ``baseline.sumgnn`` import.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, params) -> None:
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(params.gcn_dropout)
        self.activation = F.relu
        self.aggregator_type = params.gcn_aggregator_type
        self.num_nodes = params.num_nodes
        self.num_rels = params.aug_num_rels
        self.emb_dim = params.emb_dim
        self.num_gcn_layers = params.num_gcn_layers

        self.pre_embed = nn.Parameter(torch.Tensor(self.num_nodes, self.emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.pre_embed, gain=nn.init.calculate_gain('relu'))

        self.rel_weght = nn.Parameter(torch.Tensor(self.num_rels, self.emb_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.rel_weght, gain=nn.init.calculate_gain('relu'))

        for i in range(self.num_gcn_layers):
            self.layers.append(SAGEConv(self.emb_dim, self.emb_dim, self.aggregator_type))

    def forward(self, g):
        h = self.pre_embed[g.ndata['idx']]
        h = self.dropout(h)
        edge_weight = self.rel_weght[g.edata['type']]
        for l, layer in enumerate(self.layers):
            g.ndata['h'] = layer(g, h, edge_weight=edge_weight)
            if l != len(self.layers) - 1:
                g.ndata['h'] = self.activation(g.ndata['h'])
                g.ndata['h'] = self.dropout(g.ndata['h'])
            h = g.ndata['h']

            if l == 0:
                x = torch.cat([self.pre_embed[g.ndata['idx']], g.ndata['h']], dim=1)
                g.ndata['repr'] = x.unsqueeze(1).reshape(-1, 2, self.emb_dim)
            else:
                g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)

        return g.ndata.pop('h')

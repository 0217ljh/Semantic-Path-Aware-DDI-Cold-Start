"""KnowDDI's ``model/Classifier_model.py`` — GraphSAGE-once + GSL + head.

Faithful port of KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/model/Classifier_model.py``.

This wires divergence-points 2/3 (per the port brief):
  * ``embedding_model`` (GraphSAGE) runs ONCE on the GLOBAL graph (:28), producing
    ``ndata['h']`` + layer-concat ``ndata['repr']`` for every entity.
  * Each subgraph gathers its node reprs by global id ``ndata['idx']`` (:29-30) —
    NOT a per-subgraph message-passing pass.
  * GSL refines the subgraph (:35), then the head is
    ``[ mean_nodes(repr) || head_repr || tail_repr ] -> Linear -> num_rels`` (:40-44).
  * ``use_pre_embeddings`` is hardcoded True (matches original :14), so
    ``score_dim = (1 + num_gcn_layers + num_infer_layers) * emb_dim``.

No DGL-0.6->2.x API break here: ``dgl.mean_nodes`` and ``g.nodes[idx].data[...]``
gather are unchanged on DGL 2.x. INDEPENDENT copy; no ``baseline.sumgnn`` import.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from dgl import mean_nodes

from .GraphSAGE import GraphSAGE
from .gsl_model import gsl_model


class Classifier_model(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params
        self.global_graph = params.global_graph
        self.n_rel = params.num_rels
        self.emb_dim = params.emb_dim
        self.use_pre_embeddings = True

        if self.use_pre_embeddings:
            self.score_dim = (1 + self.params.num_gcn_layers + self.params.num_infer_layers) * self.emb_dim
        else:
            self.score_dim = (self.params.num_gcn_layers + self.params.num_infer_layers) * self.emb_dim

        self.embedding_model = GraphSAGE(params)
        self.gsl_model = gsl_model(params)
        self.W_final = nn.Linear(3 * self.score_dim, self.n_rel)  # get score

    def forward(self, sub_graphs):
        g = sub_graphs
        self.global_graph.ndata['h'] = self.embedding_model(self.global_graph)
        g.ndata['h'] = self.global_graph.nodes[g.ndata['idx']].data['h']
        g.ndata['repr'] = self.global_graph.nodes[g.ndata['idx']].data['repr']

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)

        complete_graph = self.gsl_model(g)

        gsl_hidden = complete_graph.ndata['repr']
        head_hidden = gsl_hidden[head_ids].view(-1, self.score_dim)
        tail_hidden = gsl_hidden[tail_ids].view(-1, self.score_dim)
        g_out = mean_nodes(complete_graph, 'repr').view(-1, self.score_dim)

        pred = torch.cat([g_out, head_hidden, tail_hidden], dim=1)
        scores = self.W_final(pred)
        return scores

"""KnowDDI's ``model/gsl_model.py`` — the graph-structure-learning (GSL) crux.

Faithful port of KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/model/gsl_model.py``. This is the
core KnowDDI contribution and divergence-point 2/3 from SumGNN. Mechanisms
preserved 1:1:

* :meth:`gsl_model.build_full_connect_graph` (orig :151): per batched subgraph,
  build a block-diagonal FULLY connected graph, seed ``ndata h/repr/id/idx`` from
  the original subgraph, edge ``type`` = ``num_rels+1`` (the "resemble" relation)
  everywhere, then overwrite the ORIGINAL edges' ``type`` with their real relation.
* :class:`graph_structure_learner` (orig :40 / forward :86): connection-strength
  weights ``MLP( exp(-|h_u - h_v|) || rel_emb )`` over ALL node-pairs, blend in the
  original structure with weight ``lamda``, ``edge_softmax`` per node, then
  ``threshold`` sparsify.
* :class:`gsl_layer` (orig :111): message = ``src['h'] * weight``, reduce = sum,
  then ``apply_nodes`` (a small Linear+ReLU update).
* :meth:`gsl_model.forward` (orig :181): iterate ``num_infer_layers`` refinements,
  concatenating each layer's ``h`` onto ``repr``.

DGL-0.6 -> 2.x migrations (codex thread 019f25df; validated vs DGL 2.x docs):

  M-A  build_full_connect_graph: ``dgl.graph((row,col))`` -> pass explicit
       ``num_nodes=int(batch_num_nodes.sum())`` so the last node(s) are never
       under-counted. ``set_batch_num_nodes`` / ``set_batch_num_edges`` still exist
       and behave the same on 2.x (docs verified).
  M-B  ``complete_graph.edges[ori_row,ori_col].data['type'] = ...`` (endpoint-index
       setter) -> assign via ``edata['type'][eids]`` where
       ``eids = edge_ids(ori_row, ori_col)``. The endpoint-index *setter* is not
       part of the documented 2.x API; ``edge_ids`` is stable.
  M-C  graph_structure_learner.forward: the original set edge weights with
       ``complete_graph.edges[row,col].data['weight'] = weights`` where
       ``row,col = complete_graph.all_edges()`` — i.e. weights are ALREADY in
       edge-id order (all edges, in order). So this is exactly ``edata['weight'] =
       weights`` on 2.x (faithful, no reordering). The ori-graph blend similarly
       uses ``edge_ids(ori_row, ori_col)`` for the endpoint lookup.

Parallel-edge faithfulness note (codex 019f25df): ``ori_graph`` is a multigraph
(parallel KG relations between the same u,v are possible), so ``ori_row/ori_col``
can contain duplicate pairs -> ``ori_eids`` can contain duplicate indices, making
the indexed write ``edata[...][ori_eids] = ...`` a duplicate-index scatter
(PyTorch "undefined"/last-wins). This EXACTLY reproduces the original's
endpoint-index setter ``complete_graph.edges[ori_row,ori_col].data[...] = ...``,
which had the same last-wins-on-duplicates semantics — so the port is faithful.
For the weight blend ``ori_e_weight`` is all ones, so every duplicate write to the
same eid yields the identical value (harmless).

``edge_softmax``, ``update_all`` custom UDFs, and ``apply_nodes`` are all
unchanged on DGL 2.x (codex points 1/4/5). INDEPENDENT copy; no ``baseline.sumgnn``
import.
"""
from __future__ import annotations

from collections import OrderedDict

import dgl
import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, batch_norm=True, dropout=0.) -> None:
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            if l < num_layers - 1:
                layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.LeakyReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
                in_dim = hidden_dim
            else:
                layer_list['fc_score'] = nn.Linear(in_dim, 1)
        self.network = nn.Sequential(layer_list)

    def forward(self, emb):
        out = self.network(emb)
        return out


class NodeUpdateModule(nn.Module):
    def __init__(self, emb_dim) -> None:
        super(NodeUpdateModule, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.ReLU()

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(node.data['h'])
        return {'h': h}


class graph_structure_learner(torch.nn.Module):
    def __init__(self, params, rel_emb) -> None:
        super().__init__()
        self.params = params
        self.lamda = params.lamda
        self.edge_softmax = params.edge_softmax

        self.sparsify = params.sparsify
        self.threshold = params.threshold

        self.func_num = params.func_num
        self.emb_dim = params.emb_dim
        self.rel_emb = rel_emb
        self.gsl_rel_emb_dim = params.gsl_rel_emb_dim
        if self.func_num == 1:
            if self.params.gsl_has_edge_emb:
                self.MLP = MLP(inp_dim=self.emb_dim + self.gsl_rel_emb_dim, hidden_dim=params.MLP_hidden_dim,
                               num_layers=params.MLP_num_layers, batch_norm=True, dropout=params.MLP_dropout)
            else:
                self.MLP = MLP(inp_dim=self.emb_dim, hidden_dim=params.MLP_hidden_dim,
                               num_layers=params.MLP_num_layers, batch_norm=True, dropout=params.MLP_dropout)
        elif self.func_num == 2:
            self.l1_norm = nn.PairwiseDistance(p=1, keepdim=True)
        elif self.func_num == 3:
            self.l2_norm = nn.PairwiseDistance(p=2, keepdim=True)
        elif self.func_num == 4:
            self.cos = nn.CosineSimilarity(dim=1)

    def compute_similarity(self, src_hidden, dst_hidden, rel_embedding, func_num):
        if func_num == 0:  # none
            weights = torch.ones((src_hidden.shape[0], 1)).to(device=src_hidden.device)
        elif func_num == 1:  # MLP
            if self.params.gsl_has_edge_emb:
                weights = self.MLP(torch.cat([torch.exp(- torch.abs(src_hidden - dst_hidden)), rel_embedding], dim=1))
            else:
                weights = self.MLP(torch.exp(- torch.abs(src_hidden - dst_hidden)))
        elif func_num == 2:
            weights = self.l1_norm(src_hidden, dst_hidden)
        elif func_num == 3:
            weights = self.l2_norm(src_hidden, dst_hidden)
        elif func_num == 4:
            weights = self.cos(src_hidden, dst_hidden).unsqueeze(1)

        return weights

    def forward(self, complete_graph, ori_graph):
        n_feat = complete_graph.ndata['h']
        row, col = complete_graph.all_edges()
        rel_embedding = self.rel_emb(complete_graph.edata['type'])

        # compute weights for node-pairs
        weights = self.compute_similarity(n_feat[row], n_feat[col], rel_embedding, func_num=self.params.func_num)
        # DGL-2.x port (M-C): row,col = all_edges() is exactly the edge-id order, so
        # ``weights`` aligns 1:1 with edata. Original used the endpoint-index setter
        # ``complete_graph.edges[row,col].data['weight'] = weights``; on 2.x this is
        # the faithful ``edata['weight'] = weights`` (no reordering).
        complete_graph.edata['weight'] = weights

        # add origin graph structure to weight matrix
        ori_row, ori_col = ori_graph.all_edges()
        ori_e_weight = torch.ones((ori_graph.number_of_edges(), 1), dtype=torch.float, device=ori_graph.device)
        # DGL-2.x port (M-B): endpoint lookup via edge_ids instead of the
        # ``complete_graph.edges[ori_row,ori_col].data['weight']`` endpoint setter.
        ori_eids = complete_graph.edge_ids(ori_row, ori_col)
        complete_graph.edata['weight'][ori_eids] = (
            (1 - self.lamda) * complete_graph.edata['weight'][ori_eids] + self.lamda * ori_e_weight)

        # edge softmax and sparsify
        if self.edge_softmax:
            complete_graph.edata['weight'] = edge_softmax(complete_graph, complete_graph.edata['weight'])
        if self.sparsify:
            complete_graph.edata['weight'] = torch.where(
                complete_graph.edata['weight'] > self.threshold,
                complete_graph.edata['weight'],
                torch.zeros(complete_graph.edata['weight'].shape).to(complete_graph.device))

        return complete_graph


class gsl_layer(torch.nn.Module):
    def __init__(self, params, rel_emb) -> None:
        super().__init__()
        self.params = params
        self.emb_dim = params.emb_dim
        self.rel_emb = rel_emb
        self.graph_structure_learner = graph_structure_learner(self.params, self.rel_emb)
        self.apply_mod = NodeUpdateModule(self.emb_dim)

    def forward(self, complete_graph, ori_graph):
        def msg_func(edges):
            w = edges.data['weight']
            x = edges.src['h']
            msg = x * w
            return {'msg': msg}

        def reduce_func(nodes):
            return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

        complete_graph = self.graph_structure_learner(complete_graph, ori_graph)
        complete_graph.update_all(msg_func, reduce_func)
        complete_graph.apply_nodes(func=self.apply_mod)

        return complete_graph


class gsl_model(torch.nn.Module):
    def __init__(self, params) -> None:
        super().__init__()

        self.params = params
        self.ni_layer = params.num_infer_layers
        self.gsl_layers = nn.ModuleList()
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.gsl_rel_emb_dim = params.gsl_rel_emb_dim
        self.rel_emb = nn.Embedding(self.aug_num_rels, self.gsl_rel_emb_dim, sparse=False)
        for i in range(self.ni_layer):
            self.gsl_layers.append(gsl_layer(params, self.rel_emb))

    def build_full_connect_graph(self, ori_graph):
        # construct complete graphs for all graphs in the batch
        batch_num_nodes = ori_graph.batch_num_nodes()
        block_begin_idx = torch.cat([batch_num_nodes.new_zeros(1), batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
        block_end_idx = batch_num_nodes.cumsum(dim=0)
        dense_adj = torch.zeros((ori_graph.num_nodes(),
                                 ori_graph.num_nodes()),
                                dtype=torch.float,
                                device=ori_graph.device)
        for idx_b, idx_e in zip(block_begin_idx, block_end_idx):
            dense_adj[idx_b:idx_e, idx_b:idx_e] = 1.
        row, col = torch.nonzero(dense_adj).t().contiguous()

        # DGL-2.x port (M-A): pass explicit num_nodes so an all-isolated last node
        # can never shrink the graph (docs warn dgl.graph infers node count from
        # max id). block-diagonal all-ones incl. diagonal guarantees every node has
        # edges here, but explicit is safer + faithful to the intended shape.
        complete_graph = dgl.graph((row, col), num_nodes=int(batch_num_nodes.sum())).to(ori_graph.device)
        batch_num_edges = torch.pow(batch_num_nodes, 2)
        complete_graph.set_batch_num_nodes(batch_num_nodes)
        complete_graph.set_batch_num_edges(batch_num_edges)
        complete_graph.ndata['h'] = ori_graph.ndata['h']
        complete_graph.ndata['repr'] = ori_graph.ndata['repr']
        complete_graph.ndata['id'] = ori_graph.ndata['id']
        complete_graph.ndata['idx'] = ori_graph.ndata['idx']
        complete_graph.edata['type'] = torch.full(
            (complete_graph.number_of_edges(),),
            self.num_rels + 1,  # self.num_rels+1 is the rel_id of resemble in BKG
            dtype=ori_graph.edata['type'].dtype,
            device=ori_graph.device)
        ori_row, ori_col = ori_graph.all_edges()
        # DGL-2.x port (M-B): overwrite original edges' type via edge_ids lookup
        # instead of the endpoint-index setter.
        if ori_graph.number_of_edges() > 0:
            ori_eids = complete_graph.edge_ids(ori_row, ori_col)
            complete_graph.edata['type'][ori_eids] = ori_graph.edata['type']

        return complete_graph

    def forward(self, g):
        ori_graph = g
        # In order to learn the weights of all possible edges, build a fully connected graph
        complete_graph = self.build_full_connect_graph(ori_graph)

        for i in range(self.ni_layer):
            complete_graph = self.gsl_layers[i](complete_graph, ori_graph)
            complete_graph.ndata['repr'] = torch.cat(
                [complete_graph.ndata['repr'], complete_graph.ndata['h'].unsqueeze(1)], dim=1)

        return complete_graph

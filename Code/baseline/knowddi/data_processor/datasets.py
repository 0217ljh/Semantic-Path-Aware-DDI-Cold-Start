"""KnowDDI's ``data_processor/datasets.py`` — SubgraphDataset + directional pruning.

Faithful port of KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/data_processor/datasets.py``.

Divergence-point 1 (KnowDDI-only; SumGNN has NONE): :meth:`extract_r_digraph`
(orig :92) — DIRECTIONAL pruning that keeps only edges lying on head->...->tail
directed paths within ``dig_layer`` hops. This MUST be preserved.

The subgraph contract KnowDDI's GSL consumes (divergence-point 3): each item is a
directed subgraph carrying ``ndata['idx']`` (global ids, for the GraphSAGE-once
gather), ``ndata['id']`` (1=head / 2=tail / 0=other) and ``edata['type']``. The
direct head-tail edge is removed (orig :86-87) so the model can't read the label.

DGL-0.6 -> 2.x migrations (codex thread 019f25df, point 7; same know-how as the
repo's SumGNN port, borrowed as migration reference only):

  D-1  ``self.global_graph.subgraph(nodes)`` -> ``dgl.node_subgraph(g, nodes)``.
       node_subgraph auto-inherits ``edata`` (feature inheritance) and exposes the
       parent edge ids at ``sg.edata[dgl.EID]``; node order is preserved (roots
       0,1 stay at the front). We re-attach ``edata['type']`` explicitly via the
       parent EID map (identical to the original's ``.parent_eid`` reattach) and
       drop the reserved ``dgl.NID/EID`` node/edge features so downstream batching
       only carries the model's real features.
  D-2  ``edge_ids(0, 1, return_uv=True)`` still exists on 2.x and returns
       ``(u, v, eid)``; the original relied on it to find + remove the direct
       head-tail edge(s). Guard with ``has_edges_between(0, 1)`` (edge_ids without
       return_uv raises on absent edges on 2.x).
  D-3  ``dgl.edge_subgraph`` / ``dgl.node_subgraph`` preserve + relabel
       ``ndata['id']/'idx']`` and ``edata['type']`` (feature copy). We use
       ``relabel_nodes=True`` (default) so the returned graph is compactly indexed,
       matching the original's DGL-0.6 subgraph relabeling.

INDEPENDENT copy (CLAUDE.md file-independence); imports only from
``baseline.knowddi.utils``. Does NOT import ``baseline.sumgnn`` or the reproduction.
"""
from __future__ import annotations

import numpy as np
import dgl
import lmdb
import scipy.sparse as ssp
import torch
from torch.utils.data import Dataset

from ..utils.data_utils import process_files_ddi, process_files_decagon
from ..utils.graph_utils import deserialize, get_neighbors, ssp_multigraph_to_dgl

# lmdb-2.x port (SumGNN migration know-how): lmdb 2.x refuses to open the same env
# twice in one process, but KnowDDI opens the SAME db_path once per split
# (train/valid/test SubgraphDataset). Cache the read-only env per db_path and share
# it (identical semantics; the original relied on lmdb 0.98 allowing dup opens).
_ENV_CACHE: dict = {}


def _open_env(db_path: str):
    env = _ENV_CACHE.get(db_path)
    if env is None:
        env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        _ENV_CACHE[db_path] = env
    return env


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only."""

    def __init__(self, db_path, db_name, raw_data_paths=None, add_traspose_rels=None,
                 use_pre_embeddings=False, dataset='', kge_model='', ssp_graph=None,
                 id2entity=None, id2relation=None, rel=None, global_graph=None,
                 dig_layer=4, bkg_file_path=None) -> None:
        self.main_env = _open_env(db_path)
        self.db = self.main_env.open_db(db_name.encode())
        self.db_path = db_path
        self.db_name = db_name
        # use_pre_embeddings here selects KGE node features (not the pre_embed table,
        # which lives in GraphSAGE). Phase-1 keeps it False (no KGE npy available).
        self.node_features, self.kge_entity2id = (None, None)
        BKG_file = bkg_file_path

        if not ssp_graph:
            if dataset == 'drugbank' or dataset == 'drugbank_sub':
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel = \
                    process_files_ddi(raw_data_paths, BKG_file)
            elif dataset == 'BioSNAP':
                ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = \
                    process_files_decagon(raw_data_paths, BKG_file)

            self.num_rels = rel
            print('number of relations:%d' % (self.num_rels))

            # Add transpose matrices to handle both directions of relations.
            if add_traspose_rels:
                ssp_graph_t = [adj.T for adj in ssp_graph]
                ssp_graph += ssp_graph_t

            # add self loops
            ssp_graph.append(ssp.identity(len(id2entity)))

            # effective # relations after adding symmetric adjs and/or self connections
            self.aug_num_rels = len(ssp_graph)
            self.global_graph = ssp_multigraph_to_dgl(ssp_graph)
            self.ssp_graph = ssp_graph
        else:
            self.num_rels = rel
            self.aug_num_rels = len(ssp_graph)
            self.global_graph = global_graph
            self.ssp_graph = ssp_graph

        self.id2entity = id2entity
        self.id2relation = id2relation
        self.num_entity = len(id2entity)
        self.dig_layer = dig_layer
        with self.main_env.begin(db=self.db) as txn:
            self.num_graphs = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes, r_label, g_label, n_labels = deserialize(txn.get(str_id)).values()
            directed_subgraph = self._prepare_subgraphs(nodes, n_labels)
            return directed_subgraph, r_label, g_label

    def __len__(self):
        return self.num_graphs

    def _prepare_subgraphs(self, nodes, n_labels):
        # D-1: node_subgraph induces the node-relabeled subgraph (roots 0,1 stay at
        # the front) and inherits edata; re-attach edata['type'] via the parent EID
        # map, then drop reserved NID/EID so batching only carries real features.
        subgraph = dgl.node_subgraph(self.global_graph, nodes)
        subgraph.edata['type'] = self.global_graph.edata['type'][subgraph.edata[dgl.EID]]
        for _k in (dgl.NID, dgl.EID):
            if _k in subgraph.ndata:
                del subgraph.ndata[_k]
            if _k in subgraph.edata:
                del subgraph.edata[_k]
        subgraph.ndata['idx'] = torch.LongTensor(np.array(nodes))
        subgraph = self._prepare_features(subgraph, n_labels)
        # D-2: remove the direct head-tail edge(s) so the model can't read the label.
        if subgraph.has_edges_between(0, 1):
            _, _, edges_btw_roots = subgraph.edge_ids(0, 1, return_uv=True)
            subgraph.remove_edges(edges_btw_roots)

        directed_subgraph = self.extract_r_digraph(subgraph)
        return directed_subgraph

    def extract_r_digraph(self, graph):
        """Directional pruning (KnowDDI paper's algorithm; orig datasets.py:92).

        Keep only edges on directed head->...->tail paths of length <= dig_layer.
        Walk out-neighborhoods from head for ``dig_layer`` hops recording per-hop
        edges, then trace back from tail, selecting edges whose dst is a reachable
        tail-frontier node. Returns an edge-induced (or node-induced, if empty)
        subgraph. Divergence-point 1 vs SumGNN — MUST be preserved.
        """
        head_nodes = (graph.ndata['id'] == 1).nonzero().squeeze(1)
        tail_nodes = (graph.ndata['id'] == 2).nonzero().squeeze(1)

        total_nodes = torch.cat([head_nodes, tail_nodes])
        raw_layer_edges = {}
        for i in range(self.dig_layer):
            head_nodes, head_edges = get_neighbors(graph, head_nodes)
            raw_layer_edges[i] = head_edges

        layer_edges_id = torch.LongTensor([])

        for i in reversed(range(self.dig_layer)):
            select = torch.nonzero(torch.eq(raw_layer_edges[i][:, 1], tail_nodes.unsqueeze(1)))
            l_edge = raw_layer_edges[i][select[:, 1]]

            layer_edges_id = torch.cat([layer_edges_id, l_edge[:, 2]])
            tail_nodes = torch.unique(l_edge[:, 0])

        total_edges = torch.unique(layer_edges_id, dim=0, sorted=True)
        if total_edges.numel():
            r_digraph = dgl.edge_subgraph(graph, total_edges)
        else:
            # If the extracted subgraph has no edges, then the returned graph only
            # has head and tail.
            r_digraph = dgl.node_subgraph(graph, total_nodes)

        return r_digraph

    def _prepare_features(self, subgraph, n_labels):
        n_nodes = subgraph.number_of_nodes()
        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        return subgraph


__all__ = ["SubgraphDataset"]

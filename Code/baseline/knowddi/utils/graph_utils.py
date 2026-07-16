"""KnowDDI's ``utils/graph_utils.py`` — ssp<->dgl + BFS + batching helpers.

Ported from KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/utils/graph_utils.py``.
The ONLY DGL-0.6->2.x migration here is :func:`ssp_multigraph_to_dgl` (original
:42-60 used ``dgl.DGLGraph(multigraph=True)`` + ``dgl.from_networkx(...)`` which
no longer exists on DGL 2.x). We rebuild the same homogeneous multigraph directly
from every relation adjacency's COO, stacking a per-edge ``type`` feature =
relation index — identical migration to the repo's already-validated SumGNN port
(``Code/baseline/sumgnn/_core/utils/graph_utils.py:72-108``, borrowed as migration
know-how only, NOT design). Edge order is relation-major then COO order, so
downstream ``edata['type']`` alignment matches the original.

Everything else (:func:`serialize`, :func:`deserialize`, :func:`incidence_matrix`,
:func:`remove_nodes`, :func:`collate_dgl`, :func:`move_batch_to_device_dgl`,
:func:`_bfs_relational`, :func:`get_neighbors`) is a faithful port. KnowDDI's
``get_neighbors`` (:129-134) uses ``out_edges(nodes, form='all')`` which is
unchanged on DGL 2.x — it drives the directional pruning ``extract_r_digraph``.

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence). Does
NOT import ``baseline.sumgnn`` or the reproduction tree.
"""
from __future__ import annotations

import pickle
import random

import dgl
import numpy as np
import scipy.sparse as ssp
import torch


def serialize(data: dict) -> bytes:
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data: bytes) -> dict:
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def incidence_matrix(adj_list):
    """adj_list: List of sparse adjacency matrices."""
    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def ssp_multigraph_to_dgl(graph):
    """Convert an ssp multigraph (list of adjacency matrices) to a dgl multigraph.

    # DGL-0.6->2.x port: the original built
    ``g = dgl.DGLGraph(multigraph=True); g = dgl.from_networkx(nx_g, edge_attrs=['type'])``
    over a ``nx.MultiDiGraph``. On DGL 2.x ``DGLGraph(multigraph=...)`` is gone; we
    build the same graph straight from COO. Each ``(src, dst)`` becomes edge id
    ``i`` (duplicates allowed = multigraph), and ``edata['type']`` carries the
    relation index in the same edge order the original produced.
    """
    num_nodes = graph[0].shape[0]
    src_list, dst_list, type_list = [], [], []
    for rel, adj in enumerate(graph):
        adjcoo = adj.tocoo()
        src_list.append(adjcoo.row)
        dst_list.append(adjcoo.col)
        type_list.append(np.full(len(adjcoo.row), rel, dtype=np.int64))

    if src_list:
        src = np.concatenate(src_list)
        dst = np.concatenate(dst_list)
        etype = np.concatenate(type_list)
    else:
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)
        etype = np.zeros(0, dtype=np.int64)

    g_dgl = dgl.graph((torch.from_numpy(src.astype(np.int64)),
                       torch.from_numpy(dst.astype(np.int64))),
                      num_nodes=num_nodes)
    g_dgl.edata['type'] = torch.from_numpy(etype)
    # KnowDDI's original tagged every node with its global id in ndata['idx']
    # (graph_utils.py:59); Classifier/GraphSAGE index the pre_embed table by this.
    g_dgl.ndata['idx'] = torch.LongTensor(np.arange(g_dgl.num_nodes()))
    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of (subgraph, r_label, g_label) triples.
    graphs, r_labels, g_labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, r_labels, g_labels


def move_batch_to_device_dgl(batch, device, multi_type: int = 0):
    g_dgl_pos, r_labels_pos, targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    if multi_type == 1:
        r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)
    elif multi_type == 2:
        r_labels_pos = torch.FloatTensor(r_labels_pos).to(device=device)
    # DGL-2.x port: in DGL 2.x graph structure + features must share a device, so
    # ``g.to(device)`` moves everything atomically (the DGL-0.6 per-feature idiom
    # raised device-mismatch). Same fix the SumGNN port used (send_graph_to_device).
    g_dgl_pos = g_dgl_pos.to(device)

    return g_dgl_pos, r_labels_pos, targets_pos


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """BFS for graphs (modified from dgl.contrib to accommodate node sampling)."""
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:
        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(list(next_lvl), max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def get_neighbors(dgl_graphs, nodes):
    """Port of KnowDDI ``get_neighbors`` (graph_utils.py:129).

    Returns ``(new_nodes, sampled_edges)`` where ``sampled_edges`` is an
    ``(E, 3)`` LongTensor of ``[src, dst, eid]`` for every out-edge of ``nodes``.
    ``out_edges(..., form='all')`` is unchanged on DGL 2.x. Drives the directional
    pruning in ``SubgraphDataset.extract_r_digraph``.
    """
    src, dst, eid = dgl_graphs.out_edges(nodes, form='all')
    sampled_edges = torch.cat(
        [src.unsqueeze(1), dst.unsqueeze(1), eid.unsqueeze(1)], dim=1).to(device=nodes.device)
    new_nodes, new_index = torch.unique(sampled_edges[:, 1], dim=0, sorted=True, return_inverse=True)

    return new_nodes, sampled_edges


__all__ = [
    "serialize", "deserialize", "remove_nodes", "incidence_matrix",
    "ssp_multigraph_to_dgl", "collate_dgl", "move_batch_to_device_dgl",
    "_bfs_relational", "get_neighbors",
]

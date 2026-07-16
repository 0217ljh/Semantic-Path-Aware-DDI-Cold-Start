"""Graph utilities for the SumGNN reproduction.

Byte-for-byte port of SumGNN's `utils/graph_utils.py` EXCEPT the DGL-0.4.x
constructs, which are rewritten for DGL 2.4.0 (see the three functions marked
`# DGL-2.4 port`). Semantics are preserved:

* `ssp_multigraph_to_dgl` builds a single homogeneous multigraph carrying an
  edge feature `type` (the relation index), exactly as the original did via
  `dgl.DGLGraph(multigraph=True).from_networkx(edge_attrs=['type'])`.
* `send_graph_to_device` moves node/edge tensors to `device` using the 2.4 API
  (`g.ndata`/`g.edata` are direct dict-like views).
"""
import numpy as np
import scipy.sparse as ssp
import torch
import networkx as nx
import dgl
import pickle


def serialize(data):
    data_tuple = tuple(data.values())
    return pickle.dumps(data_tuple)


def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


def get_edge_count(adj_list):
    count = []
    for adj in adj_list:
        count.append(len(adj.tocoo().row.tolist()))
    return np.array(count)


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

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


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def ssp_to_torch(A, device, dense=False):
    '''
    A : Sparse adjacency matrix
    '''
    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    dat = torch.FloatTensor(A.tocoo().data)
    A = torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).to(device=device)
    return A


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.

    # DGL-2.4 port: the original used
    ``g = dgl.DGLGraph(multigraph=True); g.from_networkx(nx_g, edge_attrs=['type'])``
    which no longer exists in DGL 2.x. We build the same homogeneous multigraph
    directly from the coo of every relation adjacency, stacking a per-edge
    ``type`` feature = relation index. Edge order matches the original
    (relation-major, then coo order), so downstream ``edata['type']`` alignment is
    identical.
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

    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, g_labels_pos, r_labels_pos = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)
    return (batched_graph_pos, r_labels_pos), g_labels_pos


def move_batch_to_device_dgl(batch, device):
    (g_dgl_pos, r_labels_pos), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)

    return g_dgl_pos, r_labels_pos, targets_pos


def move_batch_to_device_dgl_ddi2(batch, device):
    (g_dgl_pos, r_labels_pos), targets_pos = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.FloatTensor(r_labels_pos).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)

    return g_dgl_pos, r_labels_pos, targets_pos


def send_graph_to_device(g, device):
    # DGL-2.4 port: in DGL 2.x the graph STRUCTURE and its features must live on the
    # same device, so per-feature ``.to(device)`` (the DGL-0.4 idiom) raises a device
    # mismatch. ``g.to(device)`` moves structure + all node/edge features atomically,
    # which is the equivalent operation.
    return g.to(device)


#  The following three functions are modified from networks source codes to
#  accomodate diameter and radius for dirercted graphs


def eccentricity(G):
    e = {}
    for n in G.nbunch_iter():
        length = nx.single_source_shortest_path_length(G, n)
        e[n] = max(length.values())
    return e


def radius(G):
    e = eccentricity(G)
    e = np.where(np.array(list(e.values())) > 0, list(e.values()), np.inf)
    return min(e)


def diameter(G):
    e = eccentricity(G)
    return max(e.values())

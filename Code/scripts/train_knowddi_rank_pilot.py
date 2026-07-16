"""Subgraph-paradigm grid entry (KnowDDI) for the warm/cold rank-transfer analysis.

Faithful port of KnowDDI (Wang, Yang & Yao 2023, arXiv:2311.15056; upstream at
Paper/Reference/Original-Code/KnowDDI) adapted to our ddi_unified binary cold_s2
task + merged KG, producing the pre-scorer pair representation for
analyze_rank_sufficiency.py. Mirrors the R-GCN / EmerGNN rank pilots so the three
encoder paradigms (node / path / subgraph) share ONE grid protocol.

WHAT IS KEPT FAITHFUL (the subgraph paradigm's defining machinery):
  - Per-pair h-hop ENCLOSING subgraph with SEAL double-radius node labeling
    (subgraph_extraction.py::subgraph_extraction_labeling).
  - GraphSAGE node encoder over the WHOLE global KG graph, node features =
    learnable per-node embedding + KG structure ONLY (knowledge-only, no SMILES /
    Morgan) -> per-node multi-layer `repr` (GraphSAGE.py).
  - Graph Structure Learning (gsl_model.py): fully-connected subgraph + MLP edge
    reweighting exp(-|src-dst|) + rel emb, lamda blend of the original edges,
    edge-softmax + threshold sparsify, num_infer_layers rounds of weighted MP.
  - Pre-scorer pair representation pred = cat([mean_pool, head_repr, tail_repr])
    (Classifier_model.py:42). THIS is the rank-analysis target (== `raw`).

WHAT IS ADAPTED (documented, spec case B: paper task != our grid task):
  - Head: 86-way multiclass (drugbank) / multilabel (BioSNAP) -> 1 logit + BCE,
    metric AUROC. The GraphSAGE+GSL encoder and pred representation are untouched.
  - Global graph = OUR merged KG (mask1, DDI-removed) instead of HetioNet, using
    the SAME MergedKG node-id space as the R-GCN pilot (grid-fair). Train POSITIVE
    DDI pairs are added back as a relation (KnowDDI adds train DDI edges to the
    message-passing graph; valid/test DDI edges are never added -> no leakage).
  - Splits reuse train_gcn_rank_pilot.load_data verbatim (identical tr / warm_val /
    warm_test / cold_test as the node-paradigm entry).

RELATION-ID LAYOUT (see _build_relation_layout): gsl_model.build_full_connect_graph
hardcodes the "resemble" (inferred fully-connected) edge type as num_rels+1. In the
original HetioNet that slot was a real resemble relation; our merged KG has none, so
we RESERVE id (num_rels+1) as a dedicated resemble slot with no KG edges assigned,
avoiding entangling a real KG relation's embedding with inferred edges.

ENGINEERING (faithful-equivalent, NOT algorithm changes):
  - The combined incidence matrix (subgraph_extraction_labeling recomputes it per
    link over all relation adjacencies) is precomputed ONCE and reused. On our
    7.1M-edge KG the per-link recompute is infeasible; the result is identical.
  - Subgraphs are extracted in-process and cached to .pkl (node ids + labels per
    pair), avoiding the upstream LMDB / multiprocessing path (cross-platform pain);
    the enclosing-subgraph algorithm is byte-faithful.
  - DGL 2.4 API (upstream is 0.6.1): graphs are built from edge tensors directly
    (equivalent to ssp_multigraph_to_dgl); edge_softmax import path unchanged.

Run (from project root, WSL conda env project_1):
  python Code/scripts/train_knowddi_rank_pilot.py --fold fold0 --epochs 30

Then (per selected checkpoint):
  python Code/scripts/analyze_rank_sufficiency.py \
      --z Code/runs/<run_id>/pilot_Z.npz --key raw --title KnowDDI-warmbest
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import dgl
from dgl import mean_nodes
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.conv import SAGEConv


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
# Reuse the EXACT node-paradigm splits + KG-id mapping for a grid-fair comparison.
from train_gcn_rank_pilot import load_data, _pairs_to_idx, _canon, DDI800  # noqa: E402


def load_cold_val(fold: str) -> pd.DataFrame:
    """Official inductive S2 validation split (UNSEEN-drug pairs). Used ONLY to select
    the cold-best checkpoint -> never leaks cold_test. Canonicalized like load_data."""
    return _canon(pd.read_parquet(DDI800 / "inductive" / "S2" / fold / "val.parquet"))


# --------------------------------------------------------------------------- #
# KG -> per-relation graph  (faithful to data_utils.process_files_ddi, but the
# node-id space is MergedKG's and edges come from the merged-KG parquet)
# --------------------------------------------------------------------------- #
def _build_relation_layout(kg_relations: list[str]) -> dict:
    """Relation-id layout (see module docstring). num_rels = 1 (single binary DDI
    relation). Reserve id (num_rels+1) as a dedicated resemble slot.

      id 0                : DDI (train positive pairs)
      id 1                : the FIRST KG relation
      id 2 (= num_rels+1) : RESEMBLE (reserved; no KG edges)  [gsl inferred edges]
      id 3 .. 3+(K-2)     : the remaining K-1 KG relations
      id (3+K-1)          : self-loop  (appended last, like datasets.py)

    aug_num_rels = 3 + K  (K = number of KG relations).
    """
    num_rels = 1                       # binary DDI = one relation
    resemble_id = num_rels + 1         # == 2, matches gsl_model.build_full_connect_graph
    kg_rel_ids: dict[str, int] = {}
    nxt = 1
    for i, r in enumerate(kg_relations):
        if nxt == resemble_id:
            nxt += 1                   # skip the reserved resemble slot
        kg_rel_ids[r] = nxt
        nxt += 1
    self_loop_id = nxt
    aug_num_rels = self_loop_id + 1
    return {
        "num_rels": num_rels,
        "resemble_id": resemble_id,
        "ddi_id": 0,
        "kg_rel_ids": kg_rel_ids,
        "self_loop_id": self_loop_id,
        "aug_num_rels": aug_num_rels,
    }


def build_graph_tensors(kg: MergedKG, edges_path: str, tr_pos_pairs: np.ndarray):
    """Build the KnowDDI global-graph edge tensors (src, dst, type) and the combined
    undirected incidence (csr) used for enclosing-subgraph BFS.

    tr_pos_pairs: (P, 2) int64 node-idx array of TRAIN POSITIVE DDI pairs (the only
    DDI edges that enter message passing; valid/test never do -> no leakage).
    """
    n = kg.n_nodes
    edges = pd.read_parquet(edges_path, columns=["src", "dst", "relation"])
    rel_names = sorted(edges["relation"].astype(str).unique().tolist())
    layout = _build_relation_layout(rel_names)

    id2i = kg.id_to_idx
    src = edges["src"].astype(str).map(id2i).to_numpy()
    dst = edges["dst"].astype(str).map(id2i).to_numpy()
    good = ~(pd.isna(src) | pd.isna(dst))
    if int((~good).sum()):
        print(f"[kg] dropping {int((~good).sum())} KG edges with endpoints outside "
              f"MergedKG node set", flush=True)
    src = src[good].astype(np.int64)
    dst = dst[good].astype(np.int64)
    rtype = (edges["relation"].astype(str).map(layout["kg_rel_ids"]).to_numpy()[good]
             ).astype(np.int64)

    # DDI train-positive edges (relation id 0), undirected (add both directions)
    ddi_src = tr_pos_pairs[:, 0].astype(np.int64)
    ddi_dst = tr_pos_pairs[:, 1].astype(np.int64)
    ddi_type = np.zeros(len(ddi_src), dtype=np.int64)  # ddi_id == 0

    # self-loops (one per node), like datasets.py's ssp.identity relation
    sl = np.arange(n, dtype=np.int64)
    sl_type = np.full(n, layout["self_loop_id"], dtype=np.int64)

    # SYMMETRIZATION (deliberate, EVIDENCE-BACKED deviation from upstream's
    # add_traspose_rels=False default). We add both src<->dst with the SAME relation id
    # (correct undirected-relation semantics; upstream's transpose flag instead mints a
    # distinct reverse relation, for a genuinely directed KG). Justification, measured on
    # the merged KG (scratch analyze_kg_direction, 2026-07-05; codex-reviewed):
    #   * 60% of edges are already stored bidirectionally (reciprocity ~1.0) -> direction
    #     is redundant there; symmetrizing is a no-op.
    #   * Much of the one-way 40% is semantically UNDIRECTED but stored once (PPI,
    #     disease_disease, drug_protein, contraindication...) -> symmetrizing RECOVERS it.
    #   * Drugs are the SOURCE in ~98.5% of their KG edges (drug->target/effect/SE/...).
    #     Under directed-forward SAGEConv 45.6% of DDI drugs (292/640) get ZERO in-edges
    #     and aggregate NO neighbors -> KnowDDI's enrichment mechanism collapses on
    #     exactly the cold/unseen drugs. So direction carries ~no gain for drug reps and
    #     directed-forward is actively harmful; undirected is the right call (also grid-
    #     fair with the R-GCN entry, which uses this KG undirected). A directed variant,
    #     IF ever tested, must be option-C (inverse relations), never directed-forward.
    # Self-loops match datasets.py's appended ssp.identity relation.
    all_src = np.concatenate([src, dst, ddi_src, ddi_dst, sl])
    all_dst = np.concatenate([dst, src, ddi_dst, ddi_src, sl])
    all_typ = np.concatenate([rtype, rtype, ddi_type, ddi_type, sl_type])

    g = dgl.graph((torch.from_numpy(all_src), torch.from_numpy(all_dst)), num_nodes=n)
    g.edata["type"] = torch.from_numpy(all_typ)
    g.ndata["idx"] = torch.arange(n, dtype=torch.long)

    # Combined undirected incidence for BFS (subgraph_extraction_labeling::
    # incidence_matrix + its transpose). Includes KG + DDI + self-loops so the
    # enclosing subgraph is computed over the SAME graph the model sees.
    inc = ssp.csc_matrix(
        (np.ones(len(all_src), dtype=np.uint8), (all_src, all_dst)), shape=(n, n))
    inc = inc + inc.T
    inc = inc.tocsr()
    inc.data[:] = 1  # binary adjacency for BFS

    print(f"[kg] nodes={n} directed_edges={len(all_src)} KG_relations={len(rel_names)} "
          f"aug_num_rels={layout['aug_num_rels']} ddi_train_pos_edges={len(ddi_src)}",
          flush=True)
    return g, inc, layout


# --------------------------------------------------------------------------- #
# Enclosing-subgraph extraction (faithful to subgraph_extraction.py, incidence
# hoisted out of the per-link loop). Returns per-pair node list + SEAL labels.
# --------------------------------------------------------------------------- #
def _sp_row_vec(idx_list, dim):
    data = np.ones(len(idx_list))
    row = np.zeros(len(idx_list))
    return ssp.csr_matrix((data, (row, list(idx_list))), shape=(1, dim))


def _get_neighbors(adj, nodes):
    sp_nodes = _sp_row_vec(list(nodes), adj.shape[1])
    return set(ssp.find(sp_nodes.dot(adj))[1])


def _bfs_relational(adj, roots, max_nodes_per_hop, rng):
    visited = set()
    current = set(roots)
    while current:
        visited |= current
        nxt = _get_neighbors(adj, current) - visited
        if max_nodes_per_hop and max_nodes_per_hop < len(nxt):
            nxt = set(rng.choice(list(nxt), max_nodes_per_hop, replace=False).tolist())
        yield nxt
        current = nxt


def _neighbor_nodes(roots, adj, hop, max_nodes_per_hop, rng):
    gen = _bfs_relational(adj, roots, max_nodes_per_hop, rng)
    lvls = []
    for _ in range(hop):
        try:
            lvls.append(next(gen))
        except StopIteration:
            break
    return set().union(*lvls) if lvls else set()


def _remove_nodes(A, nodes):
    keep = list(set(range(A.shape[1])) - set(nodes))
    return A[keep, :][:, keep]


def _node_label(sub_adj, max_distance):
    """SEAL double-radius labeling (subgraph_extraction.py::node_label)."""
    roots = [0, 1]
    sgs = [_remove_nodes(sub_adj, [r]) for r in roots]
    d = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True,
                                      limit=1e6)[:, 1:], 0, 1e7) for sg in sgs]
    dist = np.array(list(zip(d[0][0], d[1][0])), dtype=int)
    target = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target, dist)) if dist.size else target
    enclosing = np.where(np.max(labels, axis=1) <= max_distance)[0]
    return labels, enclosing


def extract_enclosing_subgraph(n1, n2, inc, hop, max_nodes_per_hop, rng):
    """Faithful subgraph_extraction_labeling for enclosing_subgraph=True, with the
    combined incidence `inc` passed in (precomputed once)."""
    n1, n2 = int(n1), int(n2)
    r1 = _neighbor_nodes({n1}, inc, hop, max_nodes_per_hop, rng)
    r2 = _neighbor_nodes({n2}, inc, hop, max_nodes_per_hop, rng)
    common = r1.intersection(r2)
    common.discard(n1)
    common.discard(n2)
    subgraph_nodes = [n1, n2] + list(common)
    sub = inc[subgraph_nodes, :][:, subgraph_nodes]
    labels, enclosing = _node_label(sub, max_distance=hop)
    pruned_nodes = np.array(subgraph_nodes)[enclosing].tolist()
    pruned_labels = labels[enclosing]
    return pruned_nodes, pruned_labels


def build_subgraph_cache(pairs: np.ndarray, inc, hop, max_nodes_per_hop, seed, tag):
    """pairs: (M,2) node-idx. Returns list of (nodes, labels) per pair."""
    rng = np.random.default_rng(seed)
    out = []
    t0 = time.time()
    for i in range(len(pairs)):
        nodes, labels = extract_enclosing_subgraph(
            pairs[i, 0], pairs[i, 1], inc, hop, max_nodes_per_hop, rng)
        out.append((nodes, labels))
        if (i + 1) % 2000 == 0:
            print(f"[subgraph:{tag}] {i+1}/{len(pairs)} "
                  f"(mean_size={np.mean([len(n) for n, _ in out]):.1f}, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"[subgraph:{tag}] done {len(pairs)} pairs, "
          f"mean_size={np.mean([len(n) for n, _ in out]):.1f}, {time.time()-t0:.0f}s",
          flush=True)
    return out


# --------------------------------------------------------------------------- #
# Model  (faithful port of GraphSAGE.py + gsl_model.py + Classifier_model.py;
# only the final head is binary and forward also returns the pre-scorer `pred`)
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, batch_norm=True, dropout=0.):
        super().__init__()
        from collections import OrderedDict
        layers = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            if l < num_layers - 1:
                layers[f"fc{l}"] = nn.Linear(in_dim, hidden_dim)
                if batch_norm:
                    layers[f"norm{l}"] = nn.BatchNorm1d(hidden_dim)
                layers[f"relu{l}"] = nn.LeakyReLU()
                if dropout > 0:
                    layers[f"drop{l}"] = nn.Dropout(p=dropout)
                in_dim = hidden_dim
            else:
                layers["fc_score"] = nn.Linear(in_dim, 1)
        self.network = nn.Sequential(layers)

    def forward(self, emb):
        return self.network(emb)


class NodeUpdateModule(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.ReLU()

    def forward(self, node):
        h = self.linear(node.data["h"])
        h = self.activation(node.data["h"])  # faithful to upstream (activates pre-linear h)
        return {"h": h}


class GraphStructureLearner(nn.Module):
    def __init__(self, params, rel_emb):
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
        # Upstream gsl_model.py supports func_num 0/1/2/3/4; the drugbank-best config
        # (and this port) uses func_num=1 (MLP similarity). Guard so any other value
        # fails loudly instead of silently mis-behaving (codex 2026-07-05 nice-to-have).
        assert self.func_num == 1, (
            f"only func_num=1 (MLP similarity) is ported; got {self.func_num}. "
            f"Port the 0/2/3/4 branches from gsl_model.py if you need them.")
        in_dim = self.emb_dim + (self.gsl_rel_emb_dim if params.gsl_has_edge_emb else 0)
        self.MLP = MLP(in_dim, params.MLP_hidden_dim, params.MLP_num_layers,
                       batch_norm=True, dropout=params.MLP_dropout)

    def compute_similarity(self, src_h, dst_h, rel_emb):
        if self.params.gsl_has_edge_emb:
            return self.MLP(torch.cat([torch.exp(-torch.abs(src_h - dst_h)), rel_emb], dim=1))
        return self.MLP(torch.exp(-torch.abs(src_h - dst_h)))

    def forward(self, complete_graph, ori_graph):
        n_feat = complete_graph.ndata["h"]
        row, col = complete_graph.all_edges()
        rel_emb = self.rel_emb(complete_graph.edata["type"])
        weights = self.compute_similarity(n_feat[row], n_feat[col], rel_emb)
        complete_graph.edges[row, col].data["weight"] = weights

        ori_row, ori_col = ori_graph.all_edges()
        ori_w = torch.ones((ori_graph.number_of_edges(), 1), dtype=torch.float,
                           device=ori_graph.device)
        complete_graph.edges[ori_row, ori_col].data["weight"] = (
            (1 - self.lamda) * complete_graph.edges[ori_row, ori_col].data["weight"]
            + self.lamda * ori_w)

        if self.edge_softmax:
            complete_graph.edata["weight"] = edge_softmax(
                complete_graph, complete_graph.edata["weight"])
        if self.sparsify:
            complete_graph.edata["weight"] = torch.where(
                complete_graph.edata["weight"] > self.threshold,
                complete_graph.edata["weight"],
                torch.zeros(complete_graph.edata["weight"].shape).to(complete_graph.device))
        return complete_graph


class GslLayer(nn.Module):
    def __init__(self, params, rel_emb):
        super().__init__()
        self.emb_dim = params.emb_dim
        self.gsl = GraphStructureLearner(params, rel_emb)
        self.apply_mod = NodeUpdateModule(self.emb_dim)

    def forward(self, complete_graph, ori_graph):
        def msg_func(edges):
            return {"msg": edges.src["h"] * edges.data["weight"]}

        def reduce_func(nodes):
            return {"h": torch.sum(nodes.mailbox["msg"], dim=1)}

        complete_graph = self.gsl(complete_graph, ori_graph)
        complete_graph.update_all(msg_func, reduce_func)
        complete_graph.apply_nodes(func=self.apply_mod)
        return complete_graph


class GslModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.ni_layer = params.num_infer_layers
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.gsl_rel_emb_dim = params.gsl_rel_emb_dim
        self.rel_emb = nn.Embedding(self.aug_num_rels, self.gsl_rel_emb_dim, sparse=False)
        self.gsl_layers = nn.ModuleList(
            [GslLayer(params, self.rel_emb) for _ in range(self.ni_layer)])

    def build_full_connect_graph(self, ori_graph):
        batch_num_nodes = ori_graph.batch_num_nodes()
        begin = torch.cat([batch_num_nodes.new_zeros(1),
                           batch_num_nodes.cumsum(dim=0)[:-1]], dim=0)
        end = batch_num_nodes.cumsum(dim=0)
        dense = torch.zeros((ori_graph.num_nodes(), ori_graph.num_nodes()),
                            dtype=torch.float, device=ori_graph.device)
        for b, e in zip(begin, end):
            dense[b:e, b:e] = 1.
        row, col = torch.nonzero(dense).t().contiguous()
        complete = dgl.graph((row, col)).to(ori_graph.device)
        complete.set_batch_num_nodes(batch_num_nodes)
        complete.set_batch_num_edges(torch.pow(batch_num_nodes, 2))
        complete.ndata["h"] = ori_graph.ndata["h"]
        complete.ndata["repr"] = ori_graph.ndata["repr"]
        complete.ndata["id"] = ori_graph.ndata["id"]
        complete.ndata["idx"] = ori_graph.ndata["idx"]
        complete.edata["type"] = torch.full(
            (complete.number_of_edges(),), self.num_rels + 1,  # resemble id
            dtype=ori_graph.edata["type"].dtype, device=ori_graph.device)
        ori_row, ori_col = ori_graph.all_edges()
        complete.edges[ori_row, ori_col].data["type"] = \
            ori_graph.edges[ori_row, ori_col].data["type"]
        return complete

    def forward(self, g):
        complete = self.build_full_connect_graph(g)
        for i in range(self.ni_layer):
            complete = self.gsl_layers[i](complete, g)
            complete.ndata["repr"] = torch.cat(
                [complete.ndata["repr"], complete.ndata["h"].unsqueeze(1)], dim=1)
        return complete


class GraphSAGEEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dropout = nn.Dropout(params.gcn_dropout)
        self.activation = F.relu
        self.num_nodes = params.num_nodes
        self.num_rels = params.aug_num_rels
        self.emb_dim = params.emb_dim
        self.num_gcn_layers = params.num_gcn_layers
        self.pre_embed = nn.Parameter(torch.Tensor(self.num_nodes, self.emb_dim))
        nn.init.xavier_uniform_(self.pre_embed, gain=nn.init.calculate_gain("relu"))
        self.rel_weght = nn.Parameter(torch.Tensor(self.num_rels, self.emb_dim))
        nn.init.xavier_uniform_(self.rel_weght, gain=nn.init.calculate_gain("relu"))
        self.layers = nn.ModuleList(
            [SAGEConv(self.emb_dim, self.emb_dim, params.gcn_aggregator_type)
             for _ in range(self.num_gcn_layers)])

    def forward(self, g):
        h = self.pre_embed[g.ndata["idx"]]
        h = self.dropout(h)
        edge_weight = self.rel_weght[g.edata["type"]]
        for l, layer in enumerate(self.layers):
            g.ndata["h"] = layer(g, h, edge_weight=edge_weight)
            if l != len(self.layers) - 1:
                g.ndata["h"] = self.dropout(self.activation(g.ndata["h"]))
            h = g.ndata["h"]
            if l == 0:
                x = torch.cat([self.pre_embed[g.ndata["idx"]], g.ndata["h"]], dim=1)
                g.ndata["repr"] = x.unsqueeze(1).reshape(-1, 2, self.emb_dim)
            else:
                g.ndata["repr"] = torch.cat(
                    [g.ndata["repr"], g.ndata["h"].unsqueeze(1)], dim=1)
        return g.ndata.pop("h")


class KnowDDIClassifier(nn.Module):
    """Faithful Classifier_model.py; binary head (1 logit) + returns pre-scorer pred."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.global_graph = params.global_graph
        self.emb_dim = params.emb_dim
        self.score_dim = (1 + params.num_gcn_layers + params.num_infer_layers) * self.emb_dim
        self.embedding_model = GraphSAGEEncoder(params)
        self.gsl_model = GslModel(params)
        self.W_final = nn.Linear(3 * self.score_dim, 1)  # binary

    def forward(self, sub_graphs, return_pred=False, skip_global_embed=False):
        # skip_global_embed: PERF hook (default False = unchanged). The full-KG GraphSAGE over
        # global_graph is INVARIANT across a fixed context, so eval can compute it ONCE and set
        # global_graph.ndata['h']/['repr'] before the batch loop, then pass skip_global_embed=True
        # to avoid recomputing it per batch. Outputs are identical (same graph, same eval weights,
        # dropout off). NOT for training (grad reuse across minibatch backwards -> double-backward).
        g = sub_graphs
        if not skip_global_embed:
            self.global_graph.ndata["h"] = self.embedding_model(self.global_graph)
        g.ndata["h"] = self.global_graph.nodes[g.ndata["idx"]].data["h"]
        g.ndata["repr"] = self.global_graph.nodes[g.ndata["idx"]].data["repr"]

        head_ids = (g.ndata["id"] == 1).nonzero().squeeze(1)
        tail_ids = (g.ndata["id"] == 2).nonzero().squeeze(1)

        complete = self.gsl_model(g)
        gsl_hidden = complete.ndata["repr"]
        head_hidden = gsl_hidden[head_ids].view(-1, self.score_dim)
        tail_hidden = gsl_hidden[tail_ids].view(-1, self.score_dim)
        g_out = mean_nodes(complete, "repr").view(-1, self.score_dim)

        pred = torch.cat([g_out, head_hidden, tail_hidden], dim=1)  # pre-scorer rep
        score = self.W_final(pred).squeeze(-1)
        if return_pred:
            return score, pred
        return score


# --------------------------------------------------------------------------- #
# DGL subgraph assembly per pair (faithful to datasets.py _prepare_subgraphs +
# extract_r_digraph), operating on a cached (nodes, labels).
# --------------------------------------------------------------------------- #
def _get_neighbors_dgl(graph, nodes):
    src, dst, eid = graph.out_edges(nodes, form="all")
    sampled = torch.stack([src, dst, eid], dim=1)
    return sampled


def prepare_subgraph(global_graph, nodes, labels, dig_layer):
    nodes_t = torch.as_tensor(np.asarray(nodes), dtype=torch.long)
    sub = global_graph.subgraph(nodes_t)
    sub.ndata["idx"] = nodes_t
    # SEAL id: head (label [0,1]) -> 1, tail (label [1,0]) -> 2, else 0
    n_ids = np.zeros(len(nodes), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab[0] == 0 and lab[1] == 1:
            n_ids[i] = 1
        elif lab[0] == 1 and lab[1] == 0:
            n_ids[i] = 2
    sub.ndata["id"] = torch.FloatTensor(n_ids)

    # Remove the direct head-tail edge(s) (the DDI edge being predicted -> no leakage).
    # Our merged KG is symmetrized (see build_graph_tensors), so the query edge exists
    # in BOTH directions; upstream removes only head->tail (datasets.py:86-87) because
    # its graph is directed. We must remove BOTH 0->1 and 1->0, else the pair's own
    # reverse DDI edge survives into ori_graph, gets the lamda boost in GSL, and leaks
    # (codex 2026-07-05 must-fix). Collect both edge-id sets BEFORE removing (ids shift).
    rm = []
    for u, v in ((0, 1), (1, 0)):
        try:
            e = sub.edge_ids(u, v, return_uv=True)[2]
            if e.numel():
                rm.append(e)
        except dgl.DGLError:
            pass
    if rm:
        sub.remove_edges(torch.cat(rm))

    # extract_r_digraph: keep only edges on head->...->tail paths within dig_layer hops
    head_nodes = (sub.ndata["id"] == 1).nonzero().squeeze(1)
    tail_nodes = (sub.ndata["id"] == 2).nonzero().squeeze(1)
    total_nodes = torch.cat([head_nodes, tail_nodes])
    raw_layer_edges = {}
    hn = head_nodes
    for i in range(dig_layer):
        sampled = _get_neighbors_dgl(sub, hn)
        raw_layer_edges[i] = sampled
        hn = torch.unique(sampled[:, 1]) if sampled.numel() else hn
    layer_edge_ids = torch.LongTensor([])
    tn = tail_nodes
    for i in reversed(range(dig_layer)):
        e = raw_layer_edges[i]
        if e.numel() and tn.numel():
            select = torch.nonzero(torch.eq(e[:, 1], tn.unsqueeze(1)))
            l_edge = e[select[:, 1]]
            layer_edge_ids = torch.cat([layer_edge_ids, l_edge[:, 2]])
            tn = torch.unique(l_edge[:, 0]) if l_edge.numel() else tn
    total_edges = torch.unique(layer_edge_ids, dim=0, sorted=True)
    if total_edges.numel():
        r_digraph = dgl.edge_subgraph(sub, total_edges)
    else:
        r_digraph = dgl.node_subgraph(sub, total_nodes)
    return r_digraph


class KnowDDIDataset(torch.utils.data.Dataset):
    def __init__(self, global_graph, subgraph_cache, y, dig_layer):
        self.global_graph = global_graph
        self.cache = subgraph_cache
        self.y = y
        self.dig_layer = dig_layer

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        nodes, labels = self.cache[idx]
        g = prepare_subgraph(self.global_graph, nodes, labels, self.dig_layer)
        return g, float(self.y[idx])


def collate(samples):
    graphs, ys = map(list, zip(*samples))
    return dgl.batch(graphs), torch.tensor(ys, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Train / extract
# --------------------------------------------------------------------------- #
def make_params(args, kg, layout, global_graph, device):
    return SimpleNamespace(
        emb_dim=args.emb_dim, num_gcn_layers=args.num_gcn_layers,
        num_infer_layers=args.num_infer_layers, num_dig_layers=args.num_dig_layers,
        gcn_dropout=args.gcn_dropout, gcn_aggregator_type=args.gcn_aggregator_type,
        MLP_hidden_dim=args.MLP_hidden_dim, MLP_num_layers=args.MLP_num_layers,
        MLP_dropout=args.MLP_dropout, func_num=args.func_num, sparsify=args.sparsify,
        threshold=args.threshold, edge_softmax=args.edge_softmax,
        gsl_rel_emb_dim=args.gsl_rel_emb_dim, lamda=args.lamda,
        gsl_has_edge_emb=args.gsl_has_edge_emb,
        num_rels=layout["num_rels"], aug_num_rels=layout["aug_num_rels"],
        num_nodes=kg.n_nodes, global_graph=global_graph, device=device)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=collate)
    logits, ys = [], []
    for g, y in loader:
        g = g.to(device)
        s = model(g)
        logits.append(s.detach().cpu().numpy())
        ys.append(y.numpy())
    logit = np.concatenate(logits)
    y = np.concatenate(ys)
    return roc_auc_score(y, 1 / (1 + np.exp(-logit))), logit, y


@torch.no_grad()
def extract_pred(model, dataset, batch_size, device):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=collate)
    preds, logits, ys = [], [], []
    for g, y in loader:
        g = g.to(device)
        s, pred = model(g, return_pred=True)
        preds.append(pred.detach().cpu().numpy().astype(np.float32))
        logits.append(s.detach().cpu().numpy().astype(np.float32))
        ys.append(y.numpy().astype(np.float32))
    return np.concatenate(preds), np.concatenate(logits), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128,
                    help="the whole-KG GraphSAGE fwd+bwd is a fixed ~5s/step cost, so "
                         "LARGER batches amortize it (128 ~= 12min/epoch; 64 would ~2x). "
                         "peak GPU ~10-12GB at 128 on the merged KG.")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lr-decay", type=float, default=0.93)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--train-sample", type=int, default=12000,
                    help="subsample train pairs for extraction cost (PCA+probe needs few)")
    ap.add_argument("--seed", type=int, default=42)
    # subgraph
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--max-nodes-per-hop", type=int, default=200)
    # GraphSAGE
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--num-gcn-layers", type=int, default=2)
    ap.add_argument("--gcn-aggregator-type", default="mean")
    ap.add_argument("--gcn-dropout", type=float, default=0.2)
    # gsl
    ap.add_argument("--num-infer-layers", type=int, default=3)
    ap.add_argument("--num-dig-layers", type=int, default=3)
    ap.add_argument("--MLP-hidden-dim", type=int, default=16)
    ap.add_argument("--MLP-num-layers", type=int, default=2)
    ap.add_argument("--MLP-dropout", type=float, default=0.2)
    ap.add_argument("--func-num", type=int, default=1)
    ap.add_argument("--sparsify", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--edge-softmax", type=int, default=1)
    ap.add_argument("--gsl-rel-emb-dim", type=int, default=32)
    ap.add_argument("--lamda", type=float, default=0.7)
    ap.add_argument("--gsl-has-edge-emb", type=int, default=1)
    ap.add_argument("--kg-nodes", default=DEFAULT_NODES_PATH)
    ap.add_argument("--kg-edges", default=DEFAULT_EDGES_PATH)
    ap.add_argument("--tag", default="knowddi_ddi800_s2")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()} dgl={dgl.__version__}",
          flush=True)

    # -- KG + splits (identical to the R-GCN node-paradigm entry) -------------
    kg = MergedKG.from_parquet(args.kg_nodes, args.kg_edges)
    tr_full, warm_val, warm_test, cold_test = load_data(args.fold, args.warm_frac, args.seed)
    cold_val = load_cold_val(args.fold)   # official S2 unseen-drug val (cold-best selection)
    id2i = kg.id_to_idx

    # The message-passing graph gets ALL train POSITIVE DDI edges (the full training
    # knowledge), independent of the extraction subsample. warm_val/warm_test/cold_*
    # are separate splits -> their edges are never added -> no leakage. Each pair's OWN
    # head-tail edge is removed at subgraph prep (prepare_subgraph) anyway.
    iaf, ibf, yf = _pairs_to_idx(tr_full, id2i)
    fpos = yf > 0.5
    tr_pos_pairs = np.vstack([iaf[fpos], ibf[fpos]]).T

    # Subsample only WHICH train pairs we extract subgraphs for + train on (extraction
    # cost; PCA + linear probe need only a few thousand). The graph is unchanged.
    tr = tr_full
    if args.train_sample and args.train_sample < len(tr_full):
        tr = tr_full.sample(n=args.train_sample, random_state=args.seed).reset_index(drop=True)
        print(f"[data] subsampled train pairs {len(tr_full)} -> {len(tr)} for extraction "
              f"cost (graph still holds all {int(fpos.sum())} train-pos DDI edges)", flush=True)

    ia_tr, ib_tr, y_tr = _pairs_to_idx(tr, id2i)
    iav, ibv, yv = _pairs_to_idx(warm_val, id2i)
    iawt, ibwt, ywt = _pairs_to_idx(warm_test, id2i)
    iac, ibc, yc = _pairs_to_idx(cold_test, id2i)
    iacv, ibcv, ycv = _pairs_to_idx(cold_val, id2i)

    # TWO copies (faithful to upstream process_dataset): CPU graph for per-pair
    # subgraph extraction in the dataset; device graph for the model's GraphSAGE.
    # Linked by node 'idx'. .to(device) returns a new graph; the CPU one is unchanged.
    global_graph_cpu, inc, layout = build_graph_tensors(kg, args.kg_edges, tr_pos_pairs)
    global_graph = global_graph_cpu.to(device)

    # -- subgraph extraction (cached) ----------------------------------------
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__train_knowddi_rank_pilot__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # Shared subgraph cache: extraction (~mins) is deterministic given (KG, split,
    # fold, warm_frac, hop, max_nodes_per_hop, seed); reuse across reruns/epochs/seeds.
    # The graph (train-pos DDI edges) is stable, so warm/cold caches are train_sample-
    # independent; the train split's cache also keys on train_sample.
    cache_root = ROOT / "Code" / "runs" / "_knowddi_subgraph_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    kg_tag = Path(args.kg_edges).parent.name  # e.g. _merged_kg

    def cache_for(name, ia, ib, extra=""):
        sig = (f"{name}__{kg_tag}__{args.fold}__wf{args.warm_frac}__hop{args.hop}"
               f"__mnph{args.max_nodes_per_hop}__seed{args.seed}{extra}")
        shared = cache_root / f"{sig}.pkl"
        if shared.is_file():
            with open(shared, "rb") as f:
                cache = pickle.load(f)
            log(f"[subgraph:{name}] cache HIT {shared.name} ({len(cache)} pairs)")
            return cache
        pairs = np.vstack([ia, ib]).T
        cache = build_subgraph_cache(pairs, inc, args.hop, args.max_nodes_per_hop,
                                     args.seed, name)
        with open(shared, "wb") as f:
            pickle.dump(cache, f)
        log(f"[subgraph:{name}] cache BUILT {shared.name}")
        return cache

    log(f"[run] {run_id}")
    log(f"[data] train={len(y_tr)} warm_val={len(yv)} warm_test={len(ywt)} "
        f"cold_val={len(ycv)} cold_test={len(yc)}")
    tr_cache = cache_for("train", ia_tr, ib_tr, extra=f"__ts{args.train_sample}")
    wv_cache = cache_for("warm_val", iav, ibv)
    wt_cache = cache_for("warm_test", iawt, ibwt)
    cv_cache = cache_for("cold_val", iacv, ibcv)
    ct_cache = cache_for("cold_test", iac, ibc)

    params = make_params(args, kg, layout, global_graph, device)
    model = KnowDDIClassifier(params).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[model] KnowDDI params={n_params} score_dim={model.score_dim} "
        f"aug_num_rels={layout['aug_num_rels']}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, args.lr_decay)

    tr_ds = KnowDDIDataset(global_graph_cpu, tr_cache, y_tr, args.num_dig_layers)
    wv_ds = KnowDDIDataset(global_graph_cpu, wv_cache, yv, args.num_dig_layers)
    wt_ds = KnowDDIDataset(global_graph_cpu, wt_cache, ywt, args.num_dig_layers)
    cv_ds = KnowDDIDataset(global_graph_cpu, cv_cache, ycv, args.num_dig_layers)
    ct_ds = KnowDDIDataset(global_graph_cpu, ct_cache, yc, args.num_dig_layers)

    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch_size,
                                            shuffle=True, collate_fn=collate)
    bce = nn.BCEWithLogitsLoss()

    # -- train, select warm-best AND cold-best (one-model rule per extraction) --
    best_warm, warm_state, bad = -1.0, None, 0
    best_cold, cold_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for b_idx, (g, y) in enumerate(tr_loader):
            g = g.to(device)
            y = y.to(device)
            opt.zero_grad()
            score = model(g)
            loss = bce(score, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
            opt.step()
            losses.append(loss.item())
            if (b_idx + 1) % 50 == 0:
                log(f"[ep {ep}/{args.epochs} step {b_idx+1}/{len(tr_loader)}] "
                    f"loss={np.mean(losses[-50:]):.4f}")
        sched.step()
        wva, _, _ = evaluate(model, wv_ds, args.batch_size, device)
        cva, _, _ = evaluate(model, cv_ds, args.batch_size, device)  # official S2 val (no test leak)
        log(f"[ep {ep}/{args.epochs}] mean_loss={np.mean(losses):.4f} "
            f"warm_val_auc={wva:.4f} cold_val_auc={cva:.4f} time={time.time()-t0:.0f}s")
        if wva > best_warm:
            best_warm, bad = wva, 0
            warm_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if cva > best_cold:
            best_cold = cva
            cold_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if bad >= args.patience:
            log(f"[early-stop] no warm_val improvement for {args.patience} epochs")
            break

    logdeg = np.log1p(np.asarray(kg.degree, dtype=np.float64))

    def extract_and_save(state, tag, sel_note):
        model.load_state_dict(state)
        packs = {"train": (tr_ds, ia_tr, ib_tr), "warm": (wt_ds, iawt, ibwt),
                 "cold": (ct_ds, iac, ibc)}
        arrs = {}
        for name, (ds, ia, ib) in packs.items():
            pred, logit, y = extract_pred(model, ds, args.batch_size, device)
            arrs[f"Z_{name}"] = pred
            arrs[f"raw_{name}"] = pred
            arrs[f"y_{name}"] = y
            arrs[f"logit_{name}"] = logit
            arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
            auc = roc_auc_score(y, 1 / (1 + np.exp(-logit)))
            log(f"[extract:{tag}] {name}: n={len(y)} head_auc={auc:.4f} pred={pred.shape}")
        np.savez_compressed(run_dir / f"pilot_Z__{tag}.npz", **arrs)
        return {"tag": tag, "selected_by": sel_note}

    selected = {}
    if warm_state is not None:
        selected["warm_best"] = extract_and_save(warm_state, "warm_best",
                                                 f"warm_val_auc={best_warm:.4f}")
    if cold_state is not None:
        selected["cold_best"] = extract_and_save(cold_state, "cold_best",
                                                 f"cold_val_auc={best_cold:.4f}")

    # default alias: warm_best -> pilot_Z.npz (analyze_rank_sufficiency default)
    if warm_state is not None:
        import shutil
        shutil.copy(run_dir / "pilot_Z__warm_best.npz", run_dir / "pilot_Z.npz")

    meta = {"run_id": run_id, "fold": args.fold, "seed": args.seed,
            "model_label": "KnowDDI", "paradigm": "subgraph",
            "best_warm_val_auc": best_warm, "best_cold_val_auc": best_cold,
            "aug_num_rels": layout["aug_num_rels"], "selected": selected,
            "note": "subgraph paradigm; knowledge-only. ONE-MODEL rule: within EACH "
                    "pilot_Z__*.npz, warm+cold are extracted from that file's single "
                    "checkpoint (warm_best and cold_best are DIFFERENT checkpoints, one "
                    "file each). warm-best selected on warm_val, cold-best on official S2 "
                    "val.parquet (unseen drugs); neither touches cold_test -> no leakage. "
                    "Merged KG symmetrized (undirected, grid-fair with R-GCN)."}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[done] {run_dir}")
    print(f"\nNEXT: python Code/scripts/analyze_rank_sufficiency.py "
          f"--z {run_dir/'pilot_Z.npz'} --key raw --title KnowDDI-warmbest", flush=True)


if __name__ == "__main__":
    main()

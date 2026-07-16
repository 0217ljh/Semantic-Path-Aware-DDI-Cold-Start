"""MKG-FENN both-unseen COLD model (3-channel fusion + nearest-seen imputation).

Faithful port of the official upstream cold model
``Paper/Reference/Original-Code/MKG-FENN/Code and Datasets/code/modeltask3.py``
(both-unseen S2; ``modeltask2.py`` is byte-identical in structure for one-unseen
S1, so this one module serves BOTH inductive regimes).

Differences vs. the warm 4-channel model (``baseline.mkg_fenn.model.MKGFENN``,
ported from ``modeltask1.py``):

1. **Three-channel fusion.** The upstream cold net is
   ``nn.Sequential(GNN1, GNN2, GNN3, FusionLayer)`` — GNN4 (molecular property)
   is defined but NOT instantiated in the net and NOT fused. ``FusionLayer``
   concatenates gnn1/gnn2/gnn3 for drugA + drugB -> ``embedding_num*3*2`` ->
   ``embedding_num*3`` -> ``embedding_num*2`` -> ``event_num`` (paper: 65).
2. **Nearest-seen-neighbour imputation.** Each GNN forward, when
   ``train_or_test == 1``, replaces every unseen test drug ``i``'s per-channel
   embedding with the mean of its nearest-SEEN-neighbour drugs' embeddings:
   ``drug_f[i] = sum(drug_f[pos]) / len(pos)`` where ``pos = test_adj[k][i][0]``.

CRITICAL channel-to-``test_adj`` indexing (verbatim from modeltask3.py):
  * GNN1 imputes from ``test_adj[0]``  (modeltask3.py:53)  -> drug_sim1 (KG1 Jaccard)
  * GNN2 imputes from ``test_adj[1]``  (modeltask3.py:120) -> drug_sim2 (KG2 Jaccard)
  * GNN3 imputes from ``test_adj[3]``  (modeltask3.py:187) -> drug_sim4 (KG4 Jaccard)

  i.e. the DDI channel (GNN3) imputes using the drug-property similarity, NOT the
  DDI similarity. This is an upstream quirk preserved for faithfulness. The
  driver builds all four ``test_adj`` channels (from drug_sim1..4) even though
  only three GNNs run; GNN4's ``test_adj[2]`` slot is simply never consumed.

Parametrization vs. the original hardcodes:
  * ``n_drug`` replaces the hardcoded ``572`` (our leaf has ~1900 drugs).
  * ``event_num`` replaces the hardcoded ``65`` (we re-vocab to the count of
    train-observed ddi_type classes; CLAUDE.md permits vocab-size change when the
    task matches the paper, which it does — 65-way -> K_train-way multiclass).
  * GNN3's ``rela_embed`` num_embeddings was hardcoded ``67`` upstream (65 DDI
    relations + ghost slack); we size it to ``n_ddi_relations + 2`` to preserve
    the same "+2 ghost slack" margin over the actual relation count.

Every tensor op, reshape, attention weighting, imputation loop, and the fusion
head are otherwise verbatim from modeltask3.py.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


class GNN1(nn.Module):
    """Drug -> entity channel (KG1). modeltask3.py:8-72."""

    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, n_drug, **kwargs):
        super().__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset1"], dict1
        self.drug_name, self.args = drug_name, args
        self.n_drug = n_drug
        self.drug_embed = nn.Embedding(num_embeddings=n_drug, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(n_drug, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(n_drug, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, datas):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        n_drug = self.n_drug
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        _dev = self.drug_embed.weight.device
        drug_name = torch.LongTensor(drug_name).to(_dev)
        adj_tail = torch.LongTensor(adj_tail).to(_dev)
        adj_relation = torch.LongTensor(adj_relation).to(_dev)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((n_drug, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((n_drug, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([weighted_ent.reshape(n_drug, args.embedding_num),
                            drug_embedding.reshape((n_drug, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        idx, train_or_test, test_adj = datas[0], datas[1], datas[2]
        if train_or_test == 1:
            for i in test_adj[0].keys():
                pos = test_adj[0][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, idx, test_adj, train_or_test

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=None):
        if n_drug is None:
            n_drug = self.n_drug
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN2(nn.Module):
    """Drug -> substructure channel (KG2, Morgan FP). modeltask3.py:74-139."""

    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, n_drug, **kwargs):
        super().__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset2"], dict1
        self.drug_name, self.args = drug_name, args
        self.n_drug = n_drug
        self.drug_embed = nn.Embedding(num_embeddings=n_drug, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(n_drug, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(n_drug, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        n_drug = self.n_drug
        gnn1_embedding, idx, test_adj, train_or_test = arguments
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        _dev = self.drug_embed.weight.device
        drug_name = torch.LongTensor(drug_name).to(_dev)
        adj_tail = torch.LongTensor(adj_tail).to(_dev)
        adj_relation = torch.LongTensor(adj_relation).to(_dev)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((n_drug, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((n_drug, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([weighted_ent.reshape(n_drug, args.embedding_num),
                            drug_embedding.reshape((n_drug, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        if train_or_test == 1:
            for i in test_adj[1].keys():
                pos = test_adj[1][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, gnn1_embedding, idx, test_adj, train_or_test

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=None):
        if n_drug is None:
            n_drug = self.n_drug
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN3(nn.Module):
    """Drug -> drug channel (KG3, DDI topology). modeltask3.py:141-215.

    Note the imputation reads ``test_adj[3]`` (drug_sim4 / property), NOT
    ``test_adj[2]`` — a verbatim upstream quirk (modeltask3.py:187).
    """

    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, n_drug, **kwargs):
        super().__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset3"], dict1
        self.drug_name, self.args = drug_name, args
        self.n_drug = n_drug
        self.drug_embed = nn.Embedding(num_embeddings=n_drug, embedding_dim=args.embedding_num)
        # Upstream hardcodes num_embeddings=67 (65 DDI relations + 2 ghost slack).
        # We size to n_relations + 2 to preserve the same margin over the actual
        # relation count (build_kg3 uses a single relation id 0; the +2 slack
        # matches the upstream ``relations_num + 1`` ghost id used in arrge).
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset3"] + 2,
                                       embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=n_drug, embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(n_drug, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(n_drug, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        n_drug = self.n_drug
        gnn2_embedding, gnn1_embedding, idx, test_adj, train_or_test = arguments
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        _dev = self.drug_embed.weight.device
        drug_name = torch.LongTensor(drug_name).to(_dev)
        adj_tail = torch.LongTensor(adj_tail).to(_dev)
        adj_relation = torch.LongTensor(adj_relation).to(_dev)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((n_drug, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((n_drug, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat([weighted_ent.reshape(n_drug, args.embedding_num),
                            drug_embedding.reshape((n_drug, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)
        if train_or_test == 1:
            for i in test_adj[3].keys():
                pos = test_adj[3][i][0]
                length = len(pos)
                drug_f[i] = torch.sum(drug_f[pos], dim=0) / length
        return drug_f, gnn2_embedding, gnn1_embedding, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=None, tails_num=None, relations_num=None):
        """Verbatim from modeltask3.py:193-215.

        The in-place ghost append fires only for drugs whose KG3 row is empty
        (``surplus``). Our ``build_kg3`` gives every drug a self-loop, so
        ``surplus`` is empty and this is a no-op (codex 019f2455: safe, no
        unbounded per-forward growth when KG3 is self-loop-complete).
        """
        if n_drug is None:
            n_drug = self.n_drug
        if tails_num is None:
            tails_num = n_drug - 2         # upstream: 570 with n_drug 572 -> tails_num+1 == n_drug-1
        if relations_num is None:
            relations_num = self.rela_embed.num_embeddings - 2
        drug_number = []
        drug_list = []
        for i in drug_name_id:
            drug_number.append(drug_name_id[i])
        for key in kg:
            drug_list.append(key)
        surplus = set(drug_number).difference(set(drug_list))
        for i in list(surplus):
            kg[i].append((tails_num + 1, relations_num + 1))
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class FusionLayer(nn.Module):
    """Three-channel fusion head. modeltask3.py:284-315.

    ``event_num`` replaces the upstream hardcoded ``65`` output dim.
    """

    def __init__(self, args, event_num):
        super().__init__()
        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num * 3 * 2, args.embedding_num * 3),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 3),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 3, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, event_num))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        gnn3_embedding, gnn2_embedding, gnn1_embedding, idx = arguments
        idx = idx.cpu().numpy().tolist()
        drugA = []
        drugB = []
        for i in idx:
            drugA.append(i[0])
            drugB.append(i[1])
        Embedding = torch.cat([gnn1_embedding[drugA], gnn2_embedding[drugA], gnn3_embedding[drugA],
                               gnn1_embedding[drugB], gnn2_embedding[drugB], gnn3_embedding[drugB]], 1).float()
        return self.fullConnectionLayer(Embedding)


class MKGFENNCold(nn.Module):
    """Wrapper mirroring the upstream ``nn.Sequential(GNN1, GNN2, GNN3, FusionLayer)``.

    ``forward`` takes ``(idx_pairs, train_or_test, test_adj)`` exactly like the
    upstream driver passes to ``net(f_input)`` (MKG-FENN-task3.py:197-201,211-215),
    and returns raw logits ``(n_pairs, event_num)``. Softmax is applied by the
    caller at eval time (upstream applies ``F.softmax`` only at eval, CE loss
    consumes raw logits at train time).
    """

    def __init__(
        self,
        *,
        kgs: dict,
        tail_len: dict,
        relation_len: dict,
        dict1: dict,
        drug_name: list,
        args,
        event_num: int,
    ) -> None:
        super().__init__()
        self.n_drug = len(dict1)
        common = dict(dataset=kgs, tail_len=tail_len, relation_len=relation_len,
                      args=args, dict1=dict1, drug_name=drug_name, n_drug=self.n_drug)
        self.gnn1 = GNN1(**common)
        self.gnn2 = GNN2(**common)
        self.gnn3 = GNN3(**common)
        self.fusion = FusionLayer(args, event_num=event_num)

    def forward(self, idx_pairs, train_or_test=0, test_adj=None):
        if test_adj is None:
            test_adj = defaultdict(list)
        out = self.gnn1((idx_pairs, train_or_test, test_adj))
        out = self.gnn2(out)
        out = self.gnn3(out)
        return self.fusion(out)


__all__ = ["GNN1", "GNN2", "GNN3", "FusionLayer", "MKGFENNCold"]

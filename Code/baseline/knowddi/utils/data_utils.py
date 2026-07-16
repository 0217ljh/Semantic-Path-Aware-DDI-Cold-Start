"""KnowDDI's ``utils/data_utils.py`` — DrugBank / decagon KG file readers.

Ported from KnowDDI's official
``Paper/Reference/Original-Code/KnowDDI/pytorch/utils/data_utils.py``
(``process_files_ddi`` :4, ``process_files_decagon`` :61). No DGL calls here,
so this is a byte-faithful port modulo ``from __future__`` + type hints and a
module docstring per this repo's style. The relation-offset scheme is the crux
of KnowDDI's data contract: DDI relations occupy ``[0, rel)``, BKG relations are
offset by ``rel`` (``rel + r``; see :55-58 of the original), so the augmented
relation space is DDI-rels ++ BKG-rels ++ (later) a self-loop.

This module is an INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md
file-independence). It does NOT import from ``baseline.sumgnn`` or the
reproduction tree.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix


def process_files_ddi(files: dict, BKG_file: str, keeptrainone: bool = False):
    """Port of KnowDDI ``process_files_ddi`` (data_utils.py:4).

    ``files``: {'train': path, 'valid': path, 'test': path} of ``h t r`` int
    triples (loaded via ``np.loadtxt``). ``BKG_file``: ``h t r`` int triples of
    the background KG. Returns ``(adj_list, triplets, entity2id, relation2id,
    id2entity, id2relation, rel)`` where ``rel`` = # DDI relations and
    ``len(relation2id)`` = # DDI + # BKG relations. ``adj_list`` has one csc
    adjacency per relation (DDI relations from train, BKG relations from
    ``BKG_file``), all shaped ``(|V|, |V|)``.
    """
    entity2id: dict = {}
    relation2id: dict = {}

    triplets: dict = {}
    kg_triple: list = []
    rel = 0

    for file_type, file_path in files.items():
        data = []
        file_data = np.loadtxt(file_path)
        # np.loadtxt on a single-row file returns a 1-D array; normalize to 2-D.
        if file_data.ndim == 1:
            file_data = file_data.reshape(-1, 3)
        # Only the TRAIN split defines the DDI relation vocabulary (relation2id / rel),
        # matching KnowDDI's contract that the adjacency + fc head are built from train
        # (official data_utils.py:51 comment, adj_list loop over range(rel)). In the
        # official DrugBank data valid/test relations are a strict subset of train, so
        # this is a no-op there. Our unified builder writes a SENTINEL relation id
        # (== K_train, one beyond the head's 0..K_train-1 range) into valid/test.txt for
        # gold classes unseen in train. Registering that sentinel here would grow `rel`
        # and thus the W_final head width to K_train+1, making the sentinel PREDICTABLE
        # (breaking the "always wrong" guarantee) and drifting the architecture. Guard:
        # register/increment only while reading train; keep OOV valid/test rows with
        # their sentinel id so a subgraph is still extracted and the evaluator counts
        # them wrong, without expanding the DDI relation space.
        register_relations = file_type == "train"
        for triplet in file_data:
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
            if triplet[2] not in relation2id and register_relations:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to known relations (official behavior) OR
            # OOV valid/test rows carrying the sentinel id (kept so they reach the
            # evaluator as a wrong gold; their r_label is never used to index adj_list).
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
            elif not register_relations:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], triplet[2]])

        triplets[file_type] = np.array(data)

    triplet_kg = np.loadtxt(BKG_file)
    if triplet_kg.ndim == 1 and triplet_kg.size:
        triplet_kg = triplet_kg.reshape(-1, 3)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t
        # same id within train/valid/test and BKG_file does not mean same relation
        if rel + r not in relation2id:
            relation2id[rel + r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    # Construct the list of adjacency matrix each corresponding to each relation.
    # Note that this is constructed from the train data and BKG data.
    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8),
             (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i - rel)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8),
             (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel


def process_files_decagon(files: dict, triple_file: str, keeptrainone: bool = False):
    """Port of KnowDDI ``process_files_decagon`` (data_utils.py:61).

    Retained for parity with the official source (BioSNAP / multilabel path). NOT
    used by the Phase-1 multiclass DrugBank pipeline, which calls
    :func:`process_files_ddi`. The BioSNAP-specific ``assert len(entity2id) == 604``
    / ``assert rel == 200`` from the original are dropped because our KG is the
    merged DrugBank+Hetionet+PrimeKG graph, not decagon's fixed 604/200.
    """
    entity2id: dict = {}
    relation2id: dict = {}

    triplets: dict = {}
    triplets_mr: dict = {}
    polarity_mr: dict = {}
    kg_triple: list = []
    triplets_train: list = []
    rel = 0

    for file_type, file_path in files.items():
        data = []
        data_mr = []
        data_pol = []
        with open(file_path, 'r') as f:
            for lines in f:
                h, t, r, p = lines.strip().split('\t')
                h, t = int(h), int(t)
                p = int(p)  # pos/neg edge
                list_r_onehot = list(map(int, r.split(',')))
                list_r = [0] if keeptrainone else [i for i, _ in enumerate(list_r_onehot) if _ == 1]
                for s in list_r:
                    triplet = [h, t, s]
                    triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if triplet[0] not in entity2id:
                        entity2id[triplet[0]] = triplet[0]
                    if triplet[1] not in entity2id:
                        entity2id[triplet[1]] = triplet[1]
                    if triplet[2] not in relation2id:
                        if keeptrainone:
                            triplet[2] = 0
                            relation2id[triplet[2]] = 0
                            rel = 1
                        else:
                            relation2id[triplet[2]] = triplet[2]
                            rel += 1
                    if triplet[2] in relation2id:
                        data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                        if file_type == 'train' and p == 1:
                            triplets_train.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                if keeptrainone:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], 0])
                else:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], list_r_onehot])
                data_pol.append(p)
        triplets_train = np.array(triplets_train)
        triplets[file_type] = np.array(data)
        triplets_mr[file_type] = data_mr
        polarity_mr[file_type] = np.array(data_pol)

    triplet_kg = np.loadtxt(triple_file)
    if triplet_kg.ndim == 1 and triplet_kg.size:
        triplet_kg = triplet_kg.reshape(-1, 3)
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t
        if rel + r not in relation2id:
            relation2id[rel + r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    adj_list = []
    for i in range(rel):
        idx = np.argwhere(triplets_train[:, 2] == i)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8),
             (triplets_train[:, 0][idx].squeeze(1), triplets_train[:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i - rel)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8),
             (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))))
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr


__all__ = ["process_files_ddi", "process_files_decagon"]

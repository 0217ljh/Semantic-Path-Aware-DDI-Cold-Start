import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, saved_relation2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(len(entity2id), len(entity2id))))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

def process_files_ddi(files, triple_file, saved_relation2id=None, keeptrainone = False):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        # with open(file_path) as f:
        #     file_data = [line.split() for line in f.read().split('\n')[:-1]]
        file_data = np.loadtxt(file_path)
        for triplet in file_data:
            #print(triplet)
            triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = triplet[0]
                #ent += 1
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = triplet[1]
                #ent += 1
            if not saved_relation2id and triplet[2] not in relation2id:
                if keeptrainone:
                    triplet[2] = 0
                    relation2id[triplet[2]] = 0
                    rel = 1
                else:
                    relation2id[triplet[2]] = triplet[2]
                    rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[2] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])

        triplets[file_type] = np.array(data)
    #print(rel)
    triplet_kg = np.loadtxt(triple_file)
    print(np.max(triplet_kg[:, -1]))
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        if not saved_relation2id and rel+r not in relation2id:
            relation2id[rel+r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    #print(relation2id, rel)

    # Baseline data-schema fix: the original drugbank data had DENSE, contiguous
    # entity ids so `len(entity2id)` == max_id+1. Our leaf-derived data can have
    # entity ids in [0, n_ent) where a few low drug ids never appear in a pair or KG
    # edge, so `len(entity2id)` < max_id+1 and the csc shape underflows. Size every
    # adjacency by max_id+1 (a fixed node span) to keep all ids in range.
    _max_id = 0
    if len(entity2id):
        _max_id = max(entity2id.keys())
    n_nodes = int(_max_id) + 1

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    #print(kg_triple)
    #for i in range(len(relation2id)):
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(n_nodes, n_nodes)))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        #print(len(idx), i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(n_nodes, n_nodes)))
    #print(adj_list)
    #assert 0
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

def process_files_decagon(files, triple_file, saved_relation2id=None, keeptrainone = False):
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    triplets_mr = {}
    polarity_mr = {}
    kg_triple = []
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        data_mr = []
        data_pol = []
        edges = {}
        # with open(file_path) as f:
        #     file_data = [line.split() for line in f.read().split('\n')[:-1]]
        #file_data = np.loadtxt(file_path)
        
        train = []
        train_edge = []
        with open(file_path, 'r') as f:
            for lines in f:
                x, y, z, w = lines.strip().split('\t')
                x, y = int(x), int(y)
                w = int(w) # pos/neg edge
                z1 = list(map(int, z.split(',')))

                
                z = [0] if keeptrainone else [i for i, _ in enumerate(z1) if _ == 1]  
                #train.append([x,y])
                #train_edge.append(z)
                for s in z:
                    #print(triplet)
                    triplet = [x,y,s]
                    triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if triplet[0] not in entity2id:
                        entity2id[triplet[0]] = triplet[0]
                        #ent += 1
                    if triplet[1] not in entity2id:
                        entity2id[triplet[1]] = triplet[1]
                        #ent += 1
                    if not saved_relation2id and triplet[2] not in relation2id:
                        if keeptrainone:
                            triplet[2] = 0
                            relation2id[triplet[2]] = 0
                            rel = 1
                        else:
                            relation2id[triplet[2]] = triplet[2]
                            rel += 1

                    # Save the triplets corresponding to only the known relations
                    if triplet[2] in relation2id:
                        data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])
                if keeptrainone:
                    #triplet[2] = 0
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], 0])
                else:
                    data_mr.append([entity2id[triplet[0]], entity2id[triplet[1]], z1])
                data_pol.append(w)
        triplets[file_type] = np.array(data)
        triplets_mr[file_type] = data_mr
        polarity_mr[file_type] = np.array(data_pol)
    # Baseline data-schema fix: the original asserts were BioSNAP-specific
    # (|drugs|==604, rel==200). Our TWOSIDES leaf has a different drug-pool size, so
    # only the keeptrainone (single train-relation bucket) invariant is kept.
    if keeptrainone:
        assert rel == 1
    #print(rel)
    triplet_kg = np.loadtxt(triple_file)
    print(np.max(triplet_kg[:, -1]))
    for (h, t, r) in triplet_kg:
        h, t, r = int(h), int(t), int(r)
        if h not in entity2id:
            entity2id[h] = h
        if t not in entity2id:
            entity2id[t] = t 
        if not saved_relation2id and rel+r not in relation2id:
            relation2id[rel+r] = rel + r
        kg_triple.append([h, t, r])
    kg_triple = np.array(kg_triple)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Baseline data-schema fix (see process_files_ddi): size adjacency by max_id+1,
    # not len(entity2id), since our leaf entity ids may be non-dense.
    _max_id = max(entity2id.keys()) if len(entity2id) else 0
    n_nodes = int(_max_id) + 1

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    #print(kg_triple)
    #for i in range(len(relation2id)):
    for i in range(rel):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(n_nodes, n_nodes)))
    for i in range(rel, len(relation2id)):
        idx = np.argwhere(kg_triple[:, 2] == i-rel)
        #print(len(idx), i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(n_nodes, n_nodes)))
    #print(adj_list)
    #assert 0
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr

def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

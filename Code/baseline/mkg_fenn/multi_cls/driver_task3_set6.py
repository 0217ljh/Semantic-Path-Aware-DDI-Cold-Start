import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sqlite3
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
# from pytorchtools import EarlyStopping
from model_task3_set6 import FusionLayer,GNN1,GNN2,GNN3,GNN4
from collections import defaultdict
import os
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import warnings

warnings.filterwarnings("ignore")

# --- ORIGINAL MKG-FENN task3 (both-unseen, multi-cls), repointed to read the
#     event.db dataset placed in baseline/mkg_fenn/_data/necessary/. ---
from pathlib import Path as _Path  # noqa: E402
_NEC = _Path(__file__).resolve().parent.parent / "_data" / "necessary"
EVENT_DB_PATH = str(_NEC / "event__official.db")
DATASET_DIR = str(_NEC)
OUT_DIR = str(_Path(__file__).resolve().parent / "_mkgfenn_task3_orig_out")

parser = argparse.ArgumentParser(description='GNN based on the whole datas')
parser.add_argument("--epoches",type=int,default=120)
parser.add_argument("--batch_size",type=int,choices=[2048,1024,512,256,128],default=1024)
parser.add_argument("--weigh_decay",type=float,choices=[1e-1,1e-2,1e-3,1e-4,1e-8],default=1e-8)
parser.add_argument("--lr",type=float,choices=[1e-3,1e-4,1e-5,4*1e-3],default=5*1e-3) #4*1e-3
# 对于不同的数据可以尝试不同采样数量
parser.add_argument("--neighbor_sample_size",choices=[4,6,10,16],type=int,default=6)
parser.add_argument("--event_num",type=int,default=65)

parser.add_argument("--n_drug",type=int,default=572)
parser.add_argument("--seed",type=int,default=1)
parser.add_argument("--dropout",type=float,default=0.5)
parser.add_argument("--embedding_num",type=int,choices=[128,64,256,32],default=256)
args = parser.parse_args()

# seed = 0

random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.use_deterministic_algorithms(True, warn_only=True)  # GPU: some ops lack deterministic kernels
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_dataset(drug_name_id,num):
    kg = defaultdict(list)
    tails = {}
    relations = {}
    drug_list=[]
    filename = DATASET_DIR + "/dataset" + str(num) + "__official.txt"
    with open(filename, encoding="utf8") as reader:
        for line in reader:
            string= line.rstrip().split('//',2)
            head=string[0]
            tail=string[1]
            relation=string[2]
            drug_list.append(drug_name_id[head])
            if tail not in tails:
                tails[tail] = len(tails)
            if relation not in relations:
                relations[relation] = len(relations)
            if num==3:
                kg[drug_name_id[head]].append((drug_name_id[tail], relations[relation]))
                kg[drug_name_id[tail]].append((drug_name_id[head], relations[relation]))
            else:
                kg[drug_name_id[head]].append((tails[tail], relations[relation]))
    return kg,len(tails),len(relations)

def prepare(mechanism, action):
    d_label = {}
    d_event = []
    new_label = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i
    for i in range(len(d_event)):
        new_label.append(d_label[d_event[i]])
    return new_label,len(count)

def l2_re(parameter):
    reg=0
    for param in parameter:
        reg+=0.5*(param**2).sum()
    return reg

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    # label_binarize:返回一个one_hot的类型
    y_one_hot = label_binarize(y_test, classes = np.arange(event_num))
    pred_one_hot = label_binarize(pred_type,classes = np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = 0.0
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = 0.0
    result_all[5] = 0.0
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = 0.0
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = 0.0
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]

def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task3'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def train(train_x,train_y,test_x,test_y,net,test_adj):
    loss_function=nn.CrossEntropyLoss()
    opti = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weigh_decay)
    test_loss, test_acc, train_l = 0, 0, 0
    train_a = []
    train_x1 = train_x.copy()
    train_x[:,[0,1]] = train_x[:,[1,0]]
    train_x_total = torch.LongTensor(np.concatenate([train_x1, train_x], axis=0))
    train_y = torch.LongTensor(np.concatenate([train_y,train_y]))
    # np.random.seed(args.seed)
    # np.random.shuffle(train_x_total)
    # np.random.seed(args.seed)
    # np.random.shuffle(train_y)
    train_data = TensorDataset(train_x_total, train_y)
    train_iter = DataLoader(train_data, args.batch_size, shuffle=True)
    test_list = []
    max_test_output = torch.zeros((0,65),dtype=torch.float)
    for epoch in range(args.epoches):
        test_loss, test_score, train_l = 0, 0, 0
        train_a = []
        net.train()
        for x, y in train_iter:
            opti.zero_grad()
            train_acc = 0
            train_label = torch.LongTensor(y).to(DEVICE)
            x = torch.LongTensor(x).to(DEVICE)
            f_input = list()
            f_input.append(x)
            f_input.append(0)
            f_input.append(defaultdict(list))
            output = net(f_input)
            l = loss_function(output, train_label)
            l.backward()
            opti.step()
            train_l += l.item()
            train_acc = accuracy_score(torch.argmax(output,dim=1).cpu(), train_label.cpu())
            train_a.append(train_acc)
        net.eval()
        with torch.no_grad():
            test_x = torch.as_tensor(test_x, dtype=torch.long, device=DEVICE)
            f_input = list()
            f_input.append(test_x)
            f_input.append(1)
            f_input.append(test_adj)
            test_output = F.softmax(net(f_input),dim=1)
            test_label = torch.LongTensor(test_y).to(DEVICE)
            loss = loss_function(test_output, test_label)
            test_loss = loss.item()
            test_score = f1_score(torch.argmax(test_output,dim=1).cpu(), test_label.cpu(),average='macro')
            test_acc = accuracy_score(torch.argmax(test_output,dim=1).cpu(), test_label.cpu())
            test_list.append(test_score)
            if test_score == max(test_list):
                max_test_output = test_output
            print("test_acc:", test_acc, "train_acc:", sum(train_a) / len(train_a),"test_score:",test_score)
        print('epoch [%d] train_loss: %.6f testing_loss: %.6f ' % (
                epoch + 1, train_l / len(train_y), test_loss / len(test_y)))
    return test_loss / len(test_y), max(test_list), train_l / len(train_y), sum(train_a) / len(
        train_a), test_list, max_test_output

    # return test_loss, max(test_list), train_l, train_a, test_list

def find_dif(raw_matrix):
    # Vectorised + dynamic equivalent of the original O(n^2*k) loop:
    # sim[i,j] = number of columns where row i equals row j; diagonal = 0.
    X = np.asarray(raw_matrix)
    sim = (X[:, None, :] == X[None, :, :]).sum(axis=2).astype(np.float64)
    np.fill_diagonal(sim, 0.0)
    return sim

def Jaccard(matrix):
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
    return numerator / denominator

def main():
    import json as _json, os as _os, sys as _sys
    from types import SimpleNamespace
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from kg_builder import build_all_kgs

    SET6 = "/mnt/e/Code/My_project/DDI_Benchmark_Pipeline/Dataset/Drugbank/Set_6"
    OUT = str(_Path(__file__).resolve().parent / "_mkgfenn_task3_set6_out")
    print("[set6] loading ...", flush=True)
    drugs = pd.read_csv(SET6 + "/drugs.csv")
    ddi = pd.read_csv(SET6 + "/ddi_edges.csv")

    dict1 = {}
    for did in drugs["drugbank_id"].astype(str):
        if did not in dict1:
            dict1[did] = len(dict1)
    id_list = list(dict1.keys())
    n_drug = len(dict1)
    drug_name = list(range(n_drug))
    drug_id2smiles = dict(zip(drugs["drugbank_id"].astype(str), drugs["smiles"].astype(str)))

    kg_ent = SimpleNamespace()
    for rel, fn in [("enzymes", "drug_enzymes.csv"), ("targets", "drug_targets.csv"),
                    ("transporters", "drug_transporters.csv"), ("carriers", "drug_carriers.csv"),
                    ("pathways", "drug_pathways.csv")]:
        p = SET6 + "/" + fn
        setattr(kg_ent, rel, pd.read_csv(p) if _os.path.isfile(p) else pd.DataFrame())

    mech = ddi["ddi_type"].astype(str).tolist()
    new_label_all, event_num = prepare(mech, ["" for _ in mech])
    new_label_all = np.array(new_label_all)
    args.event_num = event_num
    a_idx = ddi["drug_a_id"].astype(str).map(dict1)
    b_idx = ddi["drug_b_id"].astype(str).map(dict1)
    valid = a_idx.notna() & b_idx.notna()
    drugA_id = a_idx[valid].astype(int).to_numpy()
    drugB_id = b_idx[valid].astype(int).to_numpy()
    new_label = new_label_all[valid.to_numpy()]
    print("[set6] n_drug=%d n_class=%d n_pairs=%d" % (n_drug, event_num, len(new_label)), flush=True)

    temp_drugA = [[] for _ in range(event_num)]
    temp_drugB = [[] for _ in range(event_num)]
    for i in range(len(new_label)):
        temp_drugA[new_label[i]].append(int(drugA_id[i]))
        temp_drugB[new_label[i]].append(int(drugB_id[i]))
    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drugA[i])):
            drug_cro_dict[temp_drugA[i][j]] = j % 5
            drug_cro_dict[temp_drugB[i][j]] = j % 5
    FOLD = 0
    test_drug = [d for d, f in drug_cro_dict.items() if f == FOLD]
    train_drug = [d for d, f in drug_cro_dict.items() if f != FOLD]
    test_drug_set = set(test_drug)

    a_test = np.isin(drugA_id, test_drug)
    b_test = np.isin(drugB_id, test_drug)
    train_mask = (~a_test) & (~b_test)
    test_mask = a_test & b_test
    X_train = np.flatnonzero(train_mask)
    X_test = np.flatnonzero(test_mask)
    print("[set6] fold0: train_drug=%d test_drug=%d train_pairs=%d test_pairs=%d"
          % (len(train_drug), len(test_drug), len(X_train), len(X_test)), flush=True)

    train_pos_df = pd.DataFrame({
        "drug_a_id": [id_list[drugA_id[i]] for i in X_train],
        "drug_b_id": [id_list[drugB_id[i]] for i in X_train],
    })
    print("[set6] building KGs (kg_builder, KG3 from train) ...", flush=True)
    dataset, tail_len, relation_len = build_all_kgs(
        kg=kg_ent, drug_id2smiles=drug_id2smiles, dict1=dict1, train_pos_df=train_pos_df)

    temp_kg = [defaultdict(list) for _ in range(4)]
    for p, name in enumerate(["dataset1", "dataset2", "dataset3", "dataset4"]):
        for i in dataset[name].keys():
            for tup in dataset[name][i]:
                temp_kg[p][i].append(tup[0])
    fm1 = np.zeros((n_drug, tail_len["dataset1"]), dtype=float)
    fm2 = np.zeros((n_drug, tail_len["dataset2"]), dtype=float)
    fm3 = np.zeros((n_drug, tail_len["dataset4"]), dtype=float)
    fm4 = np.zeros((n_drug, n_drug), dtype=float)
    for i in dataset["dataset4"].keys():
        for pidx, v in dataset["dataset4"][i]:
            fm3[i][pidx] = v
    for i in temp_kg[0]:
        for j in temp_kg[0][i]:
            fm1[i][j] = 1
    for i in temp_kg[1]:
        for j in temp_kg[1][i]:
            fm2[i][j] = 1
    for i in temp_kg[2]:
        for j in temp_kg[2][i]:
            fm4[i][j] = 1
    drug_sim1 = Jaccard(fm1)
    drug_sim2 = Jaccard(fm2)
    drug_sim4 = Jaccard(fm4)
    drug_sim3 = find_dif(fm3)

    test_adj = [defaultdict(list) for _ in range(4)]
    sims = [drug_sim1, drug_sim2, drug_sim3, drug_sim4]
    for k in range(4):
        sim = sims[k]
        for j in test_drug:
            target_list = np.asarray(sim[j]).ravel().tolist()
            max_v = 0
            current_p = []
            for p, v in enumerate(target_list):
                if v > max_v and p not in test_drug_set and p != j:
                    max_v = v
                    current_p = [p]
                elif v == max_v and p not in test_drug_set and p != j:
                    current_p.append(p)
            test_adj[k][j].append(current_p)

    # Every drug must have >=1 neighbour in each KG the model samples
    # (arrge does np.random.choice over neighbours; empty -> error, and plain
    # dicts -> KeyError). KG2/KG3/KG4 are ghost-filled by kg_builder; KG1 is
    # not, so backfill an in-range dummy (entity 0, rel 0) for empty drugs.
    for _name in ("dataset1", "dataset2", "dataset3", "dataset4"):
        _d = defaultdict(list, dataset[_name])
        for _idx in range(n_drug):
            if not _d[_idx]:
                _d[_idx].append((0, 0))
        dataset[_name] = _d

    x_datasets = np.stack([drugA_id, drugB_id], axis=1)
    net = nn.Sequential(GNN1(dataset, tail_len, relation_len, args, dict1, drug_name),
                        GNN2(dataset, tail_len, relation_len, args, dict1, drug_name),
                        GNN3(dataset, tail_len, relation_len, args, dict1, drug_name),
                        FusionLayer(args)).to(DEVICE)
    train_x = x_datasets[X_train]
    train_y = new_label[X_train]
    test_x = x_datasets[X_test]
    test_y = new_label[X_test]
    _, _, _, _, _, test_output = train(train_x, train_y, test_x, test_y, net, test_adj)
    pred_type = torch.argmax(test_output, dim=1).cpu().numpy()
    result_all, result_eve = evaluate(pred_type, test_output.cpu().numpy(), test_y, args.event_num)
    _os.makedirs(OUT, exist_ok=True)
    metrics = {"ACC": float(result_all[0]), "micro_AUPR": float(result_all[1]),
               "micro_AUC": float(result_all[3]), "macro_F1": float(result_all[6]),
               "macro_PRE": float(result_all[8]), "macro_REC": float(result_all[10]),
               "event_num": int(event_num), "n_drug": n_drug,
               "n_train_pairs": int(len(X_train)), "n_test_pairs": int(len(X_test))}
    with open(OUT + "/metrics.json", "w") as _f:
        _json.dump(metrics, _f, indent=2)
    print("=== MKG-FENN task3 (both-unseen, Set_6 %d-class, single fold) ===" % event_num, flush=True)
    print("  ACC=%.4f micro-AUPR=%.4f micro-AUC=%.4f macro-F1=%.4f macro-PRE=%.4f macro-REC=%.4f"
          % (result_all[0], result_all[1], result_all[3], result_all[6], result_all[8], result_all[10]), flush=True)
    return


if __name__ == '__main__':
    main()

"""Per-epoch train / eval / test loops. Mirrors upstream ``train_eval.py``.

Each loop iterates over a DataLoader whose collate_fn returns 4 PyG Batches
(see :func:`utils.collate`). The model's forward returns ``(predicts, loss)``;
metrics are computed at end-of-epoch over all collected probabilities and
ground-truth labels.

Metric set: ACC, F1 (binary, label=1 positive), AUROC, AUPRC. ``acc / f1``
use threshold 0.5 on predicted prob.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _move(batches, device: str):
    return tuple(b.to(device) for b in batches)


def train(loop, model, optimizer, device: str = "cuda"):
    correct, total_loss = 0, 0.0
    model.train()
    prob_all = []
    label_all = []

    for data in loop:
        batches = _move(data, device)
        optimizer.zero_grad(set_to_none=True)
        predicts, loss = model(*batches)
        loss.backward()

        prob_all.append(predicts)
        label_all.append(batches[0].y)

        total_loss += loss.item() * num_graphs(batches[0])
        optimizer.step()

    train_loss = total_loss / max(len(loop), 1)
    label_all_np = torch.cat(label_all).cpu().detach().numpy()
    prob_all_np = torch.cat(prob_all).cpu().detach().numpy()
    train_acc, train_f1, train_auc, train_aupr = get_score(label_all_np, prob_all_np)
    return train_acc, train_f1, train_auc, train_aupr, train_loss


def eval(loader, model, device: str = "cuda"):
    correct, total_loss = 0, 0.0
    model.eval()
    prob_all = []
    label_all = []

    with torch.no_grad():
        for data in loader:
            batches = _move(data, device)
            predicts, loss = model(*batches)
            prob_all.append(predicts)
            label_all.append(batches[0].y)
            total_loss += loss.item() * num_graphs(batches[0])

    eval_loss = total_loss / max(len(loader.dataset), 1)
    label_all_np = torch.cat(label_all).cpu().detach().numpy()
    prob_all_np = torch.cat(prob_all).cpu().detach().numpy()
    eval_acc, eval_f1, eval_auc, eval_aupr = get_score(label_all_np, prob_all_np)
    return eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss


def test(loader, model, device: str = "cuda"):
    model.eval()
    prob_all = []
    label_all = []
    total_loss = 0.0

    with torch.no_grad():
        for data in loader:
            batches = _move(data, device)
            predicts, loss = model(*batches)
            prob_all.append(predicts)
            label_all.append(batches[0].y)
            total_loss += loss.item() * num_graphs(batches[0])

    test_loss = total_loss / max(len(loader.dataset), 1)
    label_all_np = torch.cat(label_all).cpu().detach().numpy()
    prob_all_np = torch.cat(prob_all).cpu().detach().numpy()
    test_acc, test_f1, test_auc, test_aupr = get_score(label_all_np, prob_all_np)
    return {
        "acc": test_acc,
        "f1": test_f1,
        "auc": test_auc,
        "aupr": test_aupr,
        "loss": test_loss,
    }


def num_graphs(data):
    if hasattr(data, "num_graphs"):
        return data.num_graphs
    return data.x.c_size


def get_score(label_all, prob_all):
    predicts_label = [1 if prob >= 0.5 else 0 for prob in prob_all]
    acc = accuracy_score(label_all, predicts_label)
    f1 = f1_score(label_all, predicts_label)
    auroc = roc_auc_score(label_all, prob_all)
    p, r, _t = precision_recall_curve(label_all, prob_all)
    auprc = auc(r, p)
    return acc, f1, auroc, auprc

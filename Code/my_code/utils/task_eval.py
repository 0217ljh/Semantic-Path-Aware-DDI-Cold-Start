"""Task-specific evaluation helpers per CLAUDE.md '训练中间 eval / save 规范'.

Three tasks supported:
  - binary:       single label per pair (interact yes/no). Eval: AUC / NLL / F1.
  - multiclass:   86-way DrugBank DDI type. Eval: top-1 acc / top-3 acc /
                  macro F1 / macro AUC across classes.
  - multilabel:   963-way TWOSIDES side-effect. Eval: mean AUC per label /
                  macro F1 / mean AUPR. (Not implemented yet; placeholder.)

Functions return dict of metrics, with keys prefixed by task type for clarity.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
def eval_binary(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Binary classification eval.

    preds:  (N,) sigmoid probabilities in [0, 1]
    labels: (N,) 0/1
    Returns: {auc, nll, f1, precision, recall, n_pos, n_neg}
    """
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    eps = 1e-7
    preds_clip = np.clip(preds, eps, 1 - eps)

    auc = float(roc_auc_score(labels, preds))
    nll = float(log_loss(labels, preds_clip))
    pred_label = (preds >= 0.5).astype(np.int32)
    tp = int(((pred_label == 1) & (labels == 1)).sum())
    fp = int(((pred_label == 1) & (labels == 0)).sum())
    fn = int(((pred_label == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "auc": auc,
        "nll": nll,
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


# ---------------------------------------------------------------------------
def eval_multiclass(
    preds: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict:
    """Multi-class single-label eval.

    preds:  (N, n_classes) softmax probabilities (rows sum to 1)
    labels: (N,) int class indices in [0, n_classes)
    Returns: {top1_acc, top3_acc, top5_acc, macro_f1, macro_auc, n_classes, n_samples}
    """
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    assert preds.ndim == 2 and preds.shape[1] == n_classes, \
        f"preds shape {preds.shape} mismatch n_classes={n_classes}"
    assert labels.ndim == 1, f"labels must be 1-D int class indices, got {labels.shape}"

    # Top-k accuracy
    def topk_acc(k):
        topk = np.argpartition(-preds, k - 1, axis=1)[:, :k]
        return float(np.mean([y in topk[i] for i, y in enumerate(labels)]))

    top1 = topk_acc(1)
    top3 = topk_acc(min(3, n_classes))
    top5 = topk_acc(min(5, n_classes))

    # Macro F1: predict argmax, then sklearn macro F1
    pred_label = preds.argmax(axis=1)
    macro_f1 = float(f1_score(labels, pred_label, average="macro", zero_division=0))

    # Macro AUC (one-vs-rest, average over classes with both pos & neg samples)
    macro_auc = float("nan")
    aucs = []
    for c in range(n_classes):
        y_c = (labels == c).astype(np.int32)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            continue  # skip degenerate
        aucs.append(roc_auc_score(y_c, preds[:, c]))
    if aucs:
        macro_auc = float(np.mean(aucs))

    return {
        "top1_acc": top1,
        "top3_acc": top3,
        "top5_acc": top5,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "n_classes": n_classes,
        "n_samples": int(len(labels)),
        "n_aucs_computed": len(aucs),
    }


# ---------------------------------------------------------------------------
def eval_multilabel(
    preds: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
) -> dict:
    """Multi-label binary eval (each label independent).

    preds:  (N, n_labels) sigmoid probabilities per label
    labels: (N, n_labels) {0, 1} k-hot
    Returns: {mean_auc, macro_f1, mean_aupr, n_labels, n_samples}
    """
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    assert preds.shape == labels.shape, f"preds {preds.shape} vs labels {labels.shape}"
    assert preds.shape[1] == n_labels

    aucs, auprs = [], []
    for c in range(n_labels):
        y_c = labels[:, c]
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            continue
        aucs.append(roc_auc_score(y_c, preds[:, c]))
        auprs.append(average_precision_score(y_c, preds[:, c]))

    pred_label = (preds >= 0.5).astype(np.int32)
    macro_f1 = float(f1_score(labels, pred_label, average="macro", zero_division=0))

    return {
        "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "mean_aupr": float(np.mean(auprs)) if auprs else float("nan"),
        "macro_f1": macro_f1,
        "n_labels": n_labels,
        "n_samples": int(labels.shape[0]),
        "n_aucs_computed": len(aucs),
    }

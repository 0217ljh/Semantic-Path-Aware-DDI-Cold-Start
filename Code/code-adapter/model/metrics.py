"""Stage 7 (used by the harness) - the full report-metrics suite.

Computes exactly TaskSpec.report_metrics from logits + labels (tex Evaluation
Metrics): binary = acc/auroc/f1; multi = acc/aupr/macro_auroc/macro_f1/
macro_precision/macro_recall. Macro metrics are robust to classes absent from a
cold-test fold (they are averaged only over classes present in y).
"""
from __future__ import annotations

import numpy as np
import scipy.special as sp
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from specs import TaskSpec


def _macro_ovr(y: np.ndarray, proba: np.ndarray, fn) -> float:
    """Macro one-vs-rest average of `fn(y_bin, score)` over classes present in y
    with both a positive and a negative example (others skipped)."""
    vals = []
    for c in np.unique(y):
        yb = (y == c).astype(int)
        if yb.min() == yb.max():          # only one class present for c -> undefined
            continue
        try:
            vals.append(float(fn(yb, proba[:, c])))
        except (ValueError, IndexError):
            continue
    return float(np.mean(vals)) if vals else float("nan")


def compute_metrics(logits: np.ndarray, y: np.ndarray, task: TaskSpec) -> dict:
    """Return {name: value} for every name in task.report_metrics."""
    y = np.asarray(y)
    out: dict[str, float] = {}
    if task.is_binary:
        p = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
        pred = (p >= 0.5).astype(int)
        for m in task.report_metrics:
            if m == "acc":
                out[m] = float(accuracy_score(y, pred))
            elif m == "auroc":
                out[m] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
            elif m == "f1":
                out[m] = float(f1_score(y, pred, zero_division=0))
        return out

    proba = sp.softmax(np.asarray(logits), axis=-1)
    pred = proba.argmax(1)
    # macro ONLY over classes present in this fold's y_true (consistent with the
    # AUC/AP macros below, which are undefined for absent classes). Absent classes
    # are not counted as zero, so a cold fold missing rare types is not penalized
    # and macro_f1 stays a stable cold-val selection signal.
    present = sorted(np.unique(y).tolist())
    for m in task.report_metrics:
        if m == "acc":
            out[m] = float(accuracy_score(y, pred))
        elif m == "macro_f1":
            out[m] = float(f1_score(y, pred, average="macro", labels=present, zero_division=0))
        elif m == "macro_precision":
            out[m] = float(precision_score(y, pred, average="macro", labels=present, zero_division=0))
        elif m == "macro_recall":
            out[m] = float(recall_score(y, pred, average="macro", labels=present, zero_division=0))
        elif m == "macro_auroc":
            out[m] = _macro_ovr(y, proba, roc_auc_score)
        elif m == "aupr":
            out[m] = _macro_ovr(y, proba, average_precision_score)
    return out


__all__ = ["compute_metrics"]

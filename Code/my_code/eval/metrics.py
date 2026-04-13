"""Compute metrics from prediction DataFrame/rows."""
from __future__ import annotations

from typing import Any, Dict, List


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def compute_metrics(rows: List[dict], metric_names: List[str]) -> Dict[str, float]:
    y_true = [r["y_true"] for r in rows]
    y_pred = [r["y_pred"] for r in rows]
    result = {}
    for name in metric_names:
        if name == "accuracy":
            result[name] = accuracy(y_true, y_pred)
    return result

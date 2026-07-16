"""Typed specs for the code-adapter experiment pipeline (self-contained copy).

Stage-1 only needs ``TaskSpec``. Copied from
``my_code/rank_analysis/specs.py`` and kept minimal on purpose (codex ruling:
do not over-copy protocol/scoring objects into the data-load stage). Other
specs are added to this file only when a later stage actually consumes them.

Metric suite aligns to the paper tex (Evaluation Metrics), which is the code's
source of truth: binary = ACC/AUROC/F1; multi = ACC/AUPR/AUC/F1/precision/recall
(F1/precision/recall macro-averaged). ``val_monitor`` is the SINGLE metric the
harness supervises for cold-val best-epoch selection; ``report_metrics`` is the
full suite the stage-7 metrics module computes and reports.
"""
from __future__ import annotations

from dataclasses import dataclass


#: full reported suites (aligned to tex Evaluation Metrics)
_BINARY_REPORT = ("acc", "auroc", "f1")
_MULTI_REPORT = ("acc", "aupr", "macro_auroc", "macro_f1", "macro_precision", "macro_recall")


@dataclass(frozen=True)
class TaskSpec:
    """Binary vs multi-class DDI, threaded through data / fit / metrics so no
    stage subclasses on task."""
    kind: str                       # "binary" | "multiclass"
    n_classes: int                  # binary -> 2, multiclass -> K
    val_monitor: str                # SINGLE supervised metric for cold-val best-epoch select
    metric: str                     # rank-curve primary metric: "auroc" | "macro_auroc"
    uses_negatives: bool            # binary True (dataset negatives), multiclass False
    report_metrics: tuple = ()      # full suite computed + reported by the metrics stage

    @property
    def is_binary(self) -> bool:
        return self.kind == "binary"

    @staticmethod
    def binary() -> "TaskSpec":
        return TaskSpec("binary", 2, "auroc", "auroc", True, _BINARY_REPORT)

    @staticmethod
    def multiclass(n_classes: int) -> "TaskSpec":
        return TaskSpec("multiclass", int(n_classes), "macro_f1", "macro_auroc", False, _MULTI_REPORT)


__all__ = ["TaskSpec"]

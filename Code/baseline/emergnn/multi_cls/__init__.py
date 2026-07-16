"""EmerGNN multi-class classification task (per-pair, predict which of K interaction types).

Matches the original paper's DrugBank setting (paper §Methods Eq. 5: softmax CE
over 86 interaction classes).
"""

from __future__ import annotations

from baseline.emergnn.multi_cls.baseline import EmerGNNMulticlassBaseline
from baseline.emergnn.multi_cls.model import EmerGNN_MC

__all__ = ["EmerGNNMulticlassBaseline", "EmerGNN_MC"]

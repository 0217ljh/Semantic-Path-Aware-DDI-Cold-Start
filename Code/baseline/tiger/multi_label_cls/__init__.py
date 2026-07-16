"""TIGER multilabel (fixed-L side-effect) task — TWOSIDES Case-B adaptation.

NOT paper-native (Su et al. AAAI 2024 is binary). The dual-channel algorithm core
(mol GraphTransformer + BKG-subgraph GraphTransformer + relation-aware attention +
MI loss) is reused UNCHANGED; only the task head/loss/metric surface is swapped.
"""
from __future__ import annotations

from baseline.tiger.multi_label_cls.baseline import TIGERMultilabelBaseline

__all__ = ["TIGERMultilabelBaseline"]

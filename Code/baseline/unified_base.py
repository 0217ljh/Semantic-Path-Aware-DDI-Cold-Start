"""Per-task baseline interface for the unified benchmark (decision B, codex 019f19fb).

A baseline trains on ONE leaf and predicts on that leaf's test pairs. The regime
(S0/S1/S2) is fixed by the leaf, so there is NO cross-regime routing — each
(dataset, regime, task, fold) trains its own model. This ABC is SEPARATE from the
legacy `baseline.base.BaselineModel` (which consumes the old PairDataset/SplitFolds);
the legacy one is untouched.

Contract:
    fit(train_df, val_df, *, resources) -> None
    predict(test_df) -> np.ndarray, shape by task:
        binary      -> (n,)        P(positive)
        multiclass  -> (n, K)      class probabilities (K = n_labels)
        multilabel  -> (n, L)      per-label probabilities (L = n_labels)

`resources` is a `data_utils.unified_loader.LeafResources` (drugs, drug_split, kg,
meta, label_vocab). Task-native label columns: binary y_bin; multiclass y_cls /
y_cls_train; multilabel is_positive / y_label_ids.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from data_utils.unified_loader import LeafResources

#: (name, task) -> subclass registry (a baseline can have one class per task; separate
#: namespace from legacy base._REGISTRY).
_UNIFIED_REGISTRY: dict[tuple[str, str], type["UnifiedBaseline"]] = {}


def register_unified(name: str):
    """Decorator: register a unified baseline under (name, cls.task). `cls.task` must be
    set in the class body (it is, before the decorator runs)."""
    def deco(cls: type["UnifiedBaseline"]) -> type["UnifiedBaseline"]:
        key = (name, cls.task)
        if key in _UNIFIED_REGISTRY:
            raise ValueError(f"unified baseline {key!r} already registered")
        cls.name = name
        _UNIFIED_REGISTRY[key] = cls
        return cls
    return deco


def get_unified(name: str, task: str) -> type["UnifiedBaseline"]:
    return _UNIFIED_REGISTRY[(name, task)]


class UnifiedBaseline(ABC):
    """Leaf-native baseline. One instance trains on one leaf.

    `task` (set by subclass) ∈ {"binary", "multiclass", "multilabel"} and must match the
    leaf's meta task. Subclasses own their model state, batching, and serialization.
    """

    name: ClassVar[str] = "unified_baseline"
    task: ClassVar[str] = ""

    @abstractmethod
    def fit(self, train_df: "pd.DataFrame", val_df: "pd.DataFrame | None" = None, *,
            resources: "LeafResources") -> None:
        """Train on one leaf's rows. `val_df` is for early stopping / model selection."""

    def fit_leaf(self, leaf) -> None:
        """Convenience: train from a whole `data_utils.unified_loader.Leaf` (forwards
        train/val/resources) so wrappers don't re-thread the pieces."""
        self.fit(leaf.train, leaf.val, resources=leaf.resources)

    @abstractmethod
    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        """Score test pairs. Shape per task (see module docstring)."""

    # save/load are free-form (like legacy base): subclasses choose serialization.
    def save(self, path) -> None:  # pragma: no cover - optional
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> "UnifiedBaseline":  # pragma: no cover - optional
        raise NotImplementedError


__all__ = ["UnifiedBaseline", "register_unified", "get_unified"]

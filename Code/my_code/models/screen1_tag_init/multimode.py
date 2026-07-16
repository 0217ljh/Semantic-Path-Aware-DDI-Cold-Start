"""Screen 1 multimode wrapper: EmerGNN multimode trainer using EmerGNN_TAG.

Mirrors `baseline.emergnn.binary_cls.baseline.EmerGNNBaseline` structure
(3 sub-models s0/s1/s2 with mode-specific shuffle_train_mode + batch +
weight_decay + val_split) but every sub-model gets the SAME `external_init`
tensor injected (because the merged KG / entity vocab is shared).

This is NOT a registered baseline (no @register). It is an experiment
class, only used by `run_screen1.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen1_tag_init._per_mode_tag import _PerModeEmerGNN_TAG  # noqa: E402

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


# Mode-specific configs — IDENTICAL to upstream's _SUBMODEL_CONFIGS except
# `feat` is forced to 'E' on the trainer (X-shape isomorphic, see
# `_per_mode_tag.py` for explanation). The external init replaces what
# 'M' / 'E' would have produced.
_SUBMODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "s0": {
        "shuffle_train_mode": "S0",
        "batch_size": 128,
        "weight_decay": 1e-6,
        "val_split": "val_s0",
    },
    "s1": {
        "shuffle_train_mode": "S1",
        "batch_size": 32,
        "weight_decay": 1e-8,
        "val_split": "val_s1",
    },
    "s2": {
        "shuffle_train_mode": "S2",
        "batch_size": 32,
        "weight_decay": 1e-8,
        "val_split": "val_s2",
    },
}


class _ModeSpecificEmerGNN_TAG(_PerModeEmerGNN_TAG):
    """Per-mode TAG trainer with configurable val_split.

    Same pattern as upstream `_ModeSpecificEmerGNN` (validates on the
    sub-model's own cold-start split, not always val_s2).
    """

    def __init__(self, *, val_split: str = "val_s2", **kwargs: Any) -> None:
        if val_split not in ("val_s0", "val_s1", "val_s2"):
            raise ValueError(f"val_split must be val_s0/s1/s2, got {val_split!r}")
        super().__init__(**kwargs)
        self._val_split = val_split

    @torch.no_grad()
    def _validate(self, val: "PairDataset") -> float:
        pos = getattr(val.splits, self._val_split)[["drug_a_id", "drug_b_id"]]
        try:
            neg = val.get_negatives(self._val_split)[["drug_a_id", "drug_b_id"]]
        except Exception:
            return float("nan")
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y_score = np.concatenate(
            [self.predict_proba(pos), self.predict_proba(neg)]
        )
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y_true, y_score))


class EmerGNNTAGBaseline:
    """Screen 1 multimode TAG trainer.

    Public surface mirrors EmerGNNBaseline so existing eval harness
    (predict_proba on test_s0/s1/s2) works unchanged.
    """

    name = "emergnn-tag-screen1"
    VERSION = "1.0"

    def __init__(
        self,
        *,
        external_init: np.ndarray | torch.Tensor,
        external_init_node_ids: list[str],
        variant_tag: str,
        freeze_init: bool = False,
        n_dim: int = 64,
        length: int = 3,
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        device: str = "auto",
        backbone_kg_source: str = "drugbank",
        merged_kg_path: str | Path | None = None,
        merged_kg_blocklist: tuple[str, ...] = (),
        shuffle_ratio: float = 0.8,
        log_step_every: int = 50,
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        save_strategy: str = "no",
        save_steps: int = 500,
        save_total_limit: int = 3,
        load_best_model_at_end: bool = True,
        run_dir: str | Path | None = None,
    ) -> None:
        self._external_init = external_init
        self._external_init_node_ids = list(external_init_node_ids)
        self._variant_tag = str(variant_tag)
        self._freeze_init = bool(freeze_init)
        self._shared_kwargs = dict(
            n_dim=n_dim,
            length=length,
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            device=device,
            backbone_kg_source=backbone_kg_source,
            merged_kg_path=merged_kg_path,
            merged_kg_blocklist=merged_kg_blocklist,
            shuffle_ratio=shuffle_ratio,
            log_step_every=log_step_every,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
        )
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self._models: dict[str, _ModeSpecificEmerGNN_TAG] = {}
        self._g1_drugs: set[str] = set()
        self._g2_drugs: set[str] = set()

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        if not hasattr(train.splits, "g1_drugs") or not hasattr(train.splits, "g2_drugs"):
            raise ValueError("requires PairDataset with g1_drugs / g2_drugs")
        self._g1_drugs = set(map(str, train.splits.g1_drugs))
        self._g2_drugs = set(map(str, train.splits.g2_drugs))
        overlap = self._g1_drugs & self._g2_drugs
        if overlap:
            raise ValueError(f"G1 ∩ G2 non-empty: {len(overlap)} drugs")

        for split_key in ("s0", "s1", "s2"):
            cfg = _SUBMODEL_CONFIGS[split_key]
            print(
                f"\n=========== [screen1/{self._variant_tag}] training {split_key} "
                f"sub-model (mode={cfg['shuffle_train_mode']}, "
                f"batch={cfg['batch_size']}, wd={cfg['weight_decay']:.0e}) ===========",
                flush=True,
            )
            sub_run_dir = (
                self._run_dir / f"submodel_{split_key}"
                if self._run_dir is not None
                else None
            )
            sub_kwargs = dict(self._shared_kwargs)
            sub_kwargs["shuffle_train_mode"] = cfg["shuffle_train_mode"]
            sub_kwargs["batch_size"] = cfg["batch_size"]
            sub_kwargs["weight_decay"] = cfg["weight_decay"]
            sub_kwargs["run_dir"] = sub_run_dir
            sub = _ModeSpecificEmerGNN_TAG(
                val_split=cfg["val_split"],
                external_init=self._external_init,
                external_init_node_ids=self._external_init_node_ids,
                freeze_init=self._freeze_init,
                **sub_kwargs,
            )
            sub.fit(train, val=val, kg=kg)
            self._models[split_key] = sub

    def _classify_pair(self, a: str, b: str) -> str:
        a_g1 = a in self._g1_drugs; a_g2 = a in self._g2_drugs
        b_g1 = b in self._g1_drugs; b_g2 = b in self._g2_drugs
        if not (a_g1 or a_g2):
            raise ValueError(f"drug_a={a} not in g1∪g2")
        if not (b_g1 or b_g2):
            raise ValueError(f"drug_b={b} not in g1∪g2")
        if a_g1 and b_g1:
            return "s0"
        if a_g2 and b_g2:
            return "s2"
        return "s1"

    @torch.no_grad()
    def predict_proba(
        self, pairs: pd.DataFrame, *, kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        # Route each pair to its sub-model
        out = np.zeros(len(pairs), dtype=np.float32)
        for split_key, model in self._models.items():
            mask = pairs.apply(
                lambda r: self._classify_pair(str(r["drug_a_id"]), str(r["drug_b_id"])) == split_key,
                axis=1,
            ).values
            if not mask.any():
                continue
            sub_pairs = pairs.iloc[mask]
            probs = model.predict_proba(sub_pairs)
            out[mask] = probs
        return out


__all__ = ["EmerGNNTAGBaseline"]

"""Stage 4 / model seam - RankModel contract + leak-free scoring objects.

A backbone (or the adapter, later) implements RankModel in DRUG-ID space; the
harness owns the protocol epoch loop, leak-free contexts + asserts, cold-val
checkpoint selection and metric computation. Ported from the rank-analysis seam
(`my_code/rank_analysis`) but aligned to code-adapter's TaskSpec (report_metrics)
and model_meta protocols.

Checkpoint is MODE-AWARE by delegation: state_dict()/load_state_dict() return/
restore only what a given adapter-integration mode trains (backbone-alone -> the
backbone; +adapter frozen -> the adapter only; +adapter joint -> both; adapter-
alone -> the adapter). The harness never inspects the content.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class ScoringContext:
    """A leak-free scoring context in DRUG-ID space. Declares which DDI FACT edges
    are present; each model owns its own bio-KG + entity mapping and combines them
    in encode_pairs. The harness guarantees + asserts no scored pair's own DDI edge
    is in `ddi_edges` (see protocol.assert_pairs_absent)."""
    name: str                            # "fact_kg" | "g_probe"
    ddi_edges: np.ndarray                # (D, 3) object [drug_a, drug_b, ddi_type]
    absent_guarantee: bool = False       # harness sets True after asserting

    def ddi_pair_set(self) -> set:
        if len(self.ddi_edges) == 0:
            return set()
        return {tuple(sorted((str(a), str(b)))) for a, b in self.ddi_edges[:, :2]}


@dataclass
class PairEncoding:
    """Output of RankModel.encode_pairs. pair_repr is optional (not every model
    exposes a pre-scorer representation); logits drive the metrics."""
    pair_ids: np.ndarray                          # (n, 2) drug-id pairs, input-aligned
    pair_repr: Optional[np.ndarray] = None        # (n, d) pre-scorer representation
    logits: Optional[np.ndarray] = None           # (n,) binary | (n, K) multiclass
    repr_kind: str = ""                           # e.g. "node_readout" | "subgraph"
    repr_stage: str = ""                          # exact stage tag
    valid_mask: Optional[np.ndarray] = None       # (n,) True where extraction ok

    def has_repr(self) -> bool:
        return self.pair_repr is not None


@dataclass
class EpochData:
    """One protocol-shuffled epoch handed to model.train_epoch."""
    fact_context: ScoringContext         # DDI edges available as facts this epoch
    target_pairs: np.ndarray             # (n, 2) drug-id training targets
    target_labels: np.ndarray            # (n,) binary 0/1 (with negs) | class ids


@dataclass
class TrainEpochOutput:
    """What a model returns from train_epoch, for the harness's logging."""
    mean_loss: float
    n_targets: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class CheckpointPolicy:
    """How the runner persists checkpoints. `state_dict()` CONTENT is the model's
    business (mode-aware); this only controls WHICH checkpoints are kept + on disk."""
    save_best: bool = True               # keep the best-cold-val state (always in-memory)
    save_last: bool = False              # also keep the final-epoch state
    to_disk: bool = False                # write best/last state to save_dir/*.pt
    save_dir: Optional[Path] = None


class RankModel(abc.ABC):
    """Contract a backbone (or the adapter) implements to run in the harness.
    DRUG-ID space. `head_from_repr` is only used by the rank-truncation analysis;
    the main experiment harness does not call it."""

    @abc.abstractmethod
    def setup(self, task, hp: dict) -> None:
        """Build the model. `hp` carries tunable hyperparameters AND, for frozen
        adapter mode, `backbone_ckpt` (a pre-trained backbone to load + freeze)."""

    @abc.abstractmethod
    def train_epoch(self, epoch: EpochData, rng: np.random.Generator) -> TrainEpochOutput: ...

    @abc.abstractmethod
    def encode_pairs(self, pairs: np.ndarray, context: ScoringContext) -> PairEncoding:
        """Leak-free per-pair scoring on the given context (logits + optional repr)."""

    @abc.abstractmethod
    def known_drugs(self) -> set:
        """Drug-ids this model can embed; the harness filters data to pairs whose
        BOTH drugs are known, so partial-coverage datasets are safe."""

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """Mode-aware trainable state (see module docstring)."""

    @abc.abstractmethod
    def load_state_dict(self, state: dict) -> None: ...

    # -- checkpoint / frozen-mode support ------------------------------------- #
    def effective_hp(self) -> dict:
        """The RESOLVED hyperparameters the model actually used (defaults merged
        with overrides), so a saved checkpoint is self-describing enough for a
        frozen wrapper to rebuild the matching backbone. Default {} -> caller uses
        the raw hp; wrappers should return their merged config."""
        return {}

    def load_pretrained_backbone(self, path, freeze: bool = True) -> None:
        """Load a saved backbone checkpoint (model.checkpoints.load_checkpoint) into
        this model's backbone and, if freeze, set its parameters requires_grad_(False).
        Called by the FROZEN adapter mode after setup builds the matching backbone.
        `hp['backbone_ckpt']` carries the path; a plain backbone-alone model does not
        implement this (it trains its own backbone)."""
        raise NotImplementedError("load_pretrained_backbone: only frozen-adapter wrappers implement this")

    # -- optional, rank-truncation analysis only (not called by the main harness) --
    def head_from_repr(self, pair_repr: np.ndarray) -> np.ndarray:
        raise NotImplementedError("head_from_repr is only needed for rank-truncation analysis")

    # -- optional, RQ2/RQ3 analysis hook (adapter exposes learned params here) --
    def adapter_state(self) -> Optional[dict]:
        """Learned adapter parameters (M_B prototypes, U_t assignments, ...) for
        parameter-space analysis. None for a plain backbone."""
        return None


__all__ = ["RankModel", "ScoringContext", "PairEncoding", "EpochData",
           "TrainEpochOutput", "CheckpointPolicy"]

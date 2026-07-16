"""Typed specs + result objects for the cold-start (S2) rank-analysis pipeline.

Codex-reviewed design (2026-07-06): the model contract exposes an OPTIONAL
pair-level representation (not a mandatory Z), the training protocol (S2 reshuffle
+ leak-free extraction + cold-val selection) is harness-owned, and task/protocol
are first-class shared objects. See framework.py for the ABC + harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    """Binary vs multi-class DDI. Threaded through data / fit / measurements so we
    never subclass on task."""
    kind: str                       # "binary" | "multiclass"
    n_classes: int                  # binary -> 2, multiclass -> K
    val_monitor: str                # cold-val selection metric: "auroc" | "macro_f1"
    metric: str                     # rank-curve metric: "auroc" | "macro_auroc"
    uses_negatives: bool            # binary True (dataset negatives), multiclass False

    @property
    def is_binary(self) -> bool:
        return self.kind == "binary"

    @staticmethod
    def binary() -> "TaskSpec":
        return TaskSpec("binary", 2, "auroc", "auroc", True)

    @staticmethod
    def multiclass(n_classes: int) -> "TaskSpec":
        return TaskSpec("multiclass", int(n_classes), "macro_f1", "macro_auroc", False)


@dataclass(frozen=True)
class S2ProtocolSpec:
    """The locked cold-start (protocol 2) knobs. Owned by the harness."""
    emerging_ratio: float = 0.8          # fraction of drugs KEPT each epoch (0.8 -> 20% emerging)
    ranks: tuple = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    transfer_k: int = 80                 # pseudo-emerging drugs per sparse-source draw
    transfer_draws: int = 6
    seed: int = 42


@dataclass
class ScoringContext:
    """A leak-free scoring context the harness hands to a model for extraction.

    Works in DRUG-ID space (strings), because different models use different KGs
    and entity-id spaces (R-GCN=MergedKG, EmerGNN=its own). The context ONLY
    declares which DDI FACT edges are present; each model owns its bio KG + entity
    mapping and combines them in encode_pairs. The harness GUARANTEES + asserts
    that no scored pair's own DDI edge is in `ddi_edges` (framework.assert_pairs_absent).
    """
    name: str                            # "fact_kg" | "g_probe"
    ddi_edges: np.ndarray                # (D, 3) object array [drug_a_id, drug_b_id, ddi_type]
    absent_guarantee: bool = False       # harness sets True after asserting

    def ddi_pair_set(self) -> set:
        """Canonical unordered {(a,b)} drug-id pairs of the DDI facts, for leak asserts."""
        if len(self.ddi_edges) == 0:
            return set()
        return {tuple(sorted((str(a), str(b)))) for a, b in self.ddi_edges[:, :2]}


@dataclass
class PairEncoding:
    """Output of RankModel.encode_pairs. pair_repr / logits are OPTIONAL because
    not every analyzable model exposes both (codex fix)."""
    pair_ids: np.ndarray                             # (n, 2) drug-id pairs, aligned to inputs
    pair_repr: Optional[np.ndarray] = None           # (n, d) pre-scorer representation
    logits: Optional[np.ndarray] = None              # (n, K) model logits
    repr_kind: str = ""                              # e.g. "node_readout" | "enc_ht" | "subgraph"
    repr_stage: str = ""                             # exact stage, e.g. "pair_mlp_out" | "pre_Wr"
    valid_mask: Optional[np.ndarray] = None          # (n,) True where extraction succeeded

    def has_repr(self) -> bool:
        return self.pair_repr is not None


@dataclass
class TrainEpochOutput:
    """What a model returns from train_epoch, for the harness's logging."""
    mean_loss: float
    n_targets: int = 0
    extra: dict = field(default_factory=dict)

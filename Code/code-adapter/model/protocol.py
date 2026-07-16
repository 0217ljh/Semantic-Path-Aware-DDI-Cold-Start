"""Stage 4 / model seam - cold-start protocol execution (P1 / P2).

Turns a (dataset, epoch) into the training EpochData a model consumes, per the
resolved Protocol (from model_meta):
  * P1_FIXED    - positives are the fixed seen-DDI set every epoch; binary negatives
                  are re-sampled per epoch (reproducible). Transductive-style.
  * P2_EMERGING - EmerGNN-style: each epoch randomly treat a subset of seen drugs
                  as unseen, drop their DDI edges from the facts, and train on the
                  both-emerging pairs (inductive cold-start simulation).
Also builds the leak-free eval fact context (all seen DDI; cold drugs are unseen so
their own edges are never present) and asserts no scored pair leaks its own edge.

Switching protocol therefore changes ONLY which EpochData the loop receives; the
training loop and the model are protocol-agnostic.
"""
from __future__ import annotations

import numpy as np

from model_meta import Protocol
from specs import TaskSpec
from .contracts import EpochData, ScoringContext


class ColdStartProtocol:
    def __init__(self, protocol: Protocol, task: TaskSpec,
                 emerging_ratio: float = 0.8, seed: int = 42,
                 binary_pos_cap: int | None = None, mc_target_cap: int | None = None):
        self.protocol = protocol
        self.task = task
        self.emerging_ratio = emerging_ratio
        self.seed = seed
        # binary P1 ONLY: cap the per-epoch positive targets to this many (negatives are
        # balanced 1:1 downstream), re-sampled each epoch. None = full seen_ddi (default).
        # Speeds up the adapter (target count drives _build_batch cost). Facts = full seen_ddi
        # regardless (only the TRAINING targets are subsampled, not the KG). No effect on
        # multiclass or P2. (user-approved 2026-07-13; adapter is slow at 142k pairs/epoch.)
        self.binary_pos_cap = binary_pos_cap
        # multiclass P1: cap per-epoch typed-positive targets (SMOKE/debug only; None=full=the
        # real main-task setting). Parallel to binary_pos_cap; subsamples targets, not facts.
        self.mc_target_cap = mc_target_cap

    # -- per-epoch training data ------------------------------------------- #
    def epoch(self, data, ep: int, rng: np.random.Generator) -> EpochData:
        if self.protocol is Protocol.P2_EMERGING:
            return self._emerging_epoch(data, rng)
        return self._fixed_epoch(data, rng)

    def _emerging_epoch(self, data, rng: np.random.Generator) -> EpochData:
        seen = data.seen_ddi
        n_keep = int(len(data.train_drugs) * self.emerging_ratio)
        removed = set(rng.choice(data.train_drugs, size=len(data.train_drugs) - n_keep,
                                 replace=False).tolist())
        a_rm = np.array([str(x) in removed for x in seen[:, 0]])
        b_rm = np.array([str(x) in removed for x in seen[:, 1]])
        fact = ScoringContext("fact_kg", seen[(~a_rm) & (~b_rm)])
        tgt_pos = seen[a_rm & b_rm]
        empty = EpochData(fact, np.empty((0, 2), object), np.array([], np.int64))
        if not self.task.is_binary:
            return EpochData(fact, tgt_pos[:, :2], tgt_pos[:, 2].astype(np.int64))
        neg_all = data.train_neg
        if len(tgt_pos) == 0 or len(neg_all) == 0:
            return empty
        na = np.array([str(x) in removed for x in neg_all[:, 0]])
        nb = np.array([str(x) in removed for x in neg_all[:, 1]])
        emerg_neg = neg_all[na & nb]
        if len(emerg_neg) == 0:
            return empty
        return self._binary_targets(fact, tgt_pos[:, :2], emerg_neg, rng)

    def _fixed_epoch(self, data, rng: np.random.Generator) -> EpochData:
        fact = ScoringContext("fact_kg", data.seen_ddi)   # facts = FULL seen_ddi (never subsampled)
        pos = data.seen_ddi[:, :2]
        if not self.task.is_binary:
            types = data.seen_ddi[:, 2].astype(np.int64)
            if self.mc_target_cap and len(pos) > int(self.mc_target_cap):   # SMOKE-only cap
                idx = rng.choice(len(pos), int(self.mc_target_cap), replace=False)
                return EpochData(fact, pos[idx], types[idx])
            return EpochData(fact, pos, types)
        if len(pos) == 0 or len(data.train_neg) == 0:
            return EpochData(fact, np.empty((0, 2), object), np.array([], np.int64))
        if self.binary_pos_cap and len(pos) > int(self.binary_pos_cap):   # per-epoch target subsample
            pos = pos[rng.choice(len(pos), int(self.binary_pos_cap), replace=False)]
        return self._binary_targets(fact, pos, data.train_neg, rng)

    @staticmethod
    def _binary_targets(fact, pos, neg_pool, rng) -> EpochData:
        """Balanced 1:1 positives + per-epoch-sampled negatives (reproducible)."""
        m = min(len(pos), len(neg_pool))
        nidx = rng.choice(len(neg_pool), size=m, replace=False)
        pairs = np.concatenate([pos[:m], neg_pool[nidx]], axis=0)
        labels = np.concatenate([np.ones(m), np.zeros(m)]).astype(np.int64)
        return EpochData(fact, pairs, labels)

    # -- leak-free eval context -------------------------------------------- #
    def fact_context(self, data) -> ScoringContext:
        """All seen DDI as facts. Cold (val/test) drugs are unseen, so their own
        DDI edges are never in seen_ddi -> the cold-inference regime, leak-free."""
        return ScoringContext("fact_kg", data.seen_ddi)

    @staticmethod
    def assert_pairs_absent(pairs: np.ndarray, context: ScoringContext) -> None:
        facts = context.ddi_pair_set()
        for a, b in pairs:
            if tuple(sorted((str(a), str(b)))) in facts:
                raise AssertionError(f"leak: pair ({a},{b}) own DDI edge present in {context.name}")
        context.absent_guarantee = True


__all__ = ["ColdStartProtocol"]

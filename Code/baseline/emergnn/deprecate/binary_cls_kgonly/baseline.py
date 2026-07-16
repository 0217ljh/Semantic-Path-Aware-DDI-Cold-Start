"""EmerGNN binary baseline — Method B: kgonly (drop shuffle_train, base_kg only).

This is one of two candidate fixes for the s0/s1 logit-collapse bug observed
on the 2026-05-18 run (see ``baseline/emergnn/_results/2026-05-18__binary_cls__seed42.md``
and the Analysis E section of ``Code/runs/_diagnose/emergnn_s0s1_collapse.json``).

Root cause recap (from full review loop 2026-05-19):
    Paper's ``shuffle_train`` exists to SIMULATE cold-start drug holdout each
    epoch — necessary in upstream DrugBank dataset because their train_ddi.txt
    has no explicit known/unknown drug labels.

    Our PairDataset has explicit ``g1_drugs`` (warm/known) vs ``g2_drugs``
    (cold/unknown) partition at the dataset construction layer. The simulation
    is redundant — the partition is already there. By dropping shuffle_train
    AND removing train_ddi edges from the KG (both training and eval), we get:

      1. A consistent training-eval graph distribution (no shift)
      2. Pure inductive prediction via biomedical-entity propagation only
      3. The exact behavior Analysis E showed lifts s0/s1 AUC from 0.49/0.40
         to 0.71/0.69 (the +0.22/+0.29 pt counterfactual lift)

Concrete deltas vs ``baseline/emergnn/binary_cls/baseline.py``:

  * **No** ``shuffle_train`` call per epoch. ``shuffle_train_mode`` /
    ``shuffle_ratio`` parameters are rejected (paper-irrelevant for our setup).
  * Per-epoch training targets = full ``train.splits.train`` positives (not
    a shuffled 20% subset).
  * Training KG = ``self._kg_triplets`` only (the static base biomedical KG;
    enzymes + targets + transporters + carriers + pathways). NO train_ddi
    edges mixed in.
  * Eval KG (``self._eval_edges``) = same static base biomedical KG. NO
    train_ddi edges mixed in.
  * Negative sampling: unchanged from current binary baseline (per-epoch
    ``train.get_train_negatives(epoch, regenerate=True)`` from G1×G1 pool
    with all 7-split positives excluded).

Companion to the multimode variant at ``baseline/emergnn/binary_cls_multimode/`` —
both are user-approved candidate fixes; the comparative run (seed 42) will
determine which becomes the final EmerGNN binary baseline.

Paper-fidelity note:
    This baseline DEVIATES from upstream EmerGNN paper by removing
    ``shuffle_train``. Justification: the simulation it provides is
    functionally subsumed by our explicit G1/G2 partition (the partition
    IS the cold-start condition; we don't need to simulate it stochastically).
    Other paper-core components are preserved: bidirectional message passing,
    attention-weighted KG paths, Adam + ReduceLROnPlateau, BCE-with-sum
    loss. Entity features default to Morgan FP (``feat='M'``, matching the
    parent EmerGNNBaseline default and upstream S1/S2 dispatch) but a
    caller may override via ``feat='E'`` if they want learned embeddings —
    that path is functionally orthogonal to Method B's KG-mask change.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.base import register
# DEPRECATED variant — kept for reference (Method B candidate that lost the
# 2026-05-20 head-to-head against Method A multimode). Imports the per-mode
# helper from its new internal location after the 2026-05-20 reorg.
from baseline.emergnn._per_mode import _PerModeEmerGNN as EmerGNNBaseline
from baseline.emergnn.model import EmerGNN
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


@register("emergnn-kgonly")
class EmerGNNKGOnlyBaseline(EmerGNNBaseline):
    """EmerGNN binary baseline — kgonly (Method B).

    Inherits from :class:`EmerGNNBaseline` to reuse ``_setup_graph``,
    ``_pair_indices``, ``_edges_on_device``, ``predict_proba``, ``save``,
    ``load``, ``_validate`` and the ``__init__`` parameter surface.

    Only :meth:`fit` is overridden — see file docstring for the 3 concrete
    deltas (no shuffle_train, full-train targets, base_kg-only train+eval graph).
    """

    name = "emergnn-kgonly"
    VERSION = "1.0"

    def __init__(self, **kwargs) -> None:
        # Reject shuffle_train-related kwargs since they are by design ignored
        # in this method. Loudly so callers don't silently expect paper behavior.
        for forbidden in ("shuffle_train_mode", "shuffle_ratio"):
            if forbidden in kwargs and kwargs[forbidden] is not None:
                # Be lenient: if caller passes the EmerGNNBaseline default
                # values, just drop them. If they pass non-default values,
                # raise so they know the method ignores them.
                if forbidden == "shuffle_train_mode" and kwargs[forbidden] == "S2":
                    kwargs.pop(forbidden)
                    continue
                if forbidden == "shuffle_ratio" and kwargs[forbidden] == 0.8:
                    kwargs.pop(forbidden)
                    continue
                raise TypeError(
                    f"EmerGNNKGOnlyBaseline ignores ``{forbidden}`` (Method B "
                    f"drops shuffle_train entirely; see file docstring). "
                    f"Got {kwargs[forbidden]!r}. Use EmerGNNBaseline directly "
                    f"if you want the paper-faithful shuffle_train behavior."
                )
        # Also drop them if they're None to avoid passing extras to parent.
        kwargs.pop("shuffle_train_mode", None)
        kwargs.pop("shuffle_ratio", None)
        super().__init__(**kwargs)
        # Sentinel: ensure the parent's shuffle_train_mode/ratio attributes
        # don't get accidentally read elsewhere. We set them to values that
        # would loudly fail if used (None breaks shuffle_train calls).
        self.shuffle_train_mode = None  # invalid, will raise if anyone calls shuffle_train
        self.shuffle_ratio = None

    # ------------------------------------------------------------------
    # Override fit — drop shuffle_train, use base_kg only as KG
    # ------------------------------------------------------------------

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        if kg is None:
            kg = train.kg
        morgan_mat, _drug_ids = self._setup_graph(train, kg)

        # Paper-faithful relation count: still expand by 1 for the "DDI"
        # relation slot so the model architecture is unchanged from upstream.
        # We just won't actually USE that slot at training time (no train_ddi
        # edges go into the KG). Keeping the slot lets us reuse EmerGNN
        # without re-architecting and lets the save/load format stay
        # interchangeable with the parent baseline.
        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi

        self._model = EmerGNN(
            n_ent=self._n_ent,
            n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim,
            length=self.length,
            feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        # ── Method B core: static KG = base_kg only (no train_ddi) ────────
        # Build edges ONCE (no per-epoch shuffle). Same graph used for both
        # training propagation AND eval predict_proba.
        esrc, edst, erel = build_edge_lists_from_triplets(
            self._kg_triplets, self._n_ent, n_base_rel_with_ddi
        )
        edge_src_static = torch.from_numpy(esrc).long().to(self.device)
        edge_dst_static = torch.from_numpy(edst).long().to(self.device)
        edge_rel_static = torch.from_numpy(erel).long().to(self.device)
        # ``_eval_edges`` is what ``predict_proba`` reads (inherited from
        # parent). Same tensors as the training KG to enforce train-eval
        # graph consistency.
        self._eval_edges = (
            edge_src_static,
            edge_dst_static,
            edge_rel_static,
        )

        # ── Training data ────────────────────────────────────────────────
        # All train positives become targets every epoch (no shuffle holdout).
        pos_df = train.splits.train[["drug_a_id", "drug_b_id"]].copy()
        n_pos = len(pos_df)
        if n_pos == 0:
            raise ValueError("EmerGNNKGOnlyBaseline.fit: train.splits.train is empty.")

        best_val_auc = -1.0
        best_state: dict | None = None

        # Training progress logger (per CLAUDE.md 训练进度/eval/save 规范).
        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints

        n_pos_plus_neg = 2 * n_pos
        steps_per_epoch = (n_pos_plus_neg + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[emergnn-kgonly] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )

        ckpt_root = (self.run_dir / "checkpoints") if self.run_dir is not None else None

        def _eval_dict() -> dict:
            auc = self._validate(val) if val is not None else float("nan")
            return {"val_auc": auc}

        def _save_ckpt(tag: str) -> None:
            if ckpt_root is None:
                return
            ckpt_path = ckpt_root / f"checkpoint-{tag}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.save(ckpt_path)
            progress.log_save(str(ckpt_path), scope="step" if "step" in str(tag) else "epoch")
            rotate_checkpoints(ckpt_root, self.save_total_limit)

        for epoch in range(self.n_epochs):
            # ── Negatives: per-epoch fresh, cross-baseline aligned ─────────
            # Same call signature as the other binary baselines:
            # ``regenerate=True`` forces fresh sample via
            # ``data_utils.negatives.build_train_negatives`` with sub-seed
            # ``base_seed + 1000 + epoch``.
            neg = train.get_train_negatives(epoch, regenerate=True)
            if len(neg) < n_pos:
                # Pad / accept whatever the sampler returned (the sampler
                # already guarantees ``len == len(train) * n_per_pos`` so this
                # should not trip; defensive only).
                pass
            else:
                # Down-sample to exactly n_pos (1:1 pos:neg ratio, matches
                # parent EmerGNN convention).
                neg = neg.iloc[:n_pos].reset_index(drop=True)

            pairs_df = (
                pd.concat(
                    [
                        pos_df.assign(label=1),
                        neg[["drug_a_id", "drug_b_id"]].assign(label=0),
                    ],
                    ignore_index=True,
                )
                .sample(frac=1, random_state=epoch)
                .reset_index(drop=True)
            )

            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head = head.to(self.device)
                tail = tail.to(self.device)
                y = torch.tensor(
                    batch["label"].to_numpy(),
                    dtype=torch.float32,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model(
                    head,
                    tail,
                    edge_src_static,
                    edge_dst_static,
                    edge_rel_static,
                )
                loss = binary_cross_entropy_with_logits(logits, y, reduction="sum")
                loss.backward()
                opt.step()
                progress.step(loss.item() / max(len(batch), 1))

                if progress.should_eval_step() and val is not None:
                    self._model.eval()
                    metrics = _eval_dict()
                    self._model.train()
                    progress.log_eval(metrics, scope="step")
                    if metrics.get("val_auc", -1) > best_val_auc:
                        best_val_auc = metrics["val_auc"]
                        best_state = copy.deepcopy(self._model.state_dict())
                if progress.should_save_step():
                    _save_ckpt(f"step{progress.step_count}")

            extra: dict = {}
            if progress.should_eval_epoch() and val is not None:
                self._model.eval()
                metrics = _eval_dict()
                self._model.train()
                progress.log_eval(metrics, scope="epoch")
                extra.update(metrics)
                if metrics.get("val_auc", -1) > best_val_auc:
                    best_val_auc = metrics["val_auc"]
                    best_state = copy.deepcopy(self._model.state_dict())
                scheduler.step(metrics.get("val_auc", 0.0))
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}")
            progress.epoch_end(extra=extra if extra else None)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[emergnn-kgonly] loaded best val_auc={best_val_auc:.4f} state",
                flush=True,
            )

"""TWOSIDES multilabel core, rspmm backend (torchdrug kernel).

rspmm twin of :class:`baseline.emergnn.multi_label_cls._core_twoside.BaseModelTwoside`.
Keeps ALL training behavior identical — same cumulative train/valid/test KG
(load_twoside_kg), same relation-slot scheme + edge build (``_build_edges``), same
per-epoch ``_shuffle_train``, paired pos/neg BCE-on-active-labels, best-checkpoint
on valid PR-AUC, Adam + ReduceLROnPlateau — and swaps ONLY:
  * the model: :class:`baseline.emergnn.multi_label_cls.model_rspmm.EmerGNN_ML_RSPMM`;
  * the KG representation: coalesced sparse COO built via
    :func:`baseline.emergnn._rspmm_utils.build_sparse_kg` from the SAME
    ``_build_edges`` output (which produces the original TWOSIDES ``(first, second,
    rel)`` layout, so generalized_rspmm reproduces the paper flow exactly).

vKG + tKG are built resident once in ``fit`` (the TWOSIDES KG is ~3.4M edges; a
coalesced sparse COO is ~0.1 GB, so three resident graphs are negligible on the
5090); the per-epoch training graph is rebuilt each epoch. Selected via
``EMERGNN_BACKEND=rspmm``.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.emergnn._rspmm_utils import build_sparse_kg
from baseline.emergnn.multi_label_cls._core_twoside import (
    BaseModelTwoside,
    _build_edges,
    _shuffle_train,
    load_twoside_kg,
)
from baseline.emergnn.multi_label_cls.model_rspmm import EmerGNN_ML_RSPMM


class BaseModelTwoside_RSPMM(BaseModelTwoside):
    """rspmm-backend twin of BaseModelTwoside (fused kernel + sparse KG)."""

    def _build_kg_sparse(
        self,
        fact_triplets: np.ndarray,
        kg_triplets: np.ndarray,
        all_ent: int,
        all_rel: int,
        eval_rel: int,
    ) -> torch.Tensor:
        """(_build_edges -> coalesced sparse COO on device). Same layout as the
        original TWOSIDES load_graph, so generalized_rspmm reproduces its flow."""
        esrc, edst, erel = _build_edges(fact_triplets, kg_triplets, all_ent, all_rel, eval_rel)
        all_rel_slots = 2 * all_rel - eval_rel + 1
        row = torch.from_numpy(esrc).long().to(self.device)
        col = torch.from_numpy(edst).long().to(self.device)
        rel = torch.from_numpy(erel).long().to(self.device)
        return build_sparse_kg(row, col, rel, all_ent, all_rel_slots)

    def fit(self, bundle: dict) -> None:
        kg = load_twoside_kg(self.kg_source, self.split_code, self.fold)
        all_ent = kg["all_ent"]
        all_rel = kg["all_rel"]
        eval_rel = self.n_labels
        train_kg = kg["train_kg"]
        valid_kg = kg["valid_kg"]
        test_kg = kg["test_kg"]
        kg_entity_set = kg["kg_entity_set"]
        morgan = kg["morgan"]
        self._all_ent, self._all_rel = all_ent, all_rel

        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=np.int64)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=np.int64)
        train_neg_y = np.asarray(bundle["train_neg_y"], dtype=np.float32)
        train_ent = set(train_pos_ht.reshape(-1).tolist())
        ddi_in_kg = train_ent & kg_entity_set
        if not ddi_in_kg:
            ddi_in_kg = train_ent

        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=np.int64)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=np.int64)
        val_neg_y = np.asarray(bundle["val_neg_y"], dtype=np.float32)

        self._model = EmerGNN_ML_RSPMM(
            n_ent=all_ent, all_rel=all_rel, eval_rel=eval_rel,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan if self.feat == "M" else None,
        ).to(self.device)

        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode="max")

        def _facts_from(pos_ht, pos_y):
            f = []
            for i in range(len(pos_ht)):
                h, t = int(pos_ht[i, 0]), int(pos_ht[i, 1])
                for s in np.nonzero(pos_y[i])[0]:
                    f.append((h, t, int(s)))
            return np.asarray(f, dtype=np.int64) if f else np.zeros((0, 3), np.int64)

        # vKG (validation): train DDI facts + valid_kg; tKG (test/predict):
        # train+valid DDI facts + test_kg. Built resident once (sparse COO).
        train_facts = _facts_from(train_pos_ht, train_pos_y)
        self._valid_edges = self._build_kg_sparse(train_facts, valid_kg, all_ent, all_rel, eval_rel)
        val_facts = _facts_from(val_pos_ht, val_pos_y)
        tv_facts = (np.concatenate([train_facts, val_facts], axis=0)
                    if len(val_facts) else train_facts)
        self._eval_edges = self._build_kg_sparse(tv_facts, test_kg, all_ent, all_rel, eval_rel)
        # Diagnostic edge counts — parity with the parent core (smoke scripts read
        # ``core._edge_counts``). train_graph_edges = full-train-fact graph (representative,
        # since the per-epoch graph size varies with shuffle_train).
        _fsrc, _, _ = _build_edges(train_facts, train_kg, all_ent, all_rel, eval_rel)
        self._edge_counts = {
            "train_graph_edges": int(len(_fsrc)),
            "valid_graph_edges": int(self._valid_edges._nnz()),
            "test_graph_edges": int(self._eval_edges._nnz()),
        }
        print(f"[emergnn_ml-rspmm] graph edges train={self._edge_counts['train_graph_edges']} "
              f"valid_nnz={self._edge_counts['valid_graph_edges']} "
              f"test_nnz={self._edge_counts['test_graph_edges']}", flush=True)

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress

        rng = np.random.default_rng(0)
        best_val_pr = -1.0
        best_state: dict | None = None
        steps_per_epoch = max(
            1, (len(train_pos_ht) + self.batch_size - 1) // self.batch_size
        )
        progress = TrainProgress(
            total_epochs=self.n_epochs, log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch, prefix="[emergnn_ml-rspmm] ",
            eval_strategy=self.eval_strategy,
        )

        for epoch in range(self.n_epochs):
            fact_triplets, pos_ht, pos_y, neg_ht, neg_y = _shuffle_train(
                train_pos_ht, train_pos_y, train_neg_ht, train_neg_y,
                train_ent, ddi_in_kg, self.split_code, rng, ratio=self.shuffle_ratio,
            )
            if len(pos_ht) == 0:
                print(f"[emergnn_ml-rspmm] epoch {epoch+1}: 0 targets after shuffle_train; skip",
                      flush=True)
                continue
            epoch_kg = self._build_kg_sparse(fact_triplets, train_kg, all_ent, all_rel, eval_rel)

            self._model.train()
            progress.epoch_start(epoch)
            N = len(pos_ht)
            for start in range(0, N, self.batch_size):
                end = min(N, start + self.batch_size)
                p_h = torch.from_numpy(pos_ht[start:end, 0]).long().to(self.device)
                p_t = torch.from_numpy(pos_ht[start:end, 1]).long().to(self.device)
                p_r = torch.from_numpy(pos_y[start:end]).float().to(self.device)
                n_h = torch.from_numpy(neg_ht[start:end, 0]).long().to(self.device)
                n_t = torch.from_numpy(neg_ht[start:end, 1]).long().to(self.device)
                n_r = torch.from_numpy(neg_y[start:end]).float().to(self.device)

                opt.zero_grad(set_to_none=True)
                p_scores = torch.sigmoid(self._model(p_h, p_t, epoch_kg))
                n_scores = torch.sigmoid(self._model(n_h, n_t, epoch_kg))
                p_sel = p_scores[p_r > 0]
                n_sel = n_scores[n_r > 0]
                scores = torch.cat([p_sel, n_sel], dim=0)
                labels = torch.cat(
                    [torch.ones(len(p_sel), device=self.device),
                     torch.zeros(len(n_sel), device=self.device)], dim=0,
                )
                if len(scores) == 0:
                    continue
                loss = torch.nn.functional.binary_cross_entropy(scores, labels)
                loss.backward()
                opt.step()
                progress.step(loss.item())

            extra: dict = {}
            if progress.should_eval_epoch() and len(val_pos_ht):
                self._model.eval()
                roc, pr = self._eval_paired(val_pos_ht, val_pos_y, val_neg_ht,
                                            val_neg_y, edges=self._valid_edges)
                self._model.train()
                metrics = {"val_roc": roc, "val_pr": pr}
                progress.log_eval(metrics, scope="epoch")
                extra.update(metrics)
                if pr > best_val_pr:
                    best_val_pr = pr
                    best_state = copy.deepcopy(self._model.state_dict())
                scheduler.step(pr)
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[emergnn_ml-rspmm] loaded best val_pr={best_val_pr:.4f} state", flush=True)

    @torch.no_grad()
    def _score_pairs(self, ht: np.ndarray, edges=None) -> np.ndarray:
        """(n,2) endpoints -> (n, eval_rel) sigmoid probs. ``edges`` is a sparse KG
        (vKG for validation, tKG default for test/predict)."""
        kg = edges if edges is not None else self._eval_edges
        self._model.eval()
        out = np.empty((len(ht), self.n_labels), dtype=np.float32)
        bs = self.test_batch_size
        for start in range(0, len(ht), bs):
            end = min(len(ht), start + bs)
            h = torch.from_numpy(ht[start:end, 0]).long().to(self.device)
            t = torch.from_numpy(ht[start:end, 1]).long().to(self.device)
            logits = self._model(h, t, kg)
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
        return out


__all__ = ["BaseModelTwoside_RSPMM"]

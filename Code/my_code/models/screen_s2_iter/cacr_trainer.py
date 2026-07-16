"""Cold-Aware Consistency Regularization (CACR) trainer.

Insight: EmerGNN flow sees rich KG context during training (warm pairs
have many neighbors) but cold drugs at test time have sparse context
(no DDI edges, often few/no protein/disease links). This train-test
mismatch is the main reason cold-start AUC plateaus around 0.74.

CACR fix: at each training step, compute the score TWICE on the same
batch:
  1. Standard forward on full KG -> `score_full`
  2. Aux forward on a "cold-simulated" KG where edges incident to s OR t
     are randomly dropped at rate p (=0.5 by default) -> `score_cold`

Loss = BCE(score_full, y) + lambda * MSE(score_full.detach(), score_cold)

The detach prevents the consistency term from collapsing both branches
to a trivial constant. The model is encouraged to give STABLE predictions
under cold simulation, which is exactly the test-time condition.

This is a TRAINING-TIME modification ONLY. EmerGNN backbone unchanged.
Reuses _PerModeEmerGNN.fit() structure but overrides the inner loss.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from baseline.emergnn._per_mode import _PerModeEmerGNN
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train
from baseline.emergnn.model import EmerGNN


class _PerModeEmerGNN_CACR(_PerModeEmerGNN):
    """EmerGNN with Cold-Aware Consistency Regularization in S2 training.

    Args extending parent:
      cacr_lambda : weight on consistency loss (default 0.5)
      cacr_drop_rate : per-batch drop probability for edges incident to s/t (default 0.5)
      cacr_warmup_epochs : epochs before CACR kicks in (default 5)
    """

    def __init__(self, *, cacr_lambda: float = 0.5,
                 cacr_drop_rate: float = 0.75,
                 cacr_warmup_epochs: int = 5,
                 cacr_drop_relations: str = "ddi_only",
                 cacr_nondrug_drop_rate: float = 0.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cacr_lambda = float(cacr_lambda)
        self.cacr_drop_rate = float(cacr_drop_rate)
        self.cacr_warmup_epochs = int(cacr_warmup_epochs)
        if cacr_drop_relations not in ("ddi_only", "all"):
            raise ValueError(f"unknown cacr_drop_relations={cacr_drop_relations!r}")
        self.cacr_drop_relations = cacr_drop_relations
        self.cacr_nondrug_drop_rate = float(cacr_nondrug_drop_rate)

    def _build_cold_edge_mask(
        self,
        head_idx: torch.Tensor,       # (B,)
        tail_idx: torch.Tensor,       # (B,)
        edge_src: torch.Tensor,       # (E,)
        edge_dst: torch.Tensor,       # (E,)
        edge_rel: torch.Tensor,       # (E,)
        drop_rate: float,
    ) -> torch.Tensor:
        """Return bool keep-mask over edges (E,) for one batch.

        Per Codex round 8: prefer DDI-edge endpoint dropout to simulate
        "cold drugs have no observed DDI edges" while preserving
        target/protein/biomedical evidence.

        DDI relation indices: forward = self._n_base_rel (dummy slot added
        by shuffle_train), reverse = self._n_base_rel + self._n_base_rel_with_ddi.
        """
        endpoint_set = torch.unique(torch.cat([head_idx, tail_idx]))
        is_endpoint_edge_src = torch.isin(edge_src, endpoint_set)
        is_endpoint_edge_dst = torch.isin(edge_dst, endpoint_set)
        is_endpoint_edge = is_endpoint_edge_src | is_endpoint_edge_dst

        n_kg_rel = self._n_base_rel
        ddi_rel_fwd = n_kg_rel
        ddi_rel_rev = n_kg_rel + self._n_base_rel_with_ddi
        is_ddi_edge = (edge_rel == ddi_rel_fwd) | (edge_rel == ddi_rel_rev)

        if self.cacr_drop_relations == "ddi_only":
            # Drop only DDI edges incident to batch endpoints
            drop_target = is_endpoint_edge & is_ddi_edge
            rand = torch.rand(edge_src.size(0), device=edge_src.device)
            drop = (rand < drop_rate) & drop_target
            # Optionally light dropout on non-DDI endpoint edges
            if self.cacr_nondrug_drop_rate > 0:
                nondrug_target = is_endpoint_edge & ~is_ddi_edge
                rand2 = torch.rand(edge_src.size(0), device=edge_src.device)
                drop = drop | ((rand2 < self.cacr_nondrug_drop_rate) & nondrug_target)
        else:  # 'all'
            rand = torch.rand(edge_src.size(0), device=edge_src.device)
            drop = (rand < drop_rate) & is_endpoint_edge
        keep = ~drop
        return keep

    def fit(self, train, val=None, *, kg=None):
        """Custom fit with CACR auxiliary loss in the inner batch loop.

        Mirrors `_PerModeEmerGNN.fit()` structure but adds aux forward.
        """
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi

        # Build model
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
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        # Train_ddi triplets
        pos_df = train.splits.train.copy()
        a_ids = pos_df["drug_a_id"].astype(str).map(self._entity2id)
        b_ids = pos_df["drug_b_id"].astype(str).map(self._entity2id)
        valid_mask = a_ids.notna() & b_ids.notna()
        train_ddi_int = np.stack([
            a_ids[valid_mask].astype(np.int64).to_numpy(),
            b_ids[valid_mask].astype(np.int64).to_numpy(),
            np.full(int(valid_mask.sum()), n_kg_rel, dtype=np.int64),
        ], axis=1)

        # Static eval KG = train_ddi + base_kg
        eval_kg_triplets = np.concatenate([train_ddi_int, self._kg_triplets], axis=0)
        esrc, edst, erel = build_edge_lists_from_triplets(
            eval_kg_triplets, self._n_ent, n_base_rel_with_ddi)
        self._eval_edges = (
            torch.from_numpy(esrc).long().to(self.device),
            torch.from_numpy(edst).long().to(self.device),
            torch.from_numpy(erel).long().to(self.device),
        )

        rng = np.random.default_rng(0)
        best_val_auc = -1.0
        best_state: dict | None = None

        for epoch in range(self.n_epochs):
            epoch_kg, train_pos_targets = shuffle_train(
                train_ddi_int, self._kg_triplets,
                self.shuffle_train_mode, ratio=self.shuffle_ratio, rng=rng,
                extra_kg_ent=self._kg_entity_set,
            )
            if len(train_pos_targets) == 0:
                continue
            esrc, edst, erel = build_edge_lists_from_triplets(
                epoch_kg, self._n_ent, n_base_rel_with_ddi)
            edge_src = torch.from_numpy(esrc).long().to(self.device)
            edge_dst = torch.from_numpy(edst).long().to(self.device)
            edge_rel = torch.from_numpy(erel).long().to(self.device)

            pre_neg = train.get_train_negatives(epoch, regenerate=True)
            n_target_pos = len(train_pos_targets)
            if len(pre_neg) >= n_target_pos:
                neg_idx = rng.choice(len(pre_neg), size=n_target_pos, replace=False)
                neg = pre_neg.iloc[neg_idx].reset_index(drop=True)
            else:
                neg = pre_neg

            id2entity = {v: k for k, v in self._entity2id.items()}
            pos_target_df = pd.DataFrame({
                "drug_a_id": [id2entity[int(h)] for h in train_pos_targets[:, 0]],
                "drug_b_id": [id2entity[int(t)] for t in train_pos_targets[:, 1]],
            })

            pairs_df = pd.concat([
                pos_target_df.assign(label=1),
                neg[["drug_a_id", "drug_b_id"]].assign(label=0),
            ], ignore_index=True).sample(frac=1, random_state=epoch).reset_index(drop=True)

            self._model.train()
            t_epoch = time.time()
            losses, cacr_losses = [], []
            cacr_active = (epoch >= self.cacr_warmup_epochs)

            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(),
                                 dtype=torch.float32, device=self.device)

                opt.zero_grad(set_to_none=True)

                # Standard forward on full KG
                logits_full = self._model(head_d, tail_d, edge_src, edge_dst, edge_rel)
                loss_bce = F.binary_cross_entropy_with_logits(logits_full, y, reduction="sum")

                # CACR aux: cold-simulated forward
                # Per codex round-10 fix: use soft-label BCE on logits (more stable
                # gradient than sigmoid-MSE which saturates). Teacher = sigmoid(logits_full)
                # detached, student logits_cold pushed toward teacher distribution.
                if cacr_active:
                    keep_mask = self._build_cold_edge_mask(
                        head_d, tail_d, edge_src, edge_dst, edge_rel,
                        self.cacr_drop_rate)
                    cold_src = edge_src[keep_mask]
                    cold_dst = edge_dst[keep_mask]
                    cold_rel = edge_rel[keep_mask]
                    logits_cold = self._model(head_d, tail_d, cold_src, cold_dst, cold_rel)
                    soft_target = torch.sigmoid(logits_full.detach())
                    loss_cacr_mean = F.binary_cross_entropy_with_logits(
                        logits_cold, soft_target, reduction="mean")
                    # BCE is sum-reduced; cacr mean -> scale to per-pair sum
                    loss = loss_bce + self.cacr_lambda * loss_cacr_mean * len(batch)
                    cacr_losses.append(loss_cacr_mean.item())
                else:
                    loss = loss_bce

                loss.backward()
                # Gradient clip safety against any residual explosion
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=10.0)
                opt.step()
                losses.append(loss_bce.item() / max(len(batch), 1))

            if val is not None:
                self._model.eval()
                v_auc = self._validate(val)
                self._model.train()
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    best_state = copy.deepcopy(self._model.state_dict())
                scheduler.step(v_auc)
                ep_time = time.time() - t_epoch
                cacr_str = f" cacr={np.mean(cacr_losses):.4f}" if cacr_losses else ""
                print(f"[cacr] [ep {epoch+1}/{self.n_epochs}] "
                      f"mean_loss={np.mean(losses):.4f}{cacr_str} "
                      f"time={ep_time:.0f}s val_auc={v_auc:.4f}", flush=True)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[cacr] loaded best val_auc={best_val_auc:.4f}")


__all__ = ["_PerModeEmerGNN_CACR"]

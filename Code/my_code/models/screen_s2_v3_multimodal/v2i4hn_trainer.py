"""E-i4-HN — MNAH + I4 mechanistic pair features + HARD-NEGATIVE MINING (codex 019e76f6).

I4 alone gave +0.84pt (real but small, clean shuffle separation). codex's diagnosis: the
cold-start ceiling is set not by missing features but by insufficient DISCRIMINATION among
mechanistically-plausible pairs. I4 tells the model WHY a pair might interact; hard negatives
force the model to learn WHEN that reason is decisive.

Hard negatives = train-negative pairs with HIGH I4 mechanistic overlap (the "would-look-like-
DDI under simple rules" negatives). Each epoch the random negative pool is reranked by the
I4 hardness score, and the bottom-fraction is replaced with the hardest. A random tail is
retained to prevent all-hard collapse.

This is the SINGLE highest-EV next experiment per codex (over I4-tuning or jumping to
backbone). Combines I4's representational gain with a sharper training signal.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v2i4_trainer import (
    _PerModeEmerGNN_V2I4, _pair_feat, PAIR_FEAT_NAMES,
)
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train
from baseline.emergnn.model import EmerGNN


class _PerModeEmerGNN_V2I4HN(_PerModeEmerGNN_V2I4):
    """v2i4 + hard-negative mining via I4 mechanistic overlap reranking."""

    def __init__(self, *, hn_frac: float = 0.5, hn_pool_mult: float = 1.0,
                 hn_warmup_epochs: int = 5, hn_seed: int = 2026, **kwargs) -> None:
        """hn_frac = fraction of negatives replaced with HARD ones each epoch (0..1).
        hn_warmup_epochs = epochs to wait before turning HN on (so model finds a baseline first).
        """
        super().__init__(**kwargs)
        self.hn_frac = float(hn_frac)
        self.hn_pool_mult = float(hn_pool_mult)
        self.hn_warmup_epochs = int(hn_warmup_epochs)
        self.hn_seed = int(hn_seed)

    def _hardness(self, pairs: pd.DataFrame) -> np.ndarray:
        """Per-pair I4 mechanistic hardness: sum of overlap features (drop jaccard)."""
        self._load_i4_assets()
        a = pairs["drug_a_id"].astype(str).tolist()
        b = pairs["drug_b_id"].astype(str).tolist()
        feats = np.stack([_pair_feat(self._i4_sets.get(ai), self._i4_sets.get(bi))
                          for ai, bi in zip(a, b)], axis=0)
        # PAIR_FEAT_NAMES last entry is jaccard_overall - drop for hardness
        return feats[:, :12].sum(axis=1)

    def _harden(self, neg_pool: pd.DataFrame, n_target: int, rng) -> pd.DataFrame:
        """Replace bottom-fraction of n_target with the HARDEST candidates from neg_pool."""
        if len(neg_pool) == 0 or n_target <= 0:
            return neg_pool.iloc[:0]
        scores = self._hardness(neg_pool)
        order_hard = np.argsort(-scores)  # descending hardness
        n_hard = int(round(n_target * self.hn_frac))
        n_rand = n_target - n_hard
        hard_idx = order_hard[:n_hard]
        # random tail: from the remaining pool
        rest_pool = np.setdiff1d(np.arange(len(neg_pool)), hard_idx, assume_unique=True)
        if len(rest_pool) >= n_rand:
            rand_idx = rng.choice(rest_pool, size=n_rand, replace=False)
        else:
            rand_idx = rest_pool
        sel = np.concatenate([hard_idx, rand_idx])
        rng.shuffle(sel)
        return neg_pool.iloc[sel].reset_index(drop=True)

    def fit(self, train, val=None, *, kg=None):
        """Copy of MNAH.fit + v2i4 pair-norm fit + HN reranking of negatives per epoch."""
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)
        try:
            train_neg_for_norm = train.get_train_negatives(0, regenerate=True)[
                ["drug_a_id", "drug_b_id"]]
        except Exception as exc:
            print(f"[v2i4hn] norm-neg load failed: {exc}", flush=True)
            train_neg_for_norm = None
        self._load_feature_cache(train.splits.train, train_neg_for_norm)
        # I4 pair-feat normalizer on TRAIN POS + TRAIN NEG (codex 019e769b)
        self._fit_pair_norm(train.splits.train[["drug_a_id", "drug_b_id"]], train_neg_for_norm)

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi
        self._model = EmerGNN(
            n_ent=self._n_ent, n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)

        self._aux_mlp = self._build_aux_head(self._aux_in_dim).to(self.device)
        if self.mnah_init_beta > 0:
            raw_init = float(np.log(np.exp(self.mnah_init_beta) - 1.0))
        else:
            raw_init = -5.0
        self._raw_beta = nn.Parameter(torch.tensor(raw_init, device=self.device))
        print(f"[v2i4hn] aux head built; HN frac={self.hn_frac} warmup={self.hn_warmup_epochs}",
              flush=True)

        params = list(self._model.parameters()) + list(self._aux_mlp.parameters()) + [self._raw_beta]
        opt = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        pos_df = train.splits.train.copy()
        a_ids = pos_df["drug_a_id"].astype(str).map(self._entity2id)
        b_ids = pos_df["drug_b_id"].astype(str).map(self._entity2id)
        valid_mask = a_ids.notna() & b_ids.notna()
        train_ddi_int = np.stack([
            a_ids[valid_mask].astype(np.int64).to_numpy(),
            b_ids[valid_mask].astype(np.int64).to_numpy(),
            np.full(int(valid_mask.sum()), n_kg_rel, dtype=np.int64),
        ], axis=1)
        eval_kg_triplets = np.concatenate([train_ddi_int, self._kg_triplets], axis=0)
        esrc, edst, erel = build_edge_lists_from_triplets(
            eval_kg_triplets, self._n_ent, n_base_rel_with_ddi)
        self._eval_edges = (
            torch.from_numpy(esrc).long().to(self.device),
            torch.from_numpy(edst).long().to(self.device),
            torch.from_numpy(erel).long().to(self.device),
        )

        rng = np.random.default_rng(self.hn_seed)
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
            # HN mining (after warmup): rerank pre_neg by I4 hardness, replace bottom-frac
            if epoch >= self.hn_warmup_epochs and self.hn_frac > 0:
                neg = self._harden(pre_neg, n_target_pos, rng)
            else:
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

            self._model.train(); self._aux_mlp.train()
            t_epoch = time.time()
            losses = []
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                combined, _, _ = self._combined_logit(head_d, tail_d, edge_src, edge_dst, edge_rel, batch)
                loss = F.binary_cross_entropy_with_logits(combined, y, reduction="sum")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                opt.step()
                losses.append(loss.item() / max(len(batch), 1))

            if val is not None:
                self._model.eval(); self._aux_mlp.eval()
                br = self._validate_branches(val)
                v_auc = br["combined"]
                self._model.train(); self._aux_mlp.train()
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    best_state = {
                        "emergnn": copy.deepcopy(self._model.state_dict()),
                        "aux": copy.deepcopy(self._aux_mlp.state_dict()),
                        "raw_beta": float(self._raw_beta.detach().cpu().item()),
                    }
                scheduler.step(v_auc)
                ep_time = time.time() - t_epoch
                hn_on = "HN" if (epoch >= self.hn_warmup_epochs and self.hn_frac > 0) else "rand"
                print(f"[v2i4hn] [ep {epoch+1}/{self.n_epochs}] loss={np.mean(losses):.4f} "
                      f"time={ep_time:.0f}s val={v_auc:.4f} val_e={br['emergnn']:.4f} "
                      f"val_a={br['aux']:.4f} {hn_on}", flush=True)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[v2i4hn] loaded best val_combined={best_val_auc:.4f}", flush=True)


__all__ = ["_PerModeEmerGNN_V2I4HN"]

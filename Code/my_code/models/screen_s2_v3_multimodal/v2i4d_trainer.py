"""Phase D — MNAH + I4 + JOINT binary + 215-class ddi_type auxiliary task (codex 019e7734).

The user's framing: binary DDI and multi-class ddi_type should be MUTUALLY REINFORCING in
cold-start. The type supervision regularizes the shared representation toward mechanism;
the binary task supplies the bulk of pair samples. I4 (validated mechanistic feature lever
in E-i4, +0.84pt) feeds BOTH heads.

Architecture:
    shared inputs = [22 normalized counts, 13 I4 pair features]
    Binary path  (UNCHANGED v2i4): emergnn + beta*count_logit + beta_i4*i4_logit
    Type head    (NEW): MLP(35 -> 64 -> 215), CE on POSITIVE pairs only
Loss:
    L = L_binary + lambda_type * L_type_positives    (lambda_type=0.3 codex default)
Inference (binary AUROC): unchanged binary head. Type macro-F1 reported separately.

When lambda_type=0, this reduces to v2i4 exactly (the 0.7804 backbone is preserved).
"""
from __future__ import annotations

import copy
import json
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
from sklearn.metrics import f1_score, roc_auc_score

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v2i4_trainer import (
    _PerModeEmerGNN_V2I4, _pair_feat, PAIR_FEAT_NAMES,
)
from my_code.models.screen_s2_v2_meetnode.mnah_trainer import AuxMLP
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train
from baseline.emergnn.model import EmerGNN

DDI_TYPE_MAP = PROJECT_ROOT / "Code/data/_cache/ddi_type_map.json"


class CountPlusI4PlusTypeHead(nn.Module):
    """Binary path (count+i4) + multi-class type head over shared inputs.
    When lambda_type=0 outside, type head's params still exist but contribute nothing -> the
    binary path is bit-equivalent to v2i4's CountPlusI4Head.
    """
    def __init__(self, *, count_in: int = 22, count_hidden: int = 32, dropout: float = 0.2,
                 i4_hidden: int = 32, beta_i4_init: float = 0.5,
                 n_pair: int = 13, n_types: int = 215, type_hidden: int = 64):
        super().__init__()
        self.count_mlp = AuxMLP(in_dim=count_in, hidden=count_hidden, dropout=dropout)
        self.i4_mlp = nn.Sequential(nn.Linear(n_pair, i4_hidden), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(i4_hidden, 1))
        self.raw_beta_i4 = nn.Parameter(torch.tensor(float(np.log(np.exp(beta_i4_init) - 1.0))))
        # type head over shared input [count_feats, pair_feat]
        self.type_mlp = nn.Sequential(
            nn.Linear(count_in + n_pair, type_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(type_hidden, n_types),
        )

    def count_logit(self, feats22):
        return self.count_mlp(feats22)

    def i4_logit(self, pair_feat):
        return self.i4_mlp(pair_feat).squeeze(-1)

    def type_logits(self, feats22, pair_feat):
        return self.type_mlp(torch.cat([feats22, pair_feat], dim=1))  # (B, n_types)

    def beta_i4(self):
        return F.softplus(self.raw_beta_i4)


class _PerModeEmerGNN_V2I4D(_PerModeEmerGNN_V2I4):
    def __init__(self, *, lambda_type: float = 0.3, type_hidden: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lambda_type = float(lambda_type)
        self.type_hidden = int(type_hidden)
        self._type_map = None
        self._n_types = 215

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        return CountPlusI4PlusTypeHead(
            count_in=22, count_hidden=self.mnah_hidden, dropout=self.mnah_dropout,
            i4_hidden=self.v2i4_hidden, beta_i4_init=self.v2i4_beta_init,
            n_pair=len(PAIR_FEAT_NAMES), n_types=self._n_types, type_hidden=self.type_hidden)

    def _load_type_map(self) -> None:
        if self._type_map is not None:
            return
        m = json.loads(DDI_TYPE_MAP.read_text(encoding="utf-8"))
        self._type_map = m["pair_to_idx"]  # "DBxxx|DByyy" -> int
        self._n_types = len(m["idx_to_type"])
        print(f"[v2i4d] ddi_type map loaded: {len(self._type_map)} pairs, "
              f"{self._n_types} types", flush=True)

    @staticmethod
    def _canon_key(a: str, b: str) -> str:
        return f"{a}|{b}" if a <= b else f"{b}|{a}"

    def _type_labels(self, batch_df: pd.DataFrame, is_pos_mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mask, labels) for positive rows that have a known ddi_type.
        mask: (B,) bool; labels: int64 indices into [0, n_types) for masked rows only."""
        self._load_type_map()
        keys = [self._canon_key(str(a), str(b))
                for a, b in zip(batch_df["drug_a_id"], batch_df["drug_b_id"])]
        labels = []
        mask = np.zeros(len(batch_df), dtype=bool)
        for i, (k, pos) in enumerate(zip(keys, is_pos_mask)):
            if pos and (k in self._type_map):
                labels.append(self._type_map[k])
                mask[i] = True
        mask_t = torch.tensor(mask, device=self.device)
        lab_t = torch.tensor(labels, dtype=torch.long, device=self.device) if labels \
                else torch.zeros(0, dtype=torch.long, device=self.device)
        return mask_t, lab_t

    def _combined_logit(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)
        feats = self._lookup_features(batch_df)
        h: CountPlusI4PlusTypeHead = self._aux_mlp
        count_logit = h.count_logit(feats)
        pair_feat = self._build_pair_feats(batch_df)
        i4_logit = h.i4_logit(pair_feat)
        combined = emergnn_logit + self._beta() * count_logit + h.beta_i4() * i4_logit
        aux = self._beta() * count_logit + h.beta_i4() * i4_logit
        # also stash for type-loss
        self._last_feats22 = feats
        self._last_pair_feat = pair_feat
        return combined, emergnn_logit, aux

    def fit(self, train, val=None, *, kg=None):
        """Copy of MNAH fit + v2i4 pair-norm fit + joint binary+type loss (Phase D)."""
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)
        try:
            train_neg_for_norm = train.get_train_negatives(0, regenerate=True)[
                ["drug_a_id", "drug_b_id"]]
        except Exception as exc:
            print(f"[v2i4d] norm-neg load failed: {exc}", flush=True)
            train_neg_for_norm = None
        self._load_feature_cache(train.splits.train, train_neg_for_norm)
        self._fit_pair_norm(train.splits.train[["drug_a_id", "drug_b_id"]], train_neg_for_norm)
        self._load_type_map()

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
        print(f"[v2i4d] aux head built; lambda_type={self.lambda_type}", flush=True)

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

            self._model.train(); self._aux_mlp.train()
            t_epoch = time.time()
            losses_bin = []
            losses_type = []
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y_np = batch["label"].to_numpy()
                y = torch.tensor(y_np, dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                combined, _, _ = self._combined_logit(head_d, tail_d, edge_src, edge_dst, edge_rel, batch)
                loss_bin = F.binary_cross_entropy_with_logits(combined, y, reduction="sum")
                # type loss on POSITIVE pairs with known ddi_type
                loss_type = combined.new_zeros(())
                if self.lambda_type > 0:
                    type_mask, type_lab = self._type_labels(batch, y_np.astype(bool))
                    if int(type_mask.sum()) > 0:
                        h: CountPlusI4PlusTypeHead = self._aux_mlp
                        type_logits_all = h.type_logits(self._last_feats22, self._last_pair_feat)
                        loss_type = F.cross_entropy(type_logits_all[type_mask], type_lab,
                                                    reduction="sum")
                loss = loss_bin + self.lambda_type * loss_type
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                opt.step()
                losses_bin.append(loss_bin.item() / max(len(batch), 1))
                losses_type.append(loss_type.item() / max(int(type_mask.sum()) if self.lambda_type > 0 else 1, 1))

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
                print(f"[v2i4d] [ep {epoch+1}/{self.n_epochs}] L_bin={np.mean(losses_bin):.4f} "
                      f"L_type={np.mean(losses_type):.3f} time={ep_time:.0f}s "
                      f"val={v_auc:.4f} val_e={br['emergnn']:.4f}", flush=True)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[v2i4d] loaded best val_combined={best_val_auc:.4f}", flush=True)

    @torch.no_grad()
    def eval_type_macro_f1(self, ds) -> dict:
        """Multi-class macro-F1 on test_s2 POSITIVES (Phase D narrative metric)."""
        self._load_type_map()
        pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].copy()
        keys = [self._canon_key(str(a), str(b))
                for a, b in zip(pos["drug_a_id"], pos["drug_b_id"])]
        y_true, idx_kept = [], []
        for i, k in enumerate(keys):
            if k in self._type_map:
                y_true.append(self._type_map[k]); idx_kept.append(i)
        if not idx_kept:
            return {"macro_f1": float("nan"), "n_eval": 0}
        pos_k = pos.iloc[idx_kept].reset_index(drop=True)
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            es, ed, er = self._eval_edges
        else:
            es, ed, er = self._edges_on_device()
        self._model.eval(); self._aux_mlp.eval()
        h: CountPlusI4PlusTypeHead = self._aux_mlp
        preds = []
        for s in range(0, len(pos_k), self.batch_size):
            b = pos_k.iloc[s:s + self.batch_size]
            feats = self._lookup_features(b)
            pf = self._build_pair_feats(b)
            tl = h.type_logits(feats, pf)
            preds.extend(tl.argmax(dim=1).cpu().tolist())
        return {"macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(y_true, preds, average="micro", zero_division=0)),
                "n_eval": int(len(y_true)), "n_types": int(self._n_types)}


__all__ = ["_PerModeEmerGNN_V2I4D", "CountPlusI4PlusTypeHead"]

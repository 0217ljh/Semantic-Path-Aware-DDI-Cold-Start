"""Per-mode trainer for Screen 3 (Meet-in-Middle).

Re-implements `_PerModeEmerGNN.fit()` with junction injection at each
forward call. Mirrors upstream shuffle_train + intermediate eval/save
hooks so the trainer is comparable to Screen 1 / upstream multimode.

Key change vs upstream:
  - Model class: EmerGNN_MIM (with feat='X' + Wr(6*n_dim, 1))
  - Forward signature: extra junctions + junction_mask args
  - Junctions are computed ONCE per pair offline (pair_index -> static set)
    then indexed per batch via a precomputed Tensor.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from baseline.emergnn._per_mode import _PerModeEmerGNN
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train

from my_code.models.screen3_meet_in_middle.emergnn_mim import EmerGNN_MIM
from my_code.models.screen3_meet_in_middle.junction_finder import (
    build_adj_int, build_junction_table,
)

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


class _PerModeEmerGNN_MIM(_PerModeEmerGNN):
    """Per-mode EmerGNN_MIM trainer with full shuffle_train + junction injection.

    Args specific to MIM:
      external_init             : Tensor[N, n_dim] node init (TAG-style)
      external_init_node_ids    : node-ID list aligned to external_init rows
      junction_type             : J1 | J3-PK | J3-PD | J3-both | J7 | Jr
      max_junctions             : K (default 32)
      merged_edges_for_junctions: path to merged-KG edges parquet (for adj)
      use_mim                   : if False, runs as standard EmerGNN_TAG (terminal readout)
      freeze_init               : freeze ent_kg.weight
    """

    def __init__(
        self,
        *,
        external_init: np.ndarray | torch.Tensor,
        external_init_node_ids: list[str],
        junction_type: str = "J1",
        max_junctions: int = 32,
        merged_edges_for_junctions: str | Path,
        use_mim: bool = True,
        freeze_init: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("feat", "E")
        super().__init__(**kwargs)
        if isinstance(external_init, np.ndarray):
            external_init = torch.from_numpy(external_init.astype(np.float32))
        self._external_init = external_init.float()
        self._external_init_node_ids = list(external_init_node_ids)
        self._junction_type = str(junction_type)
        self._max_junctions = int(max_junctions)
        self._merged_edges = Path(merged_edges_for_junctions)
        self._use_mim = bool(use_mim)
        self._freeze_init = bool(freeze_init)
        # Computed at fit time
        self._adj_int: dict[int, set[int]] | None = None
        self._id2kind_int: dict[int, str] | None = None
        self._junction_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def _align_init(self) -> torch.Tensor:
        n_ent = self._n_ent
        aligned = torch.zeros(n_ent, self._external_init.shape[1])
        id2row = {nid: r for r, nid in enumerate(self._external_init_node_ids)}
        prefixes = ("db:target:", "db:enzyme:", "db:transporter:", "db:carrier:", "db:pathway:")
        n_hit = 0
        for ent, idx in self._entity2id.items():
            row = id2row.get(ent)
            if row is None and not ent.startswith(("DB", "db:", "het:", "prime:")):
                for p in prefixes:
                    row = id2row.get(f"{p}{ent}")
                    if row is not None:
                        break
            if row is not None:
                aligned[idx] = self._external_init[row]
                n_hit += 1
        print(f"[mim/align] hit={n_hit:,}/{n_ent:,}")
        return aligned

    def _build_adj_and_kind_int(self):
        edges_df = pd.read_parquet(self._merged_edges)
        self._adj_int = build_adj_int(edges_df, self._entity2id)
        nodes_path = self._merged_edges.parent / "nodes__drugbank_hetionet_primekg.parquet"
        nodes_df = pd.read_parquet(nodes_path)
        from my_code.models.screen1_tag_init.node_text_builder import canonical_kind
        self._id2kind_int = {
            self._entity2id[str(n)]: canonical_kind(str(k))
            for n, k in zip(nodes_df["id"], nodes_df["kind"])
            if str(n) in self._entity2id
        }
        print(f"[mim/adj] adj has {len(self._adj_int):,} int nodes; "
              f"id2kind covers {len(self._id2kind_int):,}")

    def _junctions_for_batch(self, head_int: np.ndarray, tail_int: np.ndarray):
        """Return (junctions, mask) for a batch of (head, tail) int IDs.

        Uses pair-level cache (Codex #3 CRITICAL #2 fix): each pair's
        junction set is computed once and reused across epochs. Cache key
        is (head, tail) tuple. Reverse-order pair (tail, head) returns
        symmetric result since junctions are 1-hop intersections (sym).
        """
        # Build any missing pairs via batched call
        missing_pairs = []
        missing_indices = []
        for i, (h, t) in enumerate(zip(head_int.tolist(), tail_int.tolist())):
            key = (h, t) if h <= t else (t, h)
            if key not in self._junction_cache:
                missing_pairs.append((h, t))
                missing_indices.append(i)
        if missing_pairs:
            jids, jmask = build_junction_table(
                missing_pairs, self._adj_int, self._id2kind_int,
                junction_type=self._junction_type, max_junctions=self._max_junctions,
            )
            for (h, t), row_ids, row_mask in zip(missing_pairs, jids, jmask):
                key = (h, t) if h <= t else (t, h)
                self._junction_cache[key] = (row_ids, row_mask)

        # Assemble batch from cache
        B = len(head_int)
        K = self._max_junctions
        out_ids = np.full((B, K), -1, dtype=np.int64)
        out_mask = np.zeros((B, K), dtype=np.float32)
        for i, (h, t) in enumerate(zip(head_int.tolist(), tail_int.tolist())):
            key = (h, t) if h <= t else (t, h)
            row_ids, row_mask = self._junction_cache[key]
            out_ids[i] = row_ids
            out_mask[i] = row_mask
        return torch.from_numpy(out_ids).long(), torch.from_numpy(out_mask).float()

    def fit(self, train, val=None, *, kg=None):
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)
        self._build_adj_and_kind_int()
        aligned_init = self._align_init()

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi

        self._model = EmerGNN_MIM(
            n_ent=self._n_ent,
            n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim,
            length=self.length,
            external_init=aligned_init,
            freeze_init=self._freeze_init,
            feat="X",
            use_mim=self._use_mim,
        ).to(self.device)

        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        # Train_ddi triplets — same as upstream
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
            losses = []
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_np = head.numpy(); tail_np = tail.numpy()
                junctions, jmask = self._junctions_for_batch(head_np, tail_np)
                head = head.to(self.device); tail = tail.to(self.device)
                junctions = junctions.to(self.device); jmask = jmask.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                logits = self._model(head, tail, edge_src, edge_dst, edge_rel,
                                     junctions=junctions, junction_mask=jmask)
                loss = binary_cross_entropy_with_logits(logits, y, reduction="sum")
                loss.backward()
                opt.step()
                losses.append(loss.item() / max(len(batch), 1))

            # End-of-epoch eval
            if val is not None:
                self._model.eval()
                v_auc = self._validate(val)
                self._model.train()
                if v_auc > best_val_auc:
                    best_val_auc = v_auc
                    best_state = copy.deepcopy(self._model.state_dict())
                scheduler.step(v_auc)
                ep_time = time.time() - t_epoch
                print(f"[mim] [ep {epoch+1}/{self.n_epochs}] "
                      f"mean_loss={np.mean(losses):.4f} time={ep_time:.0f}s "
                      f"val_auc={v_auc:.4f}", flush=True)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mim] loaded best val_auc={best_val_auc:.4f} state")

    @torch.no_grad()
    def _validate(self, val) -> float:
        pos = val.splits.val_s2[["drug_a_id", "drug_b_id"]]
        neg = val.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        y_score = np.concatenate([self.predict_proba(pos), self.predict_proba(neg)])
        y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y_true, y_score))

    @torch.no_grad()
    def predict_proba(self, pairs, *, kg=None):
        if self._model is None:
            raise RuntimeError("must fit before predict")
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head_np = head.numpy(); tail_np = tail.numpy()
            junctions, jmask = self._junctions_for_batch(head_np, tail_np)
            head = head.to(self.device); tail = tail.to(self.device)
            junctions = junctions.to(self.device); jmask = jmask.to(self.device)
            logits = self._model(head, tail, edge_src, edge_dst, edge_rel,
                                 junctions=junctions, junction_mask=jmask)
            out[start:start + len(batch)] = torch.sigmoid(logits).detach().cpu().numpy()
        return out


__all__ = ["_PerModeEmerGNN_MIM"]

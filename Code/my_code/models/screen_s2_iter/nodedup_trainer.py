"""NodeDup-style cold-start link prediction (TMLR 2025, arXiv:2402.09711).

Insight (paraphrased to fit EmerGNN):
- Cold drugs have very few KG neighbors; their embeddings end up dominated
  by whatever signal a degree-1 propagation could deliver.
- If during training we ADD a small set of "cold-simulated drug clones"
  whose only KG edge is to their parent drug, and require the model to
  predict (parent, clone) as positive, the encoder learns to produce
  REPRESENTATIONS that survive radical degree reduction.

Implementation:
- After standard _setup_graph (which builds entity2id + edges),
  for each drug d we add a cloned entity `d__dup` whose only KG edge is
  to d via a brand-new "node_duplicate" relation (relation id = n_kg_rel + 1).
- We extend self._kg_triplets with (d_idx, d_dup_idx, dup_rel) edges.
- We extend n_base_rel by +1 (so flow uses 7 base + 1 self-loop = 15 rels).
- During training, we add (drug_a, dup_of_drug_a) pairs (and same for drug_b)
  as POSITIVE supervision in addition to the standard DDI pair (drug_a, drug_b).
- At eval time we never use the cloned nodes; predictions for test_s2 use the
  ordinary drug embeddings.

Effect: encoder is regularized to give consistent identity to a drug whether
it has all its real KG neighbors or just a single "anchor" edge to a clone.
That is exactly the cold-start condition.
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


class _PerModeEmerGNN_NodeDup(_PerModeEmerGNN):
    """EmerGNN with NodeDup auxiliary supervision.

    Extra args:
      ndup_anchor_lambda : weight on (drug, drug_dup) BCE aux (default 0.3)
      ndup_low_degree_only : if True, only duplicate drugs whose KG-degree is
                            below median (default True). Else duplicate ALL drugs.
    """

    def __init__(self, *, ndup_anchor_lambda: float = 0.1,
                 ndup_low_degree_only: bool = True,
                 ndup_detach_clone: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.ndup_anchor_lambda = float(ndup_anchor_lambda)
        self.ndup_low_degree_only = bool(ndup_low_degree_only)
        self.ndup_detach_clone = bool(ndup_detach_clone)

    def _augment_with_clones(self, train) -> tuple[np.ndarray, list[int], list[int]]:
        """Add clone entities + clone-edges to self._kg_triplets.

        Returns:
          extended_kg_triplets : ndarray with original kg edges + clone edges
          clone_pairs_drug_idx : list of (drug_entity_idx, clone_entity_idx) for
                                  drugs that got cloned
          dup_relation_id : the new relation id (= self._n_base_rel - 1 after we
                            increment n_base_rel)
        """
        assert self._entity2id is not None
        # Identify drug entities (DrugBank IDs that appear in train pairs).
        # Use train splits to determine the drug pool.
        drug_ids = set()
        for split_name, df in train.splits.items():
            if df is None:
                continue
            drug_ids.update(df["drug_a_id"].astype(str).tolist())
            drug_ids.update(df["drug_b_id"].astype(str).tolist())
        drug_idxs = [self._entity2id[d] for d in drug_ids if d in self._entity2id]

        # Compute KG-degree per drug
        if len(self._kg_triplets):
            kg_arr = self._kg_triplets
            head_count = np.bincount(kg_arr[:, 0], minlength=self._n_ent)
            tail_count = np.bincount(kg_arr[:, 1], minlength=self._n_ent)
            degree = head_count + tail_count
        else:
            degree = np.zeros(self._n_ent, dtype=np.int64)

        if self.ndup_low_degree_only:
            drug_degrees = np.array([degree[i] for i in drug_idxs])
            median = float(np.median(drug_degrees))
            chosen = [i for i, d in zip(drug_idxs, drug_degrees) if d <= median]
        else:
            chosen = list(drug_idxs)
        print(f"[ndup] duplicating {len(chosen)} drugs (low_degree_only={self.ndup_low_degree_only})")

        # Allocate clone indices: start from current n_ent
        n_orig = self._n_ent
        clone_pairs = []  # list[(orig_idx, clone_idx)]
        for k, orig in enumerate(chosen):
            clone_idx = n_orig + k
            clone_pairs.append((int(orig), int(clone_idx)))
        n_new = n_orig + len(clone_pairs)
        self._n_ent = n_new
        print(f"[ndup] extended entity vocab: {n_orig} -> {n_new}")

        # Allocate new "node_duplicate" relation id
        dup_rel_id = self._n_base_rel  # next slot after existing KG rels
        self._n_base_rel += 1
        print(f"[ndup] new node_duplicate relation id={dup_rel_id}, n_base_rel={self._n_base_rel}")

        # Build clone edges (orig, clone, dup_rel) and also reverse (clone, orig, dup_rel)
        clone_edges = []
        for orig, clone_idx in clone_pairs:
            clone_edges.append([orig, clone_idx, dup_rel_id])
            clone_edges.append([clone_idx, orig, dup_rel_id])  # symmetric so flow can traverse both directions
        clone_edges_arr = np.array(clone_edges, dtype=np.int64)
        extended_kg = np.concatenate([self._kg_triplets, clone_edges_arr], axis=0)
        return extended_kg, clone_pairs, dup_rel_id

    def fit(self, train, val=None, *, kg=None):
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)

        # Augment KG with duplicate nodes
        extended_kg, clone_pairs, dup_rel_id = self._augment_with_clones(train)
        self._kg_triplets = extended_kg

        # Expand morgan_mat with zeros for clone entities (they have no SMILES)
        if morgan_mat is not None:
            n_new = self._n_ent
            n_orig = morgan_mat.shape[0]
            if n_new > n_orig:
                extra = np.zeros((n_new - n_orig, morgan_mat.shape[1]), dtype=morgan_mat.dtype)
                morgan_mat = np.concatenate([morgan_mat, extra], axis=0)

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi

        # Build model with the new n_ent and n_base_rel
        self._model = EmerGNN(
            n_ent=self._n_ent,
            n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim,
            length=self.length,
            feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)
        opt = optim.Adam(self._model.parameters(),
                         lr=self.learning_rate, weight_decay=self.weight_decay)
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

        # Build clone-anchor pair tensor on device
        anchor_pairs = torch.tensor(clone_pairs, dtype=torch.long, device=self.device)
        n_anchors = len(clone_pairs)
        print(f"[ndup] anchor pairs: {n_anchors}")

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
            losses, anchor_losses = [], []

            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(),
                                 dtype=torch.float32, device=self.device)

                opt.zero_grad(set_to_none=True)

                # Standard DDI BCE
                logits = self._model(head_d, tail_d, edge_src, edge_dst, edge_rel)
                loss_bce = F.binary_cross_entropy_with_logits(logits, y, reduction="sum")

                # NodeDup anchor: sample anchor_batch_size anchor pairs as positives
                if n_anchors > 0:
                    n_sample = min(self.batch_size, n_anchors)
                    idx = torch.randint(0, n_anchors, (n_sample,), device=self.device)
                    anch_h = anchor_pairs[idx, 0]
                    anch_t = anchor_pairs[idx, 1]
                    anch_logits = self._model(anch_h, anch_t, edge_src, edge_dst, edge_rel)
                    anch_y = torch.ones_like(anch_logits)
                    loss_anchor = F.binary_cross_entropy_with_logits(
                        anch_logits, anch_y, reduction="sum")
                    loss = loss_bce + self.ndup_anchor_lambda * loss_anchor
                    anchor_losses.append(loss_anchor.item() / n_sample)
                else:
                    loss = loss_bce

                loss.backward()
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
                anc_str = f" anc={np.mean(anchor_losses):.4f}" if anchor_losses else ""
                print(f"[ndup] [ep {epoch+1}/{self.n_epochs}] "
                      f"mean_loss={np.mean(losses):.4f}{anc_str} "
                      f"time={ep_time:.0f}s val_auc={v_auc:.4f}", flush=True)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[ndup] loaded best val_auc={best_val_auc:.4f}")


__all__ = ["_PerModeEmerGNN_NodeDup"]

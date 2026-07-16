"""D2 trainer — EmerGNN + meeting-mediator mask bonus (pair-conditional backbone).

Round 4 D2 (per Notes/Log/d2_meet_mask_design.md §3.4; CP-1 PASS_WITH_NITS at
_reviews/2026-06-01__d2_design__round1.md).

Inherits from `_PerModeEmerGNN_V2I4` (the 0.7804 anchor's trainer). Overrides
`fit()` (duplicates parent body to swap EmerGNN → EmerGNNWithMeetMask, since
the model construction is inline in mnah_trainer.py:329-336) and
`_combined_logit` (threads meet_mask through the new forward).

Controls (per design §4 + §3.5 CLI):
  K1: --d2-shuf-mediators        cardinality-bucketed derangement
  K2a: --d2-rand-mediators-kind-matched  kind + degree (±20→50→any) matched
  K2b: --d2-rand-mediators-uniform       uniform random non-drug entities
  K3: --d2-freeze-alpha-meet     alpha_meet=0 frozen + requires_grad=False
  K4: --d2-disable               skip mask construction; v2i4 parity (CP-2 smoke)

File-independence: does NOT modify mnah_trainer.py / v2i4_trainer.py /
baseline/emergnn/*. Pure inheritance + override.
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
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v2i4_trainer import _PerModeEmerGNN_V2I4  # noqa: E402
from my_code.models.screen_s2_v3_multimodal.emergnn_meet_mask import EmerGNNWithMeetMask  # noqa: E402
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train  # noqa: E402


DEFAULT_MEDIATOR_PARQUET = (
    PROJECT_ROOT
    / "Code" / "data" / "_cache" / "meet_mediators"
    / "meet_mediators__seed42_drugbank__topk20.parquet"
)


class _PerModeEmerGNN_V3MeetMask(_PerModeEmerGNN_V2I4):
    """EmerGNN + meeting-mediator mask bonus (D2 backbone integration)."""

    def __init__(
        self,
        *,
        d2_mediator_parquet: str | Path | None = None,
        d2_disable: bool = False,
        d2_freeze_alpha_meet: bool = False,
        d2_shuf_mediators: bool = False,
        d2_rand_mediators_kind_matched: bool = False,
        d2_rand_mediators_uniform: bool = False,
        d2_alpha_init: float = 0.0,
        d2_shuffle_seed: int = 12345,
        d2_kind_degree_pct: float = 0.20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d2_mediator_parquet = (
            Path(d2_mediator_parquet) if d2_mediator_parquet else DEFAULT_MEDIATOR_PARQUET
        )
        self.d2_disable = bool(d2_disable)
        self.d2_freeze_alpha_meet = bool(d2_freeze_alpha_meet)
        self.d2_shuf_mediators = bool(d2_shuf_mediators)
        self.d2_rand_mediators_kind_matched = bool(d2_rand_mediators_kind_matched)
        self.d2_rand_mediators_uniform = bool(d2_rand_mediators_uniform)
        self.d2_alpha_init = float(d2_alpha_init)
        self.d2_shuffle_seed = int(d2_shuffle_seed)
        self.d2_kind_degree_pct = float(d2_kind_degree_pct)

        # Validate mutually exclusive flags.
        n_active = sum([
            self.d2_shuf_mediators,
            self.d2_rand_mediators_kind_matched,
            self.d2_rand_mediators_uniform,
        ])
        if n_active > 1:
            raise ValueError(
                "Only one of {d2_shuf_mediators, d2_rand_mediators_kind_matched, "
                "d2_rand_mediators_uniform} may be enabled at a time."
            )

        # Lazy-init internal state.
        self._d2_pair_to_mediator_indices: dict[tuple[str, str], np.ndarray] | None = None
        self._d2_summary: dict | None = None

    # ------------------------------------------------------------------
    # Mediator-cache loading + mutation
    # ------------------------------------------------------------------

    def _load_mediator_cache(self) -> None:
        if self._d2_pair_to_mediator_indices is not None:
            return
        if self.d2_disable:
            self._d2_pair_to_mediator_indices = {}
            return

        path = self.d2_mediator_parquet
        if not path.is_file():
            raise FileNotFoundError(
                f"D2 mediator cache not found at {path}. Run "
                "precompute_meet_mediators.py first."
            )
        print(f"[d2] loading mediator cache: {path}", flush=True)
        df = pd.read_parquet(path)
        # Translate mediator_kg_ids (strings) to entity indices on the fly.
        mapping: dict[tuple[str, str], list[int]] = {}
        n_dropped_pairs = 0
        n_dropped_mediators = 0
        for row in df.itertuples(index=False):
            a, b = str(row.drug_a_id), str(row.drug_b_id)
            if a not in self._entity2id or b not in self._entity2id:
                # Pair drug outside trained vocabulary; skip.
                n_dropped_pairs += 1
                continue
            mediator_ids = [m for m in row.mediator_kg_ids]
            mediator_indices = []
            for m in mediator_ids:
                m_str = str(m)
                if m_str in self._entity2id:
                    mediator_indices.append(self._entity2id[m_str])
                else:
                    n_dropped_mediators += 1
            mapping[(a, b)] = mediator_indices

        # Apply control mutations.
        if self.d2_shuf_mediators:
            mapping = self._mutate_shuf_mediators(mapping)
        elif self.d2_rand_mediators_kind_matched:
            mapping = self._mutate_rand_kind_matched(mapping)
        elif self.d2_rand_mediators_uniform:
            mapping = self._mutate_rand_uniform(mapping)

        # Convert lists → numpy arrays.
        self._d2_pair_to_mediator_indices = {
            k: np.asarray(v, dtype=np.int64) for k, v in mapping.items()
        }
        cardinalities = np.asarray([len(v) for v in mapping.values()])
        self._d2_summary = {
            "n_pairs_loaded": int(len(mapping)),
            "n_dropped_pairs_out_of_vocab": int(n_dropped_pairs),
            "n_dropped_mediators_out_of_vocab": int(n_dropped_mediators),
            "cardinality_mean_post_mutation": float(cardinalities.mean()),
            "cardinality_max_post_mutation": int(cardinalities.max()),
            "control_shuf_mediators": self.d2_shuf_mediators,
            "control_rand_kind_matched": self.d2_rand_mediators_kind_matched,
            "control_rand_uniform": self.d2_rand_mediators_uniform,
            "freeze_alpha_meet": self.d2_freeze_alpha_meet,
        }
        print(f"[d2] mediator-cache summary: {self._d2_summary}", flush=True)

    def _mutate_shuf_mediators(
        self, mapping: dict[tuple[str, str], list[int]]
    ) -> dict[tuple[str, str], list[int]]:
        """K1: cardinality-bucketed derangement.

        Pairs bucketed by exact n_mediators. Within each bucket of size > 1,
        apply a random derangement (no fixed points). Singleton-bucket pairs
        fall back to ±1 cardinality merged bucket (logged).
        """
        rng = np.random.default_rng(self.d2_shuffle_seed)
        pairs = list(mapping.keys())
        cards = np.asarray([len(mapping[p]) for p in pairs])
        unique_cards = np.unique(cards)

        new_mapping: dict[tuple[str, str], list[int]] = {}
        n_deranged = 0
        n_singleton_fallback = 0

        for k in unique_cards:
            idx_in_bucket = np.where(cards == k)[0]
            if len(idx_in_bucket) > 1:
                # Derangement within bucket.
                perm = self._derangement(len(idx_in_bucket), rng)
                for i, p_idx in enumerate(perm):
                    src_pair = pairs[idx_in_bucket[i]]
                    tgt_pair = pairs[idx_in_bucket[p_idx]]
                    new_mapping[src_pair] = list(mapping[tgt_pair])
                n_deranged += len(idx_in_bucket)
            else:
                # Singleton — merge with adjacent ±1 cardinality bucket.
                neighbor_indices = np.where((cards == k - 1) | (cards == k + 1))[0]
                if len(neighbor_indices) > 0:
                    swap_idx = rng.choice(neighbor_indices)
                    src_pair = pairs[idx_in_bucket[0]]
                    tgt_pair = pairs[swap_idx]
                    new_mapping[src_pair] = list(mapping[tgt_pair])
                    n_singleton_fallback += 1
                else:
                    # No neighbor bucket either — keep original.
                    src_pair = pairs[idx_in_bucket[0]]
                    new_mapping[src_pair] = list(mapping[src_pair])

        print(
            f"[d2] *** SHUF-MEDIATORS K1 *** deranged {n_deranged} pairs "
            f"(non-singleton buckets); singleton fallback (±1 card) "
            f"{n_singleton_fallback} pairs; kept-original {len(pairs) - n_deranged - n_singleton_fallback}",
            flush=True,
        )
        return new_mapping

    @staticmethod
    def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
        """Random derangement of size n (no fixed points). n must be ≥ 2.

        Up to 10 retries; falls back to a cycle shift on the 11th attempt
        (guarantees no fixed points by construction).
        """
        for _ in range(10):
            perm = rng.permutation(n)
            if not np.any(perm == np.arange(n)):
                return perm
        # Fallback: cycle shift by 1 — guaranteed no fixed points for n ≥ 2.
        return (np.arange(n) + 1) % n

    def _mutate_rand_kind_matched(
        self, mapping: dict[tuple[str, str], list[int]]
    ) -> dict[tuple[str, str], list[int]]:
        """K2a: degree+kind-matched random replacement.

        Per design §4 K2a + fallback ladder: same kind ±20% degree → ±50% →
        any-degree same-kind → K2b uniform per-mediator fallback. Counts per
        level logged.
        """
        rng = np.random.default_rng(self.d2_shuffle_seed)
        # Build per-kind entity pool with degrees from current KG edges.
        # NOTE: this needs self._edge_src and self._kg_triplets, which were set
        # up in parent _setup_graph. Compute per-entity degree once.
        n_ent = self._n_ent
        degrees = np.zeros(n_ent, dtype=np.int64)
        # Use kg_triplets head/tail to compute degree.
        for col_idx in (0, 1):
            ids = self._kg_triplets[:, col_idx]
            np.add.at(degrees, ids, 1)

        # Per-kind entity pool — rebuild from entity2id by looking up node kinds.
        # We don't have node kind here directly; we'll fall back to uniform if
        # the kind info isn't available in this code path.
        # (For now: uniform K2b for all mediators in K2a path — flag for CP-2.)
        # TODO(post-CP-2): wire id2kind into the trainer for proper K2a.
        # For CP-2 round 1 we run K2b semantics under K2a flag and log it.
        print(
            "[d2] *** K2a RAND-KIND-MATCHED *** CP-2 round 1 falls back to "
            "uniform sampling because id2kind is not wired into trainer yet. "
            "TODO: post-CP-2 add id2kind cache so K2a vs K2b can be properly "
            "separated. K2a result this round should be interpreted as a noisier "
            "K2b approximation.",
            flush=True,
        )
        return self._mutate_rand_uniform(mapping)

    def _mutate_rand_uniform(
        self, mapping: dict[tuple[str, str], list[int]]
    ) -> dict[tuple[str, str], list[int]]:
        """K2b: uniform random non-drug entities, preserving per-pair cardinality."""
        rng = np.random.default_rng(self.d2_shuffle_seed)
        # Drug indices = first n_drugs in entity2id (parent _setup_graph convention).
        # Non-drug pool = all indices except drug indices. To be safe, look up which
        # entities are drugs from kg_triplets entity-type tags would be ideal, but
        # we don't have that easily; instead use complement of all entities that
        # appear as drug_a/b in mapping keys.
        drug_indices = {
            self._entity2id[d] for pair in mapping for d in pair
        }
        non_drug_indices = np.array(
            [i for i in range(self._n_ent) if i not in drug_indices],
            dtype=np.int64,
        )
        new_mapping: dict[tuple[str, str], list[int]] = {}
        for pair, mediators in mapping.items():
            k = len(mediators)
            if k == 0:
                new_mapping[pair] = []
                continue
            if k > len(non_drug_indices):
                new_mapping[pair] = list(non_drug_indices)
            else:
                sampled = rng.choice(non_drug_indices, size=k, replace=False)
                new_mapping[pair] = sampled.tolist()
        print(
            f"[d2] *** K2 RAND-UNIFORM *** replaced mediators with uniform random "
            f"non-drug entities for {len(new_mapping)} pairs",
            flush=True,
        )
        return new_mapping

    # ------------------------------------------------------------------
    # mask construction per batch
    # ------------------------------------------------------------------

    def _build_meet_mask(self, batch_df: pd.DataFrame) -> torch.Tensor | None:
        """Return (n_ent, B) {0, 1} float mask for the current batch. None if disabled."""
        if self.d2_disable:
            return None
        self._load_mediator_cache()
        B = len(batch_df)
        mask = torch.zeros(self._n_ent, B, dtype=torch.float32, device=self.device)
        for b_idx, (a, b) in enumerate(
            zip(batch_df["drug_a_id"].astype(str), batch_df["drug_b_id"].astype(str))
        ):
            ca, cb = (a, b) if a <= b else (b, a)
            indices = self._d2_pair_to_mediator_indices.get((ca, cb))
            if indices is None or len(indices) == 0:
                continue
            mask[indices, b_idx] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Override _combined_logit to thread mask through new forward.
    # ------------------------------------------------------------------

    def _combined_logit(self, head, tail, edge_src, edge_dst, edge_rel, batch_df):
        meet_mask = self._build_meet_mask(batch_df)
        # Call EmerGNNWithMeetMask.forward with mask kwarg.
        emergnn_logit = self._model(
            head, tail, edge_src, edge_dst, edge_rel, meet_mask=meet_mask
        )
        feats = self._lookup_features(batch_df)
        h = self._aux_mlp
        count_logit = h.count_logit(feats)
        pair_feat = self._build_pair_feats(batch_df)
        i4_logit = h.i4_logit(pair_feat)
        combined = (
            emergnn_logit + self._beta() * count_logit + h.beta_i4() * i4_logit
        )
        aux = self._beta() * count_logit + h.beta_i4() * i4_logit
        return combined, emergnn_logit, aux

    # ------------------------------------------------------------------
    # Override fit() to construct EmerGNNWithMeetMask instead of EmerGNN.
    # Mirrors mnah_trainer.py:310-456 with the single model-class swap.
    # ------------------------------------------------------------------

    def fit(self, train, val=None, *, kg=None):
        # Pre-fit pair-norm (v2i4 behavior).
        try:
            tr_pos = train.splits.train[["drug_a_id", "drug_b_id"]]
            try:
                tr_neg = train.get_train_negatives(0, regenerate=True)[
                    ["drug_a_id", "drug_b_id"]
                ]
            except Exception as exc:
                print(f"[d2] couldn't load train negatives for norm: {exc}", flush=True)
                tr_neg = None
            self._fit_pair_norm(tr_pos, tr_neg)
        except Exception as exc:
            print(f"[d2] pair-norm fit failed: {exc}", flush=True)

        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)

        # Load mediator cache AFTER _setup_graph so entity2id is ready.
        self._load_mediator_cache()

        # Load count-feature cache (mnah_trainer behavior).
        try:
            train_neg = train.get_train_negatives(0, regenerate=True)[
                ["drug_a_id", "drug_b_id"]
            ]
        except Exception as exc:
            print(f"[d2] could not load train negatives for normalizer: {exc}", flush=True)
            train_neg = None
        self._load_feature_cache(train.splits.train, train_neg)

        # Build the D2-aware backbone (instead of parent's EmerGNN).
        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi
        self._model = EmerGNNWithMeetMask(
            n_ent=self._n_ent, n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
            alpha_meet_init=self.d2_alpha_init,
            freeze_alpha_meet=self.d2_freeze_alpha_meet,
        ).to(self.device)

        # Build aux head (count + i4) — v2i4 behavior.
        self._aux_mlp = self._build_aux_head(self._aux_in_dim).to(self.device)
        if self.mnah_init_beta > 0:
            raw_init = float(np.log(np.exp(self.mnah_init_beta) - 1.0))
        else:
            raw_init = -5.0
        self._raw_beta = nn.Parameter(torch.tensor(raw_init, device=self.device))
        print(
            f"[d2] aux head: in={self._aux_in_dim} hidden={self.mnah_hidden} "
            f"drop={self.mnah_dropout}",
            flush=True,
        )
        print(
            f"[d2] raw_beta init={raw_init:.3f} -> beta={self.mnah_init_beta:.3f}; "
            f"alpha_meet init={self.d2_alpha_init:.3f} per layer "
            f"(L={self.length}), frozen={self.d2_freeze_alpha_meet}",
            flush=True,
        )

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
            eval_kg_triplets, self._n_ent, n_base_rel_with_ddi
        )
        self._eval_edges = (
            torch.from_numpy(esrc).long().to(self.device),
            torch.from_numpy(edst).long().to(self.device),
            torch.from_numpy(erel).long().to(self.device),
        )

        rng = np.random.default_rng(0)
        best_val_auc = -1.0
        best_state = None

        for epoch in range(self.n_epochs):
            epoch_kg, train_pos_targets = shuffle_train(
                train_ddi_int, self._kg_triplets,
                self.shuffle_train_mode, ratio=self.shuffle_ratio, rng=rng,
                extra_kg_ent=self._kg_entity_set,
            )
            if len(train_pos_targets) == 0:
                continue
            esrc, edst, erel = build_edge_lists_from_triplets(
                epoch_kg, self._n_ent, n_base_rel_with_ddi
            )
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
            losses = []
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(),
                                 dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                combined, _, _ = self._combined_logit(
                    head_d, tail_d, edge_src, edge_dst, edge_rel, batch
                )
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
                beta_val = float(self._beta().detach().cpu().item())
                alpha_vals = self._model.alpha_meet.detach().cpu().numpy().tolist()
                print(
                    f"[d2] [ep {epoch+1}/{self.n_epochs}] "
                    f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
                    f"val_combined={v_auc:.4f} val_emer={br['emergnn']:.4f} "
                    f"val_aux={br['aux']:.4f} beta={beta_val:.3f} "
                    f"alpha_meet={alpha_vals}",
                    flush=True,
                )

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[d2] loaded best val_combined={best_val_auc:.4f}", flush=True)


    # ------------------------------------------------------------------
    # save/load override (CP-2 round 2 — complete trainer-level roundtrip)
    # ------------------------------------------------------------------
    #
    # CP-2 round 1 fixed `load()` to construct `EmerGNNWithMeetMask`.
    # CP-2 round 2: extends to a complete D2 trainer roundtrip — also
    # persist+restore the v2i4 readout heads (`_aux_mlp`, `_raw_beta`),
    # feature normalizer stats (count cache mean/std and i4 mean/std), and
    # D2-specific config (alpha_meet flags, mediator parquet path).
    #
    # Parent `_PerModeEmerGNN.save()` writes only the backbone + entity2id;
    # parent `_PerModeEmerGNN.load()` hardcodes `EmerGNN()`. Both need to be
    # overridden for D2 to roundtrip correctly.

    def save(self, path) -> None:  # type: ignore[override]
        import json
        import pickle
        from pathlib import Path as _Path

        from baseline.base import write_manifest

        if self._model is None or self._entity2id is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = _Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Backbone (parent EmerGNN compat layout)
        torch.save(self._model.state_dict(), out / "model.pt")
        n_base_rel_save = int(
            getattr(self, "_n_base_rel_with_ddi", None) or self._n_base_rel
        )
        eval_edges_np = None
        if getattr(self, "_eval_edges", None) is not None:
            eval_edges_np = {
                "src": self._eval_edges[0].detach().cpu().numpy(),
                "dst": self._eval_edges[1].detach().cpu().numpy(),
                "rel": self._eval_edges[2].detach().cpu().numpy(),
            }
        with (out / "graph.pkl").open("wb") as f:
            pickle.dump({
                "entity2id": self._entity2id,
                "n_ent": self._n_ent,
                "edge_src": self._edge_src,
                "edge_dst": self._edge_dst,
                "edge_rel": self._edge_rel,
                "morgan_features": (
                    self._model.ent_feat.cpu().numpy() if self.feat == "M" else None
                ),
                "n_base_rel_with_ddi": n_base_rel_save,
                "n_base_rel_kg": int(self._n_base_rel),
                "eval_edges": eval_edges_np,
            }, f)

        # 2. v2i4 readout heads (aux + beta)
        if self._aux_mlp is not None:
            torch.save(self._aux_mlp.state_dict(), out / "aux.pt")
        if self._raw_beta is not None:
            torch.save(
                {"raw_beta": float(self._raw_beta.detach().cpu().item())},
                out / "beta.pt",
            )

        # 3. Feature normalizer stats + count-feature lookup/matrix (CP-2 round 3 fix)
        #
        # `_combined_logit` calls `_lookup_features` (inherited from
        # `mnah_trainer.py:198-221`) which dereferences `self._feat_lookup`
        # (dict) and `self._feat_matrix` (Tensor on device). Both are built by
        # `_load_feature_cache` during `fit()`; for `predict_proba()` to work
        # after `load()`, we persist them too.
        feat_matrix_np = (
            self._feat_matrix.detach().cpu().numpy()
            if self._feat_matrix is not None else None
        )
        norm_state = {
            "feat_mean": getattr(self, "_feat_mean", None),
            "feat_std": getattr(self, "_feat_std", None),
            "i4_mean": getattr(self, "_i4_mean", None),
            "i4_std": getattr(self, "_i4_std", None),
            "aux_in_dim": getattr(self, "_aux_in_dim", 22),
            "feat_lookup": getattr(self, "_feat_lookup", None),
            "feat_matrix": feat_matrix_np,
        }
        with (out / "normalizer.pkl").open("wb") as f:
            pickle.dump(norm_state, f)

        # 4. D2-specific config snapshot
        d2_cfg = {
            "d2_mediator_parquet": str(self.d2_mediator_parquet),
            "d2_disable": self.d2_disable,
            "d2_freeze_alpha_meet": self.d2_freeze_alpha_meet,
            "d2_shuf_mediators": self.d2_shuf_mediators,
            "d2_rand_mediators_kind_matched": self.d2_rand_mediators_kind_matched,
            "d2_rand_mediators_uniform": self.d2_rand_mediators_uniform,
            "d2_alpha_init": self.d2_alpha_init,
            "d2_shuffle_seed": self.d2_shuffle_seed,
            "d2_kind_degree_pct": self.d2_kind_degree_pct,
        }
        (out / "d2_config.json").write_text(json.dumps(d2_cfg, indent=2))

        write_manifest(
            out,
            baseline_name="_PerModeEmerGNN_V3MeetMask",
            extra={
                "version": "D2-1.0",
                "hyperparameters": {
                    "n_dim": self.n_dim, "length": self.length, "feat": self.feat,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size, "n_epochs": self.n_epochs,
                    "mnah_hidden": self.mnah_hidden,
                    "mnah_dropout": self.mnah_dropout,
                    "v2i4_hidden": self.v2i4_hidden,
                    **d2_cfg,
                },
                "environment": {"torch_version": torch.__version__},
                "graph_metadata": {
                    "n_ent": int(self._n_ent),
                    "n_base_rel": int(self._n_base_rel),
                    "n_base_rel_with_ddi": n_base_rel_save,
                    "backbone_kg_source": self.backbone_kg_source,
                },
            },
        )

    @classmethod
    def load(cls, path) -> "_PerModeEmerGNN_V3MeetMask":  # type: ignore[override]
        import json
        import pickle
        from pathlib import Path as _Path

        from baseline.emergnn.kg_builder import N_BASE_REL  # noqa: E402

        p = _Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        # Filter D2-specific keys that are passed via __init__.
        d2_keys = {
            "d2_mediator_parquet", "d2_disable", "d2_freeze_alpha_meet",
            "d2_shuf_mediators", "d2_rand_mediators_kind_matched",
            "d2_rand_mediators_uniform", "d2_alpha_init", "d2_shuffle_seed",
            "d2_kind_degree_pct",
        }
        init_kwargs = {k: v for k, v in hparams.items() if k in d2_keys
                       or k in {"n_dim", "length", "feat", "learning_rate",
                                "batch_size", "n_epochs", "mnah_hidden",
                                "mnah_dropout", "v2i4_hidden"}}
        inst = cls(**init_kwargs)

        with (p / "graph.pkl").open("rb") as f:
            graph = pickle.load(f)
        inst._entity2id = graph["entity2id"]
        inst._n_ent = graph["n_ent"]
        inst._edge_src = graph["edge_src"]
        inst._edge_dst = graph["edge_dst"]
        inst._edge_rel = graph["edge_rel"]
        n_base_rel_kg = (
            graph.get("n_base_rel_kg")
            or manifest.get("graph_metadata", {}).get("n_base_rel")
            or N_BASE_REL
        )
        inst._n_base_rel = int(n_base_rel_kg)
        n_base_rel_with_ddi = (
            graph.get("n_base_rel_with_ddi")
            or manifest.get("graph_metadata", {}).get("n_base_rel_with_ddi")
            or (int(n_base_rel_kg) + 1)
        )
        inst._n_base_rel_with_ddi = int(n_base_rel_with_ddi)
        morgan = graph.get("morgan_features")

        # 1. Backbone (D2 subclass; alpha_meet restored via load_state_dict)
        inst._model = EmerGNNWithMeetMask(
            n_ent=inst._n_ent,
            n_base_rel=int(n_base_rel_with_ddi),
            n_dim=inst.n_dim,
            length=inst.length,
            feat=inst.feat,
            morgan_features=morgan,
            alpha_meet_init=inst.d2_alpha_init,
            freeze_alpha_meet=inst.d2_freeze_alpha_meet,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()

        # 2. Restore eval edges
        eval_edges = graph.get("eval_edges")
        if eval_edges is not None:
            inst._eval_edges = (
                torch.from_numpy(eval_edges["src"]).long().to(inst.device),
                torch.from_numpy(eval_edges["dst"]).long().to(inst.device),
                torch.from_numpy(eval_edges["rel"]).long().to(inst.device),
            )
        else:
            inst._eval_edges = None

        # 3. Restore feature normalizer stats + count-feature cache (CP-2 round 3 fix)
        norm_path = p / "normalizer.pkl"
        if norm_path.is_file():
            with norm_path.open("rb") as f:
                norm = pickle.load(f)
            inst._feat_mean = norm.get("feat_mean")
            inst._feat_std = norm.get("feat_std")
            inst._i4_mean = norm.get("i4_mean")
            inst._i4_std = norm.get("i4_std")
            inst._aux_in_dim = int(norm.get("aux_in_dim", 22))
            # Restore count-feature lookup + matrix for predict_proba().
            inst._feat_lookup = norm.get("feat_lookup")
            feat_matrix_np = norm.get("feat_matrix")
            if feat_matrix_np is not None:
                inst._feat_matrix = (
                    torch.from_numpy(feat_matrix_np.astype(np.float32)).to(inst.device)
                )

        # 4. Restore aux MLP + raw_beta
        aux_path = p / "aux.pt"
        if aux_path.is_file():
            inst._aux_mlp = inst._build_aux_head(inst._aux_in_dim).to(inst.device)
            inst._aux_mlp.load_state_dict(
                torch.load(aux_path, map_location=inst.device)
            )
            inst._aux_mlp.eval()
        beta_path = p / "beta.pt"
        if beta_path.is_file():
            beta_state = torch.load(beta_path, map_location=inst.device)
            inst._raw_beta = nn.Parameter(
                torch.tensor(float(beta_state["raw_beta"]), device=inst.device)
            )

        return inst


__all__ = ["_PerModeEmerGNN_V3MeetMask", "DEFAULT_MEDIATOR_PARQUET"]

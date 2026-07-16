"""R3 PMP-style readout trainer — mediator-pooled MLP_score (replaces additive fusion).

Architecture decision (per Notes/Log/THEORIST_VERDICT.md §2 and
`ITERATE_STATE.json` selected_proposals[R3_pmp_replace_readout]; paper Section 4
PMP commitment):

  Inputs concat into MLP_score (101-d):
    [ emergnn_logit (1) ,  m_pool (64) ,  pair_feat (13) ,  feats22 (22) ,
      m_present_bit (1) ]

  m_pool = Linear(1024, 64) ( mean( morgan_feature[ mediator_indices ] ) )
  m_present_bit = 1.0 if len(mediators) > 0 else 0.0
  MLP_score: 101 -> 64 -> 1, ReLU + dropout 0.2

  combined = MLP_score(input)         -- REPLACES additive fusion;
                                         drops the count + i4 logit branches.

Per Theorist §2, R3 is the superset of v2i4 under this augmented input —
the MLP can learn an additive sum, so R3 ⊇ v2i4. Identity-ish init keeps
combined ≈ emergnn_logit at epoch 0 so we start from a known good anchor.

Per Theorist §4 R3 risk: empty-mediator (70 % of Path B pairs) mode collapse;
mitigated by the explicit `m_present_bit ∈ {0,1}` input. Empty-mediator
pairs still get a sensible MLP score because m_present_bit=0 signals "the
mediator branch carries no information for this pair."

Pragmatic m_pool builder (per Theorist's pragmatic note): use the morgan
feature buffer attached to the EmerGNN backbone as the per-entity embedding
table, then project 1024 → 64. This avoids needing trained GNN outputs as
inputs (which would require a forward pass detour through the backbone) and
matches the morgan-feature fallback path explicitly approved in the
handoff brief.

Controls (per design):
  K1 (--r3-shuf-mediator-pool):  shuffle m_pool batch rows
  K2 (--r3-zero-m-pool):         force m_pool to zeros (still pass m_present_bit)
  K3 (--r3-additive-init):       init MLP_score to mimic the v2i4 additive
                                 anchor (emergnn + small linear over the rest)

File-independence: does NOT modify v2i4_trainer.py, mnah_trainer.py, or
baseline/emergnn/*. Pure inheritance from _PerModeEmerGNN_V2I4.
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

from my_code.models.screen_s2_v3_multimodal.v2i4_trainer import (  # noqa: E402
    _PerModeEmerGNN_V2I4,
    N_PAIR,
)
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train  # noqa: E402
from baseline.emergnn.model import EmerGNN  # noqa: E402


DEFAULT_MEDIATOR_PARQUET = (
    PROJECT_ROOT
    / "Code" / "data" / "_cache" / "meet_mediators"
    / "meet_mediators__seed42_drugbank__topk20.parquet"
)

# Morgan FP width — matches baseline.emergnn.model.morgan_feat_dim.
MORGAN_DIM = 1024
# m_pool projection dim.
M_POOL_DIM = 64
# MLP_score concat dim: emergnn_logit (1) + m_pool (64) + pair_feat (13)
# + feats22 (22) + m_present_bit (1) = 101.
MLP_SCORE_IN = 1 + M_POOL_DIM + N_PAIR + 22 + 1
MLP_SCORE_HIDDEN = 64

# Position of emergnn_logit in the concat input (used by --r3-additive-init).
_EMERGNN_IDX = 0


class MLPScoreHead(nn.Module):
    """101 -> 64 -> 1 MLP with optional additive-style init.

    Also owns the m_pool projection (1024 -> 64) so save/load handles both
    components as a single state_dict.
    """

    def __init__(self, *, dropout: float = 0.2,
                 additive_init: bool = False):
        super().__init__()
        self.m_pool_proj = nn.Linear(MORGAN_DIM, M_POOL_DIM)
        self.fc1 = nn.Linear(MLP_SCORE_IN, MLP_SCORE_HIDDEN)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(MLP_SCORE_HIDDEN, 1)

        # Init MLP_score so combined ≈ emergnn_logit at epoch 0.
        # Strategy (matches Theorist R3 §"identity-ish at init"):
        #   fc1: zero weights on all input positions EXCEPT a single neuron that
        #        passes emergnn_logit through with weight 1, bias 0. Then fc2 has
        #        weight 1 on that neuron, zero elsewhere, bias 0.
        # Net effect: at init, MLP_score(x) ≈ ReLU(emergnn_logit). For positive
        # emergnn_logit this equals emergnn_logit (good anchor); for negative it
        # clips to 0. To avoid the clip, we use a TWO-neuron pass-through:
        # neuron 0 = +emergnn_logit, neuron 1 = -emergnn_logit, fc2 = [+1, -1].
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        with torch.no_grad():
            self.fc1.weight[0, _EMERGNN_IDX] = 1.0
            self.fc1.weight[1, _EMERGNN_IDX] = -1.0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        with torch.no_grad():
            self.fc2.weight[0, 0] = 1.0
            self.fc2.weight[0, 1] = -1.0

        if additive_init:
            # K3: in addition to the emergnn pass-through, plant small weights
            # on the count (feats22) and i4 (pair_feat) inputs to mimic v2i4's
            # softplus-additive anchor. Using small magnitudes so the
            # combined score still starts close to emergnn_logit and learns
            # additively from there.
            with torch.no_grad():
                # Use remaining neurons (2..MLP_SCORE_HIDDEN-1) split between
                # count_logit-style and i4_logit-style passes.
                # Half assigned to feats22, half to pair_feat.
                # Indices in concat: emergnn(0), m_pool(1..64), pair_feat(65..77),
                # feats22(78..99), m_present_bit(100).
                pf_start = 1 + M_POOL_DIM
                pf_end = pf_start + N_PAIR
                f22_start = pf_end
                f22_end = f22_start + 22
                free = MLP_SCORE_HIDDEN - 2  # neurons 2..H-1
                n_half = free // 2
                # Plant +0.1 average weights from feats22 to half of free neurons
                # and +0.1 weights from pair_feat to the other half.
                self.fc1.weight[2:2 + n_half, f22_start:f22_end] = 0.05
                self.fc1.weight[2 + n_half:2 + 2 * n_half, pf_start:pf_end] = 0.05
                self.fc2.weight[0, 2:2 + n_half] = 0.05
                self.fc2.weight[0, 2 + n_half:2 + 2 * n_half] = 0.05

    def project_m_pool(self, raw_pool: torch.Tensor) -> torch.Tensor:
        return self.m_pool_proj(raw_pool)

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(mlp_input))
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


class _PerModeEmerGNN_V3PMP(_PerModeEmerGNN_V2I4):
    """EmerGNN + PMP-style MLP_score readout (R3 replacement of additive fusion)."""

    def __init__(
        self,
        *,
        r3_mediator_parquet: str | Path | None = None,
        r3_dropout: float = 0.2,
        r3_shuf_mediator_pool: bool = False,
        r3_zero_m_pool: bool = False,
        r3_additive_init: bool = False,
        r3_shuffle_seed: int = 12345,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.r3_mediator_parquet = (
            Path(r3_mediator_parquet) if r3_mediator_parquet else DEFAULT_MEDIATOR_PARQUET
        )
        self.r3_dropout = float(r3_dropout)
        self.r3_shuf_mediator_pool = bool(r3_shuf_mediator_pool)
        self.r3_zero_m_pool = bool(r3_zero_m_pool)
        self.r3_additive_init = bool(r3_additive_init)
        self.r3_shuffle_seed = int(r3_shuffle_seed)

        # Internal state.
        self._r3_pair_to_mediator_indices: dict[tuple[str, str], np.ndarray] | None = None
        self._r3_summary: dict | None = None
        self._r3_last_m_present_mean: float | None = None

    # ------------------------------------------------------------------
    # Mediator-cache loading.
    # ------------------------------------------------------------------

    def _load_r3_mediator_cache(self) -> None:
        if self._r3_pair_to_mediator_indices is not None:
            return
        path = self.r3_mediator_parquet
        if not path.is_file():
            raise FileNotFoundError(
                f"R3 mediator cache not found at {path}. Run "
                "precompute_meet_mediators.py first."
            )
        print(f"[r3] loading mediator cache: {path}", flush=True)
        df = pd.read_parquet(path)
        mapping: dict[tuple[str, str], list[int]] = {}
        n_dropped_pairs = 0
        n_dropped_mediators = 0
        for row in df.itertuples(index=False):
            a, b = str(row.drug_a_id), str(row.drug_b_id)
            if a not in self._entity2id or b not in self._entity2id:
                n_dropped_pairs += 1
                continue
            mediator_indices: list[int] = []
            for m in row.mediator_kg_ids:
                m_str = str(m)
                if m_str in self._entity2id:
                    mediator_indices.append(self._entity2id[m_str])
                else:
                    n_dropped_mediators += 1
            mapping[(a, b)] = mediator_indices

        self._r3_pair_to_mediator_indices = {
            k: np.asarray(v, dtype=np.int64) for k, v in mapping.items()
        }
        cardinalities = np.asarray([len(v) for v in mapping.values()])
        n_empty = int((cardinalities == 0).sum())
        self._r3_summary = {
            "n_pairs_loaded": int(len(mapping)),
            "n_dropped_pairs_out_of_vocab": int(n_dropped_pairs),
            "n_dropped_mediators_out_of_vocab": int(n_dropped_mediators),
            "cardinality_mean": float(cardinalities.mean()),
            "cardinality_max": int(cardinalities.max()),
            "n_empty_pairs": n_empty,
            "frac_empty_pairs": float(n_empty / max(len(mapping), 1)),
        }
        print(f"[r3] mediator-cache summary: {self._r3_summary}", flush=True)

    # ------------------------------------------------------------------
    # Build per-batch m_pool + m_present_bit using morgan features.
    # ------------------------------------------------------------------

    def _build_m_pool(self, batch_df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_pool [B, 1024], m_present_bit [B, 1]).

        raw_pool = mean of morgan_feature[mediator_indices] per pair, or zeros
        if no mediators. m_present_bit = 1.0 if mediator list non-empty.
        """
        self._load_r3_mediator_cache()
        B = len(batch_df)
        raw_pool = torch.zeros(B, MORGAN_DIM, dtype=torch.float32, device=self.device)
        present = torch.zeros(B, 1, dtype=torch.float32, device=self.device)

        # Access morgan features from backbone (registered buffer ent_feat).
        if not hasattr(self._model, "ent_feat"):
            # Backbone not yet built or feat != "M"; m_pool stays zero.
            return raw_pool, present
        ent_feat = self._model.ent_feat  # (n_ent, 1024)

        a_ids = batch_df["drug_a_id"].astype(str).tolist()
        b_ids = batch_df["drug_b_id"].astype(str).tolist()
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            ca, cb = (a, b) if a <= b else (b, a)
            indices = self._r3_pair_to_mediator_indices.get((ca, cb))
            if indices is None or len(indices) == 0:
                continue
            idx_t = torch.from_numpy(indices).long().to(self.device)
            raw_pool[i] = ent_feat.index_select(0, idx_t).mean(dim=0)
            present[i, 0] = 1.0
        return raw_pool, present

    # ------------------------------------------------------------------
    # Override aux head construction: use MLPScoreHead.
    # _build_aux_head ignores in_dim for R3 since the structure is fixed.
    # ------------------------------------------------------------------

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        return MLPScoreHead(dropout=self.r3_dropout, additive_init=self.r3_additive_init)

    # ------------------------------------------------------------------
    # _combined_logit override.
    # Return tuple: (combined, emergnn_logit, MLP_score - emergnn_logit)
    # The third slot mirrors v2i4's "aux" branch (difference between final
    # logit and emergnn_logit) so existing diagnostics keep working.
    # ------------------------------------------------------------------

    def _combined_logit(self, head_idx, tail, edge_src, edge_dst, edge_rel, batch_df):
        emergnn_logit = self._model(head_idx, tail, edge_src, edge_dst, edge_rel)  # (B,)
        feats22 = self._lookup_features(batch_df)  # (B, 22)
        pair_feat = self._build_pair_feats(batch_df)  # (B, 13)
        raw_pool, m_present_bit = self._build_m_pool(batch_df)  # (B, 1024), (B, 1)

        # Control mutations on m_pool input.
        if self.r3_zero_m_pool:
            raw_pool = torch.zeros_like(raw_pool)
        elif self.r3_shuf_mediator_pool:
            B = raw_pool.shape[0]
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.r3_shuffle_seed)
            perm = torch.randperm(B, generator=gen).to(self.device)
            raw_pool = raw_pool.index_select(0, perm)

        h: MLPScoreHead = self._aux_mlp  # type: ignore[assignment]
        m_pool_64 = h.project_m_pool(raw_pool)  # (B, 64)

        # Assemble 101-d MLP input.
        mlp_input = torch.cat([
            emergnn_logit.unsqueeze(-1),  # (B, 1)
            m_pool_64,                    # (B, 64)
            pair_feat,                    # (B, 13)
            feats22,                      # (B, 22)
            m_present_bit,                # (B, 1)
        ], dim=-1)
        combined = h(mlp_input)  # (B,)

        # Track diagnostic stats.
        with torch.no_grad():
            self._r3_last_m_present_mean = float(m_present_bit.mean().item())

        aux_proxy = combined - emergnn_logit  # diagnostic equivalent of v2i4 aux
        return combined, emergnn_logit, aux_proxy

    # ------------------------------------------------------------------
    # predict_channels override — replicate v2i4 schema (combined / emergnn /
    # count / i4) so downstream eval reuses the same diagnostic surface. For
    # R3 we don't have separate count / i4 logits, so we report the
    # "MLP_score - emergnn" residual under both keys with a label note.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_channels(self, pairs: pd.DataFrame) -> dict:
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            es, ed, er = self._eval_edges
        else:
            es, ed, er = self._edges_on_device()
        self._model.eval(); self._aux_mlp.eval()
        h: MLPScoreHead = self._aux_mlp  # type: ignore[assignment]
        keys = ["combined", "emergnn", "count", "i4"]
        out = {k: np.empty(len(pairs), dtype=np.float32) for k in keys}
        for s in range(0, len(pairs), self.batch_size):
            b = pairs.iloc[s:s + self.batch_size]
            hd, tl = self._pair_indices(b); hd = hd.to(self.device); tl = tl.to(self.device)
            emer = self._model(hd, tl, es, ed, er)
            feats = self._lookup_features(b)
            pf = self._build_pair_feats(b)
            raw_pool, m_present_bit = self._build_m_pool(b)
            m_pool_64 = h.project_m_pool(raw_pool)
            mlp_input = torch.cat([
                emer.unsqueeze(-1), m_pool_64, pf, feats, m_present_bit,
            ], dim=-1)
            comb = h(mlp_input)
            residual = comb - emer
            out["combined"][s:s + len(b)] = comb.cpu().numpy()
            out["emergnn"][s:s + len(b)] = emer.cpu().numpy()
            # No separate count/i4 in R3; report residual under both keys for
            # downstream diagnostic compatibility.
            out["count"][s:s + len(b)] = residual.cpu().numpy()
            out["i4"][s:s + len(b)] = residual.cpu().numpy()
        return out

    # ------------------------------------------------------------------
    # fit() override — same body as parent / v3_meet_mask but without the
    # count + i4 logit branches in the loss path (R3 replaces them).
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
                print(f"[r3] couldn't load train negatives for norm: {exc}", flush=True)
                tr_neg = None
            self._fit_pair_norm(tr_pos, tr_neg)
        except Exception as exc:
            print(f"[r3] pair-norm fit failed: {exc}", flush=True)

        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)

        # Load count-feature cache (mnah_trainer behavior, still needed for feats22).
        try:
            train_neg = train.get_train_negatives(0, regenerate=True)[
                ["drug_a_id", "drug_b_id"]
            ]
        except Exception as exc:
            print(f"[r3] could not load train negatives for normalizer: {exc}", flush=True)
            train_neg = None
        self._load_feature_cache(train.splits.train, train_neg)

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi
        # Vanilla EmerGNN backbone (R3 replaces only the readout).
        self._model = EmerGNN(
            n_ent=self._n_ent, n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)

        # Load R3 mediator cache AFTER _setup_graph (entity2id ready).
        self._load_r3_mediator_cache()

        self._aux_mlp = self._build_aux_head(self._aux_in_dim).to(self.device)
        # _raw_beta kept for save/load schema parity with parent, but unused
        # in the loss path. Initialize to a benign value.
        if self.mnah_init_beta > 0:
            raw_init = float(np.log(np.exp(self.mnah_init_beta) - 1.0))
        else:
            raw_init = -5.0
        self._raw_beta = nn.Parameter(torch.tensor(raw_init, device=self.device))

        print(
            f"[r3] readout: MLPScoreHead {MLP_SCORE_IN}->{MLP_SCORE_HIDDEN}->1 "
            f"dropout={self.r3_dropout} additive_init={self.r3_additive_init}",
            flush=True,
        )
        print(
            f"[r3] controls: shuf_mediator_pool={self.r3_shuf_mediator_pool} "
            f"zero_m_pool={self.r3_zero_m_pool} additive_init={self.r3_additive_init}",
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
            m_present_batch_means = []
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head_t, tail_t = self._pair_indices(batch)
                head_d = head_t.to(self.device); tail_d = tail_t.to(self.device)
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
                if self._r3_last_m_present_mean is not None:
                    m_present_batch_means.append(self._r3_last_m_present_mean)

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
                m_present_mean = (
                    float(np.mean(m_present_batch_means))
                    if m_present_batch_means else float("nan")
                )
                print(
                    f"[r3] [ep {epoch+1}/{self.n_epochs}] "
                    f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
                    f"val_combined={v_auc:.4f} val_emer={br['emergnn']:.4f} "
                    f"val_aux={br['aux']:.4f} "
                    f"m_present_mean(train)={m_present_mean:.4f}",
                    flush=True,
                )

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[r3] loaded best val_combined={best_val_auc:.4f}", flush=True)

    # ------------------------------------------------------------------
    # save / load — mirror v3_meet_mask pattern (full trainer roundtrip).
    # ------------------------------------------------------------------

    def save(self, path) -> None:  # type: ignore[override]
        import json as _json
        import pickle
        from pathlib import Path as _Path

        from baseline.base import write_manifest

        if self._model is None or self._entity2id is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = _Path(path)
        out.mkdir(parents=True, exist_ok=True)

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

        if self._aux_mlp is not None:
            # MLPScoreHead.state_dict() contains m_pool_proj + fc1/fc2.
            torch.save(self._aux_mlp.state_dict(), out / "aux.pt")
        if self._raw_beta is not None:
            torch.save(
                {"raw_beta": float(self._raw_beta.detach().cpu().item())},
                out / "beta.pt",
            )

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

        r3_cfg = {
            "r3_mediator_parquet": str(self.r3_mediator_parquet),
            "r3_dropout": self.r3_dropout,
            "r3_shuf_mediator_pool": self.r3_shuf_mediator_pool,
            "r3_zero_m_pool": self.r3_zero_m_pool,
            "r3_additive_init": self.r3_additive_init,
            "r3_shuffle_seed": self.r3_shuffle_seed,
        }
        (out / "r3_config.json").write_text(_json.dumps(r3_cfg, indent=2))

        write_manifest(
            out,
            baseline_name="_PerModeEmerGNN_V3PMP",
            extra={
                "version": "R3-1.0",
                "hyperparameters": {
                    "n_dim": self.n_dim, "length": self.length, "feat": self.feat,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size, "n_epochs": self.n_epochs,
                    "mnah_hidden": self.mnah_hidden,
                    "mnah_dropout": self.mnah_dropout,
                    "v2i4_hidden": self.v2i4_hidden,
                    **r3_cfg,
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
    def load(cls, path) -> "_PerModeEmerGNN_V3PMP":  # type: ignore[override]
        import json as _json
        import pickle
        from pathlib import Path as _Path

        from baseline.emergnn.kg_builder import N_BASE_REL  # noqa: E402

        p = _Path(path)
        manifest = _json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        r3_keys = {
            "r3_mediator_parquet", "r3_dropout", "r3_shuf_mediator_pool",
            "r3_zero_m_pool", "r3_additive_init", "r3_shuffle_seed",
        }
        init_kwargs = {k: v for k, v in hparams.items() if k in r3_keys
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

        inst._model = EmerGNN(
            n_ent=inst._n_ent,
            n_base_rel=int(n_base_rel_with_ddi),
            n_dim=inst.n_dim,
            length=inst.length,
            feat=inst.feat,
            morgan_features=morgan,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()

        eval_edges = graph.get("eval_edges")
        if eval_edges is not None:
            inst._eval_edges = (
                torch.from_numpy(eval_edges["src"]).long().to(inst.device),
                torch.from_numpy(eval_edges["dst"]).long().to(inst.device),
                torch.from_numpy(eval_edges["rel"]).long().to(inst.device),
            )
        else:
            inst._eval_edges = None

        norm_path = p / "normalizer.pkl"
        if norm_path.is_file():
            with norm_path.open("rb") as f:
                norm = pickle.load(f)
            inst._feat_mean = norm.get("feat_mean")
            inst._feat_std = norm.get("feat_std")
            inst._i4_mean = norm.get("i4_mean")
            inst._i4_std = norm.get("i4_std")
            inst._aux_in_dim = int(norm.get("aux_in_dim", 22))
            inst._feat_lookup = norm.get("feat_lookup")
            feat_matrix_np = norm.get("feat_matrix")
            if feat_matrix_np is not None:
                inst._feat_matrix = (
                    torch.from_numpy(feat_matrix_np.astype(np.float32)).to(inst.device)
                )

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

        # Reload mediator cache (entity2id is already restored).
        inst._load_r3_mediator_cache()

        return inst


__all__ = [
    "_PerModeEmerGNN_V3PMP", "MLPScoreHead", "DEFAULT_MEDIATOR_PARQUET",
    "MLP_SCORE_IN", "MLP_SCORE_HIDDEN",
]

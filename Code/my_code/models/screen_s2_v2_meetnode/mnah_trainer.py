"""Meeting-Node Auxiliary Head (MNAH) trainer — v2 Stage 1.

Subclass of `_PerModeEmerGNN` that adds an auxiliary head over pre-computed
22-dim shared-mediator features (per E7 LR baseline). Logit-level late fusion:

    combined_logit = emergnn_logit + softplus(raw_beta) * aux_mlp(features)

Per codex round 11:
- alpha fixed at 1.0 (EmerGNN contribution unchanged in scale)
- beta = softplus(raw_beta), raw_beta init ~ 0.541 so initial beta ~= 1.0
- Features pre-normalized with TRAIN mean/std (fit at __init__ time)
- BCE on combined sigmoid (no separate aux supervision)
- Per-branch AUC logging at each eval epoch

The aux MLP is a small 22 -> 32 -> 1 ReLU network with dropout=0.2.

See README.md in this folder for full design rationale.
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

from baseline.emergnn._per_mode import _PerModeEmerGNN
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train
from baseline.emergnn.model import EmerGNN


CACHE_PATH = PROJECT_ROOT / "Code/data/_cache/meet_feat_drugbank_seed42_kgonly_v1.parquet"


class AuxMLP(nn.Module):
    """22 -> 32 -> 1 MLP for shared-mediator features."""

    def __init__(self, in_dim: int = 22, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        return self.fc2(h).squeeze(-1)


class _PerModeEmerGNN_MNAH(_PerModeEmerGNN):
    """EmerGNN + Meeting-Node Auxiliary Head (v2 Stage 1).

    Args extending parent:
      mnah_feat_cache : path to precomputed 22-dim feature parquet
      mnah_hidden     : MLP hidden dim (default 32)
      mnah_dropout    : MLP dropout (default 0.2)
      mnah_init_beta  : initial softplus(raw_beta) value (default 1.0)
    """

    def __init__(
        self,
        *,
        mnah_feat_cache: str | None = None,
        mnah_hidden: int = 32,
        mnah_dropout: float = 0.2,
        mnah_init_beta: float = 1.0,
        mnah_shuffle_control: bool = False,
        mnah_shuffle_seed: int = 12345,
        mnah_text_cache: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.mnah_feat_cache = mnah_feat_cache or str(CACHE_PATH)
        self.mnah_hidden = int(mnah_hidden)
        self.mnah_dropout = float(mnah_dropout)
        self.mnah_init_beta = float(mnah_init_beta)
        # Stage 2 (i4): optional PubMedBERT shared-mediator text features,
        # concatenated to the 22 counts. None = Stage 1 behavior (counts only).
        self.mnah_text_cache = mnah_text_cache
        self._aux_in_dim = 22  # updated in _load_feature_cache if text added
        # Codex round 13 control: when True, permute the pair->feature row
        # assignment so each pair gets ANOTHER pair's feature vector. Feature
        # DISTRIBUTION is identical; pair-specific mediator structure is
        # destroyed. If combined collapses to emergnn-only, the gain is real
        # pair-specific signal (not extra params / optimizer artifacts).
        self.mnah_shuffle_control = bool(mnah_shuffle_control)
        self.mnah_shuffle_seed = int(mnah_shuffle_seed)
        self._feat_lookup: dict[tuple[str, str], int] | None = None
        self._feat_matrix: torch.Tensor | None = None  # normalized features
        self._aux_mlp: AuxMLP | None = None
        self._raw_beta: nn.Parameter | None = None

    # ------------------------------------------------------------------
    # Feature cache + normalizer
    # ------------------------------------------------------------------

    def _canonical(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        """Hook for subclasses to swap the aux head. Default = single AuxMLP."""
        return AuxMLP(in_dim=in_dim, hidden=self.mnah_hidden, dropout=self.mnah_dropout)

    def _load_feature_cache(self, train_pos_pairs: pd.DataFrame,
                            train_neg_pairs: pd.DataFrame | None = None) -> None:
        """Load pre-computed feature parquet, fit train-only normalizer.

        Per codex round 12 FIX-MINOR: normalizer fit on train positives +
        train negatives, not positives only (positives have systematically
        more shared mediators than negatives, which would bias normalization).
        """
        cache_path = Path(self.mnah_feat_cache)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"MNAH feature cache not found at {cache_path}. "
                f"Run precompute_meet_features.py first."
            )
        print(f"[mnah] loading feature cache: {cache_path}", flush=True)
        df = pd.read_parquet(cache_path)
        feat_cols = [c for c in df.columns if c.startswith("f_")]
        if len(feat_cols) != 22:
            raise ValueError(f"expected 22 feature cols, got {len(feat_cols)}: {feat_cols}")

        X = df[feat_cols].to_numpy(dtype=np.float32)  # (N_pairs, 22)

        # Stage 2: optionally concat PubMedBERT shared-mediator text features.
        # Merge by canonical pair so row order can't silently misalign.
        if self.mnah_text_cache:
            tpath = Path(self.mnah_text_cache)
            if not tpath.exists():
                raise FileNotFoundError(f"MNAH text cache not found: {tpath}")
            tdf = pd.read_parquet(tpath)
            tcols = [c for c in tdf.columns if c.startswith("t_")]
            print(f"[mnah] loading text cache: {tpath} ({len(tcols)} dims)", flush=True)
            merged = df[["drug_a_id", "drug_b_id"]].merge(
                tdf[["drug_a_id", "drug_b_id"] + tcols],
                on=["drug_a_id", "drug_b_id"], how="left", validate="one_to_one")
            if merged[tcols].isna().any().any():
                n_miss = int(merged[tcols].isna().any(axis=1).sum())
                raise ValueError(f"{n_miss} count-pairs missing from text cache — "
                                 f"caches built from different pair universe?")
            Xtext = merged[tcols].to_numpy(dtype=np.float32)
            X = np.concatenate([X, Xtext], axis=1)  # (N, 22+64)
            print(f"[mnah] feature matrix is now {X.shape[1]}d (22 counts + {len(tcols)} text)", flush=True)
        self._aux_in_dim = X.shape[1]

        keys = list(zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)))
        self._feat_lookup = {k: i for i, k in enumerate(keys)}

        # Combine train pos + train neg for normalizer fit (codex r12).
        train_idx = []
        for src_df in (train_pos_pairs, train_neg_pairs):
            if src_df is None:
                continue
            for _, row in src_df.iterrows():
                key = self._canonical(str(row["drug_a_id"]), str(row["drug_b_id"]))
                if key in self._feat_lookup:
                    train_idx.append(self._feat_lookup[key])
        train_idx = np.array(train_idx, dtype=np.int64)
        if len(train_idx) == 0:
            raise RuntimeError("0 train pairs matched in feature cache — alignment bug?")
        train_X = X[train_idx]
        mean = train_X.mean(axis=0)
        std = train_X.std(axis=0) + 1e-6
        print(f"[mnah] normalizer fitted on {len(train_idx)} train pairs", flush=True)
        print(f"[mnah]   feature means range [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"[mnah]   feature stds  range [{std.min():.3f}, {std.max():.3f}]")

        X_norm = (X - mean[None, :]) / std[None, :]
        self._feat_matrix = torch.from_numpy(X_norm.astype(np.float32)).to(self.device)
        self._feat_mean = mean
        self._feat_std = std

        # Codex round 13 SHUFFLE CONTROL: remap each pair to a permuted feature
        # row. Feature matrix/distribution unchanged; pair->feature binding is
        # scrambled. Expected: combined collapses to ~emergnn-only branch.
        if self.mnah_shuffle_control:
            n = self._feat_matrix.shape[0]
            perm = np.random.default_rng(self.mnah_shuffle_seed).permutation(n)
            self._feat_lookup = {k: int(perm[v]) for k, v in self._feat_lookup.items()}
            n_fixed = int((perm == np.arange(n)).sum())
            print(f"[mnah] *** SHUFFLE CONTROL ACTIVE *** permuted {n} pair->feat "
                  f"bindings (seed={self.mnah_shuffle_seed}, {n_fixed} fixed points)",
                  flush=True)

    def _lookup_features(self, batch: pd.DataFrame) -> torch.Tensor:
        """Return (B, 22) normalized features for a batch of pairs.

        Per codex round 12 FIX-CRITICAL: cache miss raises KeyError (no silent
        zero-feature fallback). Recompute cache for split/seed/KG-source if
        this fires.
        """
        idx = []
        missing: list[tuple[str, str]] = []
        for a, b in zip(batch["drug_a_id"].astype(str), batch["drug_b_id"].astype(str)):
            key = self._canonical(a, b)
            row = self._feat_lookup.get(key)
            if row is None:
                missing.append(key)
            else:
                idx.append(row)
        if missing:
            raise KeyError(
                f"MNAH feature cache missing {len(missing)} pair(s); "
                f"examples={missing[:5]}. Recompute the cache for this "
                f"split/seed/KG source."
            )
        idx_arr = np.asarray(idx, dtype=np.int64)
        return self._feat_matrix[torch.from_numpy(idx_arr).to(self.device)]

    # ------------------------------------------------------------------
    # Score helpers
    # ------------------------------------------------------------------

    def _beta(self) -> torch.Tensor:
        return F.softplus(self._raw_beta)

    def _combined_logit(
        self, head: torch.Tensor, tail: torch.Tensor,
        edge_src: torch.Tensor, edge_dst: torch.Tensor, edge_rel: torch.Tensor,
        batch_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (combined_logit, emergnn_logit, aux_logit) per batch."""
        emergnn_logit = self._model(head, tail, edge_src, edge_dst, edge_rel)
        feats = self._lookup_features(batch_df)
        aux_logit = self._aux_mlp(feats)
        combined = emergnn_logit + self._beta() * aux_logit
        return combined, emergnn_logit, aux_logit

    # ------------------------------------------------------------------
    # predict_proba override
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, pairs: pd.DataFrame, *, kg=None) -> np.ndarray:
        if self._model is None or self._aux_mlp is None:
            raise RuntimeError("MNAH trainer not yet fitted")
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval()
        self._aux_mlp.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device); tail = tail.to(self.device)
            combined, _, _ = self._combined_logit(
                head, tail, edge_src, edge_dst, edge_rel, batch)
            probs = torch.sigmoid(combined).detach().cpu().numpy()
            out[start:start + len(batch)] = probs
        return out

    @torch.no_grad()
    def _predict_branches(self, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (combined_probs, emergnn_probs, aux_probs) for diagnostics."""
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval()
        self._aux_mlp.eval()
        c_out = np.empty(len(pairs), dtype=np.float32)
        e_out = np.empty(len(pairs), dtype=np.float32)
        a_out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device); tail = tail.to(self.device)
            comb, emer, aux = self._combined_logit(
                head, tail, edge_src, edge_dst, edge_rel, batch)
            c_out[start:start + len(batch)] = torch.sigmoid(comb).detach().cpu().numpy()
            e_out[start:start + len(batch)] = torch.sigmoid(emer).detach().cpu().numpy()
            a_out[start:start + len(batch)] = torch.sigmoid(self._beta() * aux).detach().cpu().numpy()
        return c_out, e_out, a_out

    # ------------------------------------------------------------------
    # Per-epoch log line formatter — extracted from fit() so subclasses can
    # override the label semantics without copying the entire fit() body.
    # Default returns the original f-string verbatim (no behavior change).
    # ------------------------------------------------------------------

    def _format_epoch_log(
        self,
        *,
        epoch: int,
        losses: list,
        ep_time: float,
        v_auc: float,
        br: dict,
        beta_val: float,
    ) -> str:
        """Build the per-epoch val log line. Override to relabel.

        Args:
          epoch: 0-indexed epoch counter (display +1 for human-readable).
          losses: list of per-batch loss values for this epoch (mean -> log).
          ep_time: wall-clock seconds for this epoch.
          v_auc: val combined branch AUC.
          br: full dict returned by `_validate_branches`. Must contain
            'combined', 'emergnn', 'aux' keys; subclasses may stuff extra
            keys (e.g. 'cluster_only', 'within_only') and key off them.
          beta_val: softplus(raw_beta), the fused-residual scalar.

        Returns:
          A single-line string ready for `print(..., flush=True)`.
        """
        return (
            f"[mnah] [ep {epoch + 1}/{self.n_epochs}] "
            f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
            f"val_combined={v_auc:.4f} val_emer={br['emergnn']:.4f} "
            f"val_aux={br['aux']:.4f} beta={beta_val:.3f}"
        )

    @torch.no_grad()
    def _validate_branches(self, val) -> dict:
        """Return dict with combined/emergnn/aux val AUCs."""
        pos = val.splits.val_s2[["drug_a_id", "drug_b_id"]]
        neg = val.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 or len(neg) == 0:
            return {"combined": float("nan"), "emergnn": float("nan"), "aux": float("nan")}
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, e, a = self._predict_branches(all_pairs)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return {
            "combined": float(roc_auc_score(y, c)),
            "emergnn": float(roc_auc_score(y, e)),
            "aux": float(roc_auc_score(y, a)),
        }

    # ------------------------------------------------------------------
    # fit() override — mirrors parent but adds aux head + branch logging
    # ------------------------------------------------------------------

    def fit(self, train, val=None, *, kg=None):
        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)

        # Load feature cache and fit normalizer (BEFORE building model).
        # Include train negatives in normalizer fit (codex r12 FIX-MINOR):
        # positives have systematically more shared mediators than negatives,
        # so positives-only normalization biases the mean high.
        try:
            train_neg = train.get_train_negatives(0, regenerate=True)[
                ["drug_a_id", "drug_b_id"]
            ]
        except Exception as exc:
            print(f"[mnah] could not load train negatives for normalizer: {exc}", flush=True)
            train_neg = None
        self._load_feature_cache(train.splits.train, train_neg)

        # Build EmerGNN backbone (same as parent)
        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi
        self._model = EmerGNN(
            n_ent=self._n_ent, n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)

        # Build aux head
        self._aux_mlp = self._build_aux_head(self._aux_in_dim).to(self.device)
        # raw_beta s.t. softplus(raw_beta) = mnah_init_beta
        # softplus(x) = log(1+exp(x)); inverse: log(exp(y)-1)
        if self.mnah_init_beta > 0:
            raw_init = float(np.log(np.exp(self.mnah_init_beta) - 1.0))
        else:
            raw_init = -5.0
        self._raw_beta = nn.Parameter(torch.tensor(raw_init, device=self.device))
        print(f"[mnah] aux head: in=22 hidden={self.mnah_hidden} drop={self.mnah_dropout}", flush=True)
        print(f"[mnah] raw_beta init={raw_init:.3f} -> beta={self.mnah_init_beta:.3f}", flush=True)

        # Joint optimizer over EmerGNN + aux MLP + raw_beta
        params = list(self._model.parameters()) + list(self._aux_mlp.parameters()) + [self._raw_beta]
        opt = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        # Build train DDI int + static eval KG (same as parent)
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

            self._model.train()
            self._aux_mlp.train()
            t_epoch = time.time()
            losses = []

            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start:start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head_d = head.to(self.device); tail_d = tail.to(self.device)
                y = torch.tensor(batch["label"].to_numpy(),
                                 dtype=torch.float32, device=self.device)
                opt.zero_grad(set_to_none=True)
                combined, emer, aux = self._combined_logit(
                    head_d, tail_d, edge_src, edge_dst, edge_rel, batch)
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
                # Subclasses may override `_format_epoch_log` to change the log
                # labels (e.g. v1.2 reinterprets emergnn / aux slots as
                # cluster_only / within_only). Default behavior unchanged.
                print(
                    self._format_epoch_log(
                        epoch=epoch,
                        losses=losses,
                        ep_time=ep_time,
                        v_auc=v_auc,
                        br=br,
                        beta_val=beta_val,
                    ),
                    flush=True,
                )

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[mnah] loaded best val_combined={best_val_auc:.4f}", flush=True)


__all__ = ["_PerModeEmerGNN_MNAH", "AuxMLP"]

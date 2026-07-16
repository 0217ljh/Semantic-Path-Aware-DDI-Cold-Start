"""R1 Gated Fusion trainer — softmax 3-way gate replacement fusion (round 4 attempt 2).

Architecture decision (per Notes/Log/THEORIST_VERDICT.md §2 and
`ITERATE_STATE.json` selected_proposals[R1_gated_fusion]):

  emergnn_logit, count_logit, i4_logit  -- same three branches as v2i4
  g = softmax(GateMLP(pair_feat))       -- 3-way mixture weights conditioned on
                                          the 13-d mechanistic pair_feat
  combined = g[0] * emergnn_logit + g[1] * count_logit + g[2] * i4_logit

This REPLACES v2i4's softplus-additive fusion. Theorist: under non-constant
Bayes-optimal weights across pairs (D2 K3 q1/q2 gap confirms this exists),
gated fusion strictly dominates additive. Expected combined AUC 0.785+-0.004
single seed.

Risk per Theorist §4: gate collapse to (1, 0, 0). Mitigation: zero-init the
last layer of GateMLP so g starts at uniform (1/3, 1/3, 1/3); log gate
entropy each epoch as a watchdog.

Controls (per design):
  K1 (--r1-shuf-pair-feat):           shuffle pair_feat batch row order before
                                      passing to GateMLP (input scrambled, but
                                      branches still get the correct rows)
  K2 (--r1-freeze-gate-uniform):      hardcode g = [1/3, 1/3, 1/3] frozen
  K3 (--r1-zero-gate-emergnn-only):   hardcode g = [1, 0, 0] frozen

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
    CountPlusI4Head,
    N_PAIR,
)
from my_code.models.screen_s2_v2_meetnode.mnah_trainer import AuxMLP  # noqa: E402
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets, shuffle_train  # noqa: E402
from baseline.emergnn.model import EmerGNN  # noqa: E402


class GatedFusionHead(nn.Module):
    """22→32→1 count MLP + 13→32→1 i4 MLP + 13→32→3 gate MLP (softmax 3-way).

    Gate last-layer bias init=0 + zero-init weights => uniform softmax at
    epoch 0 (Theorist mitigation against gate collapse).
    """

    def __init__(self, *, count_in: int = 22, count_hidden: int = 32,
                 dropout: float = 0.2, i4_hidden: int = 32,
                 gate_hidden: int = 32):
        super().__init__()
        # Same 22→32→1 count head as v2i4 CountPlusI4Head.count_mlp.
        self.count_mlp = AuxMLP(in_dim=count_in, hidden=count_hidden, dropout=dropout)
        # Same 13→32→1 i4 head as v2i4 CountPlusI4Head.i4_mlp.
        self.i4_mlp = nn.Sequential(
            nn.Linear(N_PAIR, i4_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(i4_hidden, 1),
        )
        # NEW: 13→32→3 gate MLP. Zero-init last layer for uniform softmax start.
        self.gate_mlp = nn.Sequential(
            nn.Linear(N_PAIR, gate_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )
        last = self.gate_mlp[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def count_logit(self, feats22: torch.Tensor) -> torch.Tensor:
        return self.count_mlp(feats22)

    def i4_logit(self, pair_feat: torch.Tensor) -> torch.Tensor:
        return self.i4_mlp(pair_feat).squeeze(-1)

    def gate(self, pair_feat: torch.Tensor) -> torch.Tensor:
        """Return softmax gate weights, shape (B, 3). Order: emergnn, count, i4."""
        return F.softmax(self.gate_mlp(pair_feat), dim=-1)


class _PerModeEmerGNN_V3GatedFusion(_PerModeEmerGNN_V2I4):
    """EmerGNN + gated 3-way replacement fusion (R1)."""

    def __init__(
        self,
        *,
        r1_gate_hidden: int = 32,
        r1_shuf_pair_feat: bool = False,
        r1_freeze_gate_uniform: bool = False,
        r1_zero_gate_emergnn_only: bool = False,
        r1_shuffle_seed: int = 12345,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.r1_gate_hidden = int(r1_gate_hidden)
        self.r1_shuf_pair_feat = bool(r1_shuf_pair_feat)
        self.r1_freeze_gate_uniform = bool(r1_freeze_gate_uniform)
        self.r1_zero_gate_emergnn_only = bool(r1_zero_gate_emergnn_only)
        self.r1_shuffle_seed = int(r1_shuffle_seed)

        n_active = sum([
            self.r1_freeze_gate_uniform,
            self.r1_zero_gate_emergnn_only,
        ])
        if n_active > 1:
            raise ValueError(
                "Only one of {r1_freeze_gate_uniform, r1_zero_gate_emergnn_only} "
                "may be enabled at a time."
            )

        # Per-epoch gate diagnostics (updated by fit()).
        self._r1_last_gate_entropy: float | None = None
        self._r1_last_gate_means: list[float] | None = None

    # ------------------------------------------------------------------
    # Override aux head construction: use GatedFusionHead.
    # ------------------------------------------------------------------

    def _build_aux_head(self, in_dim: int) -> nn.Module:
        return GatedFusionHead(
            count_in=22, count_hidden=self.mnah_hidden,
            dropout=self.mnah_dropout, i4_hidden=self.v2i4_hidden,
            gate_hidden=self.r1_gate_hidden,
        )

    # ------------------------------------------------------------------
    # Gate computation with control mutations.
    # ------------------------------------------------------------------

    def _compute_gate(self, head: GatedFusionHead, pair_feat: torch.Tensor) -> torch.Tensor:
        """Return (B, 3) gate weights. Applies K1/K2/K3 mutations if set."""
        if self.r1_freeze_gate_uniform:
            B = pair_feat.shape[0]
            return torch.full(
                (B, 3), 1.0 / 3.0, dtype=pair_feat.dtype, device=pair_feat.device,
            )
        if self.r1_zero_gate_emergnn_only:
            B = pair_feat.shape[0]
            g = torch.zeros(B, 3, dtype=pair_feat.dtype, device=pair_feat.device)
            g[:, 0] = 1.0
            return g
        gate_input = pair_feat
        if self.r1_shuf_pair_feat:
            # K1: shuffle batch row order in pair_feat before gate_mlp.
            # Branches (count, i4) still see the correct (un-shuffled) inputs;
            # only the gate is fed a scrambled batch.
            B = pair_feat.shape[0]
            # Per-call seeded permutation for reproducibility within a run.
            g = torch.Generator(device="cpu")
            g.manual_seed(self.r1_shuffle_seed)
            perm = torch.randperm(B, generator=g).to(pair_feat.device)
            gate_input = pair_feat.index_select(0, perm)
        return head.gate(gate_input)

    # ------------------------------------------------------------------
    # Override _combined_logit with replacement fusion.
    # Return signature: (combined, emergnn_logit, aux) — matches parent.
    # ------------------------------------------------------------------

    def _combined_logit(self, head_idx, tail, edge_src, edge_dst, edge_rel, batch_df):
        emergnn_logit = self._model(head_idx, tail, edge_src, edge_dst, edge_rel)
        feats22 = self._lookup_features(batch_df)
        h: GatedFusionHead = self._aux_mlp  # type: ignore[assignment]
        count_logit = h.count_logit(feats22)
        pair_feat = self._build_pair_feats(batch_df)
        i4_logit = h.i4_logit(pair_feat)
        g = self._compute_gate(h, pair_feat)  # (B, 3)
        combined = (
            g[:, 0] * emergnn_logit
            + g[:, 1] * count_logit
            + g[:, 2] * i4_logit
        )
        # aux = non-emergnn portion (for diagnostic parity with v2i4).
        aux = g[:, 1] * count_logit + g[:, 2] * i4_logit
        # Track latest gate stats on the model instance for epoch-end logging.
        with torch.no_grad():
            self._last_gate = g.detach()
        return combined, emergnn_logit, aux

    # ------------------------------------------------------------------
    # predict_channels override — replicate v2i4 schema (combined / emergnn /
    # count / i4) with the gated fusion weights for downstream diagnostics.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_channels(self, pairs: pd.DataFrame) -> dict:
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            es, ed, er = self._eval_edges
        else:
            es, ed, er = self._edges_on_device()
        self._model.eval()
        self._aux_mlp.eval()
        h: GatedFusionHead = self._aux_mlp  # type: ignore[assignment]
        keys = ["combined", "emergnn", "count", "i4"]
        out = {k: np.empty(len(pairs), dtype=np.float32) for k in keys}
        for s in range(0, len(pairs), self.batch_size):
            b = pairs.iloc[s:s + self.batch_size]
            hd, tl = self._pair_indices(b)
            hd = hd.to(self.device); tl = tl.to(self.device)
            emer = self._model(hd, tl, es, ed, er)
            feats = self._lookup_features(b)
            cl = h.count_logit(feats)
            pf = self._build_pair_feats(b)
            il = h.i4_logit(pf)
            g = self._compute_gate(h, pf)
            comb = g[:, 0] * emer + g[:, 1] * cl + g[:, 2] * il
            out["combined"][s:s + len(b)] = comb.cpu().numpy()
            out["emergnn"][s:s + len(b)] = emer.cpu().numpy()
            out["count"][s:s + len(b)] = (g[:, 1] * cl).cpu().numpy()
            out["i4"][s:s + len(b)] = (g[:, 2] * il).cpu().numpy()
        return out

    # ------------------------------------------------------------------
    # Gate-statistics helper for epoch logging.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _gate_stats(self, val) -> dict:
        """Compute gate entropy + per-branch mean weights over val pairs."""
        pos = val.splits.val_s2[["drug_a_id", "drug_b_id"]]
        neg = val.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]
        if len(pos) == 0 and len(neg) == 0:
            return {"gate_entropy": float("nan"), "gate_means": [float("nan")] * 3}
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        h: GatedFusionHead = self._aux_mlp  # type: ignore[assignment]
        self._aux_mlp.eval()
        entropies = []
        means_accum = np.zeros(3, dtype=np.float64)
        n_seen = 0
        for s in range(0, len(all_pairs), self.batch_size):
            b = all_pairs.iloc[s:s + self.batch_size]
            pf = self._build_pair_feats(b)
            g = self._compute_gate(h, pf)
            ent = -(g * (g + 1e-12).log()).sum(dim=-1)  # (B,)
            entropies.append(ent.cpu().numpy())
            means_accum += g.sum(dim=0).cpu().numpy()
            n_seen += g.shape[0]
        all_ent = np.concatenate(entropies)
        means = means_accum / max(n_seen, 1)
        return {
            "gate_entropy": float(all_ent.mean()),
            "gate_means": means.tolist(),
        }

    # ------------------------------------------------------------------
    # fit() override — same body as parent v2i4 / mnah, but with gate-entropy
    # logging at each epoch end. (We duplicate the loop because both the
    # parent loop in mnah_trainer.py and the v3_meet_mask loop in this
    # subpackage do the same; copying keeps CLAUDE.md "no edits to v2i4_trainer
    # or mnah_trainer" intact.)
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
                print(f"[r1] couldn't load train negatives for norm: {exc}", flush=True)
                tr_neg = None
            self._fit_pair_norm(tr_pos, tr_neg)
        except Exception as exc:
            print(f"[r1] pair-norm fit failed: {exc}", flush=True)

        if kg is None:
            kg = train.kg
        morgan_mat, _ = self._setup_graph(train, kg)

        # Load count-feature cache (mnah_trainer behavior).
        try:
            train_neg = train.get_train_negatives(0, regenerate=True)[
                ["drug_a_id", "drug_b_id"]
            ]
        except Exception as exc:
            print(f"[r1] could not load train negatives for normalizer: {exc}", flush=True)
            train_neg = None
        self._load_feature_cache(train.splits.train, train_neg)

        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi
        # Same backbone as v2i4: vanilla EmerGNN (no meet-mask, no hypernet).
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
        print(
            f"[r1] aux head: GatedFusionHead in22 hidden={self.mnah_hidden} "
            f"i4_hidden={self.v2i4_hidden} gate_hidden={self.r1_gate_hidden} "
            f"drop={self.mnah_dropout}",
            flush=True,
        )
        print(
            f"[r1] controls: shuf_pair_feat={self.r1_shuf_pair_feat} "
            f"freeze_gate_uniform={self.r1_freeze_gate_uniform} "
            f"zero_gate_emergnn_only={self.r1_zero_gate_emergnn_only}",
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

            if val is not None:
                self._model.eval(); self._aux_mlp.eval()
                br = self._validate_branches(val)
                gstats = self._gate_stats(val)
                self._r1_last_gate_entropy = gstats["gate_entropy"]
                self._r1_last_gate_means = gstats["gate_means"]
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
                print(
                    f"[r1] [ep {epoch+1}/{self.n_epochs}] "
                    f"loss={np.mean(losses):.4f} time={ep_time:.0f}s "
                    f"val_combined={v_auc:.4f} val_emer={br['emergnn']:.4f} "
                    f"val_aux={br['aux']:.4f} beta={beta_val:.3f} "
                    f"gate_entropy={gstats['gate_entropy']:.4f} "
                    f"gate_means={[round(x, 4) for x in gstats['gate_means']]}",
                    flush=True,
                )

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state["emergnn"])
            self._aux_mlp.load_state_dict(best_state["aux"])
            with torch.no_grad():
                self._raw_beta.copy_(torch.tensor(best_state["raw_beta"], device=self.device))
            print(f"[r1] loaded best val_combined={best_val_auc:.4f}", flush=True)

    # ------------------------------------------------------------------
    # save / load — mirrors v3_meet_mask pattern for trainer roundtrip.
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

        r1_cfg = {
            "r1_gate_hidden": self.r1_gate_hidden,
            "r1_shuf_pair_feat": self.r1_shuf_pair_feat,
            "r1_freeze_gate_uniform": self.r1_freeze_gate_uniform,
            "r1_zero_gate_emergnn_only": self.r1_zero_gate_emergnn_only,
            "r1_shuffle_seed": self.r1_shuffle_seed,
        }
        (out / "r1_config.json").write_text(_json.dumps(r1_cfg, indent=2))

        write_manifest(
            out,
            baseline_name="_PerModeEmerGNN_V3GatedFusion",
            extra={
                "version": "R1-1.0",
                "hyperparameters": {
                    "n_dim": self.n_dim, "length": self.length, "feat": self.feat,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size, "n_epochs": self.n_epochs,
                    "mnah_hidden": self.mnah_hidden,
                    "mnah_dropout": self.mnah_dropout,
                    "v2i4_hidden": self.v2i4_hidden,
                    **r1_cfg,
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
    def load(cls, path) -> "_PerModeEmerGNN_V3GatedFusion":  # type: ignore[override]
        import json as _json
        import pickle
        from pathlib import Path as _Path

        from baseline.emergnn.kg_builder import N_BASE_REL  # noqa: E402

        p = _Path(path)
        manifest = _json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        r1_keys = {
            "r1_gate_hidden", "r1_shuf_pair_feat", "r1_freeze_gate_uniform",
            "r1_zero_gate_emergnn_only", "r1_shuffle_seed",
        }
        init_kwargs = {k: v for k, v in hparams.items() if k in r1_keys
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

        return inst


__all__ = ["_PerModeEmerGNN_V3GatedFusion", "GatedFusionHead"]

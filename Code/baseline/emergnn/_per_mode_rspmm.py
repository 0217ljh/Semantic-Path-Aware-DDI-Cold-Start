"""EmerGNN per-mode training core for the rspmm backend (torchdrug kernel).

Parallel to :class:`baseline.emergnn._per_mode._PerModeEmerGNN` (the pure-PyTorch
chunk core). This subclass keeps ALL training behavior identical — same
``_setup_graph`` / KG cache, same paper-faithful per-epoch ``shuffle_train``, same
negatives, batch size, Adam + ReduceLROnPlateau, sum-reduction BCE, best-val-AUC
checkpoint selection, and progress logging — and changes ONLY:

  * the model class: :class:`baseline.emergnn.model_rspmm.EmerGNN_RSPMM` (fused
    ``generalized_rspmm`` forward) instead of the chunk ``EmerGNN``;
  * the KG representation handed to the model: a coalesced 3D sparse-COO tensor
    (built once per KG change, from TRIPLETS via
    :func:`baseline.emergnn._rspmm_utils.build_sparse_kg_from_triplets`) instead
    of dense ``edge_src/dst/rel`` lists.

Everything else is inherited from ``_PerModeEmerGNN`` verbatim (no parent code
modified, no signatures changed). Selected via ``EMERGNN_BACKEND=rspmm``.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn  # noqa: F401 — manifest env version
import torch
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.base import write_manifest
from baseline.emergnn._per_mode import _PerModeEmerGNN
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets
from baseline.emergnn.kg_builder import N_BASE_REL
from baseline.emergnn.model_rspmm import EmerGNN_RSPMM
from baseline.emergnn.shuffle_utils import shuffle_train

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


class _PerModeEmerGNN_RSPMM(_PerModeEmerGNN):
    """rspmm-backend twin of :class:`_PerModeEmerGNN` (fused kernel + sparse KG)."""

    VERSION = "1.0-rspmm"

    def _build_kg(self, triplets: np.ndarray, n_rel: int) -> torch.Tensor:
        """Coalesced sparse KG on self.device from (h,t,r) triplets (original convention)."""
        return build_sparse_kg_from_triplets(
            np.asarray(triplets, dtype=np.int64), self._n_ent, n_rel, device=self.device
        )

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        if kg is None:
            kg = train.kg
        morgan_mat, _drug_ids = self._setup_graph(train, kg)

        # Paper-faithful DDI relation slot (see _PerModeEmerGNN.fit).
        n_kg_rel = self._n_base_rel
        n_base_rel_with_ddi = n_kg_rel + 1
        self._n_base_rel_with_ddi = n_base_rel_with_ddi

        self._model = EmerGNN_RSPMM(
            n_ent=self._n_ent,
            n_base_rel=n_base_rel_with_ddi,
            n_dim=self.n_dim,
            length=self.length,
            feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        # train positives -> (ent_a, ent_b, dummy_ddi_rel=n_kg_rel) int triplets.
        pos_df = train.splits.train.copy()
        a_ids = pos_df["drug_a_id"].astype(str).map(self._entity2id)
        b_ids = pos_df["drug_b_id"].astype(str).map(self._entity2id)
        valid_mask = a_ids.notna() & b_ids.notna()
        if not valid_mask.all():
            print(
                f"[emergnn-rspmm] dropping {(~valid_mask).sum()} train pos rows with unknown drugs",
                file=sys.stderr,
            )
        train_ddi_int = np.stack(
            [
                a_ids[valid_mask].astype(np.int64).to_numpy(),
                b_ids[valid_mask].astype(np.int64).to_numpy(),
                np.full(int(valid_mask.sum()), n_kg_rel, dtype=np.int64),
            ],
            axis=1,
        )

        # Static eval KG (= train_ddi + base_kg) as sparse, built once.
        eval_kg_triplets = np.concatenate([train_ddi_int, self._kg_triplets], axis=0)
        self._eval_kg_triplets = eval_kg_triplets
        self._eval_kg = self._build_kg(eval_kg_triplets, n_base_rel_with_ddi)

        rng = np.random.default_rng(0)
        pos = train.splits.train.copy()
        best_val_auc = -1.0
        best_state: dict | None = None

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        n_pos_plus_neg = 2 * len(pos)
        steps_per_epoch = (n_pos_plus_neg + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[emergnn-rspmm] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (self.run_dir / "checkpoints") if self.run_dir is not None else None

        def _eval_dict() -> dict:
            auc = self._validate(val) if val is not None else float("nan")
            return {"val_auc": auc}

        def _save_ckpt(tag: str) -> None:
            if ckpt_root is None:
                return
            ckpt_path = ckpt_root / f"checkpoint-{tag}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.save(ckpt_path)
            progress.log_save(str(ckpt_path), scope="step" if "step" in str(tag) else "epoch")
            rotate_checkpoints(ckpt_root, self.save_total_limit)

        # Optional early stopping (env-var driven, default OFF = paper-faithful).
        # EMERGNN_EARLY_STOP_PATIENCE=N -> stop after N consecutive epochs with no
        # best-val-AUC improvement. load_best_model_at_end already restores the
        # best-val checkpoint, so early stopping only skips wasted late epochs; it
        # does NOT change the selected model. Set to 0/unset to disable.
        _es_patience = int(os.environ.get("EMERGNN_EARLY_STOP_PATIENCE", "0"))
        _epochs_no_improve = 0

        for epoch in range(self.n_epochs):
            _best_at_epoch_start = best_val_auc
            epoch_kg, train_pos_targets = shuffle_train(
                train_ddi_int,
                self._kg_triplets,
                self.shuffle_train_mode,
                ratio=self.shuffle_ratio,
                rng=rng,
                extra_kg_ent=self._kg_entity_set,
            )
            if len(train_pos_targets) == 0:
                print(
                    f"[emergnn-rspmm] epoch {epoch+1}: 0 train targets after "
                    f"shuffle_train (mode={self.shuffle_train_mode}); skip",
                    flush=True,
                )
                continue
            # This epoch's KG as a sparse tensor (built once per epoch).
            epoch_kg_sparse = self._build_kg(epoch_kg, n_base_rel_with_ddi)

            pre_neg = train.get_train_negatives(epoch, regenerate=True)
            n_target_pos = len(train_pos_targets)
            if len(pre_neg) >= n_target_pos:
                neg_idx = rng.choice(len(pre_neg), size=n_target_pos, replace=False)
                neg = pre_neg.iloc[neg_idx].reset_index(drop=True)
            else:
                neg = pre_neg

            id2entity = {v: k for k, v in self._entity2id.items()}
            pos_target_df = pd.DataFrame(
                {
                    "drug_a_id": [id2entity[int(h)] for h in train_pos_targets[:, 0]],
                    "drug_b_id": [id2entity[int(t)] for t in train_pos_targets[:, 1]],
                }
            )
            pairs_df = pd.concat(
                [
                    pos_target_df.assign(label=1),
                    neg[["drug_a_id", "drug_b_id"]].assign(label=0),
                ],
                ignore_index=True,
            ).sample(frac=1, random_state=epoch).reset_index(drop=True)

            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(pairs_df), self.batch_size):
                batch = pairs_df.iloc[start : start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head = head.to(self.device)
                tail = tail.to(self.device)
                y = torch.tensor(
                    batch["label"].to_numpy(), dtype=torch.float32, device=self.device
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model(head, tail, epoch_kg_sparse)
                loss = binary_cross_entropy_with_logits(logits, y, reduction="sum")
                loss.backward()
                opt.step()
                progress.step(loss.item() / max(len(batch), 1))

                if progress.should_eval_step() and val is not None:
                    self._model.eval()
                    metrics = _eval_dict()
                    self._model.train()
                    progress.log_eval(metrics, scope="step")
                    if metrics.get("val_auc", -1) > best_val_auc:
                        best_val_auc = metrics["val_auc"]
                        best_state = copy.deepcopy(self._model.state_dict())
                if progress.should_save_step():
                    _save_ckpt(f"step{progress.step_count}")

            extra: dict = {}
            if progress.should_eval_epoch() and val is not None:
                self._model.eval()
                metrics = _eval_dict()
                self._model.train()
                progress.log_eval(metrics, scope="epoch")
                extra.update(metrics)
                if metrics.get("val_auc", -1) > best_val_auc:
                    best_val_auc = metrics["val_auc"]
                    best_state = copy.deepcopy(self._model.state_dict())
                scheduler.step(metrics.get("val_auc", 0.0))
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}")
            progress.epoch_end(extra=extra if extra else None)

            # Early stopping (only if enabled + we actually evaluated this epoch,
            # signalled by `extra` carrying val_auc — avoids re-calling the
            # possibly-stateful progress.should_eval_epoch()).
            if _es_patience > 0 and val is not None and "val_auc" in extra:
                if best_val_auc > _best_at_epoch_start + 1e-9:
                    _epochs_no_improve = 0
                else:
                    _epochs_no_improve += 1
                if _epochs_no_improve >= _es_patience:
                    print(
                        f"[emergnn-rspmm] EARLY STOP @ epoch {epoch+1}/{self.n_epochs}: "
                        f"val_auc no improvement for {_es_patience} epochs "
                        f"(best val_auc={best_val_auc:.4f})",
                        flush=True,
                    )
                    break

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[emergnn-rspmm] loaded best val_auc={best_val_auc:.4f} state", flush=True)

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: pd.DataFrame,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("_PerModeEmerGNN_RSPMM must be fitted (or loaded) before prediction.")
        # Static eval KG (= train_ddi + base_kg) built in fit(); rebuild from
        # persisted triplets if only those survived (e.g. after load()).
        if getattr(self, "_eval_kg", None) is None:
            tri = getattr(self, "_eval_kg_triplets", None)
            if tri is None:
                raise RuntimeError("no eval KG available; fit() must run before predict.")
            self._eval_kg = self._build_kg(tri, self._n_base_rel_with_ddi)
        self._model.eval()
        out = np.empty(len(pairs), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device)
            tail = tail.to(self.device)
            logits = self._model(head, tail, self._eval_kg)
            out[start : start + len(batch)] = torch.sigmoid(logits).detach().cpu().numpy()
        return out

    # ------------------------------------------------------------------
    # Checkpoint I/O — rspmm-specific (parent's persists chunk edges +
    # rebuilds the chunk EmerGNN class, which is wrong for this backend).
    # Persist the eval-KG TRIPLETS (not edge lists) and rebuild EmerGNN_RSPMM.
    # ------------------------------------------------------------------
    def save(self, path: "Path | str") -> None:
        if self._model is None or self._entity2id is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        n_base_rel_save = int(
            getattr(self, "_n_base_rel_with_ddi", None) or self._n_base_rel
        )
        eval_tri = getattr(self, "_eval_kg_triplets", None)
        with (out / "graph.pkl").open("wb") as f:
            pickle.dump(
                {
                    "backend": "rspmm",
                    "entity2id": self._entity2id,
                    "n_ent": self._n_ent,
                    "morgan_features": (
                        self._model.ent_feat.cpu().numpy() if self.feat == "M" else None
                    ),
                    "n_base_rel_with_ddi": n_base_rel_save,
                    "n_base_rel_kg": int(self._n_base_rel),
                    "eval_kg_triplets": (
                        np.asarray(eval_tri, dtype=np.int64) if eval_tri is not None else None
                    ),
                },
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "backend": "rspmm",
                "hyperparameters": {
                    "n_dim": self.n_dim,
                    "length": self.length,
                    "feat": self.feat,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
                "environment": {
                    "torch_version": torch.__version__,
                    "sklearn_version": sklearn.__version__,
                },
                "graph_metadata": {
                    "n_ent": int(self._n_ent),
                    "n_base_rel": int(self._n_base_rel),
                    "n_base_rel_with_ddi": n_base_rel_save,
                    "backbone_kg_source": self.backbone_kg_source,
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "_PerModeEmerGNN_RSPMM":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        inst = cls(**hparams)
        with (p / "graph.pkl").open("rb") as f:
            graph = pickle.load(f)
        inst._entity2id = graph["entity2id"]
        inst._n_ent = graph["n_ent"]
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
        inst._model = EmerGNN_RSPMM(
            n_ent=inst._n_ent,
            n_base_rel=int(n_base_rel_with_ddi),
            n_dim=inst.n_dim,
            length=inst.length,
            feat=inst.feat,
            morgan_features=morgan,
        ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        # Persisted eval-KG triplets -> rebuilt lazily on first predict.
        eval_tri = graph.get("eval_kg_triplets")
        inst._eval_kg_triplets = eval_tri
        inst._eval_kg = None
        return inst


__all__ = ["_PerModeEmerGNN_RSPMM"]

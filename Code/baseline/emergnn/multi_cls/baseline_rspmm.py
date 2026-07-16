"""EmerGNN multi-class core, rspmm backend (torchdrug kernel).

rspmm twin of :class:`baseline.emergnn.multi_cls.baseline.EmerGNNMulticlassBaseline`.
Keeps ALL multi-class training behavior identical — same ddi_type vocab, same
``n_base_rel_with_ddi = n_kg_rel + n_classes`` relation-slot expansion, positives-only
training, sum-reduction cross-entropy, macro-F1 best-checkpoint selection, Adam +
ReduceLROnPlateau — and swaps ONLY:
  * the model: :class:`baseline.emergnn.multi_cls.model_rspmm.EmerGNN_MC_RSPMM`;
  * the KG representation: coalesced sparse COO built per epoch from TRIPLETS via
    :func:`baseline.emergnn._rspmm_utils.build_sparse_kg_from_triplets`.
Selected via ``EMERGNN_BACKEND=rspmm``.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.base import write_manifest
from baseline.emergnn._per_mode import N_BASE_REL
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets
from baseline.emergnn.multi_cls.baseline import EmerGNNMulticlassBaseline
from baseline.emergnn.multi_cls.model_rspmm import EmerGNN_MC_RSPMM
from baseline.emergnn.shuffle_utils import shuffle_train

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


class EmerGNNMulticlassBaseline_RSPMM(EmerGNNMulticlassBaseline):
    """rspmm-backend twin of EmerGNNMulticlassBaseline (fused kernel + sparse KG)."""

    VERSION = "1.0-mc-rspmm"

    def _build_kg(self, triplets: np.ndarray, n_rel: int) -> torch.Tensor:
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

        if "ddi_type" not in train.splits.train.columns:
            raise ValueError(
                "Multi-class EmerGNN requires `ddi_type` column in train.splits.train. "
                "Got columns: " + str(list(train.splits.train.columns))
            )
        types = sorted(train.splits.train["ddi_type"].astype(str).unique())
        self._ddi_type_to_idx = {t: i for i, t in enumerate(types)}
        self._idx_to_ddi_type = list(types)
        observed_n = len(types)
        if observed_n != self.n_classes:
            print(
                f"[emergnn_mc-rspmm] WARNING: observed {observed_n} ddi_types in train, "
                f"but n_classes={self.n_classes}. Adjusting n_classes={observed_n}.",
                file=sys.stderr,
            )
            self.n_classes = observed_n

        morgan_mat, drug_id_list = self._setup_graph(train, kg)

        n_kg_rel = self._n_base_rel
        self._n_base_rel_with_ddi = n_kg_rel + self.n_classes

        self._model = EmerGNN_MC_RSPMM(
            n_ent=self._n_ent,
            n_base_rel=self._n_base_rel_with_ddi,
            n_classes=self.n_classes,
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

        pos = train.splits.train.copy()
        a_ids = pos["drug_a_id"].astype(str).map(self._entity2id)
        b_ids = pos["drug_b_id"].astype(str).map(self._entity2id)
        valid_mask = a_ids.notna() & b_ids.notna()
        if not valid_mask.all():
            print(
                f"[emergnn_mc-rspmm] dropping {(~valid_mask).sum()} train rows with unknown drugs",
                file=sys.stderr,
            )
        pos_clean = pos[valid_mask].reset_index(drop=True)
        ddi_idx = pos_clean["ddi_type"].astype(str).map(self._ddi_type_to_idx).to_numpy()
        train_ddi_int = np.stack(
            [
                a_ids[valid_mask].astype(np.int64).to_numpy(),
                b_ids[valid_mask].astype(np.int64).to_numpy(),
                (n_kg_rel + ddi_idx).astype(np.int64),
            ],
            axis=1,
        )

        # Static eval KG (= train_ddi + base_kg) as sparse, built once.
        eval_kg_triplets = np.concatenate([train_ddi_int, self._kg_triplets], axis=0)
        self._eval_kg_triplets = eval_kg_triplets
        self._eval_kg = self._build_kg(eval_kg_triplets, self._n_base_rel_with_ddi)

        rng = np.random.default_rng(0)
        best_val_macro_f1 = -1.0
        best_state: dict | None = None

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints

        n_train = len(pos)
        steps_per_epoch = (n_train + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[emergnn_mc-rspmm] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (self.run_dir / "checkpoints") if self.run_dir is not None else None

        def _eval_dict() -> dict:
            if val is None:
                return {}
            try:
                from my_code.utils.task_eval import eval_multiclass
            except ImportError:
                sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
                from my_code.utils.task_eval import eval_multiclass
            val_pos = val.splits.val_s2[["drug_a_id", "drug_b_id", "ddi_type"]]
            if len(val_pos) == 0:
                return {}
            preds = self.predict_proba(val_pos[["drug_a_id", "drug_b_id"]])
            mask = val_pos["ddi_type"].astype(str).isin(self._ddi_type_to_idx)
            preds = preds[mask.values]
            labels = np.array(
                [self._ddi_type_to_idx[str(t)] for t in val_pos.loc[mask, "ddi_type"]]
            )
            if len(labels) == 0:
                return {}
            m = eval_multiclass(preds, labels, self.n_classes)
            return {"val_top1": m["top1_acc"], "val_macro_f1": m["macro_f1"]}

        def _save_ckpt(tag: str, scope: str) -> None:
            if ckpt_root is None:
                return
            ckpt_path = ckpt_root / f"checkpoint-{tag}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.save(ckpt_path)
            progress.log_save(str(ckpt_path), scope=scope)
            rotate_checkpoints(ckpt_root, self.save_total_limit)

        for epoch in range(self.n_epochs):
            epoch_kg, train_targets = shuffle_train(
                train_ddi_int,
                self._kg_triplets,
                self.shuffle_train_mode,
                ratio=self.shuffle_ratio,
                rng=rng,
                extra_kg_ent=self._kg_entity_set,
            )
            if len(train_targets) == 0:
                print(
                    f"[emergnn_mc-rspmm] epoch {epoch+1}: 0 train targets after "
                    f"shuffle_train (mode={self.shuffle_train_mode}); skip",
                    flush=True,
                )
                continue
            epoch_kg_sparse = self._build_kg(epoch_kg, self._n_base_rel_with_ddi)

            head_all = torch.from_numpy(train_targets[:, 0]).long().to(self.device)
            tail_all = torch.from_numpy(train_targets[:, 1]).long().to(self.device)
            y_all = torch.from_numpy(
                (train_targets[:, 2] - n_kg_rel).astype(np.int64)
            ).long().to(self.device)

            self._model.train()
            progress.epoch_start(epoch)
            n_targets = len(train_targets)
            for start in range(0, n_targets, self.batch_size):
                end = min(n_targets, start + self.batch_size)
                opt.zero_grad(set_to_none=True)
                logits = self._model(head_all[start:end], tail_all[start:end], epoch_kg_sparse)
                loss = F.cross_entropy(logits, y_all[start:end], reduction="sum")
                loss.backward()
                opt.step()
                progress.step(loss.item() / max(end - start, 1))

                if progress.should_eval_step() and val is not None:
                    self._model.eval()
                    metrics = _eval_dict()
                    self._model.train()
                    if metrics:
                        progress.log_eval(metrics, scope="step")
                        if metrics.get("val_macro_f1", -1) > best_val_macro_f1:
                            best_val_macro_f1 = metrics["val_macro_f1"]
                            best_state = copy.deepcopy(self._model.state_dict())
                if progress.should_save_step():
                    _save_ckpt(f"step{progress.step_count}", scope="step")

            extra: dict = {}
            if progress.should_eval_epoch() and val is not None:
                self._model.eval()
                metrics = _eval_dict()
                self._model.train()
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_f1", -1) > best_val_macro_f1:
                        best_val_macro_f1 = metrics["val_macro_f1"]
                        best_state = copy.deepcopy(self._model.state_dict())
                    scheduler.step(metrics.get("val_macro_f1", 0.0))
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}", scope="epoch")
            progress.epoch_end(extra=extra if extra else None)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[emergnn_mc-rspmm] loaded best val_macro_f1={best_val_macro_f1:.4f} state",
                flush=True,
            )

    @torch.no_grad()
    def predict_proba(
        self,
        pairs,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EmerGNNMulticlassBaseline_RSPMM.fit() must be called before predict_proba.")
        if getattr(self, "_eval_kg", None) is None:
            tri = getattr(self, "_eval_kg_triplets", None)
            if tri is None:
                raise RuntimeError("no eval KG available; fit() must run before predict.")
            self._eval_kg = self._build_kg(tri, self._n_base_rel_with_ddi)
        self._model.eval()
        out = np.empty((len(pairs), self.n_classes), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            head, tail = self._pair_indices(batch)
            head = head.to(self.device)
            tail = tail.to(self.device)
            logits = self._model(head, tail, self._eval_kg)
            out[start : start + len(batch)] = F.softmax(logits, dim=-1).cpu().numpy()
        return out

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._entity2id is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        n_base_rel_save = int(
            getattr(self, "_n_base_rel_with_ddi", None)
            or (self._n_base_rel + self.n_classes)
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
                    "ddi_type_to_idx": self._ddi_type_to_idx,
                    "idx_to_ddi_type": self._idx_to_ddi_type,
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
                "task": "multiclass",
                "backend": "rspmm",
                "n_classes": self.n_classes,
                "hyperparameters": {
                    "n_dim": self.n_dim,
                    "length": self.length,
                    "feat": self.feat,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                    "backbone_kg_source": self.backbone_kg_source,
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
    def load(cls, path: "Path | str") -> "EmerGNNMulticlassBaseline_RSPMM":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        n_classes = manifest.get("n_classes", 86)
        inst = cls(n_classes=n_classes, **hparams)
        with (p / "graph.pkl").open("rb") as f:
            graph = pickle.load(f)
        inst._entity2id = graph["entity2id"]
        inst._n_ent = graph["n_ent"]
        inst._ddi_type_to_idx = graph["ddi_type_to_idx"]
        inst._idx_to_ddi_type = graph["idx_to_ddi_type"]
        n_base_rel_kg = (
            graph.get("n_base_rel_kg")
            or manifest.get("graph_metadata", {}).get("n_base_rel")
            or N_BASE_REL
        )
        inst._n_base_rel = int(n_base_rel_kg)
        n_base_rel_with_ddi = (
            graph.get("n_base_rel_with_ddi")
            or manifest.get("graph_metadata", {}).get("n_base_rel_with_ddi")
            or (int(n_base_rel_kg) + int(n_classes))
        )
        inst._n_base_rel_with_ddi = int(n_base_rel_with_ddi)
        morgan = graph.get("morgan_features")
        inst._model = EmerGNN_MC_RSPMM(
            n_ent=inst._n_ent,
            n_base_rel=int(n_base_rel_with_ddi),
            n_classes=n_classes,
            n_dim=inst.n_dim,
            length=inst.length,
            feat=inst.feat,
            morgan_features=morgan,
        ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        inst._eval_kg_triplets = graph.get("eval_kg_triplets")
        inst._eval_kg = None
        return inst


__all__ = ["EmerGNNMulticlassBaseline_RSPMM"]

"""EmerGNN multi-class variant — DrugBank 86 DDI types.

NOTE — paper-vs-port task formulation:
  * This variant **matches the original paper** (Zhang et al., Nat Comp
    Sci 2023) on DrugBank: per-pair multi-class over 86 interaction
    types, softmax CE (paper uses softmax-margin which is mathematically
    equivalent), metrics = macro F1 + ACC + Cohen κ.
  * The companion ``baseline.py`` (binary EmerGNNBaseline) is a
    ColdDDI-aligned simplification, NOT the paper's task.

Lifts the task formulation from the original paper (LARS-research/EmerGNN):
  - Model output dim = n_classes (86 for DrugBank)
  - Training loss = cross-entropy on ground-truth ddi_type
  - Eval = top-k acc + macro F1 + macro AUC (via task_eval.eval_multiclass)
  - Trains on POSITIVE pairs only (no drug-replacement negatives; multi-class
    setup doesn't need a "no-interaction" class — task is 'which interaction type')

Inherits binary EmerGNNBaseline; overrides:
  - __init__ adds n_classes
  - _setup_graph: builds ddi_type_to_idx mapping
  - fit: cross-entropy on ddi_type-index labels, positives only
  - predict_proba: returns (n_pairs, n_classes) softmax probabilities
  - save/load: persists ddi_type_to_idx + n_classes

Reference: EmerGNN/DrugBank/base_model.py uses softmax margin loss; we use
plain cross-entropy as it's equivalent (up to scaling) and simpler to reproduce.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.base import register, write_manifest
# Per-mode binary trainer lives at ``baseline/emergnn/_per_mode.py`` after
# the 2026-05-20 promotion of the multimode wrapper to ``binary_cls/``.
# Multi-class inherits from the per-mode class (not from the multimode
# wrapper) because the K-way head replaces the entire training loop
# anyway — multimode dispatch is binary-task-specific.
from baseline.emergnn._per_mode import (
    _PerModeEmerGNN as EmerGNNBaseline,  # keep the alias so the rest of
                                          # this file's references (which
                                          # inherit and call .super()) keep
                                          # working without churn.
    N_BASE_REL,
    _drug_smiles_dict,
    _kg_to_kb_dict,
)
from baseline.emergnn.kg_builder import (
    build_kg_from_kb,
    build_sparse_adj,
    edges_as_dense_lists,
)
from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet
from baseline.emergnn.morgan_features import compute_morgan_matrix
from baseline.emergnn.multi_cls.model import EmerGNN_MC
from baseline.emergnn.shuffle_utils import (
    build_edge_lists_from_triplets,
    shuffle_train,
)

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


@register("emergnn_mc")
class EmerGNNMulticlassBaseline(EmerGNNBaseline):
    """EmerGNN multi-class (86 DrugBank DDI types).

    Hyper-params identical to binary parent + `n_classes`.
    """

    VERSION = "1.0-mc"

    def __init__(
        self,
        *,
        n_classes: int = 86,
        # Paper-faithful additions (CLAUDE.md §baseline-from-reproduction):
        # - weight_decay: paper hyperopt best on DrugBank S1/S2 = 1e-8
        # - shuffle_train_mode: which inductive simulation per epoch
        #   ('S0' = random 80/20; 'S1' = one-new targets; 'S2' = two-new targets).
        #   Default 'S2' = hardest cold-start simulation (covers S1 capability).
        # - shuffle_ratio: paper default 0.8 (keep 80% as old, 20% as emerging)
        # - n_epochs: paper hyperopt S1/S2 best = 100. Pin explicitly here
        #   so future binary-parent default changes don't silently drift this.
        weight_decay: float = 1e-8,
        shuffle_train_mode: str = "S2",
        shuffle_ratio: float = 0.8,
        n_epochs: int = 100,
        **kw,
    ) -> None:
        super().__init__(n_epochs=n_epochs, **kw)
        self.n_classes = int(n_classes)
        self.weight_decay = float(weight_decay)
        self.shuffle_train_mode = str(shuffle_train_mode)
        self.shuffle_ratio = float(shuffle_ratio)
        self._ddi_type_to_idx: dict[str, int] | None = None
        self._idx_to_ddi_type: list[str] | None = None
        # n_base_rel WITH DDI relations included (= self._n_base_rel + n_classes).
        # Set in fit() after entity2id is built so we can offset DDI relation
        # IDs past the existing KG relation IDs.
        self._n_base_rel_with_ddi: int | None = None
        # base KG triplets as numpy int64 (h, t, r), saved by parent
        # _setup_graph as self._kg_triplets — used by per-epoch shuffle_train.

    # ------------------------------------------------------------------
    # Override fit: multi-class training
    # ------------------------------------------------------------------
    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        if kg is None:
            kg = train.kg

        # Build ddi_type vocab from training positives
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
                f"[emergnn_mc] WARNING: observed {observed_n} ddi_types in train, "
                f"but n_classes={self.n_classes}. Adjusting n_classes={observed_n}.",
                file=sys.stderr,
            )
            self.n_classes = observed_n

        # Setup KG + entity vocab + morgan (same as parent's _setup_graph,
        # but we construct EmerGNN_MC instead of EmerGNN)
        morgan_mat, drug_id_list = self._setup_graph(train, kg)

        # Paper-faithful: expand the model's relation vocab to include all
        # n_classes DDI relations (offset past existing KG relations). This
        # lets per-epoch ``shuffle_train`` add positive DDI fact triplets
        # back into the KG with their own relation embeddings (paper §Methods
        # ``kg_triplets = concat(fact_triplet, train_kg)``).
        n_kg_rel = self._n_base_rel
        self._n_base_rel_with_ddi = n_kg_rel + self.n_classes

        # Override: re-init self._model as EmerGNN_MC with EXPANDED n_base_rel
        self._model = EmerGNN_MC(
            n_ent=self._n_ent,
            n_base_rel=self._n_base_rel_with_ddi,
            n_classes=self.n_classes,
            n_dim=self.n_dim,
            length=self.length,
            feat=self.feat,
            morgan_features=morgan_mat if self.feat == "M" else None,
        ).to(self.device)

        # Paper base_model.py: Adam(lr, weight_decay=lamb) + ReduceLROnPlateau
        # (mode='max', factor=0.1, patience=10 PyTorch defaults). Paper
        # evaluate.py S1/S2 override: lamb=1e-8.
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Paper base_model.py steps ReduceLROnPlateau once per
        # ``epoch_per_test`` (=5) epochs with ``patience=10`` → effective
        # tolerance ≈ 50 raw epochs. We step every epoch, so scale
        # ``patience`` by 5 to preserve the same effective tolerance.
        scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=50)

        pos = train.splits.train.copy()
        # Ensure ddi_type present in pos
        if "ddi_type" not in pos.columns:
            raise ValueError("train.splits.train missing 'ddi_type' for multi-class fit")

        # ── Build train_ddi triplets in (entity_id_a, entity_id_b, n_kg_rel + ddi_idx) form ─
        # Drop any rows whose drug ids aren't in entity2id (shouldn't happen
        # post-_setup_graph, but defensive).
        a_ids = pos["drug_a_id"].astype(str).map(self._entity2id)
        b_ids = pos["drug_b_id"].astype(str).map(self._entity2id)
        valid_mask = a_ids.notna() & b_ids.notna()
        if not valid_mask.all():
            print(
                f"[emergnn_mc] dropping {(~valid_mask).sum()} train rows with unknown drugs",
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

        # Static eval KG = train_ddi + base_kg (built once, used for val/test).
        # Matches paper convention vKG = train_ddi + train_kg (cumulative).
        eval_kg_triplets = np.concatenate([train_ddi_int, self._kg_triplets], axis=0)
        eval_src, eval_dst, eval_rel = build_edge_lists_from_triplets(
            eval_kg_triplets, self._n_ent, self._n_base_rel_with_ddi
        )
        self._eval_edges = (
            torch.from_numpy(eval_src).long().to(self.device),
            torch.from_numpy(eval_dst).long().to(self.device),
            torch.from_numpy(eval_rel).long().to(self.device),
        )

        rng = np.random.default_rng(0)

        # Paper selects best ckpt by macro F1 (the primary metric reported in
        # Table 1), NOT top-1 accuracy.
        best_val_macro_f1 = -1.0
        best_state: dict | None = None

        # Training progress logger (shared helper per CLAUDE.md)
        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints

        n_train = len(pos)
        steps_per_epoch = (n_train + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[emergnn_mc] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (self.run_dir / "checkpoints") if self.run_dir is not None else None

        def _eval_dict() -> dict:
            if val is None:
                return {}
            # use val_s2 positives for cold-start val
            try:
                from my_code.utils.task_eval import eval_multiclass
            except ImportError:
                sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
                from my_code.utils.task_eval import eval_multiclass
            val_pos = val.splits.val_s2[["drug_a_id", "drug_b_id", "ddi_type"]]
            if len(val_pos) == 0:
                return {}
            preds = self.predict_proba(val_pos[["drug_a_id", "drug_b_id"]])
            # Skip val rows whose ddi_type was not in train vocab (OOV)
            mask = val_pos["ddi_type"].astype(str).isin(self._ddi_type_to_idx)
            preds = preds[mask.values]
            labels = np.array(
                [self._ddi_type_to_idx[str(t)] for t in val_pos.loc[mask, "ddi_type"]]
            )
            if len(labels) == 0:
                return {}
            m = eval_multiclass(preds, labels, self.n_classes)
            return {
                "val_top1": m["top1_acc"],
                "val_macro_f1": m["macro_f1"],  # paper's primary metric
            }

        def _save_ckpt(tag: str, scope: str) -> None:
            if ckpt_root is None:
                return
            ckpt_path = ckpt_root / f"checkpoint-{tag}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.save(ckpt_path)
            progress.log_save(str(ckpt_path), scope=scope)
            rotate_checkpoints(ckpt_root, self.save_total_limit)

        for epoch in range(self.n_epochs):
            # ── Paper-faithful per-epoch shuffle_train (CLAUDE.md spec) ──
            # Splits train_ddi into fact (added to KG) vs train_targets
            # (prediction labels) under S2 inductive simulation by default.
            # Without this, training degenerates to transductive.
            epoch_kg, train_targets = shuffle_train(
                train_ddi_int,
                self._kg_triplets,
                self.shuffle_train_mode,
                ratio=self.shuffle_ratio,
                rng=rng,
                # Paper: process_files_kg unions all 3 KG splits when
                # populating ``ddi_in_kg``. We mirror that with the full
                # static KG entity set computed in _setup_graph().
                extra_kg_ent=self._kg_entity_set,
            )
            if len(train_targets) == 0:
                print(
                    f"[emergnn_mc] epoch {epoch+1}: 0 train targets after "
                    f"shuffle_train (mode={self.shuffle_train_mode}); skip",
                    flush=True,
                )
                continue
            # Build this epoch's KG edges (forward + reverse + self-loop)
            esrc, edst, erel = build_edge_lists_from_triplets(
                epoch_kg, self._n_ent, self._n_base_rel_with_ddi
            )
            edge_src = torch.from_numpy(esrc).long().to(self.device)
            edge_dst = torch.from_numpy(edst).long().to(self.device)
            edge_rel = torch.from_numpy(erel).long().to(self.device)

            # Paper does NOT extra-shuffle targets after shuffle_train —
            # the per-epoch randomness comes from shuffle_train itself.
            # Targets' relation column is in expanded space (n_kg_rel + idx);
            # un-offset for CE loss labels (which expect 0..n_classes-1).
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
                logits = self._model(
                    head_all[start:end],
                    tail_all[start:end],
                    edge_src,
                    edge_dst,
                    edge_rel,
                )
                # Paper base_model.py: loss = loss.sum() then backward
                # (NOT mean). Equivalent to reduction='sum' for CE.
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
                        # Paper selects best by macro F1 (primary metric)
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
                    # Paper selects best by macro F1
                    if metrics.get("val_macro_f1", -1) > best_val_macro_f1:
                        best_val_macro_f1 = metrics["val_macro_f1"]
                        best_state = copy.deepcopy(self._model.state_dict())
                    # Paper base_model.py: scheduler.step(v_f1) — step on val F1
                    scheduler.step(metrics.get("val_macro_f1", 0.0))
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}", scope="epoch")
            progress.epoch_end(extra=extra if extra else None)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[emergnn_mc] loaded best val_macro_f1={best_val_macro_f1:.4f} state",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Override predict_proba: returns (n_pairs, n_classes)
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        pairs: "pd.DataFrame",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("EmerGNNMulticlassBaseline.fit() must be called before predict_proba.")
        # Use the static eval KG (= train_ddi + base_kg) per paper convention.
        # Falls back to base-KG-only if fit() didn't set up _eval_edges (e.g.
        # after load() without re-fit).
        if hasattr(self, "_eval_edges") and self._eval_edges is not None:
            edge_src, edge_dst, edge_rel = self._eval_edges
        else:
            edge_src, edge_dst, edge_rel = self._edges_on_device()
        self._model.eval()
        out = np.empty((len(pairs), self.n_classes), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs.iloc[start : start + self.batch_size]
                head, tail = self._pair_indices(batch)
                head = head.to(self.device)
                tail = tail.to(self.device)
                logits = self._model(head, tail, edge_src, edge_dst, edge_rel)
                # logits shape (B, n_classes); apply softmax for probability
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                out[start : start + len(batch)] = probs
        return out

    # ------------------------------------------------------------------
    # Override save: persist ddi_type vocab + n_classes
    # ------------------------------------------------------------------
    def save(self, path: "Path | str") -> None:
        if self._model is None or self._entity2id is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        # The model was built with ``n_base_rel = kg_rel + n_classes``;
        # falling back to the raw KG count corrupts load() with a shape
        # mismatch on Wr.
        n_base_rel_save = int(
            getattr(self, "_n_base_rel_with_ddi", None)
            or (self._n_base_rel + self.n_classes)
        )
        eval_edges_np = None
        if getattr(self, "_eval_edges", None) is not None:
            eval_edges_np = {
                "src": self._eval_edges[0].detach().cpu().numpy(),
                "dst": self._eval_edges[1].detach().cpu().numpy(),
                "rel": self._eval_edges[2].detach().cpu().numpy(),
            }
        with (out / "graph.pkl").open("wb") as f:
            pickle.dump(
                {
                    "entity2id": self._entity2id,
                    "n_ent": self._n_ent,
                    "edge_src": self._edge_src,
                    "edge_dst": self._edge_dst,
                    "edge_rel": self._edge_rel,
                    "morgan_features": (
                        self._model.ent_feat.cpu().numpy() if self.feat == "M" else None
                    ),
                    "ddi_type_to_idx": self._ddi_type_to_idx,
                    "idx_to_ddi_type": self._idx_to_ddi_type,
                    "n_base_rel_with_ddi": n_base_rel_save,
                    "n_base_rel_kg": int(self._n_base_rel),
                    "eval_edges": eval_edges_np,
                },
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "task": "multiclass",
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
    def load(cls, path: "Path | str") -> "EmerGNNMulticlassBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        n_classes = manifest.get("n_classes", 86)
        inst = cls(n_classes=n_classes, **hparams)
        with (p / "graph.pkl").open("rb") as f:
            graph = pickle.load(f)
        inst._entity2id = graph["entity2id"]
        inst._n_ent = graph["n_ent"]
        inst._edge_src = graph["edge_src"]
        inst._edge_dst = graph["edge_dst"]
        inst._edge_rel = graph["edge_rel"]
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
        inst._model = EmerGNN_MC(
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
        eval_edges = graph.get("eval_edges")
        if eval_edges is not None:
            inst._eval_edges = (
                torch.from_numpy(eval_edges["src"]).long().to(inst.device),
                torch.from_numpy(eval_edges["dst"]).long().to(inst.device),
                torch.from_numpy(eval_edges["rel"]).long().to(inst.device),
            )
        else:
            inst._eval_edges = None
        return inst

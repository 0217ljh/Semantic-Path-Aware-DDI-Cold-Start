"""TIGER multi-class variant — DrugBank 86 DDI types.

NOTE — paper-vs-port task formulation:
  * Original paper (Su et al., AAAI 2024) is **binary** classification
    (ACC/F1/AUC/AUPR with random complement negatives). The 86 types
    of DrugBank DDIs are NOT addressed in the paper as a prediction
    target.
  * THIS multi-class variant is an EXTENSION we added by parameterizing
    ``fc2``'s final ``Linear(512, n_classes)`` (the released Blair1213
    code hard-codes ``Linear(512, 2)``). Training uses softmax CE on
    ddi_type index, positives only. If reporting numbers from this
    variant, label it explicitly as "TIGER (multi-class adaptation)".

Lifts the task formulation:
  - Model output dim = ``n_classes`` (86 for DrugBank). TIGER's ``fc2``
    final layer is already parameterized via ``n_classes`` in
    :class:`baseline.tiger.model.TIGER`, so there is no
    separate ``model_multiclass.py`` — we just construct the same
    :class:`TIGER` with ``n_classes=86``.
  - Training loss = cross-entropy on ground-truth ``ddi_type`` index
    (computed inside the model's ``forward`` via
    ``nll_loss(log_softmax(score), y)``).
  - Training data: POSITIVE pairs only (no drug-replacement negatives
    needed because the model predicts WHICH relation, not yes/no).
  - Eval: top-k acc + macro F1 + macro AUC via
    :func:`my_code.utils.task_eval.eval_multiclass`.

Inherits binary :class:`TIGERBaseline`; overrides:
  - ``__init__`` adds ``n_classes``.
  - ``fit``: builds the ddi_type vocab, constructs ``TIGER(n_classes=K)``,
    trains on positives only with CE on ddi_type indices.
  - ``predict_proba``: returns ``(n_pairs, n_classes)`` softmax probs
    (binary parent took ``[:, 1]``).
  - ``save`` / ``load``: persists ``ddi_type_to_idx`` + ``n_classes``.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch_geometric.data import Batch

from baseline.base import register, write_manifest
from baseline.tiger.binary_cls.baseline import TIGERBaseline
from baseline.tiger.mol_features import ATOM_FEATURE_DIM
from baseline.tiger.model import TIGER

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


@register("tiger_mc")
class TIGERMulticlassBaseline(TIGERBaseline):
    """TIGER multi-class (DrugBank 86 DDI types) — same dual-channel
    architecture as the binary parent, only the output head + loss
    differ.
    """

    VERSION = "2.0-mc"

    def __init__(self, *, n_classes: int = 86, **kw) -> None:
        super().__init__(**kw)
        self.n_classes = int(n_classes)
        self._ddi_type_to_idx: dict[str, int] | None = None
        self._idx_to_ddi_type: list[str] | None = None

    # ------------------------------------------------------------------
    # Override fit: positives-only, ddi_type labels, CE loss
    # ------------------------------------------------------------------

    def fit(
        self,
        train: "PairDataset",
        val: "PairDataset | None" = None,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        if "ddi_type" not in train.splits.train.columns:
            raise ValueError(
                "Multi-class TIGER requires `ddi_type` column in train.splits.train."
            )
        types = sorted(train.splits.train["ddi_type"].astype(str).unique())
        self._ddi_type_to_idx = {t: i for i, t in enumerate(types)}
        self._idx_to_ddi_type = list(types)
        observed_n = len(types)
        if observed_n != self.n_classes:
            print(
                f"[tiger_mc] WARNING: observed {observed_n} ddi_types, "
                f"but n_classes={self.n_classes}. Adjusting n_classes={observed_n}.",
                file=sys.stderr,
            )
            self.n_classes = observed_n

        # ── 1. SMILES graphs ──────────────────────────────────────────
        mol_rel_required = self._build_mol_graphs(train)
        if self.num_relations_mol is None:
            num_rel_mol = max(mol_rel_required + 4, 32)
        else:
            num_rel_mol = self.num_relations_mol
            if num_rel_mol < mol_rel_required:
                raise ValueError(
                    f"num_relations_mol={num_rel_mol} but observed "
                    f"sp_edge_rel requires >= {mol_rel_required}"
                )
        self._effective_num_rel_mol = num_rel_mol

        # ── 2. BKG + per-drug subgraphs (only when dual-channel) ──────
        if self.mol_only:
            num_rel_graph = 2
            n_total_nodes = 2
            max_deg_node = 2
        else:
            graph_rel_required, max_deg_node_obs, n_total_nodes = (
                self._build_bkg_and_subgraphs(train, verbose=True)
            )
            if self.num_relations_graph is None:
                num_rel_graph = max(graph_rel_required + 4, 32)
            else:
                num_rel_graph = self.num_relations_graph
                if num_rel_graph < graph_rel_required:
                    raise ValueError(
                        f"num_relations_graph={num_rel_graph} but observed "
                        f"max sp_rel requires >= {graph_rel_required}"
                    )
            max_deg_node = max(self.max_degree_node, max_deg_node_obs + 1)
        self._effective_num_rel_graph = num_rel_graph

        # ── 3. Build model with n_classes=K ───────────────────────────
        self._model = TIGER(
            max_layer=self.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=n_total_nodes,
            num_relations_mol=num_rel_mol,
            num_relations_graph=num_rel_graph,
            output_dim=self.output_dim,
            max_degree_graph=self.max_degree_graph,
            max_degree_node=max_deg_node,
            sub_coeff=self.sub_coeff,
            mi_coeff=self.mi_coeff,
            dropout=self.dropout,
            device=self.device,
            mol_only=self.mol_only,
            n_classes=self.n_classes,
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        pos = train.splits.train.copy()
        # Map ddi_type strings to class indices (drop any rows with
        # OOV ddi_type — shouldn't happen for train since vocab is from
        # train itself, but defensive)
        pos = pos[pos["ddi_type"].astype(str).isin(self._ddi_type_to_idx)].reset_index(
            drop=True
        )
        pos["_class_idx"] = pos["ddi_type"].astype(str).map(self._ddi_type_to_idx)

        best_val_top1 = -1.0
        best_state: dict | None = None

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
            from my_code.utils.task_eval import eval_multiclass
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
            from my_code.utils.task_eval import eval_multiclass

        n_train = len(pos)
        steps_per_epoch = (n_train + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[tiger_mc] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (
            (self.run_dir / "checkpoints") if self.run_dir is not None else None
        )

        def _eval_dict() -> dict:
            if val is None:
                return {}
            val_pos = val.splits.val_s2[["drug_a_id", "drug_b_id", "ddi_type"]]
            if len(val_pos) == 0:
                return {}
            preds = self.predict_proba(val_pos[["drug_a_id", "drug_b_id"]])
            mask = val_pos["ddi_type"].astype(str).isin(self._ddi_type_to_idx)
            preds = preds[mask.values]
            labels = np.array(
                [
                    self._ddi_type_to_idx[str(t)]
                    for t in val_pos.loc[mask, "ddi_type"]
                ]
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
            shuffled = pos.sample(frac=1, random_state=epoch).reset_index(drop=True)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled.iloc[start : start + self.batch_size]
                labels = batch["_class_idx"].to_numpy(dtype=np.int64)
                mol1, sub1, mol2, sub2, idx1, idx2, mask = self._make_pair_batch(
                    batch[["drug_a_id", "drug_b_id"]], labels=labels
                )
                if mol1 is None:
                    continue
                opt.zero_grad(set_to_none=True)
                # Loss is computed inside the model's forward
                # (nll_loss on log_softmax of the K-dim score).
                _probs, loss = self._forward(mol1, sub1, mol2, sub2, idx1, idx2)
                loss.backward()
                opt.step()
                progress.step(loss.item())

                if progress.should_eval_step() and val is not None:
                    self._model.eval()
                    metrics = _eval_dict()
                    self._model.train()
                    if metrics:
                        progress.log_eval(metrics, scope="step")
                        if metrics.get("val_top1", -1) > best_val_top1:
                            best_val_top1 = metrics["val_top1"]
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
                    if metrics.get("val_top1", -1) > best_val_top1:
                        best_val_top1 = metrics["val_top1"]
                        best_state = copy.deepcopy(self._model.state_dict())
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}", scope="epoch")
            progress.epoch_end(extra=extra if extra else None)

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[tiger_mc] loaded best val_top1={best_val_top1:.4f} state",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Override predict_proba: returns full (n_pairs, n_classes) softmax
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(
        self,
        pairs: pd.DataFrame,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError(
                "TIGERMulticlassBaseline must be fitted (or loaded) before prediction."
            )
        self._model.eval()
        out = np.zeros((len(pairs), self.n_classes), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start : start + self.batch_size]
            zero_labels = np.zeros(len(batch), dtype=np.int64)
            mol1, sub1, mol2, sub2, idx1, idx2, mask = self._make_pair_batch(
                batch, labels=zero_labels
            )
            if mol1 is None:
                continue
            probs, _loss = self._forward(mol1, sub1, mol2, sub2, idx1, idx2)
            probs = probs.detach().cpu().numpy()  # (B_kept, n_classes)
            kept_idx_in_batch = np.where(mask)[0]
            for i, kept_i in enumerate(kept_idx_in_batch):
                out[start + kept_i] = probs[i]
        return out

    # ------------------------------------------------------------------
    # Override save / load: persist ddi_type vocab + n_classes
    # ------------------------------------------------------------------

    def save(self, path: "Path | str") -> None:
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump(
                {
                    "mol_graphs": self._mol_graphs,
                    "mol_missing": self._mol_missing,
                    "subgraphs": self._subgraphs,
                    "drug_to_idx": self._drug_to_idx,
                    "unseen_ids": list(self._unseen_ids),
                    "n_total_nodes": (
                        self._bkg["n_total_nodes"] if self._bkg else 2
                    ),
                    "num_rel_graph": self._effective_num_rel_graph,
                    "ddi_type_to_idx": self._ddi_type_to_idx,
                    "idx_to_ddi_type": self._idx_to_ddi_type,
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
                    "max_layer": self.max_layer,
                    "output_dim": self.output_dim,
                    "max_degree_graph": self.max_degree_graph,
                    "max_degree_node": self.max_degree_node,
                    "num_relations_mol": self._effective_num_rel_mol,
                    "num_relations_graph": self._effective_num_rel_graph,
                    "sub_coeff": self.sub_coeff,
                    "mi_coeff": self.mi_coeff,
                    "dropout": self.dropout,
                    "mol_only": self.mol_only,
                    "kg_source": self.kg_source,
                    "rw_num_walks": self.rw_num_walks,
                    "rw_walk_length": self.rw_walk_length,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "TIGERMulticlassBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        n_classes = manifest.get("n_classes", 86)
        inst = cls(n_classes=n_classes, **hparams)
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._mol_graphs = payload["mol_graphs"]
        inst._mol_missing = payload.get("mol_missing", [])
        inst._subgraphs = payload.get("subgraphs")
        inst._drug_to_idx = payload.get("drug_to_idx")
        inst._unseen_ids = set(payload.get("unseen_ids", []))
        inst._ddi_type_to_idx = payload["ddi_type_to_idx"]
        inst._idx_to_ddi_type = payload["idx_to_ddi_type"]
        inst._effective_num_rel_mol = hparams.get("num_relations_mol")
        inst._effective_num_rel_graph = hparams.get("num_relations_graph")
        n_total_nodes = payload.get("n_total_nodes", 2)
        inst._model = TIGER(
            max_layer=inst.max_layer,
            num_features_drug=ATOM_FEATURE_DIM,
            num_nodes=n_total_nodes,
            num_relations_mol=inst._effective_num_rel_mol,
            num_relations_graph=inst._effective_num_rel_graph,
            output_dim=inst.output_dim,
            max_degree_graph=inst.max_degree_graph,
            max_degree_node=inst.max_degree_node,
            sub_coeff=inst.sub_coeff,
            mi_coeff=inst.mi_coeff,
            dropout=inst.dropout,
            device=inst.device,
            mol_only=inst.mol_only,
            n_classes=n_classes,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst

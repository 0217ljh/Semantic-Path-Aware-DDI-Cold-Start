"""HDN-DDI multi-class baseline — DrugBank K-way DDI type prediction.

NOTE — paper-vs-port task formulation:
  * Original paper (Sun & Zheng, BMC Bioinformatics 2025) does
    **per-triplet binary** classification — "does (drug_a, rel, drug_b)
    exist?" — with sigmoid + BCE. The K relations are INPUTS to RESCAL
    scoring, NOT a prediction target.
  * THIS multi-class variant is an EXTENSION we added: predict which of
    the K relations is correct for a given pair, softmax CE on
    ddi_type index, positives only, top-k acc / macro F1 metrics. This
    formulation is NOT in the original paper. If you report numbers
    from this variant, label it explicitly as "HDN-DDI (multi-class
    adaptation)".

**Paper core contributions preserved** (per CLAUDE.md §"从 reproduction
派生 baseline 的 review 规范":  "不允许借口'新任务'省略 paper 算法核心"):
  1. refined BRICS substructure extraction via
     :func:`baseline.hdn_ddi.mol_features.build_drug_graphs`
  2. 3-level hierarchical molecular graph (atoms y=0, frag y=1, super y=2)
  3. Bipartite inter-drug graph restricted to y==1 substructure nodes
     via :func:`baseline.hdn_ddi._shared.bipartite_edge_index_y1`
  4. LambdaLR(0.96 ** epoch) scheduler

**Task-relevant changes** (allowed per CLAUDE.md):
  - Output head: per-pair (B, n_classes) softmax scores (vs binary's
    per-pair scalar) via :class:`HDN_DDI_MC` in :mod:`.model`
  - Loss: cross-entropy on ddi_type index (vs BCE)
  - Eval: top-k acc + macro F1 (vs binary's ROC-AUC)
  - Best ckpt selected on val_macro_f1 (vs binary's val_auc) — matches
    paper's main K-way metric per CLAUDE.md §"Baseline 规范" §4 case A
  - Trains on POSITIVE pairs only (multi-class, no neg needed)

Inherits :class:`HDNDDIBaseline` (the BRICS-aware binary baseline); only
overrides ``fit`` / ``predict_proba`` / ``save`` / ``load`` to swap in
the multi-class head and metrics.
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
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR

from baseline.base import register, write_manifest
from baseline.hdn_ddi.binary_cls.baseline import HDNDDIBaseline
from baseline.hdn_ddi.multi_cls.model import HDN_DDI_MC

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


@register("hdn_ddi_mc")
class HDNDDIMulticlassBaseline(HDNDDIBaseline):
    """HDN-DDI multi-class (K DrugBank DDI types), BRICS-aware.

    All BRICS-aware contributions (3-level graph, 66-dim features,
    y==1 bipartite, LambdaLR scheduler) are inherited from
    :class:`HDNDDIBaseline`.  Only the task surface (head dim, loss,
    metrics, best-ckpt selection) is overridden here.
    """

    VERSION = "2.0-mc"  # paper-faithful BRICS-aware multi-class (replaces 1.0 flat)

    def __init__(self, *, n_classes: int = 86, **kw) -> None:
        super().__init__(**kw)
        self.n_classes = int(n_classes)
        self._ddi_type_to_idx: dict[str, int] | None = None
        self._idx_to_ddi_type: list[str] | None = None

    # ------------------------------------------------------------------
    # Override fit: multi-class training (BRICS-aware via inherited helpers)
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
                "Multi-class HDN-DDI requires `ddi_type` column in train.splits.train."
            )
        # Fail-fast on missing ddi_type in training rows. If the upstream
        # `_augment_splits_with_ddi_type` couldn't recover ddi_type for
        # some train pairs, leaving NaN would silently degrade `astype(str)`
        # into a fake class like "nan" / "None" and pollute the K-way
        # vocabulary (codex review 2026-05-18). Refuse to proceed.
        train_ddi = train.splits.train["ddi_type"]
        n_missing = int(train_ddi.isna().sum())
        if n_missing > 0:
            raise ValueError(
                f"Multi-class HDN-DDI: {n_missing} / {len(train_ddi)} train "
                f"rows have missing ddi_type. Most likely "
                f"`_augment_splits_with_ddi_type` couldn't resolve these pairs "
                f"from ddi_edges.csv. Fix upstream or filter before passing in."
            )
        # Build ddi_type vocab from the training split (positives only).
        types = sorted(train_ddi.astype(str).unique())
        self._ddi_type_to_idx = {t: i for i, t in enumerate(types)}
        self._idx_to_ddi_type = list(types)
        observed_n = len(types)
        if observed_n != self.n_classes:
            print(
                f"[hdn_ddi_mc] WARNING: observed {observed_n} ddi_types, "
                f"but n_classes={self.n_classes}. Adjusting n_classes={observed_n}.",
                file=sys.stderr,
            )
            self.n_classes = observed_n

        # Build BRICS-aware per-drug graphs. Delegate to inherited
        # ``_build_graphs`` so the ``mol_pkl_path`` branch (CLAUDE.md
        # §"Baseline 规范" §2 step 3: auto-detect + auto-build via local
        # `_data/necessary/build_hierarchical_pkl.py`) is honoured
        # exactly like the binary baseline. Previously this body did its
        # own ``build_drug_graphs(smiles)`` and ignored ``mol_pkl_path``
        # — codex round-1 finding 2026-05-17.
        self._build_graphs(train)

        # Multi-class model: same encoder as binary, K-way RESCAL einsum head.
        self._model = HDN_DDI_MC(
            in_features=self.in_features,
            hidd_dim=self.hidd_dim,
            kge_dim=self.kge_dim,
            rel_total=self.n_classes,
            heads_out_feat_params=list(self.heads_out_feat_params),
            blocks_params=list(self.blocks_params),
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Paper §Parameters: LambdaLR(0.96^epoch).  Same scheduler as the
        # binary variant; preserved because paper algorithm core is
        # optimizer-side, not task-side.
        scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 0.96 ** epoch)

        pos = train.splits.train.copy()
        best_val_macro_f1 = -1.0
        best_state: dict | None = None

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
            prefix="[hdn_ddi_mc] ",
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
                sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
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
            shuffled = pos.sample(frac=1, random_state=epoch).reset_index(drop=True)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled.iloc[start : start + self.batch_size]
                # `_make_pair_batch` is inherited from HDNDDIBaseline and
                # already builds y==1-filtered bipartite via
                # `_bipartite_edge_index_y1` -- so BRICS-aware path is
                # automatic here.
                h_b, t_b, rels_dummy, b_b, mask = self._make_pair_batch(batch)
                if h_b is None:
                    continue
                h_b = h_b.to(self.device)
                t_b = t_b.to(self.device)
                rels_dummy = rels_dummy.to(self.device)  # ignored by HDN_DDI_MC
                b_b = b_b.to(self.device)
                y = torch.tensor(
                    [self._ddi_type_to_idx[str(t)] for t in batch.loc[mask, "ddi_type"]],
                    dtype=torch.long,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                logits = self._model((h_b, t_b, rels_dummy, b_b))  # (B_kept, n_classes)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                progress.step(loss.item())

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
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}", scope="epoch")
            progress.epoch_end(extra=extra if extra else None)
            scheduler.step()

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[hdn_ddi_mc] loaded best val_macro_f1={best_val_macro_f1:.4f} state",
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
        if self._model is None or self._graphs is None:
            raise RuntimeError(
                "HDNDDIMulticlassBaseline.fit() must be called before predict_proba."
            )
        self._model.eval()
        out = np.zeros((len(pairs), self.n_classes), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs.iloc[start : start + self.batch_size]
                h_b, t_b, rels_dummy, b_b, mask = self._make_pair_batch(batch)
                if h_b is None:
                    continue
                h_b = h_b.to(self.device)
                t_b = t_b.to(self.device)
                rels_dummy = rels_dummy.to(self.device)
                b_b = b_b.to(self.device)
                logits = self._model((h_b, t_b, rels_dummy, b_b))  # (B_kept, n_classes)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                row_offset = start
                kept_idx_in_batch = np.where(mask)[0]
                for i, kept_i in enumerate(kept_idx_in_batch):
                    out[row_offset + kept_i] = probs[i]
        return out

    # ------------------------------------------------------------------
    # Override save: persist ddi_type vocab + n_classes
    # ------------------------------------------------------------------
    def save(self, path: "Path | str") -> None:
        if self._model is None or self._graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump(
                {
                    "graphs": self._graphs,
                    "missing": self._missing,
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
                    "in_features": self.in_features,
                    "hidd_dim": self.hidd_dim,
                    "kge_dim": self.kge_dim,
                    "heads_out_feat_params": list(self.heads_out_feat_params),
                    "blocks_params": list(self.blocks_params),
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "n_epochs": self.n_epochs,
                },
            },
        )

    @classmethod
    def load(cls, path: "Path | str") -> "HDNDDIMulticlassBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hparams = manifest.get("hyperparameters", {})
        n_classes = manifest.get("n_classes", 86)
        if "heads_out_feat_params" in hparams:
            hparams["heads_out_feat_params"] = tuple(hparams["heads_out_feat_params"])
        if "blocks_params" in hparams:
            hparams["blocks_params"] = tuple(hparams["blocks_params"])
        inst = cls(n_classes=n_classes, **hparams)
        with (p / "graphs.pkl").open("rb") as f:
            graphs = pickle.load(f)
        inst._graphs = graphs["graphs"]
        inst._missing = graphs.get("missing", [])
        inst._ddi_type_to_idx = graphs["ddi_type_to_idx"]
        inst._idx_to_ddi_type = graphs["idx_to_ddi_type"]
        inst._model = HDN_DDI_MC(
            in_features=inst.in_features,
            hidd_dim=inst.hidd_dim,
            kge_dim=inst.kge_dim,
            rel_total=n_classes,
            heads_out_feat_params=list(inst.heads_out_feat_params),
            blocks_params=list(inst.blocks_params),
        ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst

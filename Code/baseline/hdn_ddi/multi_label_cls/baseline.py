"""HDN-DDI MULTILABEL baseline — TWOSIDES 200-side-effect prediction (Case-B).

NOTE — paper-vs-port task formulation:
  * Original paper (Sun & Zheng, BMC Bioinformatics 2025) does **per-triplet
    binary** classification with sigmoid + BCE. It is molecular-graph only
    (BRICS 3-level hierarchical encoder + y==1 substructure bipartite +
    co-attention + RESCAL); the K relations are INPUTS to RESCAL scoring.
  * THIS multilabel variant is a **Case-B adaptation** (CLAUDE.md §"从
    reproduction 派生 baseline" case B): the paper never did multilabel, but
    the paper's ALGORITHM CORE (encoder + y==1 bipartite + co-attention +
    RESCAL all-relation scoring) is reused UNCHANGED. Only the task surface —
    head activation (sigmoid not softmax), loss (masked BCE not CE), targets
    (fixed 200-label multihot not a single ddi_type idx), and the val metric
    (macro-AUPRC not macro-F1) — is swapped.

**Paper algorithm core preserved UNCHANGED** (reused from the binary /
multiclass HDN-DDI cores, per CLAUDE.md §"不允许借口'新任务'省略 paper 算法核心"):
  1. refined BRICS substructure extraction + 3-level hierarchical molecular
     graph (atoms y=0, frag y=1, super y=2) — inherited ``_build_graphs`` /
     mol-graph pkl cache.
  2. Bipartite inter-drug graph restricted to y==1 substructure nodes —
     inherited ``_make_pair_batch`` (calls ``_bipartite_edge_index_y1``).
  3. IntraGraphAttention / InterGraphAttention encoder blocks + CoAttentionLayer
     + RESCAL all-relations einsum head — the SAME :class:`HDN_DDI_MC` model
     the multiclass core uses (``rel_total=n_labels``), scored with sigmoid.
  4. Adam + LambdaLR(0.96 ** epoch) scheduler — same optimizer/scheduler style.

**Case-B task-surface changes** (allowed):
  - fixed ``n_labels`` (200 from leaf metadata); NO per-fold re-vocab, NO
    ``_ddi_type_to_idx`` (unseen-in-train labels still occupy fixed columns).
  - SIGMOID per-label activation (vs multiclass softmax).
  - masked BCE: the bundle's neg_y is the POS multihot reused as an
    ACTIVE-LABEL MASK, NOT a BCE target. For each pair, only the columns where
    the positive multihot > 0 contribute: pos gets target 1, the paired
    endpoint-corrupted neg gets target 0 on the SAME columns.
  - best ckpt selected on val macro-AUPRC (matches the test metric semantics).
  - consumes the ``make_multilabel_bundle`` paired pos/neg bundle (NEVER the
    binary negative samplers).

Independence: per CLAUDE.md §"文件级独立性" this module imports ONLY
``baseline.hdn_ddi.*`` + ``data_utils`` — no ``reproductions/`` and no
``emergnn/`` (the EmerGNN multilabel core was READ as a reference pattern but
is not imported).
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
from sklearn.metrics import average_precision_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import LambdaLR

from baseline.base import register, write_manifest
from baseline.hdn_ddi.multi_cls.baseline import HDNDDIMulticlassBaseline
from baseline.hdn_ddi.multi_cls.model import HDN_DDI_MC

if TYPE_CHECKING:
    from data_utils.leaf_adapter import _LeafDataset
    from data_utils.protocols import KnowledgeGraphProtocol


_PAIR = ["drug_a_id", "drug_b_id"]


@register("hdn_ddi_ml")
class HDNDDIMultilabelBaseline(HDNDDIMulticlassBaseline):
    """HDN-DDI multilabel (fixed L side-effect labels), BRICS-aware, Case-B.

    All BRICS-aware paper contributions (3-level graph + mol-graph pkl cache,
    y==1 bipartite, encoder blocks, co-attention, RESCAL all-relation head,
    LambdaLR scheduler) are inherited from :class:`HDNDDIMulticlassBaseline`
    (which in turn inherits the BRICS binary core). Only the task surface
    (fixed label space, sigmoid + masked BCE loss, paired pos/neg bundle,
    macro-AUPRC selection) is overridden here.
    """

    VERSION = "1.0-ml"  # Case-B multilabel adaptation of the BRICS-aware core

    def __init__(self, *, n_labels: int = 200, **kw) -> None:
        # Route the fixed label count through the parent's ``n_classes`` slot so
        # the inherited RESCAL head (``HDN_DDI_MC`` with ``rel_total=n_classes``)
        # is sized to ``n_labels``. NO per-fold re-vocab: ``_ddi_type_to_idx``
        # stays None (unused in the multilabel path).
        super().__init__(n_classes=int(n_labels), **kw)
        self.n_labels = int(n_labels)

    # ------------------------------------------------------------------
    # Override fit: multilabel training on the paired pos/neg bundle
    # ------------------------------------------------------------------
    def fit(  # type: ignore[override]
        self,
        ds: "_LeafDataset",
        bundle: dict,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        """Train on the ``make_multilabel_bundle`` output.

        ``ds`` is the duck-typed leaf dataset (only used for ``_build_graphs``,
        i.e. to build/load the BRICS mol-graph cache from ``drugs[[drugbank_id,
        smiles]]``). ``bundle`` carries the paired pos/neg multihot targets:
        keys ``{train,val}_{pos,neg}_{ht,y}`` (``*_ht`` (n,2) int endpoints,
        ``*_y`` (n, n_labels) float32 multihot; neg_y == pos_y == the ACTIVE-
        LABEL MASK, NOT a target).
        """
        # 1) BRICS-aware per-drug graphs (inherited; honours mol_pkl auto-build).
        self._build_graphs(ds)

        # 2) Fixed L-way RESCAL head — SAME model the multiclass core uses,
        #    sized to n_labels, scored with sigmoid (not softmax) downstream.
        #    NO re-vocab; n_classes == n_labels is the fixed column space.
        self.n_classes = self.n_labels
        self._model = HDN_DDI_MC(
            in_features=self.in_features,
            hidd_dim=self.hidd_dim,
            kge_dim=self.kge_dim,
            rel_total=self.n_labels,
            heads_out_feat_params=list(self.heads_out_feat_params),
            blocks_params=list(self.blocks_params),
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Paper §Parameters: LambdaLR(0.96^epoch). Same scheduler as binary /
        # multiclass; preserved because it is optimizer-side (task-agnostic).
        scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 0.96 ** epoch)

        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=np.int64)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=np.int64)
        # neg_y is the POS multihot reused as an ACTIVE-LABEL MASK (codex
        # correction): never fed as a BCE target. We keep the reference to make
        # the mask semantics explicit, but the loss uses (pos_y > 0) as the mask
        # and forces the neg target to 0.
        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=np.int64)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=np.int64)

        n_train = len(train_pos_ht)
        best_val_macro_auprc = -1.0
        best_state: dict | None = None

        try:
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress
            from my_code.utils.checkpoint import rotate_checkpoints

        steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[hdn_ddi_ml] ",
            eval_strategy=self.eval_strategy,
            eval_steps=self.eval_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        ckpt_root = (self.run_dir / "checkpoints") if self.run_dir is not None else None

        def _eval_dict() -> dict:
            if len(val_pos_ht) == 0:
                return {}
            roc, pr = self._eval_paired(val_pos_ht, val_pos_y, val_neg_ht)
            return {"val_macro_auroc": roc, "val_macro_auprc": pr}

        def _save_ckpt(tag: str, scope: str) -> None:
            if ckpt_root is None:
                return
            ckpt_path = ckpt_root / f"checkpoint-{tag}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            self.save(ckpt_path)
            progress.log_save(str(ckpt_path), scope=scope)
            rotate_checkpoints(ckpt_root, self.save_total_limit)

        for epoch in range(self.n_epochs):
            # Paper-faithful per-epoch shuffle of the training order (the
            # multiclass/binary cores shuffle each epoch too). Permute the
            # paired index so pos[i] stays aligned with its neg[i].
            perm = np.random.RandomState(epoch).permutation(n_train)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                pos_logits, pos_kept = self._batch_logits(train_pos_ht[idx])
                neg_logits, neg_kept = self._batch_logits(train_neg_ht[idx])
                # STRICT pair alignment: a pair is scored only if BOTH the
                # positive AND its paired negative kept a mol-graph. Drop any
                # index missing on either side from BOTH sides before loss.
                keep = pos_kept & neg_kept
                if not keep.any():
                    continue
                # Map the batch-local keep mask onto the kept-row order that
                # ``_batch_logits`` returned (logits rows follow kept order).
                pos_take = keep[pos_kept]   # which pos rows survive the AND
                neg_take = keep[neg_kept]   # which neg rows survive the AND
                pos_logits = pos_logits[pos_take]   # (K, n_labels)
                neg_logits = neg_logits[neg_take]   # (K, n_labels)
                # ACTIVE-LABEL MASK from the POS multihot (NOT a target).
                y_pos = torch.from_numpy(train_pos_y[idx][keep]).float().to(self.device)
                mask = y_pos > 0
                pos_sel = pos_logits[mask]                      # (M,)
                neg_sel = neg_logits[mask]                      # (M,)
                if pos_sel.numel() == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                # masked BCE: pos active labels -> 1, neg active labels -> 0.
                loss = (
                    binary_cross_entropy_with_logits(
                        pos_sel, torch.ones_like(pos_sel)
                    )
                    + binary_cross_entropy_with_logits(
                        neg_sel, torch.zeros_like(neg_sel)
                    )
                )
                loss.backward()
                opt.step()
                progress.step(loss.item())

                if progress.should_eval_step() and len(val_pos_ht):
                    self._model.eval()
                    metrics = _eval_dict()
                    self._model.train()
                    if metrics:
                        progress.log_eval(metrics, scope="step")
                        if metrics.get("val_macro_auprc", -1) > best_val_macro_auprc:
                            best_val_macro_auprc = metrics["val_macro_auprc"]
                            best_state = copy.deepcopy(self._model.state_dict())
                if progress.should_save_step():
                    _save_ckpt(f"step{progress.step_count}", scope="step")

            extra: dict = {}
            if progress.should_eval_epoch() and len(val_pos_ht):
                self._model.eval()
                metrics = _eval_dict()
                self._model.train()
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_auprc", -1) > best_val_macro_auprc:
                        best_val_macro_auprc = metrics["val_macro_auprc"]
                        best_state = copy.deepcopy(self._model.state_dict())
            if progress.should_save_epoch():
                _save_ckpt(f"ep{epoch+1:03d}", scope="epoch")
            progress.epoch_end(extra=extra if extra else None)
            scheduler.step()

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[hdn_ddi_ml] loaded best val_macro_auprc="
                f"{best_val_macro_auprc:.4f} state",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Batch scoring helper: (n,2) int endpoints -> (kept, n_labels) logits + keep mask
    # ------------------------------------------------------------------
    def _batch_logits(self, ht: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Run the inherited encoder + RESCAL all-relation head on one batch of
        endpoint pairs. Returns ``(logits, kept_mask)`` where ``logits`` has one
        row per KEPT pair (mol-graph present for both endpoints), in kept order,
        and ``kept_mask`` is a bool array over the input rows (True = kept)."""
        pairs = pd.DataFrame(
            {
                "drug_a_id": [str(int(x)) for x in ht[:, 0]],
                "drug_b_id": [str(int(x)) for x in ht[:, 1]],
            }
        )
        h_b, t_b, rels_dummy, b_b, mask = self._make_pair_batch(pairs)
        if h_b is None:
            empty = torch.zeros((0, self.n_labels), device=self.device)
            return empty, mask
        h_b = h_b.to(self.device)
        t_b = t_b.to(self.device)
        rels_dummy = rels_dummy.to(self.device)  # ignored by HDN_DDI_MC
        b_b = b_b.to(self.device)
        logits = self._model((h_b, t_b, rels_dummy, b_b))  # (n_kept, n_labels)
        return logits, mask

    # ------------------------------------------------------------------
    # Paired eval (macro AUROC / AUPRC over active labels) — matches test metric
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_paired(
        self,
        pos_ht: np.ndarray,
        pos_y: np.ndarray,
        neg_ht: np.ndarray,
    ) -> tuple[float, float]:
        """Per-label ROC-AUC / PR-AUC over paired pos/neg, averaged over labels
        with at least one active positive. Mirrors the runner's
        ``_multilabel_metrics`` semantics (positives = pos-pair score at column
        r for rows with multihot[r]>0; negatives = the paired neg-pair score at
        column r for the SAME rows). STRICT pair alignment: only rows where BOTH
        the pos and its paired neg kept a mol-graph are scored."""
        pos_scores, pos_kept = self._score_pairs(pos_ht)
        neg_scores, neg_kept = self._score_pairs(neg_ht)
        keep = pos_kept & neg_kept
        if not keep.any():
            return float("nan"), float("nan")
        pos_scores = pos_scores[keep]
        neg_scores = neg_scores[keep]
        y = np.asarray(pos_y, dtype=np.float32)[keep]
        rocs, prs = [], []
        for r in range(self.n_labels):
            idx = y[:, r] > 0
            k = int(idx.sum())
            if k == 0:
                continue
            score = np.concatenate([pos_scores[idx, r], neg_scores[idx, r]])
            label = np.concatenate([np.ones(k), np.zeros(k)])
            if label.min() == label.max():
                continue
            try:
                from sklearn.metrics import roc_auc_score
                rocs.append(roc_auc_score(label, score))
            except ValueError:
                pass
            prs.append(average_precision_score(label, score))
        return (
            float(np.mean(rocs)) if rocs else float("nan"),
            float(np.mean(prs)) if prs else float("nan"),
        )

    @torch.no_grad()
    def _score_pairs(self, ht: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n,2) int endpoints -> ((n, n_labels) sigmoid probs, keep_mask).
        Rows whose mol-graph is missing get a 0.5 probability row (and keep=False
        so callers can drop them for strict pair alignment)."""
        self._model.eval()
        out = np.full((len(ht), self.n_labels), 0.5, dtype=np.float32)
        keep = np.zeros(len(ht), dtype=bool)
        for start in range(0, len(ht), self.batch_size):
            chunk = ht[start : start + self.batch_size]
            logits, mask = self._batch_logits(chunk)
            if logits.shape[0] == 0:
                continue
            probs = torch.sigmoid(logits).cpu().numpy()
            kept_local = np.where(mask)[0]
            for i, kept_i in enumerate(kept_local):
                out[start + kept_i] = probs[i]
                keep[start + kept_i] = True
        return out, keep

    # ------------------------------------------------------------------
    # Override predict_proba: (n, n_labels) sigmoid probabilities
    # ------------------------------------------------------------------
    def predict_proba(  # type: ignore[override]
        self,
        pairs: "pd.DataFrame | np.ndarray",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None or self._graphs is None:
            raise RuntimeError(
                "HDNDDIMultilabelBaseline.fit() must be called before predict_proba."
            )
        if isinstance(pairs, pd.DataFrame):
            ht = pairs[_PAIR].to_numpy()
        else:
            ht = np.asarray(pairs)
        ht = ht.astype(np.int64) if ht.dtype != object else ht
        # Normalize to int64 endpoints (string ids are cast via int()).
        ht_int = np.empty((len(ht), 2), dtype=np.int64)
        for i in range(len(ht)):
            ht_int[i, 0] = int(ht[i, 0])
            ht_int[i, 1] = int(ht[i, 1])
        scores, _keep = self._score_pairs(ht_int)
        return scores

    # ------------------------------------------------------------------
    # Override save: persist n_labels + graphs + state
    # ------------------------------------------------------------------
    def save(self, path: "Path | str") -> None:
        if self._model is None or self._graphs is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "graphs.pkl").open("wb") as f:
            pickle.dump(
                {"graphs": self._graphs, "missing": self._missing},
                f,
            )
        write_manifest(
            out,
            baseline_name=self.name,
            extra={
                "version": self.VERSION,
                "task": "multilabel",
                "n_labels": self.n_labels,
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
    def load(cls, path: "Path | str") -> "HDNDDIMultilabelBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        extra = manifest.get("extra", manifest)
        hparams = dict(extra.get("hyperparameters", {}))
        n_labels = int(extra.get("n_labels", 200))
        if "heads_out_feat_params" in hparams:
            hparams["heads_out_feat_params"] = tuple(hparams["heads_out_feat_params"])
        if "blocks_params" in hparams:
            hparams["blocks_params"] = tuple(hparams["blocks_params"])
        inst = cls(n_labels=n_labels, **hparams)
        with (p / "graphs.pkl").open("rb") as f:
            graphs = pickle.load(f)
        inst._graphs = graphs["graphs"]
        inst._missing = graphs.get("missing", [])
        inst.n_classes = n_labels
        inst._model = HDN_DDI_MC(
            in_features=inst.in_features,
            hidd_dim=inst.hidd_dim,
            kge_dim=inst.kge_dim,
            rel_total=n_labels,
            heads_out_feat_params=list(inst.heads_out_feat_params),
            blocks_params=list(inst.blocks_params),
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst


__all__ = ["HDNDDIMultilabelBaseline"]

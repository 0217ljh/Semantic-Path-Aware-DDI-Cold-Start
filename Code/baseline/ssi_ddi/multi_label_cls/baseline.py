"""SSI-DDI MULTILABEL baseline — TWOSIDES 200-side-effect prediction (Case-B).

NOTE — paper-vs-port task formulation:
  * Original SSI-DDI (Nyamabo et al., Briefings in Bioinformatics 2022) scores
    triples ``(h, t, r)`` over many DDI relation types with a margin loss +
    corruption-based negatives. It is molecular-graph only (substructure-aware
    GAT blocks + co-attention + RESCAL); the R relations are INPUTS to RESCAL
    scoring.
  * THIS multilabel variant is a **Case-B adaptation** (CLAUDE.md §"从
    reproduction 派生 baseline" case B): the paper never did multilabel, but
    the paper's ALGORITHM CORE (GAT encoder + co-attention + RESCAL all-relation
    scoring) is reused UNCHANGED. Only the task surface — head activation
    (sigmoid not margin/softmax), loss (masked BCE), targets (fixed 200-label
    multihot not a single ddi_type idx), and the val metric (macro-AUPRC) — is
    swapped.

**Paper algorithm core preserved UNCHANGED** (reused from the binary /
multiclass SSI-DDI cores, per CLAUDE.md §"不允许借口'新任务'省略 paper 算法核心"):
  1. per-drug PyG molecular graphs from ``drugs[[drugbank_id, smiles]]`` —
     inherited ``_build_graphs`` (:func:`baseline.ssi_ddi.mol_features.
     build_drug_graphs`).
  2. Substructure-aware GAT blocks + CoAttentionLayer + RESCAL all-relation
     einsum head — the SAME :class:`SSI_DDI_MC` model the multiclass core uses
     (``rel_total=n_labels``), scored with sigmoid.
  3. pair batching (``_make_pair_batch``) with cold-start drop of pairs whose
     mol-graph is missing — inherited from the binary core.
  4. Adam optimizer (same lr / weight_decay) — task-agnostic optimizer setting.

**Case-B task-surface changes** (allowed):
  - fixed ``n_labels`` (200 from leaf metadata); NO per-fold re-vocab, NO
    ``_ddi_type_to_idx`` / ``_idx_to_ddi_type`` (unseen-in-train labels still
    occupy fixed columns; both stay None in the multilabel path).
  - SIGMOID per-label activation (vs multiclass softmax).
  - masked BCE: the bundle's neg_y is the POS multihot reused as an
    ACTIVE-LABEL MASK, NOT a BCE target. For each pair, only the columns where
    the positive multihot > 0 contribute: pos gets target 1, the paired
    endpoint-corrupted neg gets target 0 on the SAME columns.
  - best ckpt selected on val macro-AUPRC (matches the test metric semantics).
  - consumes the ``make_multilabel_bundle`` paired pos/neg bundle (NEVER the
    binary negative samplers).

Independence: per CLAUDE.md §"文件级独立性" this module imports ONLY
``baseline.ssi_ddi.*`` + ``data_utils`` — no ``reproductions/`` and no
``hdn_ddi/`` / ``emergnn/`` (the HDN-DDI multilabel core was READ as a reference
pattern but is not imported).
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

from baseline.base import register, write_manifest
from baseline.ssi_ddi.multi_cls.baseline import SSIDDIMulticlassBaseline
from baseline.ssi_ddi.multi_cls.model import SSI_DDI_MC

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


_PAIR = ["drug_a_id", "drug_b_id"]


@register("ssi_ddi_ml")
class SSIDDIMultilabelBaseline(SSIDDIMulticlassBaseline):
    """SSI-DDI multilabel (fixed L side-effect labels), Case-B.

    All SSI-DDI paper contributions (per-drug PyG mol-graphs, substructure-aware
    GAT blocks, co-attention, RESCAL all-relation head) are inherited from
    :class:`SSIDDIMulticlassBaseline` (which in turn inherits the binary core's
    featurization). Only the task surface (fixed label space, sigmoid + masked
    BCE loss, paired pos/neg bundle, macro-AUPRC selection) is overridden here.
    """

    VERSION = "1.0-ml"  # Case-B multilabel adaptation of the SSI-DDI core

    def __init__(self, *, n_labels: int = 200, log_step_every: int = 50, **kw) -> None:
        # Route the fixed label count through the parent's ``n_classes`` slot so
        # the inherited RESCAL head (``SSI_DDI_MC`` with ``rel_total=n_classes``)
        # is sized to ``n_labels``. NO per-fold re-vocab: ``_ddi_type_to_idx`` /
        # ``_idx_to_ddi_type`` stay None (unused in the multilabel path).
        super().__init__(n_classes=int(n_labels), **kw)
        self.n_labels = int(n_labels)
        self.log_step_every = int(log_step_every)

    # ------------------------------------------------------------------
    # Override fit: multilabel training on the paired pos/neg bundle
    # ------------------------------------------------------------------
    def fit(  # type: ignore[override]
        self,
        ds: "PairDataset",
        bundle: dict,
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> None:
        """Train on the ``make_multilabel_bundle`` output.

        ``ds`` is the duck-typed leaf dataset (only used for ``_build_graphs``,
        i.e. to build the SSI-DDI per-drug PyG graphs from ``drugs[[drugbank_id,
        smiles]]``). ``bundle`` carries the paired pos/neg multihot targets:
        keys ``{train,val}_{pos,neg}_{ht,y}`` (``*_ht`` (n,2) int endpoints,
        ``*_y`` (n, n_labels) float32 multihot; neg_y == pos_y == the ACTIVE-
        LABEL MASK, NOT a target).
        """
        # 1) SSI-DDI per-drug PyG graphs (inherited from the binary core).
        self._build_graphs(ds)

        # 2) Fixed L-way RESCAL head — SAME model the multiclass core uses
        #    (via ``_build_model``), sized to n_labels, scored with sigmoid (not
        #    softmax) downstream. NO re-vocab; n_classes == n_labels is the fixed
        #    column space.
        self.n_classes = self.n_labels
        self._model = self._build_model()  # inherited: SSI_DDI_MC(rel_total=n_classes)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=np.int64)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=np.int64)
        # neg_y is the POS multihot reused as an ACTIVE-LABEL MASK (codex
        # correction): never fed as a BCE target. The loss uses (pos_y > 0) as the
        # mask and forces the neg target to 0.
        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=np.int64)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=np.int64)

        n_train = len(train_pos_ht)
        best_val_macro_auprc = -1.0
        best_state: dict | None = None

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress

        steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(
            total_epochs=self.n_epochs,
            log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch,
            prefix="[ssi_ddi_ml] ",
        )

        def _eval_dict() -> dict:
            if len(val_pos_ht) == 0:
                return {}
            roc, pr = self._eval_paired(val_pos_ht, val_pos_y, val_neg_ht)
            return {"val_macro_auroc": roc, "val_macro_auprc": pr}

        for epoch in range(self.n_epochs):
            # Per-epoch shuffle of the training order (the multiclass/binary cores
            # shuffle each epoch too). Permute the paired index so pos[i] stays
            # aligned with its neg[i].
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

            extra: dict = {}
            if len(val_pos_ht):
                self._model.eval()
                metrics = _eval_dict()
                self._model.train()
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_auprc", -1) > best_val_macro_auprc:
                        best_val_macro_auprc = metrics["val_macro_auprc"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[ssi_ddi_ml] loaded best val_macro_auprc="
                f"{best_val_macro_auprc:.4f} state",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Batch scoring helper: (n,2) int endpoints -> (kept, n_labels) logits + keep mask
    # ------------------------------------------------------------------
    def _batch_logits(self, ht: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        """Run the inherited GAT encoder + co-attention + RESCAL all-relation head
        on one batch of endpoint pairs. Returns ``(logits, kept_mask)`` where
        ``logits`` has one row per KEPT pair (mol-graph present for both
        endpoints), in kept order, and ``kept_mask`` is a bool array over the
        input rows (True = kept)."""
        pairs = pd.DataFrame(
            {
                "drug_a_id": [str(int(x)) for x in ht[:, 0]],
                "drug_b_id": [str(int(x)) for x in ht[:, 1]],
            }
        )
        # SSI-DDI's _make_pair_batch returns (h_batch, t_batch, rels, mask);
        # the all-relation head only needs (h_batch, t_batch).
        h_b, t_b, _rels, mask = self._make_pair_batch(pairs)
        if h_b is None:
            empty = torch.zeros((0, self.n_labels), device=self.device)
            return empty, mask
        h_b = h_b.to(self.device)
        t_b = t_b.to(self.device)
        logits = self._model.forward_all((h_b, t_b))  # (n_kept, n_labels)
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
                "SSIDDIMultilabelBaseline.fit() must be called before predict_proba."
            )
        if isinstance(pairs, pd.DataFrame):
            ht = pairs[_PAIR].to_numpy()
        else:
            ht = np.asarray(pairs)
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
    def load(cls, path: "Path | str") -> "SSIDDIMultilabelBaseline":
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
        inst._model = SSI_DDI_MC(
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


__all__ = ["SSIDDIMultilabelBaseline"]

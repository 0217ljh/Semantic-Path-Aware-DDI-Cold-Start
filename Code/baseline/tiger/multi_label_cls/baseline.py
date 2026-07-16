"""TIGER MULTILABEL baseline — TWOSIDES 200-side-effect prediction (Case-B).

NOTE — paper-vs-port task formulation:
  * Original paper (Su et al., AAAI 2024) is **binary** classification (pos pair
    vs random negative), softmax CE on a ``(B, 2)`` head, metrics ACC/F1/AUC/AUPR.
    TIGER is DUAL-CHANNEL — a molecular GraphTransformer + a biomedical-KG (BKG)
    subgraph GraphTransformer, joined by relation-aware attention with a
    mutual-information (MI) auxiliary loss. The paper never did multilabel.
  * THIS multilabel variant is a **Case-B adaptation** (CLAUDE.md §"从 reproduction
    派生 baseline" case B): the paper never did multilabel, but the paper's
    ALGORITHM CORE (mol GraphTransformer + BKG-subgraph GraphTransformer +
    relation-aware attention + MI loss + dual-channel fusion + ``fc2`` head) is
    reused UNCHANGED. Only the task surface — head activation (sigmoid not
    softmax), loss (masked BCE not CE), targets (fixed 200-label multihot not a
    single ddi_type idx), and the val metric (macro-AUPRC not F1) — is swapped.

**Paper algorithm core preserved UNCHANGED** (reused from the binary TIGER core
:class:`baseline.tiger.binary_cls.baseline.TIGERBaseline`, per CLAUDE.md
§"不允许借口'新任务'省略 paper 算法核心"):
  1. molecular GraphTransformer encoder + SMILES→graph featurization + mol-graph
     pkl cache — inherited ``_build_mol_graphs``.
  2. BKG (from the FULL merged DrugBank+Hetionet+PrimeKG KG) + per-drug DeepWalk
     subgraph extraction — inherited ``_build_bkg_and_subgraphs`` (which routes
     through :mod:`baseline.tiger.kg_builder` / ``_shared`` cache builders).
  3. dual-channel fusion (``fc1``) + relation-aware GraphTransformer attention +
     the two MI auxiliary losses (drug↔mol-atom, drug↔sub-node) + ``fc2`` head —
     reused via the ADDITIVE :meth:`baseline.tiger.model.TIGER.forward_logits`
     (raw logits + MI aux loss; NO log_softmax/nll — the existing softmax
     ``forward`` used by binary/multiclass is UNTOUCHED).
  4. Adam optimizer (same lr / weight_decay). Pair batching + cold-start
     center-node patch hook + the strict-drop of pairs whose mol-graph/BKG entry
     is missing — inherited ``_make_pair_batch``.

**Case-B task-surface changes** (allowed):
  - fixed ``n_labels`` (200 from leaf metadata); NO per-fold re-vocab, NO
    ``_ddi_type_to_idx`` (this class does NOT subclass the multiclass core, whose
    ``fit`` re-vocabs). ``fc2`` is sized to ``n_labels`` once.
  - SIGMOID per-label activation (vs binary/multiclass softmax).
  - masked BCE: the bundle's neg_y is the POS multihot reused as an ACTIVE-LABEL
    MASK, NOT a BCE target. For each pair, only the columns where the positive
    multihot > 0 contribute: pos gets target 1, the paired endpoint-corrupted neg
    gets target 0 on the SAME columns.
  - best ckpt selected on val macro-AUPRC (matches the test metric semantics).
  - the MI auxiliary loss TIGER uses in training is KEPT (added to the masked BCE
    each step, same ``sub_coeff`` / ``mi_coeff`` weighting).
  - consumes a paired pos/neg multihot bundle (``{train,val}_{pos,neg}_{ht,y}``
    with STRING TIGER-id endpoints — see the unified wrapper's id-remap).

Data limitation (documented, not hidden): ~45% of TWOSIDES drugs (269/604) carry
no DrugBank id and therefore appear in the merged KG only as an ISOLATED
self-loop node (built by :mod:`baseline.tiger.kg_builder`); their KG channel
carries no neighbourhood signal. The molecular channel still covers ALL 604
drugs. These drugs are NOT dropped — dropping them would make ``_make_pair_batch``
drop every pair that touches them.

Independence: per CLAUDE.md §"文件级独立性" this module imports ONLY
``baseline.tiger.*`` + ``data_utils`` + framework — no ``reproductions/`` and no
``hdn_ddi/`` / ``ssi_ddi/`` (their multilabel heads were READ as a reference
pattern but are NOT imported).
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
from baseline.tiger.binary_cls.baseline import TIGERBaseline
from baseline.tiger.mol_features import ATOM_FEATURE_DIM
from baseline.tiger.model import TIGER

if TYPE_CHECKING:
    from data_utils.dataset import PairDataset
    from data_utils.protocols import KnowledgeGraphProtocol


_PAIR = ["drug_a_id", "drug_b_id"]


@register("tiger_ml")
class TIGERMultilabelBaseline(TIGERBaseline):
    """TIGER multilabel (fixed L side-effect labels), dual-channel, Case-B.

    All TIGER paper contributions (mol GraphTransformer + BKG-subgraph
    GraphTransformer + relation-aware attention + dual-channel fusion + MI loss)
    are inherited from :class:`baseline.tiger.binary_cls.baseline.TIGERBaseline`
    (build helpers, ``_make_pair_batch``, mol/BKG/subgraph caches). This class
    does NOT subclass the multiclass core (whose ``fit`` re-vocabs the ddi_type
    space). Only the task surface (fixed label space, ``fc2``→L, sigmoid + masked
    BCE, paired pos/neg bundle, macro-AUPRC selection) is overridden here, and the
    model uses the ADDITIVE raw-logits path ``TIGER.forward_logits``.
    """

    VERSION = "1.0-ml"  # Case-B multilabel adaptation of the dual-channel core

    def __init__(self, *, n_labels: int = 200, **kw) -> None:
        super().__init__(**kw)
        self.n_labels = int(n_labels)

    # ------------------------------------------------------------------
    # Raw-logits forward (uses the ADDITIVE TIGER.forward_logits path)
    # ------------------------------------------------------------------
    def _forward_logits(self, mol1, sub1, mol2, sub2, idx1, idx2):
        """Return ``(logits (B, n_labels), aux_MI_loss)`` via the additive
        raw-logits path — never the softmax ``forward``."""
        if self.mol_only:
            return self._model.forward_logits(
                mol1.to(self.device),
                mol1.to(self.device),
                mol2.to(self.device),
                mol2.to(self.device),
            )
        return self._model.forward_logits(
            mol1.to(self.device),
            sub1.to(self.device),
            mol2.to(self.device),
            sub2.to(self.device),
            batch_idx1=idx1,
            batch_idx2=idx2,
            unseen_ids=self._unseen_ids,
        )

    def _batch_logits(
        self, ht: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Run the inherited dual-channel encoder + ``fc2`` (raw logits) on one
        batch of endpoint pairs. ``ht`` is an ``(n, 2)`` array of STRING TIGER
        ids. Returns ``(logits, aux_loss, kept_mask)`` where ``logits`` has one
        row per KEPT pair (mol-graph AND BKG subgraph present for both
        endpoints), in kept order; ``aux_loss`` is the TIGER MI auxiliary loss
        for this batch (a scalar tensor, 0.0 when nothing is kept); and
        ``kept_mask`` is a bool array over the input rows."""
        pairs = pd.DataFrame(
            {"drug_a_id": [str(x) for x in ht[:, 0]],
             "drug_b_id": [str(x) for x in ht[:, 1]]}
        )
        zero_labels = np.zeros(len(pairs), dtype=np.int64)
        mol1, sub1, mol2, sub2, idx1, idx2, mask = self._make_pair_batch(
            pairs, labels=zero_labels
        )
        if mol1 is None:
            empty = torch.zeros((0, self.n_labels), device=self.device)
            return empty, torch.zeros((), device=self.device), mask
        logits, aux = self._forward_logits(mol1, sub1, mol2, sub2, idx1, idx2)
        return logits, aux, mask

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
        """Train on a ``make_multilabel_bundle``-style paired bundle.

        ``ds`` is the duck-typed leaf dataset (used to build mol graphs + the BKG
        + per-drug subgraphs, all keyed by the TIGER-id string surface set up by
        the unified wrapper). ``bundle`` carries keys
        ``{train,val}_{pos,neg}_{ht,y}``: ``*_ht`` ``(n, 2)`` OBJECT arrays of
        STRING TIGER ids, ``*_y`` ``(n, n_labels)`` float32 multihot. neg_y ==
        pos_y == the ACTIVE-LABEL MASK, NOT a target.
        """
        # ── 1. molecular graphs (inherited; honours mol_pkl auto-build) ──
        mol_rel_required = self._build_mol_graphs(ds)
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

        # ── 2. BKG + per-drug subgraphs (inherited; full merged KG) ──────
        if self.mol_only:
            num_rel_graph = 2
            n_total_nodes = 2
            max_deg_node = 2
        else:
            graph_rel_required, max_deg_node_obs, n_total_nodes = (
                self._build_bkg_and_subgraphs(ds, verbose=True)
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

        # ── 3. Fixed L-way model — same dual-channel arch, n_classes=n_labels ──
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
            n_classes=self.n_labels,
        ).to(self.device)
        opt = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Endpoints stay OBJECT (string TIGER ids); the multihots are float32.
        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=object)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=object)
        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=object)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=object)

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
            prefix="[tiger_ml] ",
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
            # Per-epoch shuffle of the training order (the binary/multiclass cores
            # shuffle each epoch too). Permute the paired index so pos[i] stays
            # aligned with its neg[i].
            perm = np.random.RandomState(epoch).permutation(n_train)
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                pos_logits, pos_aux, pos_kept = self._batch_logits(train_pos_ht[idx])
                neg_logits, neg_aux, neg_kept = self._batch_logits(train_neg_ht[idx])
                # STRICT pair alignment: a pair is scored only if BOTH the
                # positive AND its paired negative kept a mol-graph + BKG entry.
                # Drop any index missing on either side from BOTH sides.
                keep = pos_kept & neg_kept
                if not keep.any():
                    continue
                # Map the batch-local keep mask onto the kept-row order that
                # ``_batch_logits`` returned (logits rows follow kept order).
                # The pos and neg batches were built independently, so their
                # kept-row counts can differ; the AND (``keep``) restricts both
                # to the intersection, giving equal-length aligned rows.
                pos_take = keep[pos_kept]
                neg_take = keep[neg_kept]
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
                bce = (
                    binary_cross_entropy_with_logits(
                        pos_sel, torch.ones_like(pos_sel)
                    )
                    + binary_cross_entropy_with_logits(
                        neg_sel, torch.zeros_like(neg_sel)
                    )
                )
                # Keep TIGER's MI auxiliary loss from BOTH the pos and neg
                # forward passes (mirrors the two forward calls binary training
                # makes). Already computed inside ``_batch_logits`` and attached
                # to the same graph as the logits, so no redundant re-forward.
                loss = bce + pos_aux + neg_aux
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

        if self.load_best_model_at_end and best_state is not None:
            self._model.load_state_dict(best_state)
            print(
                f"[tiger_ml] loaded best val_macro_auprc="
                f"{best_val_macro_auprc:.4f} state",
                flush=True,
            )

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
        ``_multilabel_metrics``. STRICT pair alignment: only rows where BOTH the
        pos and its paired neg kept a mol-graph + BKG entry are scored."""
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
        """``(n, 2)`` STRING-id endpoints -> ``((n, n_labels) sigmoid probs,
        keep_mask)``. Rows whose mol-graph / BKG entry is missing get a 0.5 row
        (and keep=False so callers can drop them for strict pair alignment)."""
        self._model.eval()
        out = np.full((len(ht), self.n_labels), 0.5, dtype=np.float32)
        keep = np.zeros(len(ht), dtype=bool)
        for start in range(0, len(ht), self.batch_size):
            chunk = ht[start : start + self.batch_size]
            logits, _aux, mask = self._batch_logits(chunk)
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
    @torch.no_grad()
    def predict_proba(  # type: ignore[override]
        self,
        pairs: "pd.DataFrame | np.ndarray",
        *,
        kg: "KnowledgeGraphProtocol | None" = None,
    ) -> np.ndarray:
        if self._model is None or self._mol_graphs is None:
            raise RuntimeError(
                "TIGERMultilabelBaseline.fit() must be called before predict_proba."
            )
        if isinstance(pairs, pd.DataFrame):
            ht = pairs[_PAIR].to_numpy()
        else:
            ht = np.asarray(pairs)
        # Keep endpoints as strings (TIGER ids); no int cast (ids may be
        # 'DB00813' / 'twoside:12').
        ht_str = np.empty((len(ht), 2), dtype=object)
        for i in range(len(ht)):
            ht_str[i, 0] = str(ht[i, 0])
            ht_str[i, 1] = str(ht[i, 1])
        scores, _keep = self._score_pairs(ht_str)
        return scores

    # ------------------------------------------------------------------
    # Override save / load: persist n_labels + graphs + state
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
                    "n_labels": self.n_labels,
                },
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
    def load(cls, path: "Path | str") -> "TIGERMultilabelBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        extra = manifest.get("extra", manifest)
        hparams = dict(extra.get("hyperparameters", {}))
        n_labels = int(extra.get("n_labels", 200))
        inst = cls(n_labels=n_labels, **hparams)
        with (p / "graphs.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._mol_graphs = payload["mol_graphs"]
        inst._mol_missing = payload.get("mol_missing", [])
        inst._subgraphs = payload.get("subgraphs")
        inst._drug_to_idx = payload.get("drug_to_idx")
        inst._unseen_ids = set(payload.get("unseen_ids", []))
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
            n_classes=n_labels,
        ).to(inst.device)
        inst._model.load_state_dict(
            torch.load(p / "model.pt", map_location=inst.device)
        )
        inst._model.eval()
        return inst


__all__ = ["TIGERMultilabelBaseline"]

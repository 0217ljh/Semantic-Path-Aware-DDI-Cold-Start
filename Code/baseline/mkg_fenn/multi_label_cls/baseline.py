"""MKG-FENN MULTILABEL core (regime-aware: S0 warm 4-channel, S1/S2 cold 3-channel).

TWOSIDES 200-side-effect prediction (Case-B). The paper's real task is 65-event
MULTICLASS with softmax cross-entropy (``--event_num default=65``; MKG-FENN-task
{1,2,3}.py:38-39, CE loss task3.py:172). This module is a **Case-B adaptation**
(CLAUDE.md §"从 reproduction 派生 baseline" case B): the paper never did
multilabel, but the paper's COLD-START ALGORITHM CORE is reused UNCHANGED — only
the task surface (head activation, loss, targets, val metric) is swapped.

**Paper algorithm core preserved UNCHANGED** (reused from ``multi_cls``, per
CLAUDE.md §"不允许借口'新任务'省略 paper 算法核心"):
  * S0 (transductive)  -> 4-channel warm model (``baseline.mkg_fenn.model.MKGFENN``,
    ported from modeltask1.py), ``event_num = n_labels``. No imputation, no test_adj.
  * S1/S2 (inductive)  -> 3-channel cold model (``model_cold.MKGFENNCold``, ported
    from modeltask2/3.py), ``event_num = n_labels``, with per-channel nearest-seen-
    neighbour imputation via ``test_adj``.
  * The 4 intrinsic KGs (``kg_builder.build_all_kgs``), the cold plumbing helpers
    (``find_dif`` / ``jaccard`` / ``_build_feature_matrices`` / ``_build_test_adj`` /
    ``_ghost_pad_kg1`` / ``_drug_smiles_dict``), and the deterministic seeding are
    IMPORTED from ``multi_cls.baseline`` (single source of truth; ``multi_cls`` left
    untouched). CLAUDE.md §baseline allows same-package intra-baseline reuse.
  * Training protocol (task3.py, task-agnostic — kept): deterministic seeding,
    symmetric pair augmentation (a,b)+(b,a), Adam (same lr / weight_decay, no
    scheduler, no early-stop), per-epoch shuffle, TrainProgress logging.

**Case-B task-surface changes** (allowed):
  * FIXED ``n_labels`` (200 from leaf metadata); NO per-fold re-vocab, NO
    ``_ddi_type_to_idx`` (unseen-in-train labels still occupy fixed columns).
  * SIGMOID per-label activation (vs multiclass softmax); ONE sigmoid at inference.
  * masked BCE: the paired bundle's ``neg_y`` is the POS multihot reused as an
    ACTIVE-LABEL MASK, NOT a BCE target. For each pair, only the columns where the
    positive multihot > 0 contribute: pos gets target 1, the paired endpoint-
    corrupted neg gets target 0 on the SAME columns (make_multilabel_bundle-style).
  * best ckpt selected on VAL macro-AUPRC (matches the test metric semantics).
  * ``predict_proba`` returns ``(n, n_labels)`` sigmoid probabilities.

Endpoints live in the SAME pool-id (dict1 index) space as the other 3 KGs. The
unified wrapper re-keys KG1 (drugbank_id -> pool id) so all 4 KGs share one space;
unmapped drugs keep a ghost KG1 slot (``_ghost_pad_kg1``) and are NOT dropped.
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim
from torch.nn.functional import binary_cross_entropy_with_logits

from baseline.mkg_fenn.kg_builder import build_all_kgs
from baseline.mkg_fenn.model import MKGFENN
from baseline.mkg_fenn.multi_cls.model_cold import MKGFENNCold
# Same-package intra-baseline import of the pure cold-plumbing helpers (single
# source of truth; multi_cls left untouched). CLAUDE.md §baseline allows this.
from baseline.mkg_fenn.multi_cls.baseline import (
    PAPER_HYPERPARAMS,
    _build_feature_matrices,
    _build_test_adj,
    _drug_smiles_dict,
    _ghost_pad_kg1,
    find_dif,
    jaccard,
)

if TYPE_CHECKING:  # pragma: no cover
    from data_utils.kg import KnowledgeGraph

_PAIR = ["drug_a_id", "drug_b_id"]

#: Fixed TWOSIDES multilabel head dimension (per-label sigmoid). Overridable via
#: the leaf's ``labels.n_labels`` metadata (200 for TWOSIDES).
DEFAULT_N_LABELS = 200


class MKGFENNMultilabelBaseline:
    """Regime-aware MKG-FENN multilabel core (S0 warm / S1-S2 cold), event_num=n_labels.

    Not a legacy ``BaselineModel`` — consumed by the unified wrapper. The wrapper
    passes the native (re-keyed) KG1 object as ``kg``, the leaf's ``cold`` flag /
    unseen drug ids / drug vocab (``dict1``), and the paired pos/neg multilabel
    bundle (``make_multilabel_bundle`` output; endpoints already in dict1 pool-id
    space). ``neg_y`` is the POS multihot reused as an ACTIVE-LABEL MASK.
    """

    VERSION = "1.0-ml-cold"

    def __init__(
        self,
        *,
        n_labels: int = DEFAULT_N_LABELS,
        cold: bool = False,
        embedding_num: int = 128,
        neighbor_sample_size: int = 6,
        dropout: float = 0.3,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-8,
        batch_size: int = 256,
        n_epochs: int = 50,
        fp_radius: int = 2,
        fp_nbits: int = 512,
        n_bins: int = 10,
        seed: int = 1,
        device: str = "auto",
        log_step_every: int = 50,
        run_dir: "str | Path | None" = None,
    ) -> None:
        self.n_labels = int(n_labels)
        self.cold = bool(cold)
        self.embedding_num = embedding_num
        self.neighbor_sample_size = neighbor_sample_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.n_bins = n_bins
        self.seed = seed
        self.device = self._resolve_device(device)
        self.log_step_every = log_step_every
        self.run_dir = Path(run_dir) if run_dir is not None else None

        self._model: nn.Module | None = None
        self._dict1: dict[str, int] | None = None
        self._test_adj: list[dict] | None = None
        self._unseen_ids: list[int] = []
        self._kgs: dict | None = None
        self._tail_len: dict | None = None
        self._relation_len: dict | None = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            embedding_num=self.embedding_num,
            neighbor_sample_size=self.neighbor_sample_size,
            dropout=self.dropout,
        )

    def _seed_all(self) -> None:
        """Deterministic seeding + determinism flags (official task3.py:48-57).

        Identical block to the multiclass/binary cores (warn_only=True faithful-
        equivalent; cuDNN deterministic/benchmark/enabled). The per-leaf runner
        launches one process per baseline, so these process-global flags do not
        leak across baselines.
        """
        import os
        import random
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # very old torch without warn_only kwarg
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    def _build_dict1(self, train) -> dict[str, int]:
        """All drugs across splits + drugs table -> contiguous idx (baseline.py:113-123).

        Unseen drugs KEPT so cold drugs stay reachable at predict time. Keyed on the
        pool ``drug_id`` (leaf_adapter renames it to ``drugbank_id`` on the drugs
        frame, but for a re-keyed twoside leaf the value is the pool-id string).
        """
        drug_ids: set[str] = set()
        for _name, df in train.splits.items():
            drug_ids.update(df["drug_a_id"].astype(str))
            drug_ids.update(df["drug_b_id"].astype(str))
        if train.drugs is not None and "drugbank_id" in train.drugs.columns:
            drug_ids.update(train.drugs["drugbank_id"].astype(str))
        return {did: idx for idx, did in enumerate(sorted(drug_ids))}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        train,
        bundle: dict,
        *,
        kg: "KnowledgeGraph | None" = None,
        dict1: dict[str, int] | None = None,
        unseen_ids: list[int] | None = None,
    ) -> None:
        """Train on one leaf's paired pos/neg multilabel bundle.

        Parameters
        ----------
        train
            Duck-typed ``_LeafDataset`` (leaf_adapter, task="multilabel"). Used for
            the drug vocab / SMILES / the KG3 train-DDI positive pairs
            (``train.splits.train``).
        bundle
            ``make_multilabel_bundle`` output: keys ``{train,val}_{pos,neg}_{ht,y}``
            (``*_ht`` (n,2) int64 dict1-index endpoints, ``*_y`` (n,n_labels) float32
            multihot; ``neg_y == pos_y ==`` the ACTIVE-LABEL MASK, NOT a target).
        kg
            Native (re-keyed) drug->entity KnowledgeGraph (KG1 source). The wrapper
            passes this since the adapter sets ``_LeafDataset.kg = None``.
        dict1
            Precomputed drug vocab (all drugs -> idx). Built here if None.
        unseen_ids
            Drug ids (dict1 index space) unseen at train time (cold regimes). Empty
            for S0.
        """
        self._seed_all()
        if kg is None:
            kg = train.kg
        if kg is None:
            raise ValueError("MKG-FENN multilabel requires a KG1 object (drug->entity).")

        self._dict1 = dict1 if dict1 is not None else self._build_dict1(train)
        n_drug = len(self._dict1)
        self._unseen_ids = list(unseen_ids) if unseen_ids else []
        event_num = self.n_labels

        # KG3 train-DDI positive pairs (drives the drug->drug channel + fm4).
        pos = train.splits.train

        # Build the 4 intrinsic KGs (reused UNCHANGED). build_kg3 self-loops all drugs.
        smiles = _drug_smiles_dict(train.drugs)
        kgs, tail_len, relation_len = build_all_kgs(
            kg=kg, drug_id2smiles=smiles, dict1=self._dict1,
            train_pos_df=pos[_PAIR], fp_radius=self.fp_radius,
            fp_nbits=self.fp_nbits, n_bins=self.n_bins,
        )
        drug_name = list(range(n_drug))
        args = self._args()

        # Cold plumbing: drug_sim1..4 + test_adj (only when there are unseen drugs).
        # Built from the UN-padded KG1 so the ghost slot doesn't distort similarity.
        # Verbatim reuse of the multiclass core's faithful helpers.
        if self.cold and self._unseen_ids:
            fm1, fm2, fm3, fm4 = _build_feature_matrices(kgs, tail_len, n_drug)
            drug_sim1 = np.asarray(jaccard(fm1))
            drug_sim2 = np.asarray(jaccard(fm2))
            drug_sim3 = find_dif(fm3, n_drug)
            drug_sim4 = np.asarray(jaccard(fm4))
            self._test_adj = _build_test_adj(
                [drug_sim1, drug_sim2, drug_sim3, drug_sim4], self._unseen_ids, n_drug
            )
        else:
            self._test_adj = None

        # Ghost-pad KG1 empties for the COLD model (Option A). Warm MKGFENN ghost-
        # handles itself, so only the cold path pads (avoids a double ghost slot).
        if self.cold:
            kgs, tail_len, relation_len = _ghost_pad_kg1(kgs, tail_len, relation_len, self._dict1)
        self._kgs, self._tail_len, self._relation_len = kgs, tail_len, relation_len

        # Model (regime switch, event_num=n_labels).
        if self.cold:
            self._model = MKGFENNCold(
                kgs=kgs, tail_len=tail_len, relation_len=relation_len,
                dict1=self._dict1, drug_name=drug_name, args=args, event_num=event_num,
            ).to(self.device)
        else:
            self._model = MKGFENN(
                kgs=kgs, tail_len=tail_len, relation_len=relation_len,
                dict1=self._dict1, drug_name=drug_name,
                embedding_num=self.embedding_num,
                neighbor_sample_size=self.neighbor_sample_size,
                dropout=self.dropout, event_num=event_num,
            ).to(self.device)
            self._model.precompute_adj()

        # Paired bundle arrays (endpoints already in dict1 pool-id index space).
        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=np.int64)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=np.int64)
        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=np.int64)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=np.int64)

        n_train = len(train_pos_ht)

        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress

        steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)
        progress = TrainProgress(
            total_epochs=self.n_epochs, log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch, prefix="[mkg_fenn_ml] ",
        )

        best_val_macro_auprc = -1.0
        best_state: dict | None = None
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.n_epochs):
            # Per-epoch shuffle of the PAIRED index (pos[i] stays aligned with neg[i]).
            perm = rng.permutation(n_train)
            if not self.cold:
                self._model.precompute_adj()           # warm path re-samples neighbours
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                pos_ht = train_pos_ht[idx]
                neg_ht = train_neg_ht[idx]
                y_mask = train_pos_y[idx]              # ACTIVE-LABEL MASK (NOT target)
                if len(pos_ht) < 2:                    # BatchNorm1d needs >=2 rows
                    continue
                pos_logits = self._forward_train(pos_ht)   # (B, n_labels)
                neg_logits = self._forward_train(neg_ht)   # (B, n_labels)
                mask = torch.from_numpy(y_mask > 0).to(self.device)
                pos_sel = pos_logits[mask]
                neg_sel = neg_logits[mask]
                if pos_sel.numel() == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                # masked BCE: pos active labels -> 1, neg active labels -> 0.
                loss = (
                    binary_cross_entropy_with_logits(pos_sel, torch.ones_like(pos_sel))
                    + binary_cross_entropy_with_logits(neg_sel, torch.zeros_like(neg_sel))
                )
                loss.backward()
                opt.step()
                progress.step(loss.item())
            extra: dict = {}
            if len(val_pos_ht):
                metrics = self._eval_paired(val_pos_ht, val_pos_y, val_neg_ht)
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_auprc", -1) > best_val_macro_auprc:
                        best_val_macro_auprc = metrics["val_macro_auprc"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mkg_fenn_ml] loaded best val_macro_auprc={best_val_macro_auprc:.4f}",
                  flush=True)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def _forward_train(self, ht: np.ndarray) -> torch.Tensor:
        """Train-time forward on (B,2) int endpoints (no imputation; cold t_or_t=0)."""
        idx_tensor = torch.as_tensor(ht, dtype=torch.long)
        if self.cold:
            return self._model(idx_tensor, 0, None)
        return self._model(idx_tensor)

    def _forward_eval(self, ht: np.ndarray) -> torch.Tensor:
        """Eval-time forward. Cold path enables nearest-seen imputation (t_or_t=1)."""
        idx_tensor = torch.as_tensor(ht, dtype=torch.long)
        if self.cold and self._test_adj is not None:
            return self._model(idx_tensor, 1, self._test_adj)
        if self.cold:
            return self._model(idx_tensor, 0, None)
        return self._model(idx_tensor)

    # ------------------------------------------------------------------
    # eval (macro-AUPRC / AUROC over active labels) — matches test metric
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_paired(
        self, pos_ht: np.ndarray, pos_y: np.ndarray, neg_ht: np.ndarray
    ) -> dict:
        """Per-label ROC-AUC / PR-AUC over paired pos/neg, averaged over labels with
        >=1 active positive. Mirrors the runner's ``_multilabel_metrics`` semantics
        (positives = pos-pair score at column r for rows with multihot[r]>0;
        negatives = the paired neg-pair score at column r for the SAME rows)."""
        self._model.eval()
        pos_scores = self._score_pairs(pos_ht)
        neg_scores = self._score_pairs(neg_ht)
        self._model.train()
        y = np.asarray(pos_y, dtype=np.float32)
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
                rocs.append(roc_auc_score(label, score))
            except ValueError:
                pass
            prs.append(average_precision_score(label, score))
        if not prs:
            return {}
        return {
            "val_macro_auroc": float(np.mean(rocs)) if rocs else float("nan"),
            "val_macro_auprc": float(np.mean(prs)),
        }

    @torch.no_grad()
    def _score_pairs(self, ht: np.ndarray) -> np.ndarray:
        """(n,2) int endpoints -> (n, n_labels) sigmoid probs. Rows whose drugs are
        absent from dict1 get a 0.5 probability row."""
        self._model.eval()
        out = np.full((len(ht), self.n_labels), 0.5, dtype=np.float32)
        n_drug = len(self._dict1) if self._dict1 else 0
        for start in range(0, len(ht), self.batch_size):
            chunk = np.asarray(ht[start:start + self.batch_size], dtype=np.int64)
            in_vocab = (chunk[:, 0] >= 0) & (chunk[:, 0] < n_drug) \
                & (chunk[:, 1] >= 0) & (chunk[:, 1] < n_drug)
            if not in_vocab.any():
                continue
            kept = chunk[in_vocab]
            logits = self._forward_eval(kept)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            block_pos = np.where(in_vocab)[0]
            out[start + block_pos] = probs
        return out

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, ht: "np.ndarray | pd.DataFrame") -> np.ndarray:
        """(n, n_labels) sigmoid probabilities aligned to the input rows.

        ``ht`` is (n,2) int64 dict1-index endpoints (the wrapper maps the test
        drug ids -> dict1 idx first). OOV rows get a 0.5 row.
        """
        if self._model is None or self._dict1 is None:
            raise RuntimeError("fit() before predict_proba().")
        if isinstance(ht, pd.DataFrame):
            ht = ht[_PAIR].to_numpy()
        return self._score_pairs(np.asarray(ht, dtype=np.int64))

    # ------------------------------------------------------------------
    # save / load (persist n_labels)
    # ------------------------------------------------------------------
    def save(self, path: "str | Path") -> None:
        if self._model is None or self._dict1 is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), out / "model.pt")
        with (out / "state.pkl").open("wb") as f:
            pickle.dump({
                "dict1": self._dict1,
                "n_labels": self.n_labels,
                "cold": self.cold,
                "test_adj": self._test_adj,
                "unseen_ids": self._unseen_ids,
                "kgs": self._kgs,
                "tail_len": self._tail_len,
                "relation_len": self._relation_len,
            }, f)
        (out / "manifest.json").write_text(json.dumps({
            "baseline": "mkg_fenn", "task": "multilabel", "version": self.VERSION,
            "n_labels": self.n_labels, "cold": self.cold,
            "hyperparameters": {
                "embedding_num": self.embedding_num,
                "neighbor_sample_size": self.neighbor_sample_size,
                "dropout": self.dropout, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "batch_size": self.batch_size,
                "n_epochs": self.n_epochs, "fp_radius": self.fp_radius,
                "fp_nbits": self.fp_nbits, "n_bins": self.n_bins, "seed": self.seed,
            },
        }, indent=2))

    @classmethod
    def load(cls, path: "str | Path") -> "MKGFENNMultilabelBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hp = manifest.get("hyperparameters", {})
        inst = cls(n_labels=manifest["n_labels"], cold=manifest["cold"], **hp)
        with (p / "state.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._dict1 = payload["dict1"]
        inst.n_labels = payload["n_labels"]
        inst.cold = payload["cold"]
        inst._test_adj = payload["test_adj"]
        inst._unseen_ids = payload["unseen_ids"]
        inst._kgs = payload["kgs"]
        inst._tail_len = payload["tail_len"]
        inst._relation_len = payload["relation_len"]
        drug_name = list(range(len(inst._dict1)))
        args = inst._args()
        if inst.cold:
            inst._model = MKGFENNCold(
                kgs=payload["kgs"], tail_len=payload["tail_len"],
                relation_len=payload["relation_len"], dict1=inst._dict1,
                drug_name=drug_name, args=args, event_num=inst.n_labels,
            ).to(inst.device)
        else:
            inst._model = MKGFENN(
                kgs=payload["kgs"], tail_len=payload["tail_len"],
                relation_len=payload["relation_len"], dict1=inst._dict1,
                drug_name=drug_name, embedding_num=inst.embedding_num,
                neighbor_sample_size=inst.neighbor_sample_size,
                dropout=inst.dropout, event_num=inst.n_labels,
            ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst


__all__ = ["MKGFENNMultilabelBaseline", "PAPER_HYPERPARAMS", "DEFAULT_N_LABELS"]

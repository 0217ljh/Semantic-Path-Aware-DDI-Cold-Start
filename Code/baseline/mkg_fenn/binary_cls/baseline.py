"""MKG-FENN binary core (regime-aware: S0 warm 4-channel, S1/S2 cold 3-channel).

Binary DDI-existence (2-way) adaptation of MKG-FENN. Case-B: the paper never did
binary (paper task = 65-event multiclass with softmax CE). This REUSES the
regime-aware multiclass infrastructure (``multi_cls.baseline.MKGFENNMulticlassBaseline``
is the template) and swaps ONLY the task surface:

  * ``event_num = 2`` (2-way head instead of K-way);
  * training set = per-epoch ``{pos label=1, neg label=0}`` (deterministic negatives
    from the leaf adapter, design B) instead of positives-with-ddi_type;
  * best checkpoint by VAL AUPRC (binary PRIMARY metric; run_baseline_unified
    ``_binary_metrics``) instead of val macro-F1;
  * ``predict_proba`` returns ``(n,)`` = ``softmax(logits)[:, 1]`` (positive score,
    mirroring the dead binary adapter baseline.py:281) instead of ``(n, K)``.

The COLD-START ALGORITHM CORE is TASK-AGNOSTIC and is KEPT UNCHANGED (CLAUDE.md
case-B: 不允许借口新任务省略 paper 核心):
  * S0 (transductive)  -> 4-channel warm model (``baseline.mkg_fenn.model.MKGFENN``,
    event_num=2). No imputation, no test_adj.
  * S1/S2 (inductive)  -> 3-channel cold model (``model_cold.MKGFENNCold``, event_num=2)
    with per-channel nearest-seen-neighbour imputation via ``test_adj``.

The dead binary adapter (``baseline.mkg_fenn.baseline.MKGFENNBaseline``) WRONGLY used
the 4-channel warm model for cold regimes; this core does NOT repeat that — S1/S2 use
the cold 3ch+impute model, exactly as the multiclass core does.

Reuse strategy (CLAUDE.md §baseline §3.2: same-package intra-baseline import allowed):
  * MODEL classes ``MKGFENN`` / ``MKGFENNCold`` imported from their existing modules.
  * KG builder ``build_all_kgs`` reused UNCHANGED.
  * The pure cold-plumbing helpers (``find_dif``, ``jaccard``, ``_build_feature_matrices``,
    ``_build_test_adj``, ``_ghost_pad_kg1``, ``_drug_smiles_dict``) are IMPORTED from
    ``multi_cls.baseline`` rather than duplicated. Reason: they are pure functions with a
    single verified faithful port; importing keeps ONE source of truth and avoids silent
    drift between the two task variants (CLAUDE.md file-level independence targets
    cross-directory reproduction<->baseline coupling, not same-package intra-baseline
    reuse, which is explicitly allowed). ``multi_cls`` is left untouched.

Training protocol (task3.py, task-agnostic — kept): deterministic seeding, symmetric
pair augmentation (a,b)+(b,a) applied to BOTH pos and neg rows (codex 019f2479), Adam +
CrossEntropyLoss (no scheduler, no early-stop), per-epoch shuffle + per-epoch fresh
deterministic negatives, best checkpoint by VAL AUPRC, TrainProgress logging.
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
import torch.nn.functional as F
from torch import optim

from baseline.mkg_fenn.kg_builder import build_all_kgs
from baseline.mkg_fenn.model import MKGFENN
from baseline.mkg_fenn.multi_cls.model_cold import MKGFENNCold
# Same-package intra-baseline import of the pure cold-plumbing helpers (single source
# of truth; multi_cls left untouched). CLAUDE.md §baseline allows this.
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

#: Binary head dimension (2-way softmax: 0 = no interaction, 1 = interaction).
BINARY_EVENT_NUM = 2


class MKGFENNBinaryBaseline:
    """Regime-aware MKG-FENN binary core (S0 warm / S1-S2 cold), event_num=2.

    Not a legacy ``BaselineModel`` — consumed by the unified wrapper via a
    duck-typed ``_LeafDataset`` (leaf_adapter.make_dataset(task="binary")). The
    wrapper passes the native KG1 object as ``kg`` and the leaf's ``cold`` flag /
    unseen drug ids. Per-epoch deterministic negatives come from the leaf's
    ``get_train_negatives(epoch, regenerate=True)`` provider.
    """

    VERSION = "1.0-bin-cold"

    def __init__(
        self,
        *,
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
        self.event_num = BINARY_EVENT_NUM
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

        Same block as the multiclass core (``use_deterministic_algorithms`` with
        ``warn_only=True`` faithful-equivalent; cuDNN deterministic/benchmark/enabled).
        The per-leaf runner launches one process per baseline, so these process-global
        flags do not leak across baselines.
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

    def _pair_indices(self, pairs: pd.DataFrame) -> tuple[torch.Tensor | None, np.ndarray]:
        """Map (drug_a_id, drug_b_id) pairs to dict1 index pairs; drop OOV rows."""
        assert self._dict1 is not None
        a_ids = pairs["drug_a_id"].astype(str).to_numpy()
        b_ids = pairs["drug_b_id"].astype(str).to_numpy()
        keep_idx, idx_pairs = [], []
        for i, (a, b) in enumerate(zip(a_ids, b_ids)):
            ai = self._dict1.get(a)
            bi = self._dict1.get(b)
            if ai is None or bi is None:
                continue
            keep_idx.append(i)
            idx_pairs.append((ai, bi))
        mask = np.zeros(len(pairs), dtype=bool)
        if not keep_idx:
            return None, mask
        mask[np.asarray(keep_idx)] = True
        return torch.tensor(idx_pairs, dtype=torch.long), mask

    def _build_dict1(self, train) -> dict[str, int]:
        """All drugs across splits + drugs table -> contiguous idx (baseline.py:113-123).

        Unseen drugs KEPT (KG3 self-loop) so cold drugs stay reachable at predict time.
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
        val=None,
        *,
        kg: "KnowledgeGraph | None" = None,
        dict1: dict[str, int] | None = None,
        unseen_ids: list[int] | None = None,
    ) -> None:
        """Train on one leaf's DDI pairs (binary DDI-existence).

        Parameters
        ----------
        train, val
            Duck-typed ``_LeafDataset`` (leaf_adapter, task="binary"). Positives are
            ``train.splits.train``; per-epoch deterministic negatives come from
            ``train.get_train_negatives(epoch, regenerate=True)``.
        kg
            Native drug->entity KnowledgeGraph (KG1 source). The unified wrapper passes
            this (the adapter sets ``_LeafDataset.kg = None``).
        dict1
            Optional precomputed drug vocab (all drugs -> idx). Built here if None.
        unseen_ids
            Drug ids (dict1 index space) unseen at train time (cold regimes). Empty for S0.
        """
        self._seed_all()
        if kg is None:
            kg = train.kg
        if kg is None:
            raise ValueError("MKG-FENN binary requires a KG1 object (drug->entity).")

        # Drug vocab (all drugs across splits + drugs table). baseline.py:113-123.
        self._dict1 = dict1 if dict1 is not None else self._build_dict1(train)
        n_drug = len(self._dict1)
        self._unseen_ids = list(unseen_ids) if unseen_ids else []

        # Train positives (for KG3 / feature matrices / positive supervision rows).
        pos = train.splits.train
        event_num = BINARY_EVENT_NUM

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

        # Ghost-pad KG1 empties for the COLD model (Option A). Warm MKGFENN ghost-handles
        # itself, so only the cold path pads (avoids a double ghost slot).
        if self.cold:
            kgs, tail_len, relation_len = _ghost_pad_kg1(kgs, tail_len, relation_len, self._dict1)
        self._kgs, self._tail_len, self._relation_len = kgs, tail_len, relation_len

        # Model (regime switch, event_num=2).
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

        # Positive index pairs (fixed across epochs).
        pos_x = self._pairs_to_idx(pos)

        loss_function = nn.CrossEntropyLoss()
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress

        # Rough step estimate (pos + neg, x2 for symmetric aug); TrainProgress only
        # uses this for the step-fraction display.
        est_steps = (2 * 2 * len(pos_x) + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs, log_step_every=self.log_step_every,
            total_steps_per_epoch=max(est_steps, 1), prefix="[mkg_fenn_bin] ",
        )

        best_val_auprc = -1.0
        best_state: dict | None = None
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.n_epochs):
            # Per-epoch DETERMINISTIC negatives (design B). NOT a hand-rolled sampler.
            neg_df = train.get_train_negatives(epoch, regenerate=True)
            neg_x = self._pairs_to_idx(neg_df)

            # {pos=1, neg=0}. Symmetric pair augmentation (a,b)+(b,a) applied to BOTH
            # pos and neg rows (codex 019f2479; task3.py:176-179 symmetric범 extended
            # to the binary training set). Concatenate reversed pairs, duplicate labels.
            x = np.concatenate([pos_x, neg_x], axis=0)
            y = np.concatenate([
                np.ones(len(pos_x), dtype=np.int64),
                np.zeros(len(neg_x), dtype=np.int64),
            ], axis=0)
            x_rev = x[:, [1, 0]]
            x_total = np.concatenate([x, x_rev], axis=0)
            y_total = np.concatenate([y, y], axis=0)

            n_total = len(x_total)
            perm = rng.permutation(n_total)            # per-epoch shuffle
            xt = x_total[perm]
            yt = y_total[perm]
            if not self.cold:
                self._model.precompute_adj()           # warm path re-samples neighbours
            self._model.train()
            progress.epoch_start(epoch)
            for start in range(0, n_total, self.batch_size):
                xb = xt[start:start + self.batch_size]
                yb = yt[start:start + self.batch_size]
                if len(xb) < 2:                        # BatchNorm1d needs >=2 rows
                    continue
                idx_tensor = torch.as_tensor(xb, dtype=torch.long)
                yb_t = torch.as_tensor(yb, dtype=torch.long, device=self.device)
                opt.zero_grad(set_to_none=True)
                logits = self._forward_train(idx_tensor)
                loss = loss_function(logits, yb_t)
                loss.backward()
                opt.step()
                progress.step(loss.item())
            extra = {}
            if val is not None:
                metrics = self._eval(val)
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_auprc", -1) > best_val_auprc:
                        best_val_auprc = metrics["val_auprc"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mkg_fenn_bin] loaded best val_auprc={best_val_auprc:.4f}", flush=True)

    def _pairs_to_idx(self, pairs: pd.DataFrame) -> np.ndarray:
        """(m, 2) int64 dict1-index pairs; OOV rows dropped."""
        assert self._dict1 is not None
        a = pairs["drug_a_id"].astype(str).map(self._dict1).to_numpy()
        b = pairs["drug_b_id"].astype(str).map(self._dict1).to_numpy()
        stacked = np.stack([a, b], axis=1)
        keep = ~(pd.isna(stacked).any(axis=1))
        return stacked[keep].astype(np.int64)

    def _forward_train(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        """Train-time forward (no imputation; train_or_test=0 for cold)."""
        if self.cold:
            return self._model(idx_tensor, 0, None)
        return self._model(idx_tensor)

    def _forward_eval(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        """Eval-time forward. Cold path enables nearest-seen imputation (train_or_test=1).

        Imputation is a per-drug embedding fallback for unseen drugs and is label-
        agnostic, so it applies identically to positive and negative pairs
        (codex 019f2479).
        """
        if self.cold and self._test_adj is not None:
            return self._model(idx_tensor, 1, self._test_adj)
        if self.cold:
            return self._model(idx_tensor, 0, None)
        return self._model(idx_tensor)

    @torch.no_grad()
    def _eval(self, val) -> dict:
        """Val AUPRC over {val positives + val negatives} (binary PRIMARY metric).

        Mirrors run_baseline_unified ``_binary_metrics`` (PRIMARY = auprc). OOV rows
        (drugs not in dict1) are DROPPED rather than 0.5-filled (codex 019f2479:
        filler perturbs ranking-based AUPRC and pollutes checkpoint selection).
        """
        from sklearn.metrics import average_precision_score
        pos = val.splits.val_s2
        neg = val.get_negatives("val_s2")
        if pos is None or neg is None or len(pos) == 0 or len(neg) == 0:
            return {}
        pos_score = self.predict_proba(pos[_PAIR], _fill_oov=np.nan)
        neg_score = self.predict_proba(neg[_PAIR], _fill_oov=np.nan)
        y_score = np.concatenate([pos_score, neg_score])
        y_true = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
        keep = ~np.isnan(y_score)                      # drop OOV rows
        if keep.sum() < 2 or len(set(y_true[keep].tolist())) < 2:
            return {}
        return {"val_auprc": float(average_precision_score(y_true[keep], y_score[keep]))}

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, pairs: pd.DataFrame, *, _fill_oov: float = 0.5) -> np.ndarray:
        """(n,) P(interaction) = ``softmax(logits)[:, 1]`` (dead adapter baseline.py:281).

        OOV rows (drugs absent from dict1) get ``_fill_oov`` (default 0.5, mirroring the
        dead binary adapter so the runner can score every test row). ``_eval`` passes
        ``np.nan`` so it can drop OOV rows before AUPRC (codex 019f2479).
        """
        if self._model is None or self._dict1 is None:
            raise RuntimeError("fit() before predict_proba().")
        self._model.eval()
        out = np.full(len(pairs), _fill_oov, dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            idx_tensor, mask = self._pair_indices(batch)
            if idx_tensor is None:
                continue
            idx_tensor = idx_tensor.to(self.device)
            logits = self._forward_eval(idx_tensor)
            probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            block = np.full(len(batch), _fill_oov, dtype=np.float32)
            block[mask] = probs
            out[start:start + len(batch)] = block
        return out

    # ------------------------------------------------------------------
    # save / load
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
                "event_num": self.event_num,
                "cold": self.cold,
                "test_adj": self._test_adj,
                "unseen_ids": self._unseen_ids,
                "kgs": self._kgs,
                "tail_len": self._tail_len,
                "relation_len": self._relation_len,
            }, f)
        (out / "manifest.json").write_text(json.dumps({
            "baseline": "mkg_fenn", "task": "binary", "version": self.VERSION,
            "event_num": self.event_num, "cold": self.cold,
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
    def load(cls, path: "str | Path") -> "MKGFENNBinaryBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hp = manifest.get("hyperparameters", {})
        inst = cls(cold=manifest["cold"], **hp)
        with (p / "state.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._dict1 = payload["dict1"]
        inst.event_num = payload["event_num"]
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
                drug_name=drug_name, args=args, event_num=inst.event_num,
            ).to(inst.device)
        else:
            inst._model = MKGFENN(
                kgs=payload["kgs"], tail_len=payload["tail_len"],
                relation_len=payload["relation_len"], dict1=inst._dict1,
                drug_name=drug_name, embedding_num=inst.embedding_num,
                neighbor_sample_size=inst.neighbor_sample_size,
                dropout=inst.dropout, event_num=inst.event_num,
            ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst


__all__ = ["MKGFENNBinaryBaseline", "PAPER_HYPERPARAMS", "BINARY_EVENT_NUM"]

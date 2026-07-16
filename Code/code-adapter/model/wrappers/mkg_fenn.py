"""MKGFENNRankWrapper - MKG-FENN (multimodal-KG four-channel fused end-to-end network)
backbone plugged into the code-adapter harness. Written fresh (no rank_analysis wrapper
existed), REUSING the verified regime-aware MKG-FENN cores UNCHANGED:
``baseline.mkg_fenn.binary_cls.baseline.MKGFENNBinaryBaseline`` (binary) /
``baseline.mkg_fenn.multi_cls.baseline.MKGFENNMulticlassBaseline`` (multiclass).

KEY TRICK (avoids re-implementing the intricate cold-start setup): the core's ``fit()``
runs ALL setup (dict1, the 4 intrinsic KGs, cold plumbing drug_sim1..4 + ``_build_test_adj``,
KG1 ghost-pad, ``MKGFENNCold``/``MKGFENN`` model construction) BEFORE the ``for epoch in
range(self.n_epochs)`` loop. We build the core with ``n_epochs=0``, so ``fit(ds, ds, kg,
dict1, unseen_ids)`` BUILDS everything (``core._dict1/_kgs/_test_adj/_model``) with NO
training and ``best_state`` stays ``None`` (no load). The wrapper then creates its OWN
optimizer over ``core._model.parameters()`` and drives training per-epoch via the harness,
reusing the core's ``_forward_train`` / ``_forward_eval`` / ``_pairs_to_idx``.

MKG-FENN's KG1 (drug->{enzyme,target,transporter,carrier,pathway}) + KG2 (Morgan FP) +
KG3 (train DDI, self-looped) + KG4 (property bins) are INTRINSIC drug side-info, NOT the
merged DDI-fact KG. Cold-start imputation is feature-based (nearest-seen neighbour via
``test_adj``), never DDI. So encode_pairs / train_epoch IGNORE the ScoringContext ->
leak-free by construction (a pair's own DDI edge is never an input), like ssi_ddi / tiger.
KG1 is loaded from the native filtered tables at ``Code/data/KG/drugbank/filtered/``.

The model returns RAW LOGITS (fusion head ends in ``nn.Linear(.., event_num)``, no softmax;
verified via ``predict_proba`` applying ``F.softmax`` itself). Binary head is event_num=2:
we return ``logits[:,1] - logits[:,0]`` so the harness sigmoid recovers ``softmax(logits)[1]``
= P(interaction). Multiclass: dense train-vocab probs are scattered to their GLOBAL class
columns (``core._idx_to_ddi_type[i]`` = str(global y_cls)) then log'd, so the harness softmax
recovers the scattered probabilities (same shape as the tiger multiclass path).

Needs hp['dataset']+hp['fold'] (to load the leaf for drugs/KG1/regime); the entrypoint
injects them. "wrapper" = harness glue, NOT the M_A.M_B adapter.

PROTOCOL: native P1_FIXED ONLY. KG3 (train-DDI) + the cold nearest-seen imputation
(test_adj) are built ONCE at setup from the leaf's full seen DDI and are NOT rebuilt
per epoch, so a forced ``--protocol p2`` would NOT reflect P2's per-epoch emerging-edge
removals in KG3/imputation (non-faithful training regime; cold-test itself stays
leak-free since cold drugs are unseen). Run this baseline under its native P1 only.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "Code"))
sys.path.insert(0, str(_ROOT / "Code" / "scripts"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
from data_utils.unified import DATASET_DIRS  # noqa: E402
from data_utils.kg import KnowledgeGraph  # noqa: E402
from baseline.mkg_fenn.binary_cls.baseline import MKGFENNBinaryBaseline  # noqa: E402
from baseline.mkg_fenn.multi_cls.baseline import MKGFENNMulticlassBaseline  # noqa: E402

from ..contracts import (EpochData, PairEncoding, RankModel, ScoringContext,  # noqa: E402
                         TrainEpochOutput)

#: native KG1 source (drug->side-info), intrinsic + cold-start safe (NOT the merged KG).
_KG1_FILTERED_DIR = "Code/data/KG/drugbank/filtered"

#: pair columns the cores' _pairs_to_idx reads
_PAIR = ["drug_a_id", "drug_b_id"]

#: tunable hyperparameters (exposed via hp). Defaults mirror the cores' __init__
#: (binary_cls/baseline.py:97-115, multi_cls/baseline.py:228-247 = PAPER_HYPERPARAMS).
_DEFAULTS = dict(
    embedding_num=128, neighbor_sample_size=6, dropout=0.3,
    learning_rate=1e-2, weight_decay=1e-8, batch_size=256,
    fp_radius=2, fp_nbits=512, n_bins=10, seed=42,
)


class MKGFENNRankWrapper(RankModel):
    # -- build ---------------------------------------------------------------
    def setup(self, task, hp: dict) -> None:
        self.task = task
        self.hp = {**_DEFAULTS, **(hp or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_str = "cuda" if self.device.type == "cuda" else "cpu"
        torch.manual_seed(self.hp["seed"]); np.random.seed(self.hp["seed"])

        dataset = self.hp["dataset"]; fold = self.hp["fold"]
        dir2group = {v: k for k, v in DATASET_DIRS.items()}
        if dataset not in dir2group:
            raise ValueError(f"dataset dir {dataset!r} not in DATASET_DIRS values "
                             f"{sorted(DATASET_DIRS.values())}")
        group = dir2group[dataset]
        leaf_task = "binary" if task.is_binary else "multiclass"

        unified_root = _ROOT / "Code" / "data" / "ddi_unified"
        leaf = U.load_leaf(str(unified_root), group, leaf_task, "cold_s2", fold)
        ds = make_dataset(leaf.train, leaf.val, leaf.resources, task=leaf_task)

        # Cold-setup inputs from leaf.resources (mirrors the mkg_fenn unified helpers
        # MKGFENNUnified{Binary,Multiclass}._is_cold/_build_dict1/_unseen_ids/_load_kg1;
        # replicated here to keep the wrapper self-contained, i.e. importing it does NOT
        # trigger the unified registry's import-time @register_unified side effects).
        res = leaf.resources
        cold = self._is_cold(res)                              # S2 -> True (inductive)
        dict1 = self._build_dict1(res)
        unseen_ids = self._unseen_ids(res, dict1) if cold else []
        kg1 = self._load_kg1()
        # class output width in GLOBAL id space (binary head=2; multiclass = task.n_classes,
        # matching the harness metrics' proba[:, c] global-column indexing).
        self.n_classes = 2 if task.is_binary else int(task.n_classes)

        # Build the core with n_epochs=0: fit() runs ALL setup (dict1/_kgs/_test_adj/_model)
        # before the epoch loop, so this BUILDS everything with NO training (best_state stays
        # None -> no load). VERIFIED: binary_cls/baseline.py fit() setup lines 239-306, loop
        # 326-373 skipped when n_epochs=0, best-state load guard line 375; multi_cls/baseline.py
        # setup 363-432, loop 468-498 skipped, load guard 500.
        common = dict(
            cold=cold, n_epochs=0, seed=self.hp["seed"],
            embedding_num=self.hp["embedding_num"],
            neighbor_sample_size=self.hp["neighbor_sample_size"],
            dropout=self.hp["dropout"], learning_rate=self.hp["learning_rate"],
            weight_decay=self.hp["weight_decay"], batch_size=self.hp["batch_size"],
            fp_radius=self.hp["fp_radius"], fp_nbits=self.hp["fp_nbits"],
            n_bins=self.hp["n_bins"], device=dev_str,
        )
        if task.is_binary:
            self.core = MKGFENNBinaryBaseline(**common)
        else:
            self.core = MKGFENNMulticlassBaseline(n_classes=int(task.n_classes), **common)
        self.core.fit(ds, ds, kg=kg1, dict1=dict1, unseen_ids=unseen_ids)   # builds, no train
        if self.core._model is None:
            raise RuntimeError("MKG-FENN core._model was not built by the n_epochs=0 fit()")

        # OWN optimizer over the built model (the harness drives the epoch loop).
        self.opt = torch.optim.Adam(self.core._model.parameters(),
                                    lr=self.core.learning_rate,
                                    weight_decay=self.core.weight_decay)
        self._loss_fn = nn.CrossEntropyLoss()

    # -- cold-setup helpers (mirror the mkg_fenn unified baseline helpers) ----
    @staticmethod
    def _is_cold(resources) -> bool:
        """Cold iff inductive regime (S1/S2). Transductive S0 -> warm."""
        return str(resources.meta.get("regime", "")).lower() == "inductive"

    @staticmethod
    def _build_dict1(resources) -> dict:
        """All drugs -> contiguous sorted idx (covers seen + unseen so cold drugs stay
        reachable). Matches core._build_dict1 / the unified helper."""
        drug_ids = set(resources.drugs["drug_id"].astype(str))
        return {did: i for i, did in enumerate(sorted(drug_ids))}

    @staticmethod
    def _unseen_ids(resources, dict1: dict) -> list:
        """Unseen (test/val/eval_unseen) drug ids in dict1 index space (cold regimes)."""
        roles = resources.drug_split
        g2 = roles[roles["role"].isin(["test", "val", "eval_unseen"])]["drug_id"].astype(str)
        return sorted({dict1[d] for d in g2 if d in dict1})

    def _load_kg1(self) -> KnowledgeGraph:
        """Native KG1 tables from the filtered dir (intrinsic drug side-info, NOT the
        merged DDI-fact KG). Same source as the unified helper's _load_kg1."""
        kg_dir = _ROOT / _KG1_FILTERED_DIR
        if not kg_dir.is_dir():
            raise FileNotFoundError(f"native KG1 filtered dir not found: {kg_dir}")
        return KnowledgeGraph.from_filtered_dir(kg_dir)

    # -- index helpers (dict1 space; drop OOV) -------------------------------
    def _pairs_to_idx_masked(self, pairs: np.ndarray):
        """(kept_idx (m,2) int64, mask (n,) bool) aligned to `pairs` order; OOV dropped.

        Unlike the core's _pairs_to_idx (which drops rows and returns idx WITHOUT a mask),
        this keeps a positional mask so labels + output rows can be aligned."""
        d = self.core._dict1
        keep, rows = [], []
        for i, (a, b) in enumerate(pairs[:, :2]):
            ai = d.get(str(a)); bi = d.get(str(b))
            if ai is None or bi is None:
                continue
            keep.append(i); rows.append((ai, bi))
        mask = np.zeros(len(pairs), dtype=bool)
        if keep:
            mask[np.asarray(keep)] = True
        idx = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        return idx, mask

    # -- RankModel contract (KG-free: ScoringContext ignored) ----------------
    def train_epoch(self, epoch: EpochData, rng) -> TrainEpochOutput:
        pairs = np.asarray(epoch.target_pairs)
        y = np.asarray(epoch.target_labels)
        n = len(y)
        if n == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)

        # dict1 idx aligned to labels; drop OOV rows (known_drugs filtering should prevent
        # OOV, this is defensive). For multiclass, map GLOBAL class id -> the core's dense
        # train-vocab idx (core._ddi_type_to_idx keyed by str(global id)); drop labels
        # whose class is not in the train vocab.
        idx, mask = self._pairs_to_idx_masked(pairs)
        if len(idx) == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)
        y_kept = y[mask]
        if self.task.is_binary:
            x = idx
            yt = y_kept.astype(np.int64)
        else:
            t2i = self.core._ddi_type_to_idx
            dense = np.array([t2i.get(str(int(c)), -1) for c in y_kept], dtype=np.int64)
            in_vocab = dense >= 0
            x = idx[in_vocab]
            yt = dense[in_vocab]
        if len(x) == 0:
            return TrainEpochOutput(mean_loss=0.0, n_targets=0)

        # Symmetric pair augmentation (a,b)+(b,a) with duplicated labels (core fit
        # binary_cls/baseline.py:334-346 / multi_cls/baseline.py:434-445).
        x_total = np.concatenate([x, x[:, [1, 0]]], axis=0)
        y_total = np.concatenate([yt, yt], axis=0)

        n_total = len(x_total)
        perm = rng.permutation(n_total)
        xt = x_total[perm]; yt = y_total[perm]

        if not self.core.cold:
            self.core._model.precompute_adj()            # warm path re-samples neighbours
        self.core._model.train()
        bs = int(self.hp["batch_size"]); total = 0.0; counted = 0
        for s in range(0, n_total, bs):
            xb = xt[s:s + bs]; yb = yt[s:s + bs]
            if len(xb) < 2:                              # BatchNorm1d needs >=2 rows
                continue
            idx_tensor = torch.as_tensor(xb, dtype=torch.long)
            yb_t = torch.as_tensor(yb, dtype=torch.long, device=self.device)
            self.opt.zero_grad(set_to_none=True)
            logits = self.core._forward_train(idx_tensor)          # (B, event_num) raw logits
            loss = self._loss_fn(logits, yb_t)
            loss.backward()
            self.opt.step()
            total += float(loss.item()) * len(xb); counted += len(xb)
        return TrainEpochOutput(mean_loss=total / max(counted, 1), n_targets=n)

    @torch.no_grad()
    def encode_pairs(self, pairs, context: ScoringContext) -> PairEncoding:
        pairs = np.asarray(pairs)
        self.core._model.eval()
        n = len(pairs)
        bs = int(self.hp["batch_size"])
        if self.task.is_binary:
            out = np.zeros((n,), dtype=np.float32)                 # OOV/dropped -> 0 logit
        else:
            out = np.zeros((n, self.n_classes), dtype=np.float32)  # OOV/dropped -> uniform
        idx2glob = None if self.task.is_binary else self.core._idx_to_ddi_type
        for s in range(0, n, bs):
            block = pairs[s:s + bs]
            idx, mask = self._pairs_to_idx_masked(block)
            if len(idx) == 0:
                continue
            idx_tensor = torch.as_tensor(idx, dtype=torch.long, device=self.device)
            logits = self.core._forward_eval(idx_tensor)           # (B_kept, event_num) raw logits
            if self.task.is_binary:
                lg = logits.detach().cpu().numpy()
                kept = (lg[:, 1] - lg[:, 0]).astype(np.float32)    # log-odds; sigmoid -> softmax[1]
                slot = np.zeros((len(block),), dtype=np.float32)
                slot[mask] = kept
                out[s:s + len(block)] = slot
            else:
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()   # (B_kept, event_num)
                # scatter dense train-vocab probs -> GLOBAL class columns, then log so the
                # harness softmax recovers the scattered probabilities (tiger multiclass path).
                glob = np.zeros((probs.shape[0], self.n_classes), dtype=np.float32)
                for i, t in enumerate(idx2glob):
                    gc = int(t)
                    if gc < self.n_classes:
                        glob[:, gc] = probs[:, i]
                kept = np.log(np.clip(glob, 1e-9, 1.0)).astype(np.float32)
                slot = np.zeros((len(block), self.n_classes), dtype=np.float32)
                slot[mask] = kept
                out[s:s + len(block)] = slot
        return PairEncoding(pair_ids=pairs, pair_repr=None, logits=out.astype(np.float32),
                            repr_kind="mkgfenn_4channel_fusion",
                            repr_stage=("logodds" if self.task.is_binary else "softmax_to_logit"))

    @property
    def model(self):
        """The trainable nn.Module (composer contract). MKG-FENN keeps it at core._model;
        this alias gives the composer the same `.model` handle every wrapper exposes."""
        return self.core._model

    # -- adapter-composition interface (KG channels ONLY; molecular GNN2 skipped) --------
    def _kg_pair_channels(self, pairs):
        """Grad-enabled per-KG-channel pair reps, SKIPPING the molecular channel (GNN2 / KG2
        Morgan FP) per the multimodal integration rule. Runs the model's OWN gnn1 (KG1 side-
        info) + gnn3 (KG3 DDI topology) directly; gnn3 never CONSUMES gnn2's output (only
        unpacks + passes it through), and each channel computes its own embedding, so dropping
        gnn2 is a valid KG-only readout. Respects the caller's train/eval mode + the cold
        nearest-seen imputation exactly as core._forward_train / _forward_eval do.
        NOTE (codex 019f5d66): gnn1/gnn3 sample neighbours via np.random.choice, so skipping
        gnn2 shifts the NumPy RNG state gnn3 sees vs a full gnn1->gnn2->gnn3 run -> gnn3's
        sampled neighbourhood is a DIFFERENT but equally valid draw, NOT the identical one.
        This is immaterial: the cold path re-samples every forward regardless (encode_pairs
        itself is non-deterministic) and we only need a valid KG3-channel readout. Consequently
        gnn2's params receive no gradient in joint mode -- molecular left untouched, by design.
        Returns (kg1, kg3), each [N, 2*embedding_num] = [chan_A || chan_B]."""
        idx, mask = self._pairs_to_idx_masked(np.asarray(pairs))
        if not mask.all():
            raise ValueError("MKG-FENN pair_forward/pair_streams got OOV pairs; the composer "
                             "must filter to known_drugs() first")
        m = self.core._model
        if m.training:                                    # == core._forward_train (no imputation)
            tot, tadj = 0, defaultdict(list)
        elif self.core.cold and self.core._test_adj is not None:   # == core._forward_eval (cold)
            tot, tadj = 1, self.core._test_adj
        else:
            tot, tadj = 0, defaultdict(list)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        drug_f1, idx_o, tadj_o, tot_o = m.gnn1((idx_t, tot, tadj))          # KG1 channel
        drug_f3, _g2, gnn1_emb, idx_o = m.gnn3((None, drug_f1, idx_o, tadj_o, tot_o))  # KG3 (gnn2 skipped)
        ii = idx_o.cpu().numpy().tolist() if torch.is_tensor(idx_o) else list(idx_o)
        a = [p[0] for p in ii]; b = [p[1] for p in ii]
        kg1 = torch.cat([gnn1_emb[a], gnn1_emb[b]], dim=1)   # [N, 2*emb]
        kg3 = torch.cat([drug_f3[a], drug_f3[b]], dim=1)     # [N, 2*emb]
        return kg1, kg3

    def pair_forward(self, pairs, context: ScoringContext) -> "torch.Tensor":
        """Grad-enabled KG-only pre-scorer pair rep [N, 4*embedding_num] = [KG1 || KG3] (the
        molecular KG2/Morgan channel is dropped per the multimodal rule). For design-R
        composition; respects caller train/eval mode. ScoringContext ignored (KG1/KG3 are
        intrinsic side-info; cold imputation is feature-based, leak-free) like encode_pairs."""
        kg1, kg3 = self._kg_pair_channels(pairs)
        return torch.cat([kg1, kg3], dim=1)                  # [N, 4*emb]

    def pair_streams(self, pairs, context: ScoringContext) -> "list":
        """Per-KG-channel grad-enabled streams for design-R 'default' [KG1_pair, KG3_pair],
        each [N, 2*embedding_num]. MKG-FENN is single-hop (no depth), so the natural per-
        'layer' unit is the per-KG-channel readout; the molecular KG2 channel is skipped.
        cat(pair_streams) == pair_forward by construction."""
        kg1, kg3 = self._kg_pair_channels(pairs)
        return [kg1, kg3]

    def effective_hp(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.hp.items()}

    def known_drugs(self) -> set:
        # All drugs in the model vocab. Cold/unseen drugs ARE in dict1 (scored via
        # feature-based imputation), so they count as "known"; the harness drops any
        # pair whose drug is outside this set.
        return set(self.core._dict1.keys())

    def state_dict(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.core._model.state_dict().items()}

    def load_state_dict(self, state: dict) -> None:
        self.core._model.load_state_dict(state)


__all__ = ["MKGFENNRankWrapper"]

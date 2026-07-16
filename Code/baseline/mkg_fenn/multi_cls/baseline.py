"""MKG-FENN multiclass core (regime-aware: S0 warm 4-channel, S1/S2 cold 3-channel).

Faithful multiclass (65-event-style) adaptation of MKG-FENN for the project's cold-start
DDI benchmark. The paper's real task is multiclass over interaction types with softmax
cross-entropy (``--event_num default=65``; MKG-FENN-task{1,2,3}.py:38-39, CE loss at
task3.py:172). We keep that task; ``event_num`` is re-vocabbed to the count of
train-observed ddi_type classes (CLAUDE.md case-A: vocab size follows our data).

Regime switch (Notes/Log/mkg_fenn_faithful_cold_port.md):
  * S0 (transductive)  -> 4-channel warm model (``baseline.mkg_fenn.model.MKGFENN``,
    ported from modeltask1.py). No imputation, no test_adj.
  * S1/S2 (inductive)  -> 3-channel cold model (``model_cold.MKGFENNCold``, ported from
    modeltask2/3.py) with per-channel nearest-seen-neighbour imputation via ``test_adj``.

Cold plumbing (ported from MKG-FENN-task3.py):
  * ``drug_sim1..4`` per-channel drug-drug similarity (task3.py:313-319):
      - drug_sim1 = Jaccard(feature_matrix1)   (KG1 drug->entity)
      - drug_sim2 = Jaccard(feature_matrix2)   (KG2 drug->Morgan substructure)
      - drug_sim3 = find_dif(feature_matrix3)  (KG4 drug->property, custom)
      - drug_sim4 = Jaccard(feature_matrix4)   (KG3 drug->DDI)
  * ``test_adj`` (task3.py:344-395): per unseen drug j, per channel k in {0,1,2,3},
    the argmax-similarity SEEN drug(s) p with ``p not in unseen`` and ``p != j`` (ties
    kept as a list). GNN1<-test_adj[0], GNN2<-test_adj[1], GNN3<-test_adj[3].

Training protocol (task3.py): deterministic seeding, symmetric pair augmentation
(a,b)+(b,a) (task3.py:176-179), Adam + CrossEntropyLoss (no scheduler, no early-stop),
per-epoch shuffle, best checkpoint by VAL macro-F1 (paper primary metric; task3.py
selects best by test macro-F1, we use val to avoid test leakage). TrainProgress
per-epoch loss + eval logging (CLAUDE.md).

``predict_proba`` returns ``(n, K_train)`` softmax probabilities over the dense
train-observed class vocab; the unified wrapper scatters these to global class ids.
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

if TYPE_CHECKING:  # pragma: no cover
    from data_utils.kg import KnowledgeGraph

_PAIR = ["drug_a_id", "drug_b_id"]

#: Paper-spec hyperparameters (baseline.py:48-56, App C.1 Table 8 MKG-FENN row).
PAPER_HYPERPARAMS: dict[str, object] = {
    "embedding_num":         128,
    "neighbor_sample_size":  6,
    "dropout":               0.3,
    "learning_rate":         1e-2,
    "weight_decay":          1e-8,
    "batch_size":            256,
    "n_epochs":              50,
}


# ---------------------------------------------------------------------------
# Cold-plumbing helpers — verbatim ports from MKG-FENN-task3.py
# ---------------------------------------------------------------------------
def find_dif(raw_matrix: np.ndarray, n_drug: int) -> np.ndarray:
    """Custom drug-drug similarity for the property channel. task3.py:232-245.

    Counts, over the first 20 property columns, how many entries two drugs share.
    ``n_drug`` replaces the hardcoded 572.
    """
    sim_matrix4 = np.zeros((n_drug, n_drug), dtype=float)
    n_cols = min(20, raw_matrix.shape[1])
    for i in range(n_drug):
        for j in range(n_drug):
            for k in range(n_cols):
                if i == j:
                    sim_matrix4[i, j] = 0
                    break
                else:
                    if raw_matrix[i, k] == raw_matrix[j, k]:
                        sim_matrix4[i, j] += 1
                    else:
                        sim_matrix4[i, j] += 0
    return sim_matrix4


def jaccard(matrix: np.ndarray) -> np.ndarray:
    """Jaccard similarity between rows. task3.py:247-251 (verbatim)."""
    matrix = np.mat(matrix)
    numerator = matrix * matrix.T
    denominator = (np.ones(np.shape(matrix)) * matrix.T
                   + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T)
    return numerator / denominator


def _build_feature_matrices(
    kgs: dict, tail_len: dict, n_drug: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel per-drug feature matrices. task3.py:284-310.

    Returns (fm1, fm2, fm3, fm4):
      fm1 (n, tail1)  from KG1 tails      -> drug_sim1 = Jaccard(fm1)
      fm2 (n, tail2)  from KG2 tails      -> drug_sim2 = Jaccard(fm2)
      fm3 (n, tail4)  from KG4 (prop val) -> drug_sim3 = find_dif(fm3)
      fm4 (n, n)      from KG3 tails       -> drug_sim4 = Jaccard(fm4)
    """
    # temp_kg[p][i] = list of tail ids for KG (p+1), drug i (task3.py:284-288)
    temp_kg = [dict() for _ in range(4)]
    for p, name in enumerate(("dataset1", "dataset2", "dataset3", "dataset4")):
        for i, neighbors in kgs[name].items():
            temp_kg[p].setdefault(i, [])
            for tail, _rel in neighbors:
                temp_kg[p][i].append(tail)

    fm1 = np.zeros((n_drug, tail_len["dataset1"]), dtype=float)
    fm2 = np.zeros((n_drug, tail_len["dataset2"]), dtype=float)
    fm3 = np.zeros((n_drug, tail_len["dataset4"]), dtype=float)
    fm4 = np.zeros((n_drug, n_drug), dtype=float)

    # fm3 filled from KG4 raw (prop_idx -> bin value); task3.py:296-298
    for i, neighbors in kgs["dataset4"].items():
        for prop_idx, val in neighbors:
            if prop_idx < fm3.shape[1]:
                fm3[i][prop_idx] = val

    for i in temp_kg[0]:               # KG1 -> fm1  (task3.py:300-302)
        for j in temp_kg[0][i]:
            if j < fm1.shape[1]:
                fm1[i][j] = 1
    for i in temp_kg[1]:               # KG2 -> fm2  (task3.py:304-306)
        for j in temp_kg[1][i]:
            if j < fm2.shape[1]:
                fm2[i][j] = 1
    for i in temp_kg[2]:               # KG3 (DDI) -> fm4  (task3.py:308-310)
        for j in temp_kg[2][i]:
            if j < fm4.shape[1]:
                fm4[i][j] = 1
    return fm1, fm2, fm3, fm4


def _build_test_adj(
    drug_sims: list[np.ndarray], unseen_ids: list[int], n_drug: int
) -> list[dict]:
    """Nearest-seen-neighbour structure. task3.py:344-395.

    For each unseen drug j and channel k, find the argmax-similarity SEEN drug(s)
    p (p not unseen, p != j; ties kept). Returns a 4-element list of dicts:
    ``test_adj[k][j] = [[most-similar-seen-drug-ids]]`` (the inner list matches the
    upstream ``.append(current_p)`` so the model reads ``test_adj[k][j][0]``).
    """
    from collections import defaultdict as _dd
    unseen_set = set(unseen_ids)
    test_adj = [_dd(list) for _ in range(4)]
    for k in range(4):
        sim = drug_sims[k]
        for j in unseen_ids:
            row = np.asarray(sim[j]).ravel().tolist()
            max_v = 0
            current_p = []
            for p, v in enumerate(row):
                if v > max_v and p not in unseen_set and p != j:
                    max_v = v
                    current_p = [p]
                elif v == max_v and p not in unseen_set and p != j:
                    current_p.append(p)
            test_adj[k][j].append(current_p)
    return test_adj


def _ghost_pad_kg1(
    kgs: dict, tail_len: dict, relation_len: dict, dict1: dict[str, int]
) -> tuple[dict, dict, dict]:
    """Give KG1-empty drugs a single ghost neighbour (Option A, codex 019f245c).

    Upstream KG1 (drug->entity) was total over the curated 572 drugs. Our
    ``build_kg1`` OMITS drugs with zero entity annotations (e.g. peptides), which
    would make the verbatim ``arrge`` fail on ``np.random.choice(0, ...)``. We
    mirror the repo's WARM model.py (GNN1 ``ent_with_ghost``/``rel_with_ghost``):
    add a ghost entity+relation slot and point every KG1-missing drug at it. KG2
    (Morgan) and KG4 (property) already ghost-pad in the builder; KG3 self-loops.
    Only KG1 needs this. Returns NEW dicts (originals untouched -> feature matrices
    are computed from the un-padded KG1 to keep the similarity faithful).
    """
    kg1 = {k: list(v) for k, v in kgs["dataset1"].items()}
    ghost_tail = tail_len["dataset1"]        # new slot appended past real entities
    ghost_rel = relation_len["dataset1"]
    for didx in dict1.values():
        if not kg1.get(didx):
            kg1[didx] = [(ghost_tail, ghost_rel)]
    new_kgs = dict(kgs)
    new_kgs["dataset1"] = kg1
    new_tail = dict(tail_len)
    new_tail["dataset1"] = tail_len["dataset1"] + 1
    new_rel = dict(relation_len)
    new_rel["dataset1"] = relation_len["dataset1"] + 1
    return new_kgs, new_tail, new_rel


def _drug_smiles_dict(drugs: pd.DataFrame) -> dict[str, str]:
    """{drugbank_id: smiles}. Empty string for missing SMILES (build_kg2/4 skip them)."""
    return {
        str(row["drugbank_id"]): "" if pd.isna(row["smiles"]) else str(row["smiles"])
        for _, row in drugs[["drugbank_id", "smiles"]].iterrows()
    }


class MKGFENNMulticlassBaseline:
    """Regime-aware MKG-FENN multiclass core (S0 warm / S1-S2 cold).

    Not a legacy ``BaselineModel`` — consumed by the unified wrapper via a
    duck-typed ``_LeafDataset`` (leaf_adapter.make_dataset). The wrapper passes
    the native KG1 object as ``kg`` and the leaf's ``cold`` flag / unseen drug ids.
    """

    VERSION = "1.0-mc-cold"

    def __init__(
        self,
        *,
        n_classes: int = 65,
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
        self.n_classes = int(n_classes)
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
        self._ddi_type_to_idx: dict[str, int] | None = None
        self._idx_to_ddi_type: list[str] | None = None
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

        Matches upstream's full block (torch.use_deterministic_algorithms +
        cuDNN deterministic/benchmark/enabled flags). ``use_deterministic_algorithms``
        is set with ``warn_only=True`` (faithful-equivalent): upstream uses a hard
        ``True``, but on this repo's torch/data some ops lack a deterministic kernel
        (or require CUBLAS_WORKSPACE_CONFIG), which would HARD-CRASH training; warn_only
        keeps the determinism intent without aborting the run. The per-leaf runner
        launches one process per baseline, so these process-global flags do not leak
        across baselines.
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
        """Train on one leaf's positive DDI pairs (multiclass).

        Parameters
        ----------
        train, val
            Duck-typed ``_LeafDataset`` (leaf_adapter). ``train.splits.train`` is
            the positives frame carrying a ``ddi_type`` column (= global y_cls).
        kg
            Native drug->entity KnowledgeGraph (KG1 source). The unified wrapper
            passes this since the adapter sets ``_LeafDataset.kg = None``.
        dict1
            Optional precomputed drug vocab (all drugs -> idx). Built here if None.
        unseen_ids
            Drug ids (in dict1 index space) that are unseen at train time (cold
            regimes). Empty for S0.
        """
        self._seed_all()
        if kg is None:
            kg = train.kg
        if kg is None:
            raise ValueError("MKG-FENN multiclass requires a KG1 object (drug->entity).")

        # Drug vocab (all drugs across splits + drugs table). baseline.py:113-123.
        self._dict1 = dict1 if dict1 is not None else self._build_dict1(train)
        n_drug = len(self._dict1)
        self._unseen_ids = list(unseen_ids) if unseen_ids else []

        # ddi_type vocab from train positives (re-vocab; EmerGNN mc template).
        pos = train.splits.train
        if "ddi_type" not in pos.columns:
            raise ValueError("multiclass fit needs a 'ddi_type' column in train.splits.train")
        types = sorted(pos["ddi_type"].astype(str).unique())
        self._ddi_type_to_idx = {t: i for i, t in enumerate(types)}
        self._idx_to_ddi_type = list(types)
        event_num = len(types)
        if event_num != self.n_classes:
            print(f"[mkg_fenn_mc] observed {event_num} train ddi_types (n_classes hint "
                  f"{self.n_classes}); using event_num={event_num}", file=sys.stderr)
        self.n_classes = event_num

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

        # Ghost-pad KG1 empties for the COLD model (Option A). The warm MKGFENN does
        # its own ghost handling (ent_with_ghost/rel_with_ghost + kg.get(idx, [])),
        # so we only pad on the cold path to avoid a double ghost slot.
        if self.cold:
            kgs, tail_len, relation_len = _ghost_pad_kg1(kgs, tail_len, relation_len, self._dict1)
        self._kgs, self._tail_len, self._relation_len = kgs, tail_len, relation_len

        # Model (regime switch).
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

        # Symmetric pair augmentation (a,b)+(b,a) (task3.py:176-179).
        train_x = np.stack([
            pos["drug_a_id"].astype(str).map(self._dict1).to_numpy(),
            pos["drug_b_id"].astype(str).map(self._dict1).to_numpy(),
        ], axis=1)
        train_y = pos["ddi_type"].astype(str).map(self._ddi_type_to_idx).to_numpy()
        keep = ~(np.isnan(train_x.astype(float)).any(axis=1))
        train_x = train_x[keep].astype(np.int64)
        train_y = train_y[keep].astype(np.int64)
        train_x_rev = train_x[:, [1, 0]]
        train_x_total = np.concatenate([train_x, train_x_rev], axis=0)
        train_y_total = np.concatenate([train_y, train_y], axis=0)

        loss_function = nn.CrossEntropyLoss()
        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from my_code.utils.train_progress import TrainProgress

        n_total = len(train_x_total)
        steps_per_epoch = (n_total + self.batch_size - 1) // self.batch_size
        progress = TrainProgress(
            total_epochs=self.n_epochs, log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch, prefix="[mkg_fenn_mc] ",
        )

        best_val_macro_f1 = -1.0
        best_state: dict | None = None
        rng = np.random.default_rng(self.seed)

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_total)            # per-epoch shuffle
            xt = train_x_total[perm]
            yt = train_y_total[perm]
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
                y = torch.as_tensor(yb, dtype=torch.long, device=self.device)
                opt.zero_grad(set_to_none=True)
                logits = self._forward_train(idx_tensor)
                loss = loss_function(logits, y)
                loss.backward()
                opt.step()
                progress.step(loss.item())
            extra = {}
            if val is not None:
                metrics = self._eval(val)
                if metrics:
                    progress.log_eval(metrics, scope="epoch")
                    extra.update(metrics)
                    if metrics.get("val_macro_f1", -1) > best_val_macro_f1:
                        best_val_macro_f1 = metrics["val_macro_f1"]
                        best_state = copy.deepcopy(self._model.state_dict())
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[mkg_fenn_mc] loaded best val_macro_f1={best_val_macro_f1:.4f}", flush=True)

    def _forward_train(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        """Train-time forward (no imputation; train_or_test=0 for cold)."""
        if self.cold:
            return self._model(idx_tensor, 0, None)
        return self._model(idx_tensor)

    def _forward_eval(self, idx_tensor: torch.Tensor) -> torch.Tensor:
        """Eval-time forward. Cold path enables nearest-seen imputation (train_or_test=1)."""
        if self.cold and self._test_adj is not None:
            return self._model(idx_tensor, 1, self._test_adj)
        if self.cold:
            return self._model(idx_tensor, 0, None)
        return self._model(idx_tensor)

    def _build_dict1(self, train) -> dict[str, int]:
        drug_ids: set[str] = set()
        for _name, df in train.splits.items():
            drug_ids.update(df["drug_a_id"].astype(str))
            drug_ids.update(df["drug_b_id"].astype(str))
        if train.drugs is not None and "drugbank_id" in train.drugs.columns:
            drug_ids.update(train.drugs["drugbank_id"].astype(str))
        return {did: idx for idx, did in enumerate(sorted(drug_ids))}

    @torch.no_grad()
    def _eval(self, val) -> dict:
        """Val macro-F1 over the dense train-vocab classes (paper primary metric)."""
        from sklearn.metrics import f1_score
        val_pos = val.splits.val_s2
        if val_pos is None or len(val_pos) == 0 or "ddi_type" not in val_pos.columns:
            return {}
        mask_vocab = val_pos["ddi_type"].astype(str).isin(self._ddi_type_to_idx)
        vp = val_pos[mask_vocab]
        if len(vp) == 0:
            return {}
        probs = self.predict_proba(vp[_PAIR])
        keep = ~np.all(probs == 0.0, axis=1)           # drop rows with OOV drugs (all-zero)
        if not keep.any():
            return {}
        preds = probs[keep].argmax(axis=1)
        labels = np.array([self._ddi_type_to_idx[str(t)] for t in vp["ddi_type"]])[keep]
        return {"val_macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0))}

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        """(n, K_train) softmax over the dense train-observed class vocab.

        Rows whose drugs are absent from ``dict1`` get an all-zero row (the unified
        wrapper leaves them as ~0 mass at every global class -> counted wrong).
        """
        if self._model is None or self._dict1 is None:
            raise RuntimeError("fit() before predict_proba().")
        self._model.eval()
        out = np.zeros((len(pairs), self.n_classes), dtype=np.float32)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs.iloc[start:start + self.batch_size]
            idx_tensor, mask = self._pair_indices(batch)
            if idx_tensor is None:
                continue
            idx_tensor = idx_tensor.to(self.device)
            logits = self._forward_eval(idx_tensor)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            block = np.zeros((len(batch), self.n_classes), dtype=np.float32)
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
                "ddi_type_to_idx": self._ddi_type_to_idx,
                "idx_to_ddi_type": self._idx_to_ddi_type,
                "n_classes": self.n_classes,
                "cold": self.cold,
                "test_adj": self._test_adj,
                "unseen_ids": self._unseen_ids,
                "kgs": self._kgs,
                "tail_len": self._tail_len,
                "relation_len": self._relation_len,
            }, f)
        (out / "manifest.json").write_text(json.dumps({
            "baseline": "mkg_fenn", "task": "multiclass", "version": self.VERSION,
            "n_classes": self.n_classes, "cold": self.cold,
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
    def load(cls, path: "str | Path") -> "MKGFENNMulticlassBaseline":
        p = Path(path)
        manifest = json.loads((p / "manifest.json").read_text())
        hp = manifest.get("hyperparameters", {})
        inst = cls(n_classes=manifest["n_classes"], cold=manifest["cold"], **hp)
        with (p / "state.pkl").open("rb") as f:
            payload = pickle.load(f)
        inst._dict1 = payload["dict1"]
        inst._ddi_type_to_idx = payload["ddi_type_to_idx"]
        inst._idx_to_ddi_type = payload["idx_to_ddi_type"]
        inst.n_classes = payload["n_classes"]
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
                drug_name=drug_name, args=args, event_num=inst.n_classes,
            ).to(inst.device)
        else:
            inst._model = MKGFENN(
                kgs=payload["kgs"], tail_len=payload["tail_len"],
                relation_len=payload["relation_len"], dict1=inst._dict1,
                drug_name=drug_name, embedding_num=inst.embedding_num,
                neighbor_sample_size=inst.neighbor_sample_size,
                dropout=inst.dropout, event_num=inst.n_classes,
            ).to(inst.device)
        inst._model.load_state_dict(torch.load(p / "model.pt", map_location=inst.device))
        inst._model.eval()
        return inst


__all__ = ["MKGFENNMulticlassBaseline", "PAPER_HYPERPARAMS",
           "find_dif", "jaccard"]

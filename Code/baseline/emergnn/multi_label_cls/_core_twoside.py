"""Ported TWOSIDES multilabel EmerGNN core (INTERNAL, not registered).

Paper-faithful port of the TWOSIDES reproduction's data + training pipeline
(LARS-research/EmerGNN, TWOSIDES/{load_data.py, base_model.py}). COPY+adapt, no
import of the reproduction (CLAUDE.md §文件级独立性). The only bridge to the
unified benchmark is the *DDI pairs* — those come from the leaf's parquet (via
the adapter, one dense 200-multihot row per pair, paired pos/neg). Everything
KG-side (background edges, entity/relation vocab sizes, Morgan feats) is rebuilt
from the leaf's declared KG source, which for TWOSIDES is the reproduction's
``_Original-Dataset/TWOSIDES/data`` directory (per-fold ``train_KG.txt`` +
``necessary/{entity2id,relation2id,id2drug_feat}__official``).

Ported mechanics (upstream file:line -> here):
  * load_data.py:159-163 load_graph relation-slot scheme (DDI facts -> slot 0
    both dirs; KG fwd r, KG rev r+all_rel-eval_rel; self-loop slot 2*all_rel-
    eval_rel) -> :func:`_build_edges`.
  * load_data.py:166-217 shuffle_train (S0 random 80/20; S1 one-new targets;
    S2 two-new targets; facts folded into the epoch KG) -> :func:`_shuffle_train`.
  * base_model.py:36-56 train loop: paired pos/neg carrying 200-multihot; per-
    label scores masked by (pos_label>0)/(neg_label>0); BCE over concatenated
    pos+neg; Adam(lr, weight_decay=lamb); ReduceLROnPlateau(mode='max')
    -> :meth:`BaseModelTwoside.fit`.
  * base_model.py:58-106 evaluate: per-label ROC-AUC / PR-AUC over paired
    pos/neg -> :meth:`BaseModelTwoside._eval_paired` + best-ckpt on valid PR-AUC.
  * evaluate.py:57-71 per-regime hyperparam bundle (S1 n_dim=32 lr=1e-3 batch=32;
    S2 n_dim=64 lr=3e-3 batch=64) -> chosen by the unified wrapper.
"""
from __future__ import annotations

import copy
import json
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.emergnn.multi_label_cls.model import EmerGNN_ML

if TYPE_CHECKING:
    import pandas as pd

MORGAN_NBITS = 1024


# ----------------------------------------------------------------------------
# KG-side rebuild from the leaf's declared TWOSIDES KG source
# ----------------------------------------------------------------------------
def _fold_to_repro_dirname(split_code: str, fold: str) -> str:
    """Leaf ``<S1|S2>/foldK`` -> reproduction cumulative dir ``S{1|2}_1..(K+1)``.

    Verified 1:1 against the leaf drug pools (all 5 folds, both regimes):
      S2/fold0 -> S2_1, S2/fold1 -> S2_12, ..., S2/fold4 -> S2_12345.
    """
    if split_code == "S0":
        return "S0"
    k = int(str(fold).replace("fold", ""))
    suffix = "".join(str(i + 1) for i in range(k + 1))
    return f"{split_code}_{suffix}"


def _read_kg_file(path: Path) -> np.ndarray:
    """Read one *_KG.txt as (K,3) int64 (h,t,r) background edges."""
    triplets = []
    with path.open() as f:
        for line in f:
            a = line.split()
            if len(a) < 3:
                continue
            triplets.append((int(a[0]), int(a[1]), int(a[2])))
    if not triplets:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(triplets, dtype=np.int64)


def load_twoside_kg(kg_source: str, split_code: str, fold: str) -> dict:
    """Read the reproduction fold's KG background + vocab sizes + Morgan feats.

    Faithful to upstream process_files_kg (TWOSIDES/load_data.py:104-141): loads
    ``train/valid/test_KG.txt`` separately and builds the CUMULATIVE graphs
      train_kg = train_KG                       (load_data.py:135)
      valid_kg = train_KG + valid_KG            (load_data.py:136)
      test_kg  = train_KG + valid_KG + test_KG  (load_data.py:137)
    Upstream concatenates (no dedup). ``kg_entity_set`` is the UNION of entities
    over ALL THREE files, matching how upstream populates ``ddi_in_kg`` while
    iterating every KG split (load_data.py:104-125) — this defines shuffle_train's
    removable-entity pool (load_data.py:166).

    Returns keys: all_ent, all_rel, train_kg, valid_kg, test_kg (each (K,3) int64),
    kg_entity_set (set[int]), morgan (all_ent,1024 float32).
    """
    src = Path(kg_source)
    nec = src / "necessary"
    relation2id = json.loads((nec / "relation2id__official.json").read_text())
    id2drug_feat = pickle.loads((nec / "id2drug_feat__official.pkl").read_bytes())

    repro_dir = src / _fold_to_repro_dirname(split_code, fold)
    train_path = repro_dir / "train_KG.txt"
    valid_path = repro_dir / "valid_KG.txt"
    test_path = repro_dir / "test_KG.txt"
    if not train_path.is_file():
        raise FileNotFoundError(f"TWOSIDES KG background not found: {train_path}")

    # eval_rel = 200 side-effect relations (keys 0..199); 200..522 are KG
    # relations. n_labels is authoritative; all_rel = max(relation2id)+1.
    int_keys = [int(k) for k in relation2id]
    all_rel = max(int_keys) + 1

    train_KG = _read_kg_file(train_path)
    valid_KG = _read_kg_file(valid_path) if valid_path.is_file() else np.zeros((0, 3), np.int64)
    test_KG = _read_kg_file(test_path) if test_path.is_file() else np.zeros((0, 3), np.int64)

    # cumulative graphs (upstream concatenates, no dedup)
    train_kg = train_KG
    valid_kg = np.concatenate([train_KG, valid_KG], axis=0) if len(valid_KG) else train_KG
    test_kg = (np.concatenate([train_KG, valid_KG, test_KG], axis=0)
               if (len(valid_KG) or len(test_KG)) else train_KG)

    # entity union over ALL THREE KG files (upstream ddi_in_kg is populated while
    # iterating every split). Use test_kg (= the full cumulative union).
    kg_entity_set = (set(np.unique(test_kg[:, :2]).tolist()) if len(test_kg) else set())

    # all_ent covers every KG entity AND every drug id (0..603).
    n_drug = len(id2drug_feat)
    max_ent = int(test_kg[:, :2].max()) if len(test_kg) else -1
    all_ent = max(max_ent + 1, n_drug)

    # Morgan matrix keyed on drug id (0..n_drug-1); non-drug rows are zeros.
    morgan = np.zeros((all_ent, MORGAN_NBITS), dtype=np.float32)
    for k, v in id2drug_feat.items():
        did = int(k)
        m = np.asarray(v["Morgan"], dtype=np.float32)
        morgan[did] = m
    return {
        "all_ent": int(all_ent),
        "all_rel": int(all_rel),
        "train_kg": train_kg,
        "valid_kg": valid_kg,
        "test_kg": test_kg,
        "kg_entity_set": kg_entity_set,
        "morgan": morgan,
    }


def _build_edges(
    fact_triplets: np.ndarray,   # (F,3) DDI facts (h,t,r); r ignored (slot 0)
    kg_triplets: np.ndarray,     # (K,3) KG background (h,t,r) original rel index
    all_ent: int,
    all_rel: int,
    eval_rel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of upstream load_graph (load_data.py:146-164) -> (src, dst, rel).

    DDI facts: both directions on slot 0. KG: fwd (t,h,r), rev (h,t,r+all_rel-
    eval_rel). Self-loop on slot 2*all_rel-eval_rel for every entity.
    """
    self_loop = 2 * all_rel - eval_rel
    srcs, dsts, rels = [], [], []

    if len(fact_triplets):
        fh = fact_triplets[:, 0]
        ft = fact_triplets[:, 1]
        # [t,h,0] and [h,t,0]
        srcs.append(ft); dsts.append(fh); rels.append(np.zeros(len(fh), np.int64))
        srcs.append(fh); dsts.append(ft); rels.append(np.zeros(len(fh), np.int64))

    if len(kg_triplets):
        kh = kg_triplets[:, 0]
        kt = kg_triplets[:, 1]
        kr = kg_triplets[:, 2]
        r_inv = kr + (all_rel - eval_rel)
        # [t,h,r] and [h,t,r_inv]
        srcs.append(kt); dsts.append(kh); rels.append(kr)
        srcs.append(kh); dsts.append(kt); rels.append(r_inv)

    # self-loop for all entities
    ar = np.arange(all_ent, dtype=np.int64)
    srcs.append(ar); dsts.append(ar); rels.append(np.full(all_ent, self_loop, np.int64))

    edge_src = np.concatenate(srcs).astype(np.int64)
    edge_dst = np.concatenate(dsts).astype(np.int64)
    edge_rel = np.concatenate(rels).astype(np.int64)
    return edge_src, edge_dst, edge_rel


def _shuffle_train(
    train_pos_ht: np.ndarray,     # (N,2) positive pair endpoints (drug ids)
    train_multihot: np.ndarray,   # (N,200) positive label vectors
    train_neg_ht: np.ndarray,     # (N,2) negative pair endpoints
    train_neg_multihot: np.ndarray,  # (N,200)
    train_ent: set[int],
    ddi_in_kg: set[int],
    split_code: str,
    rng: np.random.Generator,
    ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of upstream shuffle_train (load_data.py:166-217).

    Returns (fact_triplets, pos_ht, pos_multihot, neg_ht, neg_multihot) where
    fact_triplets are the epoch's KG-folded DDI facts (h,t,label_idx) and the
    pos/neg arrays are this epoch's prediction targets. For S1/S2 the paired
    negative rows track their positives (base_model.py trains paired pos/neg).
    """
    n_ent = len(ddi_in_kg)
    drop = rng.choice(list(ddi_in_kg), n_ent - int(n_ent * ratio), replace=False) \
        if n_ent > 0 else np.array([], dtype=np.int64)
    epoch_train_ent = set(train_ent) - set(int(x) for x in drop)

    N = len(train_pos_ht)
    if split_code == "S0":
        idx = rng.permutation(N)
        n_fact = int(N * 0.8)
        fact_idx = idx[:n_fact]
        tgt_idx = idx[n_fact:]
        facts = []
        for i in fact_idx:
            h, t = int(train_pos_ht[i, 0]), int(train_pos_ht[i, 1])
            for s in np.nonzero(train_multihot[i])[0]:
                facts.append((h, t, int(s)))
        fact_triplets = np.asarray(facts, dtype=np.int64) if facts else np.zeros((0, 3), np.int64)
        return (fact_triplets, train_pos_ht[tgt_idx], train_multihot[tgt_idx],
                train_neg_ht[tgt_idx], train_neg_multihot[tgt_idx])

    # S1 / S2
    facts = []
    tgt = []
    for i in range(N):
        h, t = int(train_pos_ht[i, 0]), int(train_pos_ht[i, 1])
        h_in = h in epoch_train_ent
        t_in = t in epoch_train_ent
        if h_in and t_in:
            for s in np.nonzero(train_multihot[i])[0]:
                facts.append((h, t, int(s)))
        elif split_code == "S1" and (h_in or t_in):
            tgt.append(i)
        elif split_code == "S2" and (not h_in and not t_in):
            tgt.append(i)
    fact_triplets = np.asarray(facts, dtype=np.int64) if facts else np.zeros((0, 3), np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    if len(tgt) == 0:
        return (fact_triplets, np.zeros((0, 2), np.int64), np.zeros((0, train_multihot.shape[1]), np.float32),
                np.zeros((0, 2), np.int64), np.zeros((0, train_multihot.shape[1]), np.float32))
    return (fact_triplets, train_pos_ht[tgt], train_multihot[tgt],
            train_neg_ht[tgt], train_neg_multihot[tgt])


class BaseModelTwoside:
    """Ported TWOSIDES multilabel trainer. Owns model + KG state.

    Consumes the adapter's paired multilabel bundle (dense 200-multihot per
    pair, paired pos/neg) plus the leaf's KG source. Trains with the paper's
    per-epoch shuffle_train + BCE-on-active-labels recipe. Uses THREE distinct
    graphs matching upstream (load_data.py): the per-epoch TRAINING graph
    (folded facts + train_kg), the vKG (train facts + valid_kg) for validation
    model-selection, and the tKG (train+val facts + test_kg) for
    ``predict_proba`` / test scoring.
    """

    def __init__(
        self,
        *,
        kg_source: str,
        split_code: str,
        fold: str,
        n_labels: int = 200,
        n_dim: int = 64,
        length: int = 3,
        feat: str = "M",
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 64,
        test_batch_size: int = 16,
        n_epochs: int = 100,
        shuffle_ratio: float = 0.8,
        device: str = "auto",
        log_step_every: int = 50,
        eval_strategy: str = "epoch",
        run_dir: str | Path | None = None,
    ) -> None:
        self.kg_source = str(kg_source)
        self.split_code = str(split_code)
        self.fold = str(fold)
        self.n_labels = int(n_labels)
        self.n_dim = int(n_dim)
        self.length = int(length)
        self.feat = str(feat)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.test_batch_size = int(test_batch_size)
        self.n_epochs = int(n_epochs)
        self.shuffle_ratio = float(shuffle_ratio)
        self.device = self._resolve_device(device)
        self.log_step_every = int(log_step_every)
        self.eval_strategy = str(eval_strategy)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._model: EmerGNN_ML | None = None
        self._eval_edges = None       # tKG (test / predict)
        self._valid_edges = None      # vKG (validation during training)
        self._edge_counts: dict | None = None
        self._all_ent = None
        self._all_rel = None

    @staticmethod
    def _resolve_device(d: str) -> str:
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _edges_to_device(self, esrc, edst, erel):
        return (
            torch.from_numpy(esrc).long().to(self.device),
            torch.from_numpy(edst).long().to(self.device),
            torch.from_numpy(erel).long().to(self.device),
        )

    def fit(self, bundle: dict) -> None:
        """`bundle` from the multilabel adapter (see leaf_adapter multilabel path)
        with keys: train_pos_ht, train_pos_y, train_neg_ht, train_neg_y,
        val_pos_ht, val_pos_y, val_neg_ht, val_neg_y (arrays)."""
        kg = load_twoside_kg(self.kg_source, self.split_code, self.fold)
        all_ent = kg["all_ent"]
        all_rel = kg["all_rel"]
        eval_rel = self.n_labels
        train_kg = kg["train_kg"]        # background for the per-epoch training graph
        valid_kg = kg["valid_kg"]        # train_KG + valid_KG (upstream valid_kg)
        test_kg = kg["test_kg"]          # train_KG + valid_KG + test_KG (upstream test_kg)
        kg_entity_set = kg["kg_entity_set"]  # union over all three KG files
        morgan = kg["morgan"]
        self._all_ent, self._all_rel = all_ent, all_rel

        train_pos_ht = np.asarray(bundle["train_pos_ht"], dtype=np.int64)
        train_pos_y = np.asarray(bundle["train_pos_y"], dtype=np.float32)
        train_neg_ht = np.asarray(bundle["train_neg_ht"], dtype=np.int64)
        train_neg_y = np.asarray(bundle["train_neg_y"], dtype=np.float32)
        train_ent = set(train_pos_ht.reshape(-1).tolist())
        # upstream: ddi_in_kg = train_ent entities appearing in the FULL cumulative
        # KG (populated while iterating all 3 KG files, load_data.py:104-125). This
        # is shuffle_train's removable-entity pool (load_data.py:166).
        ddi_in_kg = train_ent & kg_entity_set
        if not ddi_in_kg:
            ddi_in_kg = train_ent

        val_pos_ht = np.asarray(bundle["val_pos_ht"], dtype=np.int64)
        val_pos_y = np.asarray(bundle["val_pos_y"], dtype=np.float32)
        val_neg_ht = np.asarray(bundle["val_neg_ht"], dtype=np.int64)
        val_neg_y = np.asarray(bundle["val_neg_y"], dtype=np.float32)

        self._model = EmerGNN_ML(
            n_ent=all_ent, all_rel=all_rel, eval_rel=eval_rel,
            n_dim=self.n_dim, length=self.length, feat=self.feat,
            morgan_features=morgan if self.feat == "M" else None,
        ).to(self.device)

        opt = optim.Adam(self._model.parameters(), lr=self.learning_rate,
                         weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(opt, mode="max")

        def _facts_from(pos_ht, pos_y):
            f = []
            for i in range(len(pos_ht)):
                h, t = int(pos_ht[i, 0]), int(pos_ht[i, 1])
                for s in np.nonzero(pos_y[i])[0]:
                    f.append((h, t, int(s)))
            return np.asarray(f, dtype=np.int64) if f else np.zeros((0, 3), np.int64)

        # ── vKG (validation during training): train DDI facts + valid_kg ──
        #    upstream load_data.py:34 vKG = fact(train_pos) + valid_kg.
        train_facts = _facts_from(train_pos_ht, train_pos_y)
        vsrc, vdst, vrel = _build_edges(train_facts, valid_kg, all_ent, all_rel, eval_rel)
        self._valid_edges = self._edges_to_device(vsrc, vdst, vrel)
        # ── tKG (test / predict): train + valid DDI facts + test_kg ──
        #    upstream load_data.py:40 tKG = fact(train_pos + valid_pos) + test_kg.
        val_facts = _facts_from(val_pos_ht, val_pos_y)
        tv_facts = (np.concatenate([train_facts, val_facts], axis=0)
                    if len(val_facts) else train_facts)
        tsrc, tdst, trel = _build_edges(tv_facts, test_kg, all_ent, all_rel, eval_rel)
        self._eval_edges = self._edges_to_device(tsrc, tdst, trel)
        # edge counts for confirmation (train per-epoch graph size varies with
        # shuffle_train; report the full-train-fact graph as its representative).
        fsrc, _, _ = _build_edges(train_facts, train_kg, all_ent, all_rel, eval_rel)
        self._edge_counts = {
            "train_graph_edges": int(len(fsrc)),
            "valid_graph_edges": int(self._valid_edges[0].numel()),
            "test_graph_edges": int(self._eval_edges[0].numel()),
        }
        print(f"[emergnn_ml] graph edges train={self._edge_counts['train_graph_edges']} "
              f"valid={self._edge_counts['valid_graph_edges']} "
              f"test={self._edge_counts['test_graph_edges']}", flush=True)

        try:
            from my_code.utils.train_progress import TrainProgress
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
            from my_code.utils.train_progress import TrainProgress

        rng = np.random.default_rng(0)
        best_val_pr = -1.0
        best_state: dict | None = None
        steps_per_epoch = max(
            1, (len(train_pos_ht) + self.batch_size - 1) // self.batch_size
        )
        progress = TrainProgress(
            total_epochs=self.n_epochs, log_step_every=self.log_step_every,
            total_steps_per_epoch=steps_per_epoch, prefix="[emergnn_ml] ",
            eval_strategy=self.eval_strategy,
        )

        for epoch in range(self.n_epochs):
            fact_triplets, pos_ht, pos_y, neg_ht, neg_y = _shuffle_train(
                train_pos_ht, train_pos_y, train_neg_ht, train_neg_y,
                train_ent, ddi_in_kg, self.split_code, rng, ratio=self.shuffle_ratio,
            )
            if len(pos_ht) == 0:
                print(f"[emergnn_ml] epoch {epoch+1}: 0 targets after shuffle_train; skip",
                      flush=True)
                continue
            # per-epoch TRAINING graph: this epoch's folded DDI facts + train_kg
            # (upstream shuffle_train builds self.KG = load_graph(fact, train_kg)).
            esrc, edst, erel = _build_edges(fact_triplets, train_kg, all_ent, all_rel, eval_rel)
            edge_src, edge_dst, edge_rel = self._edges_to_device(esrc, edst, erel)

            self._model.train()
            progress.epoch_start(epoch)
            N = len(pos_ht)
            for start in range(0, N, self.batch_size):
                end = min(N, start + self.batch_size)
                p_h = torch.from_numpy(pos_ht[start:end, 0]).long().to(self.device)
                p_t = torch.from_numpy(pos_ht[start:end, 1]).long().to(self.device)
                p_r = torch.from_numpy(pos_y[start:end]).float().to(self.device)
                n_h = torch.from_numpy(neg_ht[start:end, 0]).long().to(self.device)
                n_t = torch.from_numpy(neg_ht[start:end, 1]).long().to(self.device)
                n_r = torch.from_numpy(neg_y[start:end]).float().to(self.device)

                opt.zero_grad(set_to_none=True)
                # upstream base_model.py: sigmoid then mask by label>0, BCE.
                p_scores = torch.sigmoid(
                    self._model(p_h, p_t, edge_src, edge_dst, edge_rel)
                )
                n_scores = torch.sigmoid(
                    self._model(n_h, n_t, edge_src, edge_dst, edge_rel)
                )
                p_sel = p_scores[p_r > 0]
                n_sel = n_scores[n_r > 0]
                scores = torch.cat([p_sel, n_sel], dim=0)
                labels = torch.cat(
                    [torch.ones(len(p_sel), device=self.device),
                     torch.zeros(len(n_sel), device=self.device)], dim=0,
                )
                if len(scores) == 0:
                    continue
                loss = torch.nn.functional.binary_cross_entropy(scores, labels)
                loss.backward()
                opt.step()
                progress.step(loss.item())

            extra: dict = {}
            if progress.should_eval_epoch() and len(val_pos_ht):
                self._model.eval()
                # validation uses vKG (upstream evaluate(valid, vKG)).
                roc, pr = self._eval_paired(val_pos_ht, val_pos_y, val_neg_ht,
                                            val_neg_y, edges=self._valid_edges)
                self._model.train()
                metrics = {"val_roc": roc, "val_pr": pr}
                progress.log_eval(metrics, scope="epoch")
                extra.update(metrics)
                if pr > best_val_pr:
                    best_val_pr = pr
                    best_state = copy.deepcopy(self._model.state_dict())
                scheduler.step(pr)   # upstream best-ckpt on valid PR-AUC
            progress.epoch_end(extra=extra if extra else None)

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[emergnn_ml] loaded best val_pr={best_val_pr:.4f} state", flush=True)

    @torch.no_grad()
    def _score_pairs(self, ht: np.ndarray, edges=None) -> np.ndarray:
        """(n,2) endpoints -> (n, eval_rel) sigmoid probs. `edges` selects the
        graph (default = tKG = self._eval_edges, used for test/predict)."""
        edge_src, edge_dst, edge_rel = edges if edges is not None else self._eval_edges
        self._model.eval()
        out = np.empty((len(ht), self.n_labels), dtype=np.float32)
        bs = self.test_batch_size
        for start in range(0, len(ht), bs):
            end = min(len(ht), start + bs)
            h = torch.from_numpy(ht[start:end, 0]).long().to(self.device)
            t = torch.from_numpy(ht[start:end, 1]).long().to(self.device)
            logits = self._model(h, t, edge_src, edge_dst, edge_rel)
            out[start:end] = torch.sigmoid(logits).cpu().numpy()
        return out

    @torch.no_grad()
    def _eval_paired(self, pos_ht, pos_y, neg_ht, neg_y, edges=None) -> tuple[float, float]:
        """Per-label ROC-AUC / PR-AUC over paired pos/neg (upstream evaluate).
        `edges` selects the eval graph (vKG for validation, tKG otherwise)."""
        pos_scores = self._score_pairs(np.asarray(pos_ht, dtype=np.int64), edges=edges)
        neg_scores = self._score_pairs(np.asarray(neg_ht, dtype=np.int64), edges=edges)
        pos_y = np.asarray(pos_y, dtype=np.float32)
        rocs, prs = [], []
        for r in range(self.n_labels):
            idx = pos_y[:, r] > 0
            k = int(idx.sum())
            if k == 0:
                continue
            score = np.concatenate([pos_scores[idx, r], neg_scores[idx, r]])
            label = np.concatenate([np.ones(k), np.zeros(k)])
            rocs.append(roc_auc_score(label, score))
            prs.append(average_precision_score(label, score))
        return (float(np.mean(rocs)) if rocs else float("nan"),
                float(np.mean(prs)) if prs else float("nan"))

    @torch.no_grad()
    def predict_proba(self, ht: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BaseModelTwoside.fit() before predict_proba().")
        return self._score_pairs(np.asarray(ht, dtype=np.int64))


__all__ = ["BaseModelTwoside", "load_twoside_kg", "_build_edges", "_shuffle_train",
           "_fold_to_repro_dirname"]

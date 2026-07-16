"""Binary DDI core for SumGNN (Case-B GraIL-style binary link prediction).

This module implements a BINARY-task core that REUSES SumGNN's enclosing-subgraph
extraction + R-GCN + KG-summarization self-attention + Morgan-feature fusion UNCHANGED
(everything is imported from ``baseline.sumgnn._core``; nothing in ``_core`` is edited).
The ONLY task-specific changes live here, per the codex-approved design:

  * head width ``train_rels == 1`` (a single logit) -> ``GraphClassifier`` builds
    ``fc_layer = nn.Linear(..., 1)`` untouched.
  * ``BCEWithLogitsLoss`` on the per-subgraph BINARY ``g_label`` (1 positive DDI pair,
    0 negative pair). The ``r_label`` slot is fixed to 0 (the single DDI relation).
  * validation / best-checkpoint metric = AUPRC (the binary primary metric), NOT the
    multiclass macro-F1 / BioSNAP masked-BCE of the ``_core`` Trainer/Evaluator.
  * ``predict_proba`` -> ``(n,)`` sigmoid probabilities.

Faithfulness (codex): the ``_core`` ``SubgraphDataset`` opens ``db_neg`` but yields only
``db_pos``; the ``_core`` ``links2subgraphs`` writes ``g_label=1`` for both pos and neg
extraction; the ``_core`` Trainer/Evaluator are multiclass/BioSNAP. Reusing any of them
unchanged would corrupt the binary task, so this file provides:

  * :func:`generate_binary_subgraph_datasets` -- extraction that builds the DDI adjacency
    from TRAIN POSITIVES ONLY and extracts the EXPLICIT pos and neg query sets with the
    correct per-query ``g_label`` (1 pos / 0 neg), ``r_label=0``.
  * :class:`BinarySubgraphDataset` -- a dataset that reads BOTH the ``*_pos`` and
    ``*_neg`` LMDBs and yields mixed labelled examples.
  * :class:`BinaryTrainer` -- BCEWithLogitsLoss training loop with per-epoch loss logging
    and AUPRC-on-validation best-checkpoint selection.
  * :class:`BinaryEvaluator` -- sigmoid-prob AUPRC/AUROC evaluator.

Reused-unchanged helpers from ``_core`` (imported, NOT reimplemented):
  * subgraph BFS + double-radius node labelling: ``subgraph_extraction_labeling``,
    ``get_average_subgraph_size``, ``intialize_worker``, ``extract_save_subgraph``,
    ``sample_neg`` (only borrowed for the size estimate path; we never call it to add
    extra negatives).
  * DDI adjacency + KG assembly from int-triple files: ``process_files_ddi``.
  * subgraph -> DGL graph + feature prep: ``SubgraphDataset._prepare_subgraphs`` /
    ``_prepare_features_new`` (inherited by subclassing ``SubgraphDataset``).
  * graph batching + device move: ``collate_dgl``, ``move_batch_to_device_dgl``.
  * model: ``GraphClassifier`` (head width driven by ``params.train_rels``).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import struct
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import lmdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

# baseline-local ported SumGNN core (independent copy; NOT the reproduction). Ensure the
# _core dir is importable even if this module is imported before the wrapper.
_CORE = Path(__file__).resolve().parents[1] / "_core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

# ---- reused-unchanged _core imports (nothing here edits _core) ----------
from subgraph_extraction.datasets import SubgraphDataset, _open_env
from subgraph_extraction.graph_sampler import (extract_save_subgraph,
                                               get_average_subgraph_size,
                                               intialize_worker)
from utils.data_utils import process_files_ddi
from utils.graph_utils import deserialize, serialize

if TYPE_CHECKING:
    from types import SimpleNamespace


def _load_query_pairs(path: str) -> np.ndarray:
    """Load a query file `drug_a_idx drug_b_idx 0` (r_label slot = 0) as an (n,3) int
    array. Returns an empty (0,3) array for a missing / empty file."""
    if not os.path.isfile(path):
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.loadtxt(path).reshape(-1, 3)
    return arr.astype(np.int64)


def generate_binary_subgraph_datasets(params: "SimpleNamespace",
                                       splits=("train", "valid", "test")) -> None:
    """BINARY enclosing-subgraph extraction (Case-B).

    Differs from ``_core`` ``generate_subgraph_datasets`` on two faithfulness points:

      1. Adjacency is built from ``process_files_ddi(params.file_paths, ...)`` where the
         ``train`` file contains TRAIN POSITIVES ONLY (relation slot 0). Negatives never
         enter the file paths, so they never become false DDI edges in the adjacency.
      2. For each split we extract the EXPLICIT pos query set (``g_label=1``) AND the
         EXPLICIT neg query set (``g_label=0``) read from separate on-disk files
         (``<split>_pos.txt`` / ``<split>_neg.txt``). ``_core`` would instead sample
         fresh negatives and write ``g_label=1`` for both -- both are avoided here.

    Writes the same LMDB layout the ``_core`` ``SubgraphDataset`` expects: named dbs
    ``<split>_pos`` / ``<split>_neg`` under ``params.db_path``, plus the shared
    ``max_n_label_*`` / subgraph-size stats keys.
    """
    triple_file = "data/{}/relations_2hop.txt".format(params.dataset)
    # process_files_ddi reads train/valid/test.txt (POSITIVES only) + the KG to build
    # the multigraph adjacency `adj_list` (relation 0 = train-positive DDI, KG rels
    # follow). We DISCARD its `triplets` (they are the positives) and instead drive
    # extraction from the explicit pos/neg query files, so negatives never touch adj.
    adj_list, _triplets, _e2i, relation2id, id2entity, id2relation, rel = process_files_ddi(
        params.file_paths, triple_file, None)

    # relation2id.json for the model builder (same path convention as _core).
    import json
    data_path = "data/{}/relation2id.json".format(params.dataset)
    if not os.path.isdir(data_path):
        with open(data_path, "w") as f:
            json.dump({str(k): int(v) for k, v in relation2id.items()}, f)

    # explicit pos/neg query sets per split (rows `a b 0`).
    base = os.path.dirname(params.file_paths["train"])
    query_files = {
        "train": (os.path.join(base, "train_pos.txt"), os.path.join(base, "train_neg.txt")),
        "valid": (os.path.join(base, "dev_pos.txt"), os.path.join(base, "dev_neg.txt")),
        "test": (os.path.join(base, "test_pos.txt"), os.path.join(base, "test_neg.txt")),
    }
    graphs = {}
    for split_name in splits:
        pos_f, neg_f = query_files[split_name]
        graphs[split_name] = {"pos": _load_query_pairs(pos_f), "neg": _load_query_pairs(neg_f)}

    _links2subgraphs_binary(adj_list, graphs, params, splits)


def _links2subgraphs_binary(A, graphs, params, splits) -> None:
    """BINARY variant of ``_core`` ``links2subgraphs``: extract enclosing subgraphs for
    the explicit pos + neg query sets, writing ``g_label=1`` for pos and ``g_label=0``
    for neg (the ``_core`` version writes 1 for both). Reuses ``_core`` per-subgraph
    workers (``extract_save_subgraph`` / ``intialize_worker``) and the size estimator
    (``get_average_subgraph_size``) UNCHANGED."""
    max_n_label = {"value": np.array([0, 0])}
    subgraph_sizes, enc_ratios, num_pruned_nodes = [], [], []

    # size estimate from the first split's positive queries (same as _core).
    first = graphs[splits[0]]
    sample_src = first["pos"] if len(first["pos"]) else first["neg"]
    bytes_per_datum = get_average_subgraph_size(
        min(100, len(sample_src)), sample_src, A, params) * 1.5
    links_length = 0
    for split_name in splits:
        s = graphs[split_name]
        links_length += (len(s["pos"]) + len(s["neg"])) * 2
    map_size = int(bytes_per_datum * links_length * 4) + (512 << 20)
    env = lmdb.open(params.db_path, map_size=int(map_size), max_dbs=6)

    def extraction_helper(links, g_labels, split_env):
        with env.begin(write=True, db=split_env) as txn:
            txn.put("num_graphs".encode(),
                    (len(links)).to_bytes(int.bit_length(len(links)), byteorder="little"))
        if len(links) == 0:
            return
        with mp.Pool(processes=None, initializer=intialize_worker,
                     initargs=(A, params, None)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label["value"] = np.maximum(np.max(datum["n_labels"], axis=0),
                                                  max_n_label["value"])
                subgraph_sizes.append(datum["subgraph_size"])
                enc_ratios.append(datum["enc_ratio"])
                num_pruned_nodes.append(datum["num_pruned_nodes"])
                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name in splits:
        split = graphs[split_name]
        logging.info(f"[sumgnn-bin] extracting POS subgraphs for {split_name} "
                     f"({len(split['pos'])})")
        # g_label = 1 for every positive query.
        pos_labels = np.ones(len(split["pos"]))
        extraction_helper(split["pos"], pos_labels, env.open_db((split_name + "_pos").encode()))

        logging.info(f"[sumgnn-bin] extracting NEG subgraphs for {split_name} "
                     f"({len(split['neg'])})")
        # g_label = 0 for every negative query (KEY binary difference).
        neg_labels = np.zeros(len(split["neg"]))
        extraction_helper(split["neg"], neg_labels, env.open_db((split_name + "_neg").encode()))

    with env.begin(write=True) as txn:
        bl_sub = int.bit_length(int(max_n_label["value"][0]))
        bl_obj = int.bit_length(int(max_n_label["value"][1]))
        txn.put("max_n_label_sub".encode(),
                (int(max_n_label["value"][0])).to_bytes(bl_sub, byteorder="little"))
        txn.put("max_n_label_obj".encode(),
                (int(max_n_label["value"][1])).to_bytes(bl_obj, byteorder="little"))
        for key, vals in (("subgraph_size", subgraph_sizes), ("enc_ratio", enc_ratios),
                          ("num_pruned_nodes", num_pruned_nodes)):
            txn.put(f"avg_{key}".encode(), struct.pack("f", float(np.mean(vals))))
            txn.put(f"min_{key}".encode(), struct.pack("f", float(np.min(vals))))
            txn.put(f"max_{key}".encode(), struct.pack("f", float(np.max(vals))))
            txn.put(f"std_{key}".encode(), struct.pack("f", float(np.std(vals))))


class BinarySubgraphDataset(SubgraphDataset):
    """Dataset that yields BOTH pos and neg extracted subgraphs (mixed, labelled).

    The ``_core`` ``SubgraphDataset`` opens ``db_neg`` but ``__len__``/``__getitem__``
    only cover ``db_pos``. This subclass indexes the concatenation ``[pos_db, neg_db]``:
    indices ``[0, n_pos)`` read ``db_pos`` (g_label already 1), ``[n_pos, n_pos+n_neg)``
    read ``db_neg`` (g_label already 0). Everything else (graph construction, node
    labelling, feature prep, KG multigraph) is inherited UNCHANGED from ``_core``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # _core __init__ set num_graphs_pos from db_pos; also read db_neg count.
        with self.main_env.begin(db=self.db_neg) as txn:
            raw = txn.get("num_graphs".encode())
            self.num_graphs_neg = int.from_bytes(raw, byteorder="little") if raw else 0

    def __len__(self) -> int:
        return self.num_graphs_pos + self.num_graphs_neg

    def __getitem__(self, index):
        if index < self.num_graphs_pos:
            db, local = self.db_pos, index
        else:
            db, local = self.db_neg, index - self.num_graphs_pos
        with self.main_env.begin(db=db) as txn:
            str_id = "{:08}".format(local).encode("ascii")
            nodes, r_label, g_label, n_labels = deserialize(txn.get(str_id)).values()
            subgraph = self._prepare_subgraphs(nodes, r_label, n_labels)
        # returns (subgraph, g_label [0/1 binary], r_label [0]) — same tuple shape the
        # _core collate_dgl / move_batch_to_device_dgl expect; here g_label is the BCE
        # target, r_label is the single-DDI-relation slot.
        return subgraph, g_label, r_label


class BinaryEvaluator:
    """Sigmoid-prob AUPRC / AUROC evaluator (binary primary metric = AUPRC).

    Unlike the ``_core`` ``Evaluator`` (macro-F1 over argmax) / ``Evaluator_ddi2``
    (per-label masked-BCE), this reads the single logit, applies sigmoid, and computes
    AUPRC + AUROC against the per-subgraph binary ``g_label`` (the collate maps g_label
    -> ``targets_pos``)."""

    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self):
        loader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                            num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(
                    batch, self.params.device)
                logit = self.graph_classifier(data_pos).view(-1)         # (b,) single logit
                probs.append(torch.sigmoid(logit).cpu().numpy())
                labels.append(targets_pos.cpu().numpy())                 # g_label 0/1
        if not probs:
            return {"auprc": 0.0, "auroc": 0.0}
        y_score = np.concatenate(probs)
        y_true = np.concatenate(labels).astype(np.int64)
        if len(set(y_true.tolist())) < 2:      # degenerate val -> AUPRC undefined
            return {"auprc": float(y_true.mean()), "auroc": 0.5}
        return {"auprc": float(average_precision_score(y_true, y_score)),
                "auroc": float(roc_auc_score(y_true, y_score))}


class BinaryTrainer:
    """BCEWithLogitsLoss training loop with per-epoch loss logging (CLAUDE.md) and
    AUPRC-on-validation best-checkpoint selection.

    Mirrors the ``_core`` ``Trainer`` structure (Adam, grad clip=10, per-iter eval
    hook) but with the binary loss + primary metric. It does NOT reuse ``_core``
    ``Trainer`` because that branch is CrossEntropy (drugbank) / masked-BCE (BioSNAP)
    with a macro-F1 / per-label best-checkpoint metric."""

    def __init__(self, params, graph_classifier, train, valid_evaluator=None,
                 test_evaluator=None):
        self.params = params
        self.graph_classifier = graph_classifier
        self.train_data = train
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator
        self.updates_counter = 0
        self.criterion = nn.BCEWithLogitsLoss()
        model_params = list(self.graph_classifier.parameters())
        logging.info("[sumgnn-bin] total params: %d"
                     % sum(map(lambda x: x.numel(), model_params)))
        self.optimizer = torch.optim.Adam(model_params, lr=params.lr, weight_decay=params.l2)
        self.best_metric = -1.0
        self.not_improved_count = 0

    def train_epoch(self) -> float:
        loader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                            num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        total_loss, n_batches = 0.0, 0
        bar = tqdm(enumerate(loader))
        for b_idx, batch in bar:
            data_pos, r_labels_pos, targets_pos = self.params.move_batch_to_device(
                batch, self.params.device)
            self.optimizer.zero_grad()
            logit = self.graph_classifier(data_pos).view(-1)             # (b,) single logit
            target = targets_pos.float()                                 # g_label 0/1 -> float
            loss = self.criterion(logit, target)
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            total_loss += loss.item()
            n_batches += 1
            if self.params.log_step_every and (b_idx + 1) % self.params.log_step_every == 0:
                bar.set_description(f"loss={total_loss / n_batches:.4f}")
            if (self.valid_evaluator and self.params.eval_every_iter
                    and self.updates_counter % self.params.eval_every_iter == 0):
                self._eval_and_maybe_save()
        return total_loss / max(n_batches, 1)

    def _eval_and_maybe_save(self) -> None:
        tic = time.time()
        result = self.valid_evaluator.eval()                             # {auprc, auroc}
        logging.info(f"[sumgnn-bin] val AUPRC={result['auprc']:.4f} "
                     f"AUROC={result['auroc']:.4f} in {time.time() - tic:.1f}s")
        if result["auprc"] >= self.best_metric:                          # primary = AUPRC
            self.best_metric = result["auprc"]
            self.not_improved_count = 0
            torch.save(self.graph_classifier,
                       os.path.join(self.params.exp_dir, "best_graph_classifier.pth"))
            logging.info(f"[sumgnn-bin] new best val AUPRC={self.best_metric:.4f}; saved")
        else:
            self.not_improved_count += 1

    def train(self) -> None:
        for epoch in range(1, self.params.num_epochs + 1):
            t0 = time.time()
            mean_loss = self.train_epoch()
            extra = ""
            if self.valid_evaluator:
                res = self.valid_evaluator.eval()
                if res["auprc"] >= self.best_metric:
                    self.best_metric = res["auprc"]
                    torch.save(self.graph_classifier,
                               os.path.join(self.params.exp_dir, "best_graph_classifier.pth"))
                extra = f" val_auprc={res['auprc']:.4f} val_auroc={res['auroc']:.4f}"
            logging.info(f"[sumgnn-bin] [ep {epoch}/{self.params.num_epochs}] "
                         f"mean_loss={mean_loss:.4f} time={time.time() - t0:.1f}s"
                         f" best_val_auprc={self.best_metric:.4f}{extra}")


__all__ = ["generate_binary_subgraph_datasets", "BinarySubgraphDataset",
           "BinaryEvaluator", "BinaryTrainer"]

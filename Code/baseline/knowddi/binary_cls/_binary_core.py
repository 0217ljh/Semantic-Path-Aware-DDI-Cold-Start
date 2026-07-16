"""Binary DDI core for KnowDDI (Case-B GraIL-style binary link prediction).

KnowDDI's Phase-1 core has NO binary branch (only DrugBank multiclass + BioSNAP
multilabel). This module adds a BINARY-task core that REUSES KnowDDI's enclosing-subgraph
extraction + directional ``extract_r_digraph`` pruning + GraphSAGE-once-on-global + GSL
refinement + the ``[mean||head||tail]`` head UNCHANGED (everything algorithmic is imported
from the Phase-1 core; nothing in the core is edited). The ONLY task-specific changes live
here, per the codex-approved design (thread 019f2930):

  * head width ``num_rels == 1`` (a single logit) -> ``Classifier_model`` builds
    ``W_final = nn.Linear(3*score_dim, 1)`` untouched.
  * ``BCEWithLogitsLoss`` on the per-subgraph BINARY ``g_label`` (1 positive DDI pair,
    0 negative pair). The ``r_label`` slot is fixed to 0 (the single DDI relation).
  * validation / best-checkpoint metric = AUPRC (the binary primary metric), NOT the
    multiclass macro-F1 / BioSNAP masked-BCE of the Phase-1 Trainer/Evaluator.
  * ``predict`` -> ``(n,)`` sigmoid probabilities.

Faithfulness notes:
  * The Phase-1 ``generate_subgraph_datasets`` writes one db per split with ``g_label=1``
    for every extracted link (drugbank branch), and the Phase-1 ``SubgraphDataset`` opens a
    single ``db_name`` from a CACHED env (``_open_env`` with ``max_dbs=3``). Reusing either
    unchanged would corrupt the binary task, so this file provides:
      - :func:`generate_binary_subgraph_datasets` -- extraction that builds the DDI
        adjacency from TRAIN POSITIVES ONLY and extracts the EXPLICIT pos and neg query
        sets with the correct per-query ``g_label`` (1 pos / 0 neg), ``r_label=0``, writing
        6 named dbs (``<split>_pos`` / ``<split>_neg``) into a FRESH env with ``max_dbs>=6``
        (codex: the cached ``_open_env`` caps at 3 named dbs).
      - :class:`BinaryKnowDDISubgraphDataset` -- opens BOTH the ``*_pos`` and ``*_neg`` dbs
        (fresh ``max_dbs>=6`` env) and yields mixed labelled examples, calling the Phase-1
        ``SubgraphDataset._prepare_subgraphs`` (which runs ``extract_r_digraph`` directional
        pruning) UNCHANGED.
      - :class:`BinaryEvaluator` -- sigmoid-prob AUPRC/AUROC evaluator.
      - :class:`BinaryTrainer` -- BCEWithLogitsLoss loop, per-epoch loss log (CLAUDE.md),
        AUPRC-on-validation best-checkpoint selection.

Reused-UNCHANGED from the Phase-1 core (imported, NOT reimplemented):
  * enclosing-subgraph BFS + double-radius node labelling:
    ``subgraph_extraction.subgraph_extraction_labeling`` / ``get_average_subgraph_size`` /
    ``intialize_worker`` / ``extract_save_subgraph``.
  * DDI adjacency + KG assembly from int-triple files: ``process_files_ddi``.
  * subgraph -> DGL graph + directional pruning + feature prep:
    ``SubgraphDataset._prepare_subgraphs`` / ``extract_r_digraph`` / ``_prepare_features``
    (inherited by subclassing ``SubgraphDataset``).
  * graph batching + device move: ``collate_dgl`` / ``move_batch_to_device_dgl``.
  * model: ``Classifier_model`` (head width driven by ``params.num_rels``).

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): mirrors the
SumGNN binary core SHAPE but imports only KnowDDI's own core; does NOT import
``baseline.sumgnn``.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import struct
import time
from types import SimpleNamespace

import lmdb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- reused-UNCHANGED Phase-1 core imports (nothing here edits the core) ----------
from ..data_processor import datasets as _core_datasets
from ..data_processor.datasets import SubgraphDataset
from ..data_processor.subgraph_extraction import (extract_save_subgraph,
                                                  get_average_subgraph_size,
                                                  intialize_worker)
from ..utils.data_utils import process_files_ddi
from ..utils.graph_utils import deserialize, serialize


def _get_or_install_binary_env(db_path: str):
    """Return a readonly LMDB env for ``db_path`` with ``max_dbs>=6`` (the 6 named dbs
    ``<split>_pos`` / ``<split>_neg`` over train/valid/test), PRE-SEEDING the Phase-1
    ``datasets._ENV_CACHE`` so ``SubgraphDataset.__init__``'s ``_open_env(db_path)`` cache-
    HITS this env instead of opening its own ``max_dbs=3`` handle (codex 019f2938 Opt C).

    lmdb 2.x refuses to open the SAME env twice in one process (datasets.py:49), so we must
    ensure exactly ONE env exists per db_path. Binary always uses a DEDICATED db_path (the
    wrapper's ``digraph_hop_*_BKG_file_bin``) that only THIS function ever seeds, always with
    ``max_dbs=6``, so a stale ``max_dbs=3`` entry for it cannot arise. If an env is already
    cached for this dedicated path we reuse it as-is (it was created here with max_dbs=6);
    otherwise we open a fresh readonly ``max_dbs=6`` env and cache it."""
    env = _core_datasets._ENV_CACHE.get(db_path)
    if env is None:
        env = lmdb.open(db_path, readonly=True, max_dbs=6, lock=False)
        _core_datasets._ENV_CACHE[db_path] = env
    return env


def _load_query_pairs(path: str) -> np.ndarray:
    """Load a query file ``drug_a_idx drug_b_idx 0`` (r_label slot = 0) as an (n,3) int
    array. Returns an empty (0,3) array for a missing / empty file."""
    if not os.path.isfile(path):
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.loadtxt(path)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return arr.reshape(-1, 3).astype(np.int64)


def generate_binary_subgraph_datasets(params: "SimpleNamespace",
                                       splits=("train", "valid", "test")) -> None:
    """BINARY enclosing-subgraph extraction (Case-B).

    Differs from the Phase-1 ``generate_subgraph_datasets`` on two faithfulness points:

      1. Adjacency is built from ``process_files_ddi(params.file_paths, BKG_file)`` where the
         ``train`` file contains TRAIN POSITIVES ONLY (relation slot 0). Negatives never
         enter the file paths, so they never become false DDI edges in the adjacency.
      2. For each split we extract the EXPLICIT pos query set (``g_label=1``) AND the
         EXPLICIT neg query set (``g_label=0``) read from separate on-disk files
         (``<split>_pos.txt`` / ``<split>_neg.txt``). The Phase-1 version would instead write
         ``g_label=1`` for every extracted link -- avoided here.

    Writes the LMDB layout ``BinaryKnowDDISubgraphDataset`` expects: named dbs
    ``<split>_pos`` / ``<split>_neg`` under ``params.db_path``, plus the shared
    ``max_n_label_*`` stats keys. Reuses the Phase-1 per-subgraph workers
    (``extract_save_subgraph`` / ``intialize_worker``) + size estimator
    (``get_average_subgraph_size``) UNCHANGED.
    """
    BKG_file = params.bkg_file_path
    # process_files_ddi reads train/valid/test.txt (POSITIVES only) + the BKG to build the
    # multigraph adjacency `adj_list` (relation 0 = train-positive DDI, BKG rels follow). We
    # DISCARD its `triplets` (they are the positives) and instead drive extraction from the
    # explicit pos/neg query files, so negatives never touch adj.
    adj_list, _triplets, _e2i, _relation2id, _id2entity, _id2relation, _rel = process_files_ddi(
        params.file_paths, BKG_file)

    # explicit pos/neg query sets per split (rows `a b 0`).
    base = os.path.dirname(params.file_paths["train"])
    query_files = {
        "train": (os.path.join(base, "train_pos.txt"), os.path.join(base, "train_neg.txt")),
        "valid": (os.path.join(base, "valid_pos.txt"), os.path.join(base, "valid_neg.txt")),
        "test": (os.path.join(base, "test_pos.txt"), os.path.join(base, "test_neg.txt")),
    }
    graphs = {}
    for split_name in splits:
        pos_f, neg_f = query_files[split_name]
        graphs[split_name] = {"pos": _load_query_pairs(pos_f), "neg": _load_query_pairs(neg_f)}

    _links2subgraphs_binary(adj_list, graphs, params, splits)


def _links2subgraphs_binary(A, graphs, params, splits) -> None:
    """BINARY variant of the Phase-1 ``links2subgraphs``: extract enclosing subgraphs for
    the explicit pos + neg query sets, writing ``g_label=1`` for pos and ``g_label=0`` for
    neg (the Phase-1 drugbank branch writes 1 for both). Reuses the Phase-1 per-subgraph
    workers (``extract_save_subgraph`` / ``intialize_worker``) and the size estimator
    (``get_average_subgraph_size``) UNCHANGED."""
    max_n_label = {"value": np.array([0, 0])}
    subgraph_sizes, enc_ratios, num_pruned_nodes = [], [], []

    # size estimate from the first split's positive queries (same as Phase-1).
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
        with mp.Pool(processes=params.num_workers if getattr(params, "num_workers", 0) else None,
                     initializer=intialize_worker, initargs=(A, params, None)) as p:
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
        logging.info(f"[knowddi-bin] extracting POS subgraphs for {split_name} "
                     f"({len(split['pos'])})")
        pos_labels = np.ones(len(split["pos"]))          # g_label = 1 for every positive
        extraction_helper(split["pos"], pos_labels, env.open_db((split_name + "_pos").encode()))

        logging.info(f"[knowddi-bin] extracting NEG subgraphs for {split_name} "
                     f"({len(split['neg'])})")
        neg_labels = np.zeros(len(split["neg"]))         # g_label = 0 (KEY binary difference)
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
            if vals:
                txn.put(f"avg_{key}".encode(), struct.pack("f", float(np.mean(vals))))
                txn.put(f"min_{key}".encode(), struct.pack("f", float(np.min(vals))))
                txn.put(f"max_{key}".encode(), struct.pack("f", float(np.max(vals))))
                txn.put(f"std_{key}".encode(), struct.pack("f", float(np.std(vals))))
    env.close()


class BinaryKnowDDISubgraphDataset(SubgraphDataset):
    """Dataset that yields BOTH pos and neg extracted subgraphs (mixed, labelled).

    The Phase-1 ``SubgraphDataset`` opens a single ``db_name`` from a CACHED env
    (``_open_env``, ``max_dbs=3``). This subclass instead opens a FRESH env
    (``max_dbs>=6``) and BOTH the ``<split>_pos`` and ``<split>_neg`` dbs; indices
    ``[0, n_pos)`` read the pos db (g_label already 1), ``[n_pos, n_pos+n_neg)`` read the neg
    db (g_label already 0). Everything else (graph construction, ``extract_r_digraph``
    directional pruning, node labelling, feature prep, KG multigraph) is inherited UNCHANGED
    from the Phase-1 ``SubgraphDataset._prepare_subgraphs``.

    Constructed like the Phase-1 dataset (``db_path`` + ``split`` name) but takes the split
    base name (``train`` / ``valid`` / ``test``) instead of a full ``db_name``.
    """

    def __init__(self, db_path, split, raw_data_paths=None, add_traspose_rels=None,
                 use_pre_embeddings=False, dataset="drugbank", kge_model="", ssp_graph=None,
                 id2entity=None, id2relation=None, rel=None, global_graph=None,
                 dig_layer=4, bkg_file_path=None) -> None:
        # codex 019f2938 Opt C: PRE-SEED datasets._ENV_CACHE[db_path] with our max_dbs>=6
        # readonly env BEFORE super().__init__, so the Phase-1 _open_env cache-HITS this env
        # (never opening its own max_dbs=3 handle; avoids the lmdb-2.x "open same env twice"
        # error). super().__init__ then builds ssp_graph/global_graph/id maps + sets self.db
        # = <split>_pos + self.num_graphs (pos count); we override __len__/__getitem__ to use
        # self.num_graphs_pos/neg, so the inherited self.num_graphs is harmlessly ignored.
        self.main_env = _get_or_install_binary_env(db_path)
        super().__init__(db_path=db_path, db_name=f"{split}_pos", raw_data_paths=raw_data_paths,
                         add_traspose_rels=add_traspose_rels,
                         use_pre_embeddings=use_pre_embeddings, dataset=dataset,
                         kge_model=kge_model, ssp_graph=ssp_graph, id2entity=id2entity,
                         id2relation=id2relation, rel=rel, global_graph=global_graph,
                         dig_layer=dig_layer, bkg_file_path=bkg_file_path)
        # super().__init__ reused our injected env (cache hit); attach both pos/neg dbs.
        self.db_pos = self.main_env.open_db(f"{split}_pos".encode())
        self.db_neg = self.main_env.open_db(f"{split}_neg".encode())
        with self.main_env.begin(db=self.db_pos) as txn:
            raw = txn.get("num_graphs".encode())
            self.num_graphs_pos = int.from_bytes(raw, byteorder="little") if raw else 0
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
            # Phase-1 _prepare_subgraphs runs extract_r_digraph directional pruning UNCHANGED.
            directed_subgraph = self._prepare_subgraphs(nodes, n_labels)
        # KnowDDI tuple order (codex 019f2930): (graph, r_label, g_label) -- the Phase-1
        # collate_dgl / move_batch_to_device_dgl treat the THIRD slot as the target. g_label
        # (0/1 binary) is the BCE target; r_label (0) is the single-DDI-relation slot.
        return directed_subgraph, r_label, g_label


class BinaryEvaluator:
    """Sigmoid-prob AUPRC / AUROC evaluator (binary primary metric = AUPRC).

    Unlike the Phase-1 ``Evaluator_multiclass`` (macro-F1 over argmax) /
    ``Evaluator_multilabel`` (per-label ROC-AUC), this reads the single logit, applies
    sigmoid, and computes AUPRC + AUROC against the per-subgraph binary ``g_label`` (the
    move-batch maps g_label -> ``targets_pos``)."""

    def __init__(self, params, graph_classifier, data):
        from ..utils.graph_utils import collate_dgl, move_batch_to_device_dgl
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data
        self.collate_fn = collate_dgl
        self.move_batch_to_device = move_batch_to_device_dgl

    def eval(self):
        loader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                            num_workers=self.params.num_workers, collate_fn=self.collate_fn)
        self.graph_classifier.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                data_pos, _r_labels, targets_pos = self.move_batch_to_device(
                    batch, self.params.device, multi_type=1)
                logit = self.graph_classifier(data_pos).view(-1)         # (b,) single logit
                probs.append(torch.sigmoid(logit).cpu().numpy())
                labels.append(targets_pos.cpu().numpy())                 # g_label 0/1
        if not probs:
            return {"auprc": 0.0, "auroc": 0.0}
        y_score = np.concatenate(probs)
        y_true = np.concatenate(labels).astype(np.int64)
        if len(set(y_true.tolist())) < 2:      # degenerate val -> AUPRC/AUROC undefined
            return {"auprc": float(y_true.mean()), "auroc": 0.5}
        return {"auprc": float(average_precision_score(y_true, y_score)),
                "auroc": float(roc_auc_score(y_true, y_score))}


class BinaryTrainer:
    """BCEWithLogitsLoss training loop with per-epoch loss logging (CLAUDE.md) and
    AUPRC-on-validation best-checkpoint selection.

    Mirrors the Phase-1 ``Trainer`` structure (Adam, ExponentialLR, grad clip=10, mid-epoch
    eval hook, best-ckpt to ``best_graph_classifier.pth``) but with the binary loss + primary
    metric. It does NOT reuse the Phase-1 ``Trainer`` because that branch is CrossEntropy
    (drugbank) / masked-BCE (BioSNAP) with a macro-F1 / ROC-AUC best-checkpoint metric."""

    def __init__(self, params, graph_classifier, train, valid_evaluator=None,
                 test_evaluator=None):
        from ..utils.graph_utils import collate_dgl, move_batch_to_device_dgl
        self.params = params
        self.graph_classifier = graph_classifier
        self.train_data = train
        self.valid_evaluator = valid_evaluator
        self.test_evaluator = test_evaluator
        self.collate_fn = collate_dgl
        self.move_batch_to_device = move_batch_to_device_dgl
        self.updates_counter = 0
        self.criterion = nn.BCEWithLogitsLoss()
        model_params = list(self.graph_classifier.parameters())
        logging.info("[knowddi-bin] total params: %d"
                     % sum(map(lambda x: x.numel(), model_params)))
        self.optimizer = torch.optim.Adam(model_params, lr=params.lr,
                                          weight_decay=params.weight_decay_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                params.lr_decay_rate)
        self.eval_every_iter = int(getattr(params, "eval_every_iter", 526))
        self.best_metric = -1.0
        self.not_improved_count = 0
        self.early_stop = 0

        _ROOT = _project_code_root()
        import sys
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from my_code.utils.train_progress import TrainProgress
        self.prog = TrainProgress(
            total_epochs=params.num_epochs,
            log_step_every=getattr(params, "log_step_every", 50),
            prefix="[knowddi-bin] ",
            eval_strategy="steps",
            eval_steps=self.eval_every_iter,
            save_strategy="no",
        )

    def train_epoch(self) -> float:
        loader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True,
                            num_workers=self.params.num_workers, collate_fn=self.collate_fn)
        self.graph_classifier.train()
        total_loss, n_batches = 0.0, 0
        for b_idx, batch in enumerate(loader):
            data_pos, _r_labels, targets_pos = self.move_batch_to_device(
                batch, self.params.device, multi_type=1)
            self.optimizer.zero_grad()
            logit = self.graph_classifier(data_pos).view(-1)             # (b,) single logit
            target = targets_pos.float()                                 # g_label 0/1 -> float
            loss = self.criterion(logit, target)
            loss.backward()
            clip_grad_norm_(self.graph_classifier.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.updates_counter += 1
            self.prog.step(loss.item())
            total_loss += loss.item()
            n_batches += 1
            # Mid-epoch VALID eval + best-ckpt + LR step every ``eval_every_iter`` UPDATES
            # (Phase-1 trainer cadence). Scheduler steps at the END of the eval block.
            if self.valid_evaluator and self.updates_counter % self.eval_every_iter == 0:
                self._eval_and_maybe_save()
                self.scheduler.step()
                if self.early_stop:
                    break
        return total_loss / max(n_batches, 1)

    def _eval_and_maybe_save(self) -> None:
        tic = time.time()
        result = self.valid_evaluator.eval()                             # {auprc, auroc}
        self.prog.log_eval({"val_auprc": result["auprc"], "val_auroc": result["auroc"]},
                           scope="step")
        logging.info(f"[knowddi-bin] val AUPRC={result['auprc']:.4f} "
                     f"AUROC={result['auroc']:.4f} in {time.time() - tic:.1f}s")
        if result["auprc"] >= self.best_metric:                          # primary = AUPRC
            self.best_metric = result["auprc"]
            self.not_improved_count = 0
            torch.save(self.graph_classifier,
                       os.path.join(self.params.exp_dir, "best_graph_classifier.pth"))
            logging.info(f"[knowddi-bin] new best val AUPRC={self.best_metric:.4f}; saved")
        else:
            self.not_improved_count += 1
            if self.not_improved_count >= self.params.early_stop_epoch:
                self.early_stop = 1

    def train(self) -> None:
        for epoch in range(1, self.params.num_epochs + 1):
            self.prog.epoch_start(epoch - 1)
            t0 = time.time()
            mean_loss = self.train_epoch()
            self.prog.epoch_end(extra={"mean_loss": mean_loss,
                                       "best_val_auprc": self.best_metric})
            logging.info(f"[knowddi-bin] [ep {epoch}/{self.params.num_epochs}] "
                         f"mean_loss={mean_loss:.4f} time={time.time() - t0:.1f}s "
                         f"best_val_auprc={self.best_metric:.4f}")
            if self.early_stop:
                logging.info(f"[knowddi-bin] val AUPRC didn't improve for "
                             f"{self.params.early_stop_epoch} evals. Training stops.")
                break


def _project_code_root():
    """Absolute path to ``.../Code`` (for ``my_code.utils.train_progress`` import)."""
    from pathlib import Path
    return Path(__file__).resolve().parents[3]


__all__ = ["generate_binary_subgraph_datasets", "BinaryKnowDDISubgraphDataset",
           "BinaryEvaluator", "BinaryTrainer"]

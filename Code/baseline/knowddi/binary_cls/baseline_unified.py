"""KnowDDI binary baseline for the UNIFIED benchmark (decision B, Case-B GraIL-style).

KnowDDI is a KG-subgraph GSL method -> applies to KG datasets (drugbank_latest_* via the
merged DrugBank+Hetionet+PrimeKG KG). One leaf = one regime -> a SINGLE KnowDDI model whose
GraphSAGE-once-on-global + per-pair enclosing-subgraph directional ``extract_r_digraph``
pruning + graph-structure-learning refinement + ``[mean||head||tail]`` head are REUSED
UNCHANGED from the Phase-1 core; ONLY the task head + loss + metric are binary (single
logit, BCEWithLogitsLoss on the per-subgraph binary ``g_label``, AUPRC best-checkpoint
selection). NO cross-regime routing.

This mirrors ``multi_cls/baseline_unified.py`` (KG / data-cache / full-KG-scope conventions)
with the binary data + training path from ``_binary_core`` / ``build_knowddi_binary_data``
(codex-approved design, thread 019f2930):

Pipeline per leaf:
  1. Build KnowDDI's on-disk int layout from the leaf's string-id drugs + pairs + merged KG
     via ``build_knowddi_binary_data`` (drugs indexed first; kg_scope="full" default). The
     DDI relation adjacency is built from TRAIN POSITIVES ONLY; negatives are extraction
     queries only. Cached under ``_data/necessary/bin__<hash>__<kg_scope>/``.
  2. Extract per-pair enclosing subgraphs (lmdb) for the EXPLICIT pos + neg query sets via
     ``generate_binary_subgraph_datasets`` (g_label 1 pos / 0 neg, r_label 0), then train
     with ``BinaryTrainer`` (BCEWithLogitsLoss) + ``BinaryEvaluator`` (AUPRC on validation).
     The load-time directional ``extract_r_digraph`` pruning still runs (Phase-1
     ``SubgraphDataset._prepare_subgraphs``).
  3. ``predict`` scores each test subgraph -> sigmoid prob (n,), aligned back to the full
     test_df row order via the build's ``test_manifest.json``.

Paper-faithful hyperparams are LOCKED to the DrugBank official LOG (the base method's
multiclass config; the binary task only swaps head/loss/metric): num_dig_layers=4,
num_infer_layers=3, lamda=0.5, threshold=0.05, hop=2, emb_dim=32, num_gcn_layers=2, lr=0.005,
weight_decay_rate=1e-5, lr_decay_rate=0.93, batch_size=256, num_epochs=50, early_stop_epoch=10,
eval_every_iter=526, num_workers=32, MLP_hidden_dim=32, MLP_num_layers=2, MLP_dropout=0.2,
gsl_rel_emb_dim=32, gcn_dropout=0.2, gcn_aggregator_type='mean', func_num=1, sparsify=1,
edge_softmax=1, gsl_has_edge_emb=1, max_links=250000, max_nodes_per_hop=200. ``num_nodes`` is
sized from the ACTUAL built entity count.

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): does NOT import
``baseline.sumgnn``; mirrors the SumGNN binary wrapper SHAPE with KnowDDI's own core.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "Code") not in sys.path:
    sys.path.insert(0, str(_ROOT / "Code"))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.knowddi._data.necessary.build_knowddi_binary_data import (  # noqa: E402
    build_knowddi_binary_data)

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "_data" / "necessary"


def _leaf_hash(meta: dict, fold: str) -> str:
    key = f"bin|{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{fold}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@register_unified("knowddi")
class KnowDDIUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, emb_dim: int = 32, hop: int = 2, num_gcn_layers: int = 2,
                 num_infer_layers: int = 3, num_dig_layers: int = 4, lamda: float = 0.5,
                 threshold: float = 0.05, n_epochs: int = 50, learning_rate: float = 5e-3,
                 lr_decay_rate: float = 0.93, weight_decay_rate: float = 1e-5,
                 batch_size: int = 256, max_links: int = 250000, max_nodes_per_hop: int = 200,
                 early_stop_epoch: int = 10, eval_every_iter: int = 526,
                 num_workers: int = 32, device: str = "auto",
                 log_step_every: int = 50, kg_scope: str = "full",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        # LOCKED to DrugBank LOG values (log_train.txt:2) for the shared GSL/GraphSAGE core;
        # the binary task only swaps head/loss/metric (Case-B). n_epochs/device/run_dir from
        # the runner; everything else is the paper's best-on-DrugBank config.
        self.emb_dim = emb_dim
        self.hop = hop
        self.num_gcn_layers = num_gcn_layers
        self.num_infer_layers = num_infer_layers
        self.num_dig_layers = num_dig_layers
        self.lamda = lamda
        self.threshold = threshold
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.weight_decay_rate = weight_decay_rate
        self.batch_size = batch_size
        self.max_links = max_links
        self.max_nodes_per_hop = max_nodes_per_hop
        self.early_stop_epoch = early_stop_epoch
        # official eval + LR-scheduler cadence: every ``eval_every_iter`` param updates.
        # DrugBank official value 526 (log_train.txt:2; argparse default train.py:129).
        self.eval_every_iter = eval_every_iter
        self.num_workers = num_workers
        # standing decision 2026-07-01: default to the FULL merged KG (not drug-incident).
        self.kg_scope = kg_scope
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._fold = "fold0"
        self._params: SimpleNamespace | None = None
        self._clf = None
        self._test_ds = None

    # ---- KG / data prep -------------------------------------------------
    @staticmethod
    def _merged_edges_path(resources: "LeafResources") -> Path:
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("KnowDDI requires a KG; this leaf declares none.")
        edges = Path(kg.source) / MERGED_EDGES
        if not edges.is_file():
            raise FileNotFoundError(f"merged KG edges not found: {edges}")
        return edges

    def _ensure_data(self, train_df, val_df, test_df, resources) -> Path:
        meta = resources.meta
        # scope in the cache dir so full-KG vs drug-incident caches never collide.
        data_dir = _CACHE_ROOT / f"bin__{_leaf_hash(meta, self._fold)}__{self.kg_scope}"
        stamp = data_dir / "_build_stats.json"
        if stamp.is_file():
            logging.info(f"[knowddi-bin] data cache HIT: {data_dir}")
            return data_dir
        logging.info(f"[knowddi-bin] BUILDING data at {data_dir} (kg_scope={self.kg_scope})")
        edges = pd.read_parquet(self._merged_edges_path(resources))
        stats = build_knowddi_binary_data(
            data_dir, resources.drugs, edges, train_df, val_df, test_df,
            kg_scope=self.kg_scope,
        )
        logging.info(f"[knowddi-bin] build stats: {stats}")
        return data_dir

    def fit_leaf(self, leaf: "Leaf") -> None:  # capture fold for the cache key
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources, _test=leaf.test)

    def fit(self, train_df, val_df=None, *, resources, _test=None) -> None:
        from baseline.knowddi.binary_cls._binary_core import (
            BinaryEvaluator, BinaryKnowDDISubgraphDataset, BinaryTrainer,
            generate_binary_subgraph_datasets)
        from baseline.knowddi.model.Classifier_model import Classifier_model

        meta = resources.meta
        if val_df is None:
            val_df = train_df
        # test pairs must be in the on-disk test_*.txt so their subgraphs get extracted
        test_df = _test if _test is not None else val_df
        data_dir = self._ensure_data(train_df, val_df, test_df, resources)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        params = SimpleNamespace(
            dataset="drugbank",
            # subgraph extraction
            hop=self.hop, max_links=self.max_links, max_nodes_per_hop=self.max_nodes_per_hop,
            enclosing_subgraph=True, add_traspose_rels=False,
            # trainer
            num_epochs=self.n_epochs, early_stop_epoch=self.early_stop_epoch,
            eval_every_iter=self.eval_every_iter,
            optimizer="Adam", lr=self.learning_rate, lr_decay_rate=self.lr_decay_rate,
            weight_decay_rate=self.weight_decay_rate, batch_size=self.batch_size,
            num_workers=self.num_workers, log_step_every=self.log_step_every,
            # GraphSAGE
            emb_dim=self.emb_dim, num_gcn_layers=self.num_gcn_layers,
            gcn_aggregator_type="mean", gcn_dropout=0.2,
            # gsl_model
            num_infer_layers=self.num_infer_layers, num_dig_layers=self.num_dig_layers,
            MLP_hidden_dim=32, MLP_num_layers=2, MLP_dropout=0.2, func_num=1, sparsify=1,
            threshold=self.threshold, edge_softmax=1, gsl_rel_emb_dim=32,
            lamda=self.lamda, gsl_has_edge_emb=1,
            # runtime
            use_pre_embeddings=False, kge_model="TransE",
            device=torch.device(dev),
        )
        # pos-only .txt paths for process_files_ddi (adjacency built from train POSITIVES).
        params.file_paths = {"train": str(data_dir / "train.txt"),
                             "valid": str(data_dir / "valid.txt"),
                             "test": str(data_dir / "test.txt")}
        params.bkg_file_path = str(data_dir / "BKG_file.txt")
        # DEDICATED binary db_path (codex 019f2938: binary owns its cache entry, separate
        # from the multiclass Phase-1 digraph path).
        params.db_path = str(data_dir / f"digraph_hop_{self.hop}_BKG_file_bin")
        params.exp_dir = str(data_dir / "exp")
        os.makedirs(params.exp_dir, exist_ok=True)
        self._params = params

        if not os.path.isdir(params.db_path):
            generate_binary_subgraph_datasets(params)

        train = BinaryKnowDDISubgraphDataset(
            db_path=params.db_path, split="train", raw_data_paths=params.file_paths,
            add_traspose_rels=params.add_traspose_rels,
            use_pre_embeddings=params.use_pre_embeddings, dataset="drugbank",
            kge_model="TransE", dig_layer=self.num_dig_layers,
            bkg_file_path=params.bkg_file_path)
        valid = BinaryKnowDDISubgraphDataset(
            db_path=params.db_path, split="valid",
            use_pre_embeddings=params.use_pre_embeddings, dataset="drugbank",
            kge_model="TransE", ssp_graph=train.ssp_graph, id2entity=train.id2entity,
            id2relation=train.id2relation, rel=train.num_rels,
            global_graph=train.global_graph, dig_layer=self.num_dig_layers,
            bkg_file_path=params.bkg_file_path)
        test = BinaryKnowDDISubgraphDataset(
            db_path=params.db_path, split="test",
            use_pre_embeddings=params.use_pre_embeddings, dataset="drugbank",
            kge_model="TransE", ssp_graph=train.ssp_graph, id2entity=train.id2entity,
            id2relation=train.id2relation, rel=train.num_rels,
            global_graph=train.global_graph, dig_layer=self.num_dig_layers,
            bkg_file_path=params.bkg_file_path)

        # Case-B binary head: a SINGLE logit (Classifier_model builds W_final =
        # Linear(3*score_dim, 1) via params.num_rels=1). train.num_rels here is the # DDI
        # relations from process_files_ddi (== 1: the single train-positive DDI relation,
        # slot 0), so num_rels=1 is consistent. aug_num_rels includes BKG rels + self loop.
        params.num_rels = 1
        params.aug_num_rels = train.aug_num_rels
        # num_nodes sizes GraphSAGE.pre_embed; from the ACTUAL built entity count.
        params.num_nodes = train.num_entity
        params.global_graph = train.global_graph.to(params.device)

        clf = Classifier_model(params).to(params.device)
        logging.info(f"[knowddi-bin] built Classifier_model: num_rels={params.num_rels} "
                     f"aug_num_rels={params.aug_num_rels} num_nodes={params.num_nodes}")

        # Model selection on VALIDATION, NOT test -- primary metric = AUPRC.
        valid_ev = BinaryEvaluator(params, clf, valid)   # best-ckpt selection (validation)
        test_ev = BinaryEvaluator(params, clf, test)     # logging only
        trainer = BinaryTrainer(params, clf, train, valid_ev, test_ev)
        logging.info(f"[knowddi-bin] training {self.n_epochs} epochs on {len(train)} subgraphs "
                     f"(num_rels={params.num_rels})")
        trainer.train()

        # load_best_model_at_end: reload the best (valid-AUPRC-selected) checkpoint if the
        # trainer saved one; else keep the final in-memory model.
        best = Path(params.exp_dir) / "best_graph_classifier.pth"
        if best.is_file():
            try:
                clf = torch.load(best, map_location=params.device, weights_only=False)
                logging.info(f"[knowddi-bin] loaded best valid-selected checkpoint: {best}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[knowddi-bin] best-ckpt reload failed ({e}); using final model")
        self._clf = clf
        self._test_ds = test

    # ---- prediction -----------------------------------------------------
    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        """Score test pairs -> (n,) P(positive), aligned to test_df row order.

        The test dataset yields extracted subgraphs ordered [all test_pos, then all
        test_neg]; the build's ``test_manifest.json`` records the original test_df row order +
        known/pos masks so we scatter each prob back to its row. Rows whose pair was dropped
        (unknown drug) get 0.5 (neutral; counted by the runner's metrics)."""
        if self._clf is None:
            raise RuntimeError("fit() before predict().")
        from torch.utils.data import DataLoader

        from baseline.knowddi.utils.graph_utils import collate_dgl, move_batch_to_device_dgl

        p = self._params
        loader = DataLoader(self._test_ds, batch_size=p.batch_size, shuffle=False,
                            num_workers=p.num_workers, collate_fn=collate_dgl)
        self._clf.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                data_pos, _r_labels, _targets = move_batch_to_device_dgl(
                    batch, p.device, multi_type=1)
                logit = self._clf(data_pos).view(-1)
                probs.append(torch.sigmoid(logit).cpu().numpy())
        prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
        return self._align_to_test(prob, test_df)

    def _align_to_test(self, prob: np.ndarray, test_df: "pd.DataFrame") -> np.ndarray:
        """Scatter the [pos, neg]-ordered test-query probs back onto the full test_df row
        order using the build's test_manifest.json (known_mask + is_pos)."""
        data_dir = Path(self._params.file_paths["test"]).parent
        manifest = json.loads((data_dir / "test_manifest.json").read_text())
        known = np.asarray(manifest["known_mask"], dtype=bool)
        is_pos = np.asarray(manifest["is_pos"], dtype=bool)
        n = int(manifest["n_test_rows"])
        # extraction order: kept positives (test_df row order), then kept negatives.
        pos_rows = np.where(known & is_pos)[0]
        neg_rows = np.where(known & (~is_pos))[0]
        expected = len(pos_rows) + len(neg_rows)
        full = np.full(n, 0.5, dtype=np.float32)   # dropped/unknown -> neutral 0.5
        if prob.shape[0] == expected:
            full[pos_rows] = prob[:len(pos_rows)]
            full[neg_rows] = prob[len(pos_rows):]
        else:  # defensive: shape mismatch -> best-effort fill in extraction order
            order = np.concatenate([pos_rows, neg_rows])
            m = min(len(order), prob.shape[0])
            full[order[:m]] = prob[:m]
        return full


__all__ = ["KnowDDIUnifiedBinary"]

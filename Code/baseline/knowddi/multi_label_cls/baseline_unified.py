"""KnowDDI multilabel baseline for the UNIFIED benchmark (decision B, TWOSIDES).

TWOSIDES-only: a FIXED ``n_labels``-side-effect sigmoid head + masked BCE with polarity
(KnowDDI's BioSNAP branch). One leaf = one regime -> a SINGLE KnowDDI model
(GraphSAGE-once-on-global + per-pair enclosing-subgraph directional pruning +
graph-structure-learning refinement + a 3-way ``[mean||head||tail]`` head), BCELoss over
the n_labels label axis. NO cross-regime routing.

This is mostly a WRAPPER: the KnowDDI core ALREADY has the multilabel branch
(``manager/trainer.py`` ``nn.BCELoss(reduce=False)`` :78 + ``multi_type=2`` :111-112 +
polarity masked-BCE :118-123; ``manager/evaluator.py`` ``Evaluator_multilabel`` :76 with
ROC-AUC key ``'auc'`` / AUPRC ``'auc_pr'`` :114-115; ``data_processor`` dispatches to
``process_files_decagon`` for ``dataset=='BioSNAP'``). The wrapper adds only the TWOSIDES
data build (decagon layout) + the fixed 200-label head sizing + sigmoid predict.

Pipeline per leaf (mirrors the multiclass wrapper, wired to the BioSNAP branch):
  1. Build KnowDDI's decagon layout (``train/valid/test.txt`` = ``h\\tt\\tmultihot\\tpolarity``
     + ``BKG_file.txt`` from the LEAF'S OWN int KG, NOT the merged DrugBank KG) via
     ``_data/necessary/build_knowddi_twoside.build_knowddi_twoside``. Cached under
     ``_data/necessary/ml__<hash>/``.
  2. Extract per-pair h-hop enclosing subgraphs (lmdb, ``dataset='BioSNAP'`` -> g_label =
     polarity, r_label = multihot) via ``generate_subgraph_datasets`` -> build the
     train/valid/test ``SubgraphDataset`` (directional ``extract_r_digraph`` pruning at
     load time) -> build ``Classifier_model`` (head width = 200) -> train the landed
     ``Trainer`` BioSNAP branch (masked BCE + polarity, Adam, ExponentialLR, best-ckpt on
     VAL ROC-AUC from ``Evaluator_multilabel``).
  3. ``predict`` scores each test pair's pruned subgraph -> sigmoid over the 200 labels ->
     (n, 200). NO scatter (the head axis IS the label axis, unlike multiclass).

Hyperparams are LOCKED to the BioSNAP official LOG values (verified against
``Paper/Reference/Original-Code/KnowDDI/pytorch/experiments/BioSNAP/log_train.txt:2``):
eval_every_iter=452, threshold=0.1, lamda=0.5, num_infer_layers=1, num_dig_layers=3,
gsl_rel_emb_dim=24, MLP_hidden_dim=24, MLP_num_layers=3, MLP_dropout=0.2, emb_dim=32,
num_gcn_layers=2, hop=2, batch_size=256, num_epochs=50, lr=0.005, weight_decay_rate=1e-5,
lr_decay_rate=0.93, early_stop_epoch=10, num_workers=32, gcn_dropout=0.2,
gcn_aggregator_type='mean', func_num=1, sparsify=1, edge_softmax=1, gsl_has_edge_emb=1,
max_links=250000, max_nodes_per_hop=200, use_pre_embeddings=False.
``num_nodes`` is sized from the ACTUAL built entity count. The classifier's ``score_dim``
uses the hardcoded ``use_pre_embeddings=True`` path (Classifier_model.py:36), so
``score_dim = (1 + num_gcn_layers + num_infer_layers) * emb_dim`` regardless of the
``use_pre_embeddings=False`` node-feature flag (codex 019f2930, decision 1).

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): does NOT import
``baseline.sumgnn``; mirrors the SumGNN multilabel wrapper SHAPE with KnowDDI's own core.
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
from baseline.knowddi._data.necessary.build_knowddi_twoside import (  # noqa: E402
    build_knowddi_twoside)

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "_data" / "necessary"


def _leaf_hash(meta: dict, fold: str) -> str:
    key = f"ml|{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{fold}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@register_unified("knowddi")
class KnowDDIUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(self, *, emb_dim: int = 32, hop: int = 2, num_gcn_layers: int = 2,
                 num_infer_layers: int = 1, num_dig_layers: int = 3, lamda: float = 0.5,
                 threshold: float = 0.1, n_epochs: int = 50, learning_rate: float = 5e-3,
                 lr_decay_rate: float = 0.93, weight_decay_rate: float = 1e-5,
                 batch_size: int = 256, max_links: int = 250000, max_nodes_per_hop: int = 200,
                 early_stop_epoch: int = 10, eval_every_iter: int = 452,
                 gsl_rel_emb_dim: int = 24, mlp_hidden_dim: int = 24, mlp_num_layers: int = 3,
                 mlp_dropout: float = 0.2, num_workers: int = 32, device: str = "auto",
                 log_step_every: int = 50, run_dir: str | Path | None = None,
                 **_ignored) -> None:
        # LOCKED to BioSNAP LOG values (log_train.txt:2). n_epochs/device/run_dir come from
        # the runner; everything else is the paper's best-on-BioSNAP config.
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
        # BioSNAP official value 452 (log_train.txt:2, best-params comment train.py:102).
        self.eval_every_iter = eval_every_iter
        self.gsl_rel_emb_dim = gsl_rel_emb_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.mlp_dropout = mlp_dropout
        self.num_workers = num_workers
        self.device = device
        self.log_step_every = log_step_every
        self.run_dir = run_dir
        self._fold = "fold0"
        self._n_labels: int | None = None
        self._params: SimpleNamespace | None = None
        self._clf = None
        self._test_ds = None

    # ---- KG / data prep -------------------------------------------------
    @staticmethod
    def _twoside_kg_triples(resources: "LeafResources") -> np.ndarray:
        """Load the TWOSIDES leaf's OWN int-indexed KG (``head tail rel``, drug ids
        already in ``[0, n_drugs)`` matching the leaf pairs). We use the S0 (transductive)
        ``train_KG.txt``, which spans all drugs; KnowDDI's enclosing-subgraph BFS then
        anchors per-pair subgraphs on it. This is NOT the merged DrugBank KG -- the
        TWOSIDES drug namespace is integer ids disjoint from ``DBxxxxx``. Mirrors SumGNN's
        multilabel ``_twoside_kg_triples``."""
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("KnowDDI multilabel requires a KG; this leaf declares none.")
        src = Path(kg.source)
        cand = [src / "S0" / "train_KG.txt", src / "train_KG.txt"]
        kg_file = next((c for c in cand if c.is_file()), None)
        if kg_file is None:
            raise FileNotFoundError(
                f"TWOSIDES KG (train_KG.txt) not found under {src}; KnowDDI multilabel "
                f"needs the leaf's own int-indexed KG, not the merged DrugBank KG.")
        return np.loadtxt(kg_file, dtype=np.int64).reshape(-1, 3)

    def _ensure_data(self, train_df, val_df, test_df, resources) -> Path:
        meta = resources.meta
        data_dir = _CACHE_ROOT / f"ml__{_leaf_hash(meta, self._fold)}"
        if (data_dir / "_build_stats.json").is_file():
            logging.info(f"[knowddi-ml] data cache HIT: {data_dir}")
            return data_dir
        logging.info(f"[knowddi-ml] BUILDING TWOSIDES data at {data_dir}")
        kg_triples = self._twoside_kg_triples(resources)
        stats = build_knowddi_twoside(data_dir, resources.drugs, kg_triples, train_df,
                                      val_df, test_df, n_labels=self._n_labels)
        logging.info(f"[knowddi-ml] build stats: {stats}")
        return data_dir

    def fit_leaf(self, leaf: "Leaf") -> None:  # capture fold for the cache key
        self._fold = leaf.fold
        self._n_labels = int(leaf.resources.meta["labels"]["n_labels"])
        self.fit(leaf.train, leaf.val, resources=leaf.resources, _test=leaf.test)

    def fit(self, train_df, val_df=None, *, resources, _test=None) -> None:
        from baseline.knowddi.data_processor.datasets import SubgraphDataset
        from baseline.knowddi.data_processor.subgraph_extraction import generate_subgraph_datasets
        from baseline.knowddi.model.Classifier_model import Classifier_model
        from baseline.knowddi.manager.evaluator import Evaluator_multilabel
        from baseline.knowddi.manager.trainer import Trainer

        meta = resources.meta
        if self._n_labels is None:
            self._n_labels = int(meta["labels"]["n_labels"])
        if val_df is None:
            val_df = train_df
        # test pairs must be in the on-disk test.txt so their subgraphs get extracted
        test_df = _test if _test is not None else val_df
        data_dir = self._ensure_data(train_df, val_df, test_df, resources)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        params = SimpleNamespace(
            dataset="BioSNAP",
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
            # gsl_model (BioSNAP-locked)
            num_infer_layers=self.num_infer_layers, num_dig_layers=self.num_dig_layers,
            MLP_hidden_dim=self.mlp_hidden_dim, MLP_num_layers=self.mlp_num_layers,
            MLP_dropout=self.mlp_dropout, func_num=1, sparsify=1,
            threshold=self.threshold, edge_softmax=1, gsl_rel_emb_dim=self.gsl_rel_emb_dim,
            lamda=self.lamda, gsl_has_edge_emb=1,
            # runtime
            use_pre_embeddings=False, kge_model="TransE",
            device=torch.device(dev),
        )
        params.file_paths = {"train": str(data_dir / "train.txt"),
                             "valid": str(data_dir / "valid.txt"),
                             "test": str(data_dir / "test.txt")}
        params.bkg_file_path = str(data_dir / "BKG_file.txt")
        params.db_path = str(data_dir / f"digraph_hop_{self.hop}_BKG_file")
        params.exp_dir = str(data_dir / "exp")
        os.makedirs(params.exp_dir, exist_ok=True)
        self._params = params

        if not os.path.isdir(params.db_path):
            generate_subgraph_datasets(params)

        train = SubgraphDataset(db_path=params.db_path, db_name="train_subgraph",
                                raw_data_paths=params.file_paths,
                                add_traspose_rels=params.add_traspose_rels,
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset="BioSNAP", kge_model="TransE",
                                dig_layer=self.num_dig_layers,
                                bkg_file_path=params.bkg_file_path)
        valid = SubgraphDataset(db_path=params.db_path, db_name="valid_subgraph",
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset="BioSNAP", kge_model="TransE",
                                ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                                id2relation=train.id2relation, rel=train.num_rels,
                                global_graph=train.global_graph,
                                dig_layer=self.num_dig_layers,
                                bkg_file_path=params.bkg_file_path)
        test = SubgraphDataset(db_path=params.db_path, db_name="test_subgraph",
                               use_pre_embeddings=params.use_pre_embeddings,
                               dataset="BioSNAP", kge_model="TransE",
                               ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                               id2relation=train.id2relation, rel=train.num_rels,
                               global_graph=train.global_graph,
                               dig_layer=self.num_dig_layers,
                               bkg_file_path=params.bkg_file_path)

        # FIXED 200-label head (codex 019f2930): process_files_decagon derives num_rels from
        # the labels appearing across splits, which need NOT be all 200. The BioSNAP head is
        # a FIXED n_labels-wide sigmoid, so we force ``params.num_rels = n_labels`` (NOT
        # train.num_rels). aug_num_rels includes BKG relations + self loop (from the loader).
        params.num_rels = self._n_labels
        params.aug_num_rels = train.aug_num_rels
        # num_nodes sizes GraphSAGE.pre_embed; from the ACTUAL entity count of the built graph.
        params.num_nodes = train.num_entity
        params.global_graph = train.global_graph.to(params.device)

        clf = Classifier_model(params).to(params.device)
        logging.info(f"[knowddi-ml] built Classifier_model: num_rels={params.num_rels} "
                     f"aug_num_rels={params.aug_num_rels} num_nodes={params.num_nodes}")

        # Model selection on VALIDATION, NOT test. Evaluator_multilabel best-ckpt key 'auc'
        # = mean ROC-AUC over labels (evaluator.py:114-117); the trainer BioSNAP branch keys
        # best-ckpt on result['auc'] (trainer.py:142-147). test evaluator is logging-only.
        valid_ev = Evaluator_multilabel(params, clf, valid)   # best-ckpt selection (VAL ROC-AUC)
        test_ev = Evaluator_multilabel(params, clf, test)     # logging only
        trainer = Trainer(params, clf, train, valid_ev, test_ev)
        logging.info(f"[knowddi-ml] training {self.n_epochs} epochs on {len(train)} subgraphs "
                     f"(n_labels={self._n_labels})")
        trainer.train()

        # load_best_model_at_end (valid ROC-AUC-selected); fall back to final model.
        best = Path(params.exp_dir) / "best_graph_classifier.pth"
        if best.is_file():
            try:
                clf = torch.load(best, map_location=params.device, weights_only=False)
                logging.info(f"[knowddi-ml] loaded best valid-selected checkpoint: {best}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[knowddi-ml] best-ckpt reload failed ({e}); using final model")
        self._clf = clf
        self._test_ds = test

    # ---- prediction -----------------------------------------------------
    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        """Score test pairs -> (n, n_labels) sigmoid probabilities, aligned to test_df row
        order. NO scatter: the head axis IS the label axis (codex 019f2930)."""
        if self._clf is None:
            raise RuntimeError("fit() before predict().")
        from torch.utils.data import DataLoader

        from baseline.knowddi.utils.graph_utils import collate_dgl, move_batch_to_device_dgl

        p = self._params
        loader = DataLoader(self._test_ds, batch_size=p.batch_size, shuffle=False,
                            num_workers=p.num_workers, collate_fn=collate_dgl)
        self._clf.eval()
        outs = []
        with torch.no_grad():
            for batch in loader:
                data, _r_labels, _targets = move_batch_to_device_dgl(batch, p.device, multi_type=2)
                logits = self._clf(data)
                outs.append(torch.sigmoid(logits).cpu().numpy())
        scores = (np.concatenate(outs, axis=0) if outs
                  else np.zeros((0, self._n_labels), dtype=np.float32))
        return self._align_to_test(scores, test_df)

    def _align_to_test(self, out: np.ndarray, test_df: "pd.DataFrame") -> np.ndarray:
        """The extracted test subgraphs cover only pairs whose BOTH drugs were known
        (build dropped the rest, in the SAME row order). Map predictions back onto the full
        test_df row order; dropped rows get uniform ~0 (counted by the runner's metrics)."""
        data_dir = Path(self._params.file_paths["test"]).parent
        with open(data_dir / "entity2id.pkl", "rb") as f:
            entity2id = pickle.load(f)
        a = test_df["drug_a_id"].astype(str).map(entity2id)
        b = test_df["drug_b_id"].astype(str).map(entity2id)
        keep = (a.notna() & b.notna()).to_numpy()
        full = np.zeros((len(test_df), self._n_labels), dtype=np.float32)
        if keep.sum() == out.shape[0]:
            full[keep] = out
        else:  # defensive: shape mismatch -> best-effort truncate/pad
            m = min(int(keep.sum()), out.shape[0])
            idx = np.where(keep)[0][:m]
            full[idx] = out[:m]
        return full


__all__ = ["KnowDDIUnifiedMultilabel"]

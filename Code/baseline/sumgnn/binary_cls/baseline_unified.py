"""SumGNN binary baseline for the UNIFIED benchmark (decision B, Case-B GraIL-style).

SumGNN is KG-subgraph based -> applies to KG datasets (drugbank_latest_* via the merged
DrugBank+Hetionet+PrimeKG KG). One leaf = one regime -> a SINGLE SumGNN model whose
enclosing-subgraph extraction + R-GCN + KG-summarization self-attention + Morgan-feature
fusion are REUSED UNCHANGED from ``_core`` (imported ``GraphClassifier`` / subgraph utils);
ONLY the task head + loss + metric are binary (single logit, BCEWithLogitsLoss on the
per-subgraph binary ``g_label``, AUPRC best-checkpoint selection). NO cross-regime routing.

This mirrors ``multi_cls/baseline_unified.py`` (KG / data-cache / materialize / full-KG-
scope conventions) with the binary data + training path from ``_binary_core`` /
``build_sumgnn_binary_data`` (codex-approved design):

Pipeline per leaf:
  1. Build SumGNN's on-disk int layout from the leaf's string-id drugs + pairs + merged KG
     via ``build_sumgnn_binary_data`` (drugs indexed first; kg_scope="full" default). The
     DDI relation adjacency is built from TRAIN POSITIVES ONLY; negatives are extraction
     queries only. Cached under ``_data/necessary/bin__<hash>__<kg_scope>/``.
  2. Extract per-pair enclosing subgraphs (lmdb) for the EXPLICIT pos + neg query sets via
     ``generate_binary_subgraph_datasets`` (g_label 1 pos / 0 neg, r_label 0), then train
     with ``BinaryTrainer`` (BCEWithLogitsLoss) + ``BinaryEvaluator`` (AUPRC on validation).
  3. ``predict`` scores each test subgraph -> sigmoid prob (n,), aligned back to the full
     test_df row order via the build's ``test_manifest.json``.

Paper-faithful hyperparams are the SAME as the mc wrapper (emb_dim=32, hop=2, num_bases=4,
lr=5e-3, batch=128); only task-native pieces differ (Case-B, spec §4/§case-B).
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

# baseline-local ported SumGNN core (independent copy; NOT the reproduction).
_CORE = Path(__file__).resolve().parents[1] / "_core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.sumgnn._data.necessary.build_sumgnn_binary_data import (  # noqa: E402
    build_sumgnn_binary_data)

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "_data" / "necessary"


def _leaf_hash(meta: dict, fold: str) -> str:
    key = f"bin|{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{fold}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@register_unified("sumgnn")
class SumGNNUnifiedBinary(UnifiedBaseline):
    task = "binary"

    def __init__(self, *, emb_dim: int = 32, hop: int = 2, num_bases: int = 4,
                 n_epochs: int = 50, learning_rate: float = 5e-3, batch_size: int = 128,
                 max_links: int = 250000, num_workers: int = 8, eval_every_iter: int = 526,
                 device: str = "auto", log_step_every: int = 50, materialize: bool = False,
                 kg_scope: str = "full",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        self.emb_dim = emb_dim
        self.hop = hop
        self.num_bases = num_bases
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_links = max_links
        self.num_workers = num_workers
        self.eval_every_iter = eval_every_iter
        # standing decision 2026-07-01: default to the FULL merged KG (not drug-incident).
        self.kg_scope = kg_scope
        # perf: materialize train/valid/test subgraphs into RAM once (faithful; forces
        # num_workers=0). Mirrors the mc wrapper.
        self.materialize = materialize
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
            raise ValueError("SumGNN requires a KG; this leaf declares none.")
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
            logging.info(f"[sumgnn-bin] data cache HIT: {data_dir}")
            return data_dir
        logging.info(f"[sumgnn-bin] BUILDING data at {data_dir} (kg_scope={self.kg_scope})")
        edges = pd.read_parquet(self._merged_edges_path(resources))
        stats = build_sumgnn_binary_data(
            data_dir, resources.drugs, edges, train_df, val_df, test_df,
            morgan_from="smiles", kg_scope=self.kg_scope,
        )
        logging.info(f"[sumgnn-bin] build stats: {stats}")
        return data_dir

    def fit_leaf(self, leaf: "Leaf") -> None:  # capture fold for the cache key
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources, _test=leaf.test)

    def fit(self, train_df, val_df=None, *, resources, _test=None) -> None:
        from baseline.sumgnn.binary_cls._binary_core import (
            BinaryEvaluator, BinarySubgraphDataset, BinaryTrainer,
            generate_binary_subgraph_datasets)
        from utils.graph_utils import collate_dgl, move_batch_to_device_dgl
        from model.dgl.graph_classifier import GraphClassifier as dgl_model

        meta = resources.meta
        if val_df is None:
            val_df = train_df
        # test pairs must be in the on-disk test_*.txt so their subgraphs get extracted
        test_df = _test if _test is not None else val_df
        data_dir = self._ensure_data(train_df, val_df, test_df, resources)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        params = SimpleNamespace(
            experiment_name=f"sumgnn_bin_{_leaf_hash(meta, self._fold)}",
            dataset="drugbank", main_dir=".",  # resolved against the chdir sandbox
            train_file="train", valid_file="dev", test_file="test",
            num_epochs=self.n_epochs, eval_every=3, eval_every_iter=self.eval_every_iter,
            save_every=10_000, early_stop=100, optimizer="Adam", lr=self.learning_rate,
            clip=1000, l2=1e-5, max_links=self.max_links, hop=self.hop, max_nodes_per_hop=200,
            use_kge_embeddings=False, kge_model="TransE", model_type="dgl",
            constrained_neg_prob=0.0, batch_size=self.batch_size, num_neg_samples_per_link=0,
            num_workers=self.num_workers, add_traspose_rels=False, enclosing_sub_graph=True,
            rel_emb_dim=32, attn_rel_emb_dim=32, emb_dim=self.emb_dim, num_gcn_layers=2,
            num_bases=self.num_bases, dropout=0.3, edge_dropout=0.4, gnn_agg_type="sum",
            add_ht_emb=True, add_sb_emb=True, has_attn=True, has_kg=True, feat="morgan",
            feat_dim=1024, add_feat_emb=True, add_transe_emb=True, gamma=0.2,
            log_step_every=self.log_step_every,
            device=torch.device(dev), collate_fn=collate_dgl,
            move_batch_to_device=move_batch_to_device_dgl,
        )
        # pos-only .txt paths for process_files_ddi (adjacency built from train POSITIVES).
        params.file_paths = {"train": str(data_dir / "train.txt"),
                             "valid": str(data_dir / "dev.txt"),
                             "test": str(data_dir / "test.txt")}
        params.db_path = str(data_dir / f"subgraphs_bin_en_True_neg_0_hop_{self.hop}_max{self.max_links}")
        params.exp_dir = str(data_dir / "exp")
        os.makedirs(params.exp_dir, exist_ok=True)
        self._params = params

        # ported datasets read 'data/{dataset}/...' relative to CWD -> chdir sandbox.
        self._prepare_cwd(data_dir)

        if not os.path.isdir(params.db_path):
            generate_binary_subgraph_datasets(params)

        train = BinarySubgraphDataset(params.db_path, "train_pos", "train_neg", params.file_paths,
                                      add_traspose_rels=False, num_neg_samples_per_link=0,
                                      use_kge_embeddings=False, dataset="drugbank",
                                      kge_model="TransE", file_name="train")
        test = BinarySubgraphDataset(params.db_path, "test_pos", "test_neg", params.file_paths,
                                     add_traspose_rels=False, num_neg_samples_per_link=0,
                                     use_kge_embeddings=False, dataset="drugbank",
                                     kge_model="TransE", file_name="test",
                                     ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                                     id2relation=train.id2relation, rel=train.num_rels,
                                     graph=train.graph)
        params.num_rels = train.num_rels
        params.aug_num_rels = train.aug_num_rels
        params.inp_dim = train.n_feat_dim
        # Case-B binary head: a SINGLE logit (GraphClassifier fc_layer=Linear(..., 1)).
        params.train_rels = 1
        params.num_nodes = 200000
        params.max_label_value = train.max_n_label

        clf = self._build_model(params, dgl_model, data_dir)
        # Model selection on VALIDATION (dev), NOT test — primary metric = AUPRC.
        valid = BinarySubgraphDataset(params.db_path, "valid_pos", "valid_neg", params.file_paths,
                                      add_traspose_rels=False, num_neg_samples_per_link=0,
                                      use_kge_embeddings=False, dataset="drugbank",
                                      kge_model="TransE", file_name="valid",
                                      ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                                      id2relation=train.id2relation, rel=train.num_rels,
                                      graph=train.graph)
        if self.materialize:
            from subgraph_extraction.datasets_materialized import MaterializedSubgraphDataset
            train = MaterializedSubgraphDataset(train)
            valid = MaterializedSubgraphDataset(valid)
            test = MaterializedSubgraphDataset(test)
            params.num_workers = 0   # cache is in-process; forked workers would duplicate it
            logging.info("[sumgnn-bin] materialized train/valid/test subgraphs -> num_workers=0")
        valid_ev = BinaryEvaluator(params, clf, valid)   # best-ckpt selection (validation)
        test_ev = BinaryEvaluator(params, clf, test)     # logging only
        trainer = BinaryTrainer(params, clf, train, valid_ev, test_ev)
        logging.info(f"[sumgnn-bin] training {self.n_epochs} epochs on {len(train)} subgraphs "
                     f"(num_rels={params.num_rels}, inp_dim={params.inp_dim})")
        trainer.train()
        # load_best_model_at_end: reload the best (valid-AUPRC-selected) checkpoint if the
        # trainer saved one; else keep the final in-memory model.
        best = Path(params.exp_dir) / "best_graph_classifier.pth"
        if best.is_file():
            try:
                clf = torch.load(best, map_location=params.device, weights_only=False)
                logging.info(f"[sumgnn-bin] loaded best valid-selected checkpoint: {best}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[sumgnn-bin] best-ckpt reload failed ({e}); using final model")
        self._clf = clf
        self._test_ds = test

    def _prepare_cwd(self, data_dir: Path) -> None:
        """The ported datasets.py loads entity.txt/relations_2hop.txt via
        'data/drugbank/...' relative to CWD. Create a CWD sandbox with a
        data/drugbank -> data_dir symlink and chdir into it."""
        sandbox = data_dir / "_cwd"
        (sandbox / "data").mkdir(parents=True, exist_ok=True)
        link = sandbox / "data" / "drugbank"
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink():
                    link.unlink()
            if not link.exists():
                os.symlink(data_dir, link, target_is_directory=True)
        except OSError:
            import shutil
            link.mkdir(parents=True, exist_ok=True)
            for f in ("entity.txt", "relations_2hop.txt"):
                shutil.copy(data_dir / f, link / f)
        os.chdir(sandbox)

    def _build_model(self, params, dgl_model, data_dir):
        # build the classifier directly (avoids initialize_model's cwd-relative paths).
        r2i_path = data_dir / "relation2id.json"
        relation2id = json.loads(r2i_path.read_text()) if r2i_path.is_file() else {}
        clf = dgl_model(params, relation2id).to(params.device)
        mfeat_path = data_dir / "DB_molecular_feats.pkl"
        with open(mfeat_path, "rb") as f:
            x = pickle.load(f, encoding="utf-8")
        mfeat = np.array([y for y in x["Morgan_Features"]])
        clf.drug_feat(torch.FloatTensor(mfeat).to(params.device))
        return clf

    # ---- prediction -----------------------------------------------------
    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        """Score test pairs -> (n,) P(positive), aligned to test_df row order.

        The test dataset yields extracted subgraphs ordered [all test_pos, then all
        test_neg]; the build's ``test_manifest.json`` records the original test_df row
        order + known/pos masks so we scatter each prob back to its row. Rows whose pair
        was dropped (unknown drug) get 0.5 (neutral; counted by the runner's metrics)."""
        if self._clf is None:
            raise RuntimeError("fit() before predict().")
        from torch.utils.data import DataLoader

        p = self._params
        loader = DataLoader(self._test_ds, batch_size=p.batch_size, shuffle=False,
                            num_workers=p.num_workers, collate_fn=p.collate_fn)
        self._clf.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                data_pos, r_labels_pos, targets_pos = p.move_batch_to_device(batch, p.device)
                logit = self._clf(data_pos).view(-1)
                probs.append(torch.sigmoid(logit).cpu().numpy())
        prob = np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)
        return self._align_to_test(prob, test_df)

    def _align_to_test(self, prob: np.ndarray, test_df: "pd.DataFrame") -> np.ndarray:
        """Scatter the [pos, neg]-ordered test-query probs back onto the full test_df
        row order using the build's test_manifest.json (known_mask + is_pos)."""
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


__all__ = ["SumGNNUnifiedBinary"]

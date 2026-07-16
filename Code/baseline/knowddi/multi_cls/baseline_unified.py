"""KnowDDI multiclass baseline for the UNIFIED benchmark (decision B).

KnowDDI is a KG-subgraph GSL method -> applies to KG datasets (drugbank_latest_* via
the merged DrugBank+Hetionet+PrimeKG KG). One leaf = one regime -> a SINGLE KnowDDI
multiclass model (GraphSAGE-once-on-global + per-pair enclosing-subgraph directional
pruning + graph-structure-learning refinement + a 3-way ``[mean||head||tail]`` head),
CrossEntropy over the train-observed DDI-event vocab. NO cross-regime routing.

Pipeline per leaf (mirrors the official ``train.py`` ``process_dataset`` flow, wired to
the landed KnowDDI core):
  1. Build KnowDDI's on-disk int-indexed layout (train/valid/test.txt, BKG_file.txt,
     entity2id.pkl, train_class_to_global.json) from the leaf's string-id drugs + pairs
     + the merged KG, via ``_data/necessary/build_knowddi_data.build_knowddi_data``
     (drugs indexed first; kg_scope="full" by default = the WHOLE merged KG, standing
     decision 2026-07-01). Cached under ``_data/necessary/mc__<hash>__<kg_scope>/``.
  2. Extract per-pair h-hop enclosing subgraphs (lmdb) via
     ``generate_subgraph_datasets`` -> build the train/valid/test ``SubgraphDataset``
     (which applies the directional ``extract_r_digraph`` pruning at load time) ->
     build ``Classifier_model`` -> train the landed ``Trainer`` (CE, Adam, ExponentialLR,
     best-ckpt on VAL macro-F1).
  3. ``predict`` scores each test pair's pruned subgraph -> softmax over the TRAIN
     DDI-event vocab (fc head width = K_train), scattered to the leaf's GLOBAL class axis
     -> (n, n_labels_global).

``predict`` returns (n, n_labels_GLOBAL): KnowDDI's ``W_final`` head outputs ``K_train``
logits (one per dense train-observed DDI-event id). We map each dense id back to its
global class via ``train_class_to_global.json`` and scatter. Test rows whose gold class
was unseen-in-train get ~0 mass -> counted wrong (same convention as SumGNN / EmerGNN
multiclass; surfaced via the runner's oov_target_rate). Test pairs with an unknown drug
are dropped on-disk by the builder and get uniform ~0 (also counted wrong), realigned to
the full test_df row order via entity2id.

Hyperparams are LOCKED to the DrugBank official LOG values (verified against
``Paper/Reference/Original-Code/KnowDDI/pytorch/experiments/Drugbank/log_train.txt:2``):
num_dig_layers=4, num_infer_layers=3, lamda=0.5, threshold=0.05, hop=2, emb_dim=32,
num_gcn_layers=2, lr=0.005, weight_decay_rate=1e-5, lr_decay_rate=0.93, batch_size=256,
num_epochs=50, early_stop_epoch=10, eval_every_iter=526, num_workers=32,
MLP_hidden_dim=32, MLP_num_layers=2, MLP_dropout=0.2,
gsl_rel_emb_dim=32, gcn_dropout=0.2, gcn_aggregator_type='mean', func_num=1, sparsify=1,
edge_softmax=1, gsl_has_edge_emb=1, max_links=250000, max_nodes_per_hop=200.
``num_nodes`` is sized from the ACTUAL built entity count (not the original's 35000
hardcode; GraphSAGE.py docstring locked decision).

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): does NOT
import ``baseline.sumgnn``; mirrors the SumGNN wrapper SHAPE with KnowDDI's own core.
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
from baseline.knowddi._data.necessary.build_knowddi_data import build_knowddi_data  # noqa: E402

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "_data" / "necessary"


def _leaf_hash(meta: dict, fold: str) -> str:
    key = f"{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{fold}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@register_unified("knowddi")
class KnowDDIUnifiedMulticlass(UnifiedBaseline):
    task = "multiclass"

    def __init__(self, *, emb_dim: int = 32, hop: int = 2, num_gcn_layers: int = 2,
                 num_infer_layers: int = 3, num_dig_layers: int = 4, lamda: float = 0.5,
                 threshold: float = 0.05, n_epochs: int = 50, learning_rate: float = 5e-3,
                 lr_decay_rate: float = 0.93, weight_decay_rate: float = 1e-5,
                 batch_size: int = 256, max_links: int = 250000, max_nodes_per_hop: int = 200,
                 early_stop_epoch: int = 10, eval_every_iter: int = 526,
                 num_workers: int = 32, device: str = "auto",
                 log_step_every: int = 50, kg_scope: str = "full",
                 run_dir: str | Path | None = None, **_ignored) -> None:
        # LOCKED to DrugBank LOG values (log_train.txt:2). n_epochs/device/run_dir come
        # from the runner; everything else is the paper's best-on-DrugBank config.
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
        self._n_global: int | None = None
        self._params: SimpleNamespace | None = None
        self._clf = None
        self._train = None
        self._test_ds = None
        self._idx_to_global: list[int] | None = None

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
        data_dir = _CACHE_ROOT / f"mc__{_leaf_hash(meta, self._fold)}__{self.kg_scope}"
        stamp = data_dir / "_build_stats.json"
        if stamp.is_file():
            logging.info(f"[knowddi] data cache HIT: {data_dir}")
            return data_dir
        logging.info(f"[knowddi] BUILDING KnowDDI data at {data_dir} (kg_scope={self.kg_scope})")
        edges = pd.read_parquet(self._merged_edges_path(resources))
        stats = build_knowddi_data(
            data_dir, resources.drugs, edges, train_df, val_df, test_df,
            label_col="y_cls", kg_scope=self.kg_scope,
        )
        logging.info(f"[knowddi] build stats: {stats}")
        return data_dir

    def fit_leaf(self, leaf) -> None:  # capture fold for the cache key
        self._fold = leaf.fold
        self.fit(leaf.train, leaf.val, resources=leaf.resources, _test=leaf.test)

    def fit(self, train_df, val_df=None, *, resources, _test=None) -> None:
        from baseline.knowddi.data_processor.datasets import SubgraphDataset
        from baseline.knowddi.data_processor.subgraph_extraction import generate_subgraph_datasets
        from baseline.knowddi.model.Classifier_model import Classifier_model
        from baseline.knowddi.manager.evaluator import Evaluator_multiclass
        from baseline.knowddi.manager.trainer import Trainer

        meta = resources.meta
        self._n_global = int(meta["labels"]["n_labels"])
        if val_df is None:
            val_df = train_df
        # test pairs must be in the on-disk test.txt so their subgraphs get extracted
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
        # absolute on-disk paths for the ported pipeline (which otherwise reads
        # data/{dataset}/... relative to cwd). We point every reader at data_dir.
        params.file_paths = {"train": str(data_dir / "train.txt"),
                             "valid": str(data_dir / "valid.txt"),
                             "test": str(data_dir / "test.txt")}
        params.bkg_file_path = str(data_dir / "BKG_file.txt")
        params.db_path = str(data_dir / f"digraph_hop_{self.hop}_BKG_file")
        params.exp_dir = str(data_dir / "exp")
        os.makedirs(params.exp_dir, exist_ok=True)
        self._params = params

        # dense train class -> global class map (builder remapped y_cls to a
        # contiguous [0, K_train) block; fc head index == dense id).
        self._idx_to_global = json.loads((data_dir / "train_class_to_global.json").read_text())

        if not os.path.isdir(params.db_path):
            generate_subgraph_datasets(params)

        train = SubgraphDataset(db_path=params.db_path, db_name="train_subgraph",
                                raw_data_paths=params.file_paths,
                                add_traspose_rels=params.add_traspose_rels,
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset="drugbank", kge_model="TransE",
                                dig_layer=self.num_dig_layers,
                                bkg_file_path=params.bkg_file_path)
        valid = SubgraphDataset(db_path=params.db_path, db_name="valid_subgraph",
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset="drugbank", kge_model="TransE",
                                ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                                id2relation=train.id2relation, rel=train.num_rels,
                                global_graph=train.global_graph,
                                dig_layer=self.num_dig_layers,
                                bkg_file_path=params.bkg_file_path)
        test = SubgraphDataset(db_path=params.db_path, db_name="test_subgraph",
                               use_pre_embeddings=params.use_pre_embeddings,
                               dataset="drugbank", kge_model="TransE",
                               ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                               id2relation=train.id2relation, rel=train.num_rels,
                               global_graph=train.global_graph,
                               dig_layer=self.num_dig_layers,
                               bkg_file_path=params.bkg_file_path)

        # num_rels = # DDI-event relations (== K_train dense head width); aug_num_rels
        # includes BKG relations + self loop. global_graph on device (train.py:66).
        params.num_rels = train.num_rels
        params.aug_num_rels = train.aug_num_rels
        # num_nodes sizes GraphSAGE.pre_embed; from the ACTUAL entity count of the built
        # graph (locked decision: NOT the original's 35000 hardcode).
        params.num_nodes = train.num_entity
        params.global_graph = train.global_graph.to(params.device)

        clf = Classifier_model(params).to(params.device)
        logging.info(f"[knowddi] built Classifier_model: num_rels={params.num_rels} "
                     f"aug_num_rels={params.aug_num_rels} num_nodes={params.num_nodes}")

        # Model selection on VALIDATION, NOT test (train.py builds a separate valid +
        # test evaluator; best-ckpt on valid macro-F1). test evaluator is logging-only.
        valid_ev = Evaluator_multiclass(params, clf, valid)          # best-ckpt selection
        test_ev = Evaluator_multiclass(params, clf, test, is_test=True)  # logging only
        trainer = Trainer(params, clf, train, valid_ev, test_ev)
        logging.info(f"[knowddi] training {self.n_epochs} epochs on {len(train)} subgraphs")
        trainer.train()

        # load_best_model_at_end: the trainer saves best_graph_classifier.pth on VAL
        # macro-F1 improvement (fires each epoch), so reload the best checkpoint.
        best = Path(params.exp_dir) / "best_graph_classifier.pth"
        if best.is_file():
            try:
                clf = torch.load(best, map_location=params.device, weights_only=False)
                logging.info(f"[knowddi] loaded best valid-selected checkpoint: {best}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[knowddi] best-ckpt reload failed ({e}); using final model")
        self._clf = clf
        self._train = train
        self._test_ds = test

    # ---- prediction -----------------------------------------------------
    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._clf is None:
            raise RuntimeError("fit() before predict().")
        from torch.utils.data import DataLoader
        import torch.nn.functional as F

        from baseline.knowddi.utils.graph_utils import collate_dgl, move_batch_to_device_dgl

        p = self._params
        loader = DataLoader(self._test_ds, batch_size=p.batch_size, shuffle=False,
                            num_workers=p.num_workers, collate_fn=collate_dgl)
        self._clf.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                data, _r_labels, _targets = move_batch_to_device_dgl(batch, p.device, multi_type=1)
                logits = self._clf(data)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        prob = (np.concatenate(probs, axis=0) if probs
                else np.zeros((0, len(self._idx_to_global))))

        out = np.zeros((prob.shape[0], self._n_global), dtype=np.float32)
        for i, g in enumerate(self._idx_to_global):     # dense train idx -> global id
            if 0 <= g < self._n_global:
                out[:, g] = prob[:, i]
        # align to full test_df row order (dropped-drug pairs get uniform ~0).
        return self._align_to_test(out, test_df)

    def _align_to_test(self, out: np.ndarray, test_df: "pd.DataFrame") -> np.ndarray:
        """The extracted test subgraphs cover only pairs whose BOTH drugs were known
        (build dropped the rest, in the SAME row order). Map predictions back onto the
        full test_df row order; dropped rows get uniform ~0 (counted wrong)."""
        data_dir = Path(self._params.file_paths["test"]).parent
        with open(data_dir / "entity2id.pkl", "rb") as f:
            entity2id = pickle.load(f)
        a = test_df["drug_a_id"].astype(str).map(entity2id)
        b = test_df["drug_b_id"].astype(str).map(entity2id)
        keep = (a.notna() & b.notna()).to_numpy()
        full = np.zeros((len(test_df), self._n_global), dtype=np.float32)
        if keep.sum() == out.shape[0]:
            full[keep] = out
        else:  # defensive: shape mismatch -> best-effort truncate/pad
            m = min(int(keep.sum()), out.shape[0])
            idx = np.where(keep)[0][:m]
            full[idx] = out[:m]
        return full


__all__ = ["KnowDDIUnifiedMulticlass"]

"""SumGNN multilabel baseline for the UNIFIED benchmark (decision B).

TWOSIDES-only: a FIXED n_labels-side-effect sigmoid head + BCE (SumGNN's BioSNAP branch).
One leaf = one regime -> a SINGLE SumGNN model. SumGNN is KG-subgraph based; the TWOSIDES
leaf's KG is `resources.kg.source` (merged KG, restricted to drug-incident edges by the
builder). NO cross-regime routing.

Pipeline mirrors the multiclass wrapper but uses SumGNN's decagon/BioSNAP data layout
(`write_twoside_split`): each split line is `a\\tb\\tmultihot\\tpolarity`, features live in
`id2drug_feat.pkl` ({'Morgan','rdkit2d'}), and the trainer/evaluator take the BioSNAP
branch (Sigmoid + BCE, `Evaluator_ddi2`). `predict` returns (n, n_labels) sigmoid scores.
"""
from __future__ import annotations

import hashlib
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

_CORE = Path(__file__).resolve().parents[1] / "_core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

from baseline.unified_base import UnifiedBaseline, register_unified  # noqa: E402
from baseline.sumgnn._data.necessary.build_sumgnn_data import write_twoside_split  # noqa: E402
from data_utils import unified  # noqa: E402

if TYPE_CHECKING:
    from data_utils.unified_loader import LeafResources, Leaf

_PAIR = ["drug_a_id", "drug_b_id"]
MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
_CACHE_ROOT = Path(__file__).resolve().parents[1] / "_data" / "necessary"


def _leaf_hash(meta: dict, fold: str) -> str:
    key = f"ml|{meta.get('dataset_id')}|{meta.get('split_type')}|{meta.get('split_code')}|{fold}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@register_unified("sumgnn")
class SumGNNUnifiedMultilabel(UnifiedBaseline):
    task = "multilabel"

    def __init__(self, *, emb_dim: int = 32, hop: int = 2, num_bases: int = 4,
                 n_epochs: int = 50, learning_rate: float = 5e-3, batch_size: int = 128,
                 max_links: int = 250000, num_workers: int = 8, eval_every_iter: int = 526,
                 device: str = "auto", log_step_every: int = 50, materialize: bool = False,
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
        self.device = device
        self.log_step_every = log_step_every
        self.materialize = materialize   # perf (codex 019f1fdc); forces num_workers=0
        self.run_dir = run_dir
        self._fold = "fold0"
        self._n_labels: int | None = None
        self._params: SimpleNamespace | None = None
        self._clf = None
        self._test_ds = None

    @staticmethod
    def _twoside_kg_triples(resources: "LeafResources") -> np.ndarray:
        """Load the TWOSIDES leaf's OWN SumGNN-format KG (int-indexed `head tail rel`,
        drug ids already in [0, n_drugs) matching the leaf pairs). We use the S0
        (transductive) train_KG.txt, which spans all drugs; SumGNN's subgraph BFS then
        anchors per-pair subgraphs on it. This is NOT the merged DrugBank KG — the
        TWOSIDES drug namespace is integer ids disjoint from `DBxxxxx`."""
        kg = resources.kg
        if not kg.present or not kg.source:
            raise ValueError("SumGNN multilabel requires a KG; this leaf declares none.")
        src = Path(kg.source)
        cand = [src / "S0" / "train_KG.txt", src / "train_KG.txt"]
        kg_file = next((c for c in cand if c.is_file()), None)
        if kg_file is None:
            raise FileNotFoundError(
                f"TWOSIDES KG (train_KG.txt) not found under {src}; SumGNN multilabel "
                f"needs the leaf's own int-indexed KG, not the merged DrugBank KG.")
        return np.loadtxt(kg_file, dtype=np.int64).reshape(-1, 3)

    def _locate_test(self, resources, fold: str):
        meta = resources.meta
        root = _ROOT / "Code" / "data" / "ddi_unified"
        parent = unified.layout_dir(root, meta["dataset_group"], meta["task"], meta["split_type"])
        return pd.read_parquet(parent / fold / "test.parquet")

    def _ensure_data(self, train_df, val_df, test_df, resources) -> Path:
        meta = resources.meta
        data_dir = _CACHE_ROOT / f"ml__{_leaf_hash(meta, self._fold)}"
        if (data_dir / "_build_stats.json").is_file():
            logging.info(f"[sumgnn-ml] data cache HIT: {data_dir}")
            return data_dir
        logging.info(f"[sumgnn-ml] BUILDING TWOSIDES data at {data_dir}")
        kg_triples = self._twoside_kg_triples(resources)
        stats = write_twoside_split(data_dir, resources.drugs, kg_triples, train_df, val_df,
                                    test_df, n_labels=self._n_labels)
        logging.info(f"[sumgnn-ml] build stats: {stats}")
        return data_dir

    def fit_leaf(self, leaf: "Leaf") -> None:
        self._fold = leaf.fold
        self._n_labels = int(leaf.resources.meta["labels"]["n_labels"])
        self.fit(leaf.train, leaf.val, resources=leaf.resources, _test=leaf.test)

    def fit(self, train_df, val_df=None, *, resources, _test=None) -> None:
        from subgraph_extraction.datasets import (SubgraphDataset,
                                                  generate_subgraph_datasets)
        from utils.graph_utils import collate_dgl, move_batch_to_device_dgl_ddi2
        from model.dgl.graph_classifier import GraphClassifier as dgl_model
        from managers.evaluator import Evaluator_ddi2
        from managers.trainer import Trainer

        meta = resources.meta
        if self._n_labels is None:
            self._n_labels = int(meta["labels"]["n_labels"])
        if val_df is None:
            val_df = train_df
        test_df = _test if _test is not None else val_df
        data_dir = self._ensure_data(train_df, val_df, test_df, resources)

        dev = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        params = SimpleNamespace(
            experiment_name=f"sumgnn_ml_{_leaf_hash(meta, self._fold)}", dataset="BioSNAP",
            main_dir=".",  # resolved against the chdir sandbox (data/BioSNAP symlink)
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
            device=torch.device(dev), collate_fn=collate_dgl,
            move_batch_to_device=move_batch_to_device_dgl_ddi2,
        )
        params.file_paths = {"train": str(data_dir / "train.txt"),
                             "valid": str(data_dir / "dev.txt"),
                             "test": str(data_dir / "test.txt")}
        params.db_path = str(data_dir / f"subgraphs_en_True_neg_0_hop_{self.hop}_max{self.max_links}")
        params.exp_dir = str(data_dir / "exp")
        os.makedirs(params.exp_dir, exist_ok=True)
        self._params = params
        self._prepare_cwd(data_dir)

        if not os.path.isdir(params.db_path):
            generate_subgraph_datasets(params)

        train = SubgraphDataset(params.db_path, "train_pos", "train_neg", params.file_paths,
                                add_traspose_rels=False, num_neg_samples_per_link=0,
                                use_kge_embeddings=False, dataset="BioSNAP",
                                kge_model="TransE", file_name="train")
        test = SubgraphDataset(params.db_path, "test_pos", "test_neg", params.file_paths,
                               add_traspose_rels=False, num_neg_samples_per_link=0,
                               use_kge_embeddings=False, dataset="BioSNAP",
                               kge_model="TransE", file_name="test",
                               ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                               id2relation=train.id2relation, rel=train.num_rels, graph=train.graph)
        params.num_rels = train.num_rels
        params.aug_num_rels = train.aug_num_rels
        params.inp_dim = train.n_feat_dim
        params.train_rels = self._n_labels          # BioSNAP branch: fixed n_labels head
        params.num_nodes = 200000
        params.max_label_value = train.max_n_label

        clf = self._build_model(params, dgl_model, data_dir)
        # Model selection on VALIDATION (dev), NOT test (was eval leakage). The BioSNAP
        # trainer branch also appends per-improvement results to experiments/<exp>/result.json
        # (cwd-relative) — ensure that dir exists so a best-improvement doesn't crash.
        os.makedirs(os.path.join("experiments", params.experiment_name), exist_ok=True)
        valid = SubgraphDataset(params.db_path, "valid_pos", "valid_neg", params.file_paths,
                                add_traspose_rels=False, num_neg_samples_per_link=0,
                                use_kge_embeddings=False, dataset="BioSNAP",
                                kge_model="TransE", file_name="valid",
                                ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                                id2relation=train.id2relation, rel=train.num_rels, graph=train.graph)
        if self.materialize:
            from subgraph_extraction.datasets_materialized import MaterializedSubgraphDataset
            train = MaterializedSubgraphDataset(train)
            valid = MaterializedSubgraphDataset(valid)
            test = MaterializedSubgraphDataset(test)
            params.num_workers = 0   # cache is in-process; forked workers would duplicate it
            logging.info("[sumgnn-ml] materialized train/valid/test subgraphs -> num_workers=0")
        valid_ev = Evaluator_ddi2(params, clf, valid)   # best-ckpt selection (validation)
        test_ev = Evaluator_ddi2(params, clf, test)     # logging only
        trainer = Trainer(params, clf, train, valid_ev, valid_ev, test_ev)
        logging.info(f"[sumgnn-ml] training {self.n_epochs} epochs on {len(train)} subgraphs "
                     f"(n_labels={self._n_labels}, inp_dim={params.inp_dim})")
        trainer.train()
        # load_best_model_at_end (valid-selected); fall back to final model.
        best = Path(params.exp_dir) / "best_graph_classifier.pth"
        if best.is_file():
            try:
                clf = torch.load(best, map_location=params.device, weights_only=False)
                logging.info(f"[sumgnn-ml] loaded best valid-selected checkpoint: {best}")
            except Exception as e:  # noqa: BLE001
                logging.warning(f"[sumgnn-ml] best-ckpt reload failed ({e}); using final model")
        self._clf = clf
        self._test_ds = test

    def _prepare_cwd(self, data_dir: Path) -> None:
        sandbox = data_dir / "_cwd"
        (sandbox / "data").mkdir(parents=True, exist_ok=True)
        link = sandbox / "data" / "BioSNAP"
        try:
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
        import json
        r2i_path = data_dir / "relation2id.json"
        relation2id = json.loads(r2i_path.read_text()) if r2i_path.is_file() else {}
        clf = dgl_model(params, relation2id).to(params.device)
        with open(data_dir / "id2drug_feat.pkl", "rb") as f:
            x = pickle.load(f, encoding="utf-8")
        mfeat = np.array([x[z]["Morgan"] for z in sorted(x.keys())])
        clf.drug_feat(torch.FloatTensor(mfeat).to(params.device))
        return clf

    def predict(self, test_df: "pd.DataFrame") -> "np.ndarray":
        if self._clf is None:
            raise RuntimeError("fit() before predict().")
        from torch.utils.data import DataLoader

        p = self._params
        loader = DataLoader(self._test_ds, batch_size=p.batch_size, shuffle=False,
                            num_workers=p.num_workers, collate_fn=p.collate_fn)
        self._clf.eval()
        outs = []
        with torch.no_grad():
            for batch in loader:
                data_pos, r_labels_pos, targets_pos = p.move_batch_to_device(batch, p.device)
                logits = self._clf(data_pos)
                outs.append(torch.sigmoid(logits).cpu().numpy())
        scores = np.concatenate(outs, axis=0) if outs else np.zeros((0, self._n_labels))
        return self._align_to_test(scores, test_df)

    def _align_to_test(self, out: np.ndarray, test_df: "pd.DataFrame") -> np.ndarray:
        data_dir = Path(self._params.file_paths["test"]).parent
        with open(data_dir / "entity2id.pkl", "rb") as f:
            entity2id = pickle.load(f)
        a = test_df["drug_a_id"].astype(str).map(entity2id)
        b = test_df["drug_b_id"].astype(str).map(entity2id)
        keep = (a.notna() & b.notna()).to_numpy()
        full = np.zeros((len(test_df), self._n_labels), dtype=np.float32)
        if keep.sum() == out.shape[0]:
            full[keep] = out
        else:
            m = min(keep.sum(), out.shape[0])
            idx = np.where(keep)[0][:m]
            full[idx] = out[:m]
        return full


__all__ = ["SumGNNUnifiedMultilabel"]

"""Minimal paper-faithfulness gate for the SumGNN reproduction (DrugBank, multiclass).

Runs the PORTED SumGNN core on SumGNN's OWN DrugBank data (int-indexed triples +
Hetionet 2-hop KG + Morgan feats), transductive/warm setting, for a few epochs on a
subsample, and reports Macro-F1 (the paper's DrugBank primary metric, §4.1).

Paper Table 1 target (line 590): SumGNN(Ours) DrugBank F1 86.85 +/- 0.44 (5-run mean).
This gate is a BALLPARK check that the port learns and heads toward that number, not a
full reproduction. Subgraph extraction is cached under data/drugbank/subgraphs_*.

Run from the reproduction root:
  python run_drugbank_gate.py --num_epochs 8 --max_links 20000 --gpu 0
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn import metrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets  # noqa: E402
from utils.initialization_utils import initialize_experiment, initialize_model  # noqa: E402
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl  # noqa: E402
from model.dgl.graph_classifier import GraphClassifier as dgl_model  # noqa: E402
from managers.evaluator import Evaluator  # noqa: E402
from managers.trainer import Trainer  # noqa: E402


def build_params() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # gate knobs
    p.add_argument("--num_epochs", type=int, default=8)
    p.add_argument("--max_links", type=int, default=20000,
                   help="cap on train links (subsample) to keep subgraph extraction fast")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_every_iter", type=int, default=20,
                   help="iters between val/test eval; small so eval fires on subsample")
    p.add_argument("--emb_dim", type=int, default=32)
    p.add_argument("--hop", type=int, default=2)
    p.add_argument("--num_bases", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # paper/original defaults (train.py argparse) baked in as fixed hyperparams
    params = SimpleNamespace(
        experiment_name="drugbank_gate",
        dataset="drugbank",
        gpu=args.gpu,
        disable_cuda=False,
        load_model=False,
        train_file="train",
        valid_file="dev",
        test_file="test",
        num_epochs=args.num_epochs,
        eval_every=3,
        eval_every_iter=args.eval_every_iter,
        save_every=10,
        early_stop=100,
        optimizer="Adam",
        lr=5e-3,
        clip=1000,
        l2=1e-5,
        max_links=args.max_links,
        hop=args.hop,
        max_nodes_per_hop=200,
        use_kge_embeddings=False,
        kge_model="TransE",
        model_type="dgl",
        constrained_neg_prob=0.0,
        batch_size=args.batch_size,
        num_neg_samples_per_link=0,
        num_workers=args.num_workers,
        add_traspose_rels=False,
        enclosing_sub_graph=True,
        rel_emb_dim=32,
        attn_rel_emb_dim=32,
        emb_dim=args.emb_dim,
        num_gcn_layers=2,
        num_bases=args.num_bases,
        dropout=0.3,
        edge_dropout=0.4,
        gnn_agg_type="sum",
        add_ht_emb=True,
        add_sb_emb=True,
        has_attn=True,
        has_kg=True,
        feat="morgan",
        feat_dim=1024,
        add_feat_emb=True,
        add_transe_emb=True,
        gamma=0.2,
    )
    params.seed = args.seed
    return params


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    params = build_params()
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    initialize_experiment(params, __file__)
    params.file_paths = {
        "train": os.path.join(params.main_dir, f"data/{params.dataset}/{params.train_file}.txt"),
        "valid": os.path.join(params.main_dir, f"data/{params.dataset}/{params.valid_file}.txt"),
        "test": os.path.join(params.main_dir, f"data/{params.dataset}/{params.test_file}.txt"),
    }
    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device("cuda:%d" % params.gpu)
    else:
        params.device = torch.device("cpu")
    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    params.db_path = os.path.join(
        params.main_dir,
        f"data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}"
        f"_neg_{params.num_neg_samples_per_link}_hop_{params.hop}_max{params.max_links}",
    )
    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(params.db_path, "train_pos", "train_neg", params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.train_file)
    valid = SubgraphDataset(params.db_path, "valid_pos", "valid_neg", params.file_paths,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link,
                            use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                            kge_model=params.kge_model, file_name=params.valid_file,
                            ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                            id2relation=train.id2relation, rel=train.num_rels, graph=train.graph)
    test = SubgraphDataset(params.db_path, "test_pos", "test_neg", params.file_paths,
                           add_traspose_rels=params.add_traspose_rels,
                           num_neg_samples_per_link=params.num_neg_samples_per_link,
                           use_kge_embeddings=params.use_kge_embeddings, dataset=params.dataset,
                           kge_model=params.kge_model, file_name=params.test_file,
                           ssp_graph=train.ssp_graph, id2entity=train.id2entity,
                           id2relation=train.id2relation, rel=train.num_rels, graph=train.graph)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim
    params.train_rels = params.num_rels
    params.num_nodes = 35000
    params.max_label_value = train.max_n_label

    logging.info(f"Device: {params.device}  inp_dim={params.inp_dim} num_rels={params.num_rels} "
                 f"aug_num_rels={params.aug_num_rels}")

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    import pickle
    with open(f"data/{params.dataset}/DB_molecular_feats.pkl", "rb") as f:
        x = pickle.load(f, encoding="utf-8")
    mfeat = [y for y in x["Morgan_Features"]]
    params.feat_dim = 1024
    graph_classifier.drug_feat(torch.FloatTensor(np.array(mfeat)).to(params.device))

    valid_evaluator = Evaluator(params, graph_classifier, valid)
    test_evaluator = Evaluator(params, graph_classifier, test)
    train_evaluator = Evaluator(params, graph_classifier, train)

    trainer = Trainer(params, graph_classifier, train, train_evaluator, valid_evaluator, test_evaluator)
    logging.info("Starting SumGNN gate training...")
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    # final eval on val + test with the ported Evaluator (macro-F1 = 'auc' key)
    val_res, _ = valid_evaluator.eval()
    test_res, _ = test_evaluator.eval()
    logging.info(f"[GATE] train_time={train_s:.1f}s best_val_macroF1={trainer.best_metric:.4f}")
    logging.info(f"[GATE] final val: {val_res}")
    logging.info(f"[GATE] final test: {test_res}")
    print(f"[GATE-RESULT] best_val_macro_f1={trainer.best_metric:.4f} "
          f"final_test_macro_f1={test_res['auc']:.4f} "
          f"final_test_micro_f1={test_res['microf1']:.4f} "
          f"final_test_kappa={test_res['k']:.4f} "
          f"train_time_s={train_s:.1f} max_links={params.max_links} epochs={params.num_epochs}")


if __name__ == "__main__":
    main()

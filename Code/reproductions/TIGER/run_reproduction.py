"""TIGER reproduction runner (one fold at a time, paper-faithful).

Mirrors upstream ``main.py``:
  * 5-fold StratifiedKFold over interactions with ``random_state=2023``
  * ``setup_seed(42)`` for torch + numpy + random
  * Adam(lr=1e-3, weight_decay=1e-4)
  * 50 epochs max, early-stop after 10 epochs of no val-AUC improvement
  * batch_size=128, L=2, d=64, num_heads=4, dropout=0.2
  * sub_coeff=0.1 (β1 for MG-MI), mi_coeff=0.1 (β2 for BKG-MI)
  * Best ckpt selected by val AUC; final test metrics = ACC / F1 / AUROC / AUPRC

Single-fold execution (rather than always 5-fold) is exposed for faster
iteration during reproduction debugging; specify ``--fold-only N`` to run
just fold N (0-indexed).

Example
-------
    python reproductions/TIGER/run_reproduction.py \\
        --dataset drugbank --extractor randomWalk \\
        --tag drugbank_rw --device cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# reproductions/TIGER on sys.path so local modules import.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# project root for shared utils (RunLogger).
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

from data_loader import load_data
from model.tiger import TIGER
from my_code.utils.run_logger import RunLogger
from train_eval import eval as eval_fn, test as test_fn, train as train_fn
from utils import DTADataset, collate


def init_args(user_args: list[str] | None = None):
    p = argparse.ArgumentParser(description="TIGER reproduction")

    p.add_argument("--dataset", type=str, default="drugbank",
                   help="one of drugbank / kegg / ogbl-biokg")
    p.add_argument("--data-root", type=str, default=None,
                   help="parent dir holding <dataset>/{drug_smiles,networks,ddi}.txt. "
                        "Default = data_loader.REPO_DATA_DIR = "
                        "reproductions/TIGER/_Original-Dataset/")
    p.add_argument("--cache-root", type=str, default=None,
                   help="where mol_sp.json + subgraph caches go (default = data-root)")

    # Paper hyperparams (defaults match main.py argparse defaults).
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-only", type=int, default=None,
                   help="if set, run only this fold index (0..folds-1)")
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model-episodes", type=int, default=50)
    p.add_argument("--early-stop-patience", type=int, default=10)

    p.add_argument("--extractor", type=str, default="randomWalk",
                   choices=["khop-subtree", "randomWalk", "probability"])
    p.add_argument("--graph-fixed-num", type=int, default=1,
                   help="num random-walk paths per drug (rw extractor)")
    p.add_argument("--khop", type=int, default=2)
    p.add_argument("--fixed-num", type=int, default=32,
                   help="subgraph size for probability + DeepWalk extractors "
                        "(paper Sec 'Hyper-Parameter Studies' = 32). For "
                        "k-subtree, see ``--khop-fanout`` instead.")
    p.add_argument("--khop-fanout", type=int, default=4,
                   help="k-subtree fan-out — constant #child nodes per "
                        "non-leaf node (paper Sec 'Experimental Settings' "
                        "= 4). Upstream's main.py shares a single "
                        "``fixed_num=32`` arg between the two roles, which "
                        "yields fan-out=32 — paper-inconsistent. We split.")

    p.add_argument("--d-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--sub-coeff", type=float, default=0.1,
                   help="β1 weight for L_MI(h, g) (MG channel) — paper Eq. 11")
    p.add_argument("--mi-coeff", type=float, default=0.1,
                   help="β2 weight for L_MI(h, s) (BKG channel) — paper Eq. 11")

    p.add_argument("--seed", type=int, default=42,
                   help="torch/numpy/random seed (paper main.py hardcodes 42)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--num-features-drug", type=int, default=67,
                   help="raw atom feature dim (44+11+11+1=67); paper code "
                        "ships this as 67 hardcoded in main.py init_model")

    return p.parse_args(user_args)


def setup_seed(seed: int) -> None:
    """Match upstream :func:`main.setup_seed` exactly."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def k_fold_splits(data_len: int, folds: int, y: np.ndarray):
    """Stratified k-fold + use ``test_indices[i-1]`` as val. Matches upstream
    :func:`main.k_fold` semantics."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
    test_indices = [idx for _, idx in skf.split(np.zeros(data_len), y)]
    val_indices = [test_indices[i - 1] for i in range(folds)]

    train_indices = []
    for i in range(folds):
        train_mask = torch.ones(data_len, dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def build_model(args, stats: dict, device: str) -> tuple[TIGER, torch.optim.Optimizer]:
    model = TIGER(
        max_layer=args.layer,
        num_features_drug=args.num_features_drug,
        num_nodes=stats["num_nodes"],
        num_relations_mol=stats["num_rel_mol"],
        num_relations_graph=stats["num_rel_graph"],
        output_dim=args.d_dim,
        max_degree_graph=stats["max_degree_graph"],
        max_degree_node=stats["max_degree_node"],
        sub_coeff=args.sub_coeff,
        mi_coeff=args.mi_coeff,
        dropout=args.dropout,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer


def run_one_fold(
    fold: int,
    train_idx: torch.Tensor,
    test_idx: np.ndarray,
    val_idx: np.ndarray,
    bundle,
    args,
    device: str,
) -> dict:
    print(f"================== fold {fold + 1}/{args.folds} ==================", flush=True)
    train_data = DTADataset(
        x=bundle.interactions[train_idx],
        y=bundle.labels[train_idx],
        sub_graph=bundle.drug_subgraphs,
        smile_graph=bundle.smile_graph,
    )
    test_data = DTADataset(
        x=bundle.interactions[test_idx],
        y=bundle.labels[test_idx],
        sub_graph=bundle.drug_subgraphs,
        smile_graph=bundle.smile_graph,
    )
    val_data = DTADataset(
        x=bundle.interactions[val_idx],
        y=bundle.labels[val_idx],
        sub_graph=bundle.drug_subgraphs,
        smile_graph=bundle.smile_graph,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )

    model, optimizer = build_model(args, bundle.stats, device)
    model.to(device)
    model.reset_parameters()

    train_log = {
        "train_acc": [], "train_auc": [], "train_aupr": [], "train_loss": [],
        "eval_acc": [], "eval_auc": [], "eval_aupr": [], "eval_loss": [],
    }
    best_state = None
    best_auc = 0.0
    early_stop = 0
    for i_episode in range(args.model_episodes):
        loop = tqdm(train_loader, ncols=80, desc=f"ep[{i_episode}/{args.model_episodes}]")
        tr_acc, tr_f1, tr_auc, tr_aupr, tr_loss = train_fn(loop, model, optimizer, device=device)
        ev_acc, ev_f1, ev_auc, ev_aupr, ev_loss = eval_fn(val_loader, model, device=device)
        print(
            f"[tiger-repro] fold {fold + 1} ep {i_episode + 1}/{args.model_episodes} "
            f"tr_auc={tr_auc:.4f} tr_aupr={tr_aupr:.4f} "
            f"val_auc={ev_auc:.4f} val_aupr={ev_aupr:.4f}",
            flush=True,
        )
        train_log["train_acc"].append(tr_acc)
        train_log["train_auc"].append(tr_auc)
        train_log["train_aupr"].append(tr_aupr)
        train_log["train_loss"].append(tr_loss)
        train_log["eval_acc"].append(ev_acc)
        train_log["eval_auc"].append(ev_auc)
        train_log["eval_aupr"].append(ev_aupr)
        train_log["eval_loss"].append(ev_loss)

        if ev_auc > best_auc:
            best_auc = ev_auc
            best_state = copy.deepcopy(model.state_dict())
            early_stop = 0
        else:
            early_stop += 1
            if early_stop > args.early_stop_patience:
                print(f"[tiger-repro] early stop at ep {i_episode + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    test_log = test_fn(test_loader, model, device=device)
    print(
        f"[tiger-repro] FINAL fold {fold + 1} test: "
        f"acc={test_log['acc']:.4f} f1={test_log['f1']:.4f} "
        f"auc={test_log['auc']:.4f} aupr={test_log['aupr']:.4f}",
        flush=True,
    )
    return {
        "fold": fold,
        "best_val_auc": best_auc,
        "test": test_log,
        "train_log": train_log,
    }


def main(user_args: list[str] | None = None) -> None:
    args = init_args(user_args)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tag is None:
        args.tag = f"{args.dataset}_{args.extractor}_e{args.model_episodes}"

    with RunLogger(script="tiger_reproduce", tag=args.tag, seed=args.seed) as rl:
        print(f"[run] config: {vars(args)}", flush=True)

        # Route ``--fixed-num`` for prob/DW, ``--khop-fanout`` for k-subtree —
        # see argparse help for ``--khop-fanout`` for the paper-vs-upstream
        # rationale.
        fixed_num_for_extractor = (
            args.khop_fanout if args.extractor == "khop-subtree" else args.fixed_num
        )
        bundle = load_data(
            dataset=args.dataset,
            extractor=args.extractor,
            data_root=args.data_root,
            cache_root=args.cache_root,
            khop=args.khop,
            fixed_num=fixed_num_for_extractor,
            graph_fixed_num=args.graph_fixed_num,
        )

        setup_seed(args.seed)
        train_idxs, test_idxs, val_idxs = k_fold_splits(
            len(bundle.interactions), args.folds, bundle.labels
        )

        fold_results = []
        fold_iter = (
            [args.fold_only] if args.fold_only is not None else range(args.folds)
        )
        for fold in fold_iter:
            res = run_one_fold(
                fold,
                train_idxs[fold],
                test_idxs[fold],
                val_idxs[fold],
                bundle,
                args,
                args.device,
            )
            fold_results.append(res)

        # 5-fold summary (or single-fold if --fold-only)
        if len(fold_results) > 1:
            accs = np.array([r["test"]["acc"] for r in fold_results])
            f1s = np.array([r["test"]["f1"] for r in fold_results])
            aucs = np.array([r["test"]["auc"] for r in fold_results])
            auprs = np.array([r["test"]["aupr"] for r in fold_results])
            summary = {
                "acc": [float(accs.mean()), float(accs.std())],
                "f1": [float(f1s.mean()), float(f1s.std())],
                "auc": [float(aucs.mean()), float(aucs.std())],
                "aupr": [float(auprs.mean()), float(auprs.std())],
            }
            print(f"[tiger-repro] {args.folds}-fold mean±std: {summary}", flush=True)
        else:
            summary = fold_results[0]["test"]

        out_path = rl.run_dir / "results.json"
        with out_path.open("w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "data_stats": bundle.stats,
                    "fold_results": fold_results,
                    "summary": summary,
                },
                f,
                indent=2,
            )
        print(f"[run] saved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

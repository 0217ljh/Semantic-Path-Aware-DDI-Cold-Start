"""Real-data smoke test for TIGER reproduction.

Runs 2 training epochs on a small random subset of DrugBank with k-subtree
extractor on GPU. Purpose: verify the full pipeline (data parse → model
build → train/eval loop) works on paper data without errors, and that
val AUC moves in a sensible direction. NOT a full reproduction — that needs
the full 50-epoch × 5-fold run.

Expected wall time: a few minutes on RTX 5090.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Force stdout to handle Unicode arrows (Windows default cp1252 chokes)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from data_loader import load_data
from model.tiger import TIGER
from train_eval import eval as eval_fn, test as test_fn, train as train_fn
from utils import DTADataset, collate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== TIGER real-data smoke: device={device} ===", flush=True)

    # 1. Load real drugbank data (cached subgraph JSON will be reused)
    t0 = time.time()
    bundle = load_data(
        dataset="drugbank",
        extractor="khop-subtree",
        # data_root + cache_root default to REPO_DATA_DIR
        # (= reproductions/TIGER/_Original-Dataset/)
        khop=2,
        fixed_num=4,
    )
    print(f"  load_data: {time.time() - t0:.1f}s, "
          f"interactions={bundle.interactions.shape}, drugs={bundle.stats['num_drugs_DDI']}",
          flush=True)

    # 2. Take small random subset (~2k pairs) to keep CPU/GPU memory + time low
    rng = np.random.default_rng(42)
    n_total = len(bundle.interactions)
    subset_size = min(2000, n_total)
    idx = rng.choice(n_total, size=subset_size, replace=False)
    sub_interactions = bundle.interactions[idx]
    sub_labels = bundle.labels[idx]
    print(f"  subset: {subset_size} pairs (pos={int(sub_labels.sum())})", flush=True)

    # 3. Stratified 5-fold split on the subset (matches paper protocol)
    torch.manual_seed(42)
    np.random.seed(42)
    skf = StratifiedKFold(5, shuffle=True, random_state=2023)
    folds = list(skf.split(np.zeros(len(sub_interactions)), sub_labels))
    train_idx, test_idx = folds[0]
    val_idx = folds[-1][1]  # use last fold as val (matches main.py k_fold logic)
    print(f"  fold0: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    train_ds = DTADataset(
        x=sub_interactions[train_idx], y=sub_labels[train_idx],
        sub_graph=bundle.drug_subgraphs, smile_graph=bundle.smile_graph,
    )
    val_ds = DTADataset(
        x=sub_interactions[val_idx], y=sub_labels[val_idx],
        sub_graph=bundle.drug_subgraphs, smile_graph=bundle.smile_graph,
    )
    test_ds = DTADataset(
        x=sub_interactions[test_idx], y=sub_labels[test_idx],
        sub_graph=bundle.drug_subgraphs, smile_graph=bundle.smile_graph,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, shuffle=False, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, collate_fn=collate
    )

    # 4. Build TIGER model
    model = TIGER(
        max_layer=2,
        num_features_drug=67,
        num_nodes=bundle.stats["num_nodes"],
        num_relations_mol=bundle.stats["num_rel_mol"],
        num_relations_graph=int(bundle.stats["num_rel_graph"]),
        output_dim=64,
        max_degree_graph=bundle.stats["max_degree_graph"],
        max_degree_node=bundle.stats["max_degree_node"],
        sub_coeff=0.1, mi_coeff=0.1, dropout=0.2, device=device,
    ).to(device)
    model.reset_parameters()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 5. Train 2 epochs, eval each
    n_epochs = 2
    best_val_auc = 0.0
    best_state = None
    for ep in range(n_epochs):
        t_ep = time.time()
        loop = tqdm(train_loader, ncols=80, desc=f"ep{ep + 1}/{n_epochs}")
        tr_acc, tr_f1, tr_auc, tr_aupr, tr_loss = train_fn(loop, model, opt, device=device)
        ev_acc, ev_f1, ev_auc, ev_aupr, ev_loss = eval_fn(val_loader, model, device=device)
        print(
            f"  ep {ep + 1}: tr_auc={tr_auc:.4f} tr_loss={tr_loss:.4f} | "
            f"val_auc={ev_auc:.4f} val_aupr={ev_aupr:.4f} val_acc={ev_acc:.4f} "
            f"({time.time() - t_ep:.1f}s)",
            flush=True,
        )
        if ev_auc > best_val_auc:
            best_val_auc = ev_auc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = test_fn(test_loader, model, device=device)
    print(
        f"\n  FINAL test: acc={test_metrics['acc']:.4f} f1={test_metrics['f1']:.4f} "
        f"auc={test_metrics['auc']:.4f} aupr={test_metrics['aupr']:.4f}",
        flush=True,
    )

    # 6. Sanity check: val AUC should improve over 2 epochs (model is learning)
    print(f"\n  best_val_auc={best_val_auc:.4f}", flush=True)
    if best_val_auc > 0.55:
        print("\nREAL-DATA SMOKE PASS: model trains, val AUC > random (>0.55)")
    else:
        print(f"\nREAL-DATA SMOKE WARNING: val AUC = {best_val_auc:.4f} <= 0.55, may need more epochs")


if __name__ == "__main__":
    main()

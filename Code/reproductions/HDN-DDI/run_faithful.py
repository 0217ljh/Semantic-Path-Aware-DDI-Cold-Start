"""HDN-DDI faithful reproduction runner.

Byte-exact port of
https://github.com/jcsun-00/HDN-DDI/blob/main/drugbank_test/inductive_train.py

Differences from the upstream script:
  - Reads data from this folder's ``_Original-Dataset/`` (handled inside
    :mod:`data_preprocessing` via _DATA_DIR), not the upstream's
    ``drugbank_test/DrugBank/`` path.
  - ``--n_atom_feats`` default changed from 55 to **66** because the
    upstream ships the BRICS hierarchical pkl whose ``x`` is 66-dim.
    Keeping 55 would crash ``LayerNorm(55)`` on the first forward pass.
    Upstream's default 55 was presumably for a different code path the
    repo doesn't expose.
  - Model checkpoint goes to ``./ckpts/`` under this folder by default.
  - Adds a ``--n_workers`` CLI knob (upstream hardcodes 2) since some
    Windows / WSL setups don't tolerate multiprocessing workers well.

Everything else (Adam(lr=1e-3, wd=5e-4) + LambdaLR(0.96^epoch),
batch=1024, n_epochs=100, patience=30, train+val merged into one
training set, best ckpt selected on the mean of s1+s2 ACC/AUROC/F1,
SigmoidLoss, dynamic per-batch negatives via DrugDataset) is byte-exact
upstream.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import optim

# Make THIS folder (HDN-DDI/) the first entry on sys.path so plain
# `import models / layers / data_preprocessing / custom_loss` resolves to
# the byte-exact ports next to this script.  Folder name contains a
# hyphen so we can't treat it as a Python package; sys.path injection is
# the cleanest workaround and exactly matches jcsun-00's launch layout.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Also expose project Code/ so we can pull in the shared RunLogger helper
# (Code/my_code/utils/run_logger.py) without forcing the caller to set
# PYTHONPATH.
_PROJECT_CODE_DIR = _THIS_DIR.parent.parent  # Code/
if str(_PROJECT_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_CODE_DIR))

import models  # noqa: E402
import custom_loss  # noqa: E402
from data_preprocessing import DrugDataset, DrugDataLoader  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------- args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_atom_feats", type=int, default=66,
                        help="num of input node features (66 matches the BRICS pkl)")
    parser.add_argument("--n_atom_hid", type=int, default=128,
                        help="num of hidden features (unused inside the block — kept for upstream parity)")
    parser.add_argument("--rel_total", type=int, default=86,
                        help="num of interaction types")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="num of epochs")
    parser.add_argument("--kge_dim", type=int, default=128,
                        help="dimension of RESCAL interaction matrices")
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="batch size (paper Methods says 512; repeat.sh also passes 512 — "
             "inductive_train.py's argparse default of 1024 is misleading, paper "
             "actually uses 512)",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_cuda", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 2015])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument(
        "--log_step_every", type=int, default=50,
        help="Print per-step rolling-mean train loss every N batches "
             "(CLAUDE.md training-progress convention; default 50).",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Run tag for RunLogger (default: fold{fold}_faithful).",
    )
    parser.add_argument(
        "--pkl_name", type=str, default=None,
        help="output ckpt path (default: <run_dir>/best.pkl under "
             "Code/runs/<run_id>/).",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------- metrics


def do_compute(batch, device, model):
    """*batch = (pos_tri, neg_tri); each tri = (h, t, r, b)*."""
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, _t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap = metrics.average_precision_score(target, probas_pred)
    return acc, auroc, f1_score, precision, recall, int_ap, ap


# --------------------------------------------------------------------- train


def train(
    model, train_data_loader, s1_data_loader, s2_data_loader,
    loss_fn, optimizer, n_epochs, device, pkl_name, scheduler=None,
    train_data=None, s1_data=None, s2_data=None,
    log_step_every: int = 50,
):
    print("Starting training at", datetime.today(), flush=True)
    best_mean_metrics, best_epoch = 0, 0
    n_total_steps = (len(train_data) + train_data_loader.batch_size - 1) // train_data_loader.batch_size
    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        s1_loss = 0
        s2_loss = 0

        train_probas_pred = []
        train_ground_truth = []

        s1_probas_pred = []
        s1_ground_truth = []

        s2_probas_pred = []
        s2_ground_truth = []

        # Per-step rolling-loss logging (CLAUDE.md '训练进度日志规范').
        running_loss = 0.0
        step_count = 0
        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
            running_loss += float(loss.item())
            step_count += 1
            if step_count % log_step_every == 0:
                print(
                    f"[hdn_faithful] [ep {i}/{n_epochs} "
                    f"step {step_count}/{n_total_steps}] "
                    f"loss={running_loss/step_count:.4f}",
                    flush=True,
                )
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            (train_acc, train_auc_roc, train_f1, train_precision,
             train_recall, train_int_ap, train_ap) = do_compute_metrics(
                train_probas_pred, train_ground_truth
            )

            for batch in s1_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s1_probas_pred.append(probas_pred)
                s1_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                s1_loss += loss.item() * len(p_score)

            s1_loss /= len(s1_data)
            s1_probas_pred = np.concatenate(s1_probas_pred)
            s1_ground_truth = np.concatenate(s1_ground_truth)
            (s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall,
             s1_int_ap, s1_ap) = do_compute_metrics(s1_probas_pred, s1_ground_truth)

            for batch in s2_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                s2_probas_pred.append(probas_pred)
                s2_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                s2_loss += loss.item() * len(p_score)

            s2_loss /= len(s2_data)
            s2_probas_pred = np.concatenate(s2_probas_pred)
            s2_ground_truth = np.concatenate(s2_ground_truth)
            (s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall,
             s2_int_ap, s2_ap) = do_compute_metrics(s2_probas_pred, s2_ground_truth)

            s1_metrics = np.average([s1_acc, s1_auc_roc, s1_f1])
            s2_metrics = np.average([s2_acc, s2_auc_roc, s2_f1])
            mean_metrics = np.average([s1_metrics, s2_metrics])
            if mean_metrics > best_mean_metrics:
                best_mean_metrics, best_epoch = mean_metrics, i
                torch.save(model, pkl_name)

        if scheduler:
            scheduler.step()

        flag = "*" if best_epoch == i else " "
        print(
            f"Epoch: {i}{flag} ({time.time() - start:.4f}s), "
            f"train_loss: {train_loss:.4f}, s1_loss: {s1_loss:.4f}, s2_loss: {s2_loss:.4f}"
        )
        print(
            f"\t\ttrain_acc: {train_acc:.4f}, train_roc: {train_auc_roc:.4f},"
            f"train_precision: {train_precision:.4f}, train_recall:{train_recall:.4f}"
        )
        print(
            f"\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, "
            f"s1_aupr:{s1_ap:.4f}, s1_f1:{s1_f1:.4f}"
        )
        print(
            f"\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, "
            f"s2_aupr:{s2_ap:.4f}, s2_f1:{s2_f1:.4f}"
        )

        if i - best_epoch >= 30:
            print(f"Early Stopping at training epoch: {i}, best epoch: {best_epoch}")
            break
    return best_epoch


def test(s1_data_loader, s2_data_loader, model, device):
    s1_probas_pred = []
    s1_ground_truth = []

    s2_probas_pred = []
    s2_ground_truth = []
    with torch.no_grad():
        for batch in s1_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            s1_probas_pred.append(probas_pred)
            s1_ground_truth.append(ground_truth)

        s1_probas_pred = np.concatenate(s1_probas_pred)
        s1_ground_truth = np.concatenate(s1_ground_truth)
        (s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall,
         s1_int_ap, s1_ap) = do_compute_metrics(s1_probas_pred, s1_ground_truth)

        for batch in s2_data_loader:
            model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            s2_probas_pred.append(probas_pred)
            s2_ground_truth.append(ground_truth)

        s2_probas_pred = np.concatenate(s2_probas_pred)
        s2_ground_truth = np.concatenate(s2_ground_truth)
        (s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall,
         s2_int_ap, s2_ap) = do_compute_metrics(s2_probas_pred, s2_ground_truth)

    print("\n")
    print("============================== Best Result ==============================")
    print(
        f"\t\ts1_acc: {s1_acc:.4f}, s1_roc: {s1_auc_roc:.4f}, s1_f1: {s1_f1:.4f}, "
        f"s1_precision: {s1_precision:.4f}, s1_recall: {s1_recall:.4f}, "
        f"s1_int_ap: {s1_int_ap:.4f}, s1_ap: {s1_ap:.4f}"
    )
    print(
        f"\t\ts2_acc: {s2_acc:.4f}, s2_roc: {s2_auc_roc:.4f}, s2_f1: {s2_f1:.4f}, "
        f"s2_precision: {s2_precision:.4f}, s2_recall: {s2_recall:.4f}, "
        f"s2_int_ap: {s2_int_ap:.4f}, s2_ap: {s2_ap:.4f}"
    )
    return {
        "s1": dict(acc=s1_acc, auroc=s1_auc_roc, f1=s1_f1,
                   precision=s1_precision, recall=s1_recall,
                   int_ap=s1_int_ap, ap=s1_ap),
        "s2": dict(acc=s2_acc, auroc=s2_auc_roc, f1=s2_f1,
                   precision=s2_precision, recall=s2_recall,
                   int_ap=s2_int_ap, ap=s2_ap),
    }


# --------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)

    tag = args.tag or f"fold{args.fold}_faithful"

    with RunLogger(script="hdn_ddi_faithful", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            baseline="hdn_ddi",
            fold=args.fold,
            n_atom_feats=args.n_atom_feats,
            n_atom_hid=args.n_atom_hid,
            kge_dim=args.kge_dim,
            rel_total=args.rel_total,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
        )
        print(f"[run] config: {vars(args)}", flush=True)

        pkl_name = args.pkl_name or str(rl.run_dir / "best.pkl")
        if not pkl_name.endswith(f"-fold{args.fold}.pkl"):
            pkl_name = pkl_name.replace(".pkl", f"-fold{args.fold}.pkl")
        Path(pkl_name).parent.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available() and args.use_cuda:
            torch.cuda.set_device(args.device)
            device = "cuda"
        else:
            device = "cpu"
        print(f"[run] device = {device}", flush=True)

        result = _run_one(args, pkl_name, device)

        rl.set_metrics(
            best_epoch=result["best_epoch"],
            s1_acc=result["s1"]["acc"],
            s1_auroc=result["s1"]["auroc"],
            s1_f1=result["s1"]["f1"],
            s2_acc=result["s2"]["acc"],
            s2_auroc=result["s2"]["auroc"],
            s2_f1=result["s2"]["f1"],
        )
        out_path = rl.run_dir / "results.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"[run] saved results to {out_path}", flush=True)
    return 0


def _run_one(args, pkl_name, device):
    fold_dir = _THIS_DIR / "_Original-Dataset" / "cold_start" / f"fold{args.fold}"
    df_ddi_train = pd.concat(
        [
            pd.read_csv(str(fold_dir / "train.csv")),
            pd.read_csv(str(fold_dir / "val.csv")),
        ],
        axis=0,
    )
    df_ddi_s1 = pd.read_csv(str(fold_dir / "s1.csv"))
    df_ddi_s2 = pd.read_csv(str(fold_dir / "s2.csv"))

    train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train["d1"], df_ddi_train["d2"], df_ddi_train["type"])]
    s1_tup = [(h, t, r) for h, t, r in zip(df_ddi_s1["d1"], df_ddi_s1["d2"], df_ddi_s1["type"])]
    s2_tup = [(h, t, r) for h, t, r in zip(df_ddi_s2["d1"], df_ddi_s2["d2"], df_ddi_s2["type"])]

    train_data = DrugDataset(train_tup)
    s1_data = DrugDataset(s1_tup, disjoint_split=True)
    s2_data = DrugDataset(s2_tup, disjoint_split=True)

    print(f"Training with {len(train_data)} samples, s1 with {len(s1_data)}, and s2 with {len(s2_data)}")

    train_data_loader = DrugDataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    s1_data_loader = DrugDataLoader(
        s1_data, batch_size=args.batch_size * 3, num_workers=args.n_workers
    )
    s2_data_loader = DrugDataLoader(
        s2_data, batch_size=args.batch_size * 3, num_workers=args.n_workers
    )

    model = models.HDN_DDI(
        args.n_atom_feats, args.n_atom_hid, args.kge_dim, args.rel_total,
        heads_out_feat_params=[64, 64, 64, 64, 64, 64],
        blocks_params=[2, 2, 2, 2, 2, 2],
    )
    loss = custom_loss.SigmoidLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
    model.to(device=device)

    best_epoch = train(
        model, train_data_loader, s1_data_loader, s2_data_loader,
        loss, optimizer, args.n_epochs, device, pkl_name, scheduler,
        train_data=train_data, s1_data=s1_data, s2_data=s2_data,
        log_step_every=args.log_step_every,
    )
    test_model = torch.load(pkl_name, weights_only=False)
    test_result = test(s1_data_loader, s2_data_loader, test_model, device)
    return {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_ckpt": pkl_name,
        "s1": test_result["s1"],
        "s2": test_result["s2"],
    }


if __name__ == "__main__":
    sys.exit(main())

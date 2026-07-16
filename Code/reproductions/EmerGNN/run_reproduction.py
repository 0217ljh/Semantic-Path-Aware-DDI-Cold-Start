"""EmerGNN cold-start reproduction runner (DrugBank, S1 / S2).

Trains our ported ``EmerGNN_MC`` (verified architecturally identical to
``LARS-research/EmerGNN/DrugBank/models.py``) on the paper's own
splits, with per-epoch ``shuffle_train`` (the key inductive design),
softmax CE loss (mathematically equivalent to paper's hand-coded
softmax-margin), and final eval on test set with macro F1 / accuracy /
Cohen κ.

Each run handles ONE setting (e.g. ``S1_1``, ``S2_123``); to get the
5-seed mean ± std reported in paper Table 1, run all 5 settings per
S{1,2}-family.

Example:
    python reproductions/EmerGNN/run_reproduction.py --setting S1_1 \\
        --n-epoch 100 --tag s1_1_full
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# project root on sys.path (for shared utilities only — RunLogger)
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# reproductions/EmerGNN on sys.path so all reproduction-local modules import
# directly. Per CLAUDE.md §"Baseline 规范" line 447-448 "严禁 import 互调",
# the reproduction must NOT import from baseline.emergnn — model_mc.py is now
# an independent copy (2026-05-18 migration; see _reviews/2026-05-18__paper_faithful.md).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (  # noqa: E402
    N_DDI_CLASSES,
    build_global_morgan_matrix,
    load_setting,
    load_vocab,
    shuffle_train,
)
from model_mc import EmerGNN_MC  # noqa: E402 — local copy of EmerGNN_MC
from my_code.utils.run_logger import RunLogger  # noqa: E402


# -----------------------------------------------------------------------------
# KG → edge lists
# -----------------------------------------------------------------------------


def build_kg_edges(
    triplets: np.ndarray,
    n_ent: int,
    n_base_rel: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (edge_src, edge_dst, edge_rel) LongTensors on device from a
    triplet array. Adds reverse edges + per-entity self-loops in the
    convention used by paper's ``load_data.py:double_triple + load_graph``.

    With n_rel base relations, the rel id space after this call covers
    [0, 2*n_base_rel] inclusive (forward shifted to r+n_rel, reverse
    keeps r, self-loop uses 2*n_base_rel).

    Direct edge-list construction — bypasses ``build_sparse_adj`` which
    built a ``torch.sparse_coo_tensor`` of declared size
    ``(n_ent, n_ent, 2*n_rel+1) = (34124, 34124, 219)`` then ``.to(cuda)``.
    Some PyTorch builds densify or materialize huge intermediates for
    3D sparse tensors of this size, causing >50GB allocations even
    though actual non-zero entries are only ~3.6M.
    """
    triplets = np.asarray(triplets, dtype=np.int64)
    heads = triplets[:, 0]
    tails = triplets[:, 1]
    rels = triplets[:, 2]

    # forward: (h, t, r+n_rel)
    # reverse: (t, h, r)
    # self-loop: (e, e, 2*n_rel)  for e in range(n_ent)
    fwd_r = rels + n_base_rel
    rev_r = rels
    idd = np.arange(n_ent, dtype=np.int64)
    idd_r = np.full(n_ent, 2 * n_base_rel, dtype=np.int64)

    all_src = np.concatenate([heads, tails, idd])
    all_dst = np.concatenate([tails, heads, idd])
    all_rel = np.concatenate([fwd_r, rev_r, idd_r])

    src_t = torch.from_numpy(all_src).long().to(device)
    dst_t = torch.from_numpy(all_dst).long().to(device)
    rel_t = torch.from_numpy(all_rel).long().to(device)
    return src_t, dst_t, rel_t


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_split(
    model: EmerGNN_MC,
    triplets: np.ndarray,
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    device: str,
) -> dict:
    """Eval on a (h, t, r) array using a fixed KG (vKG for val, tKG for test).

    Returns dict with macro F1, accuracy, Cohen κ (paper's 3 metrics),
    plus n_samples for sanity.
    """
    if len(triplets) == 0:
        return {"macro_f1": float("nan"), "accuracy": float("nan"), "cohen_kappa": float("nan"), "n_samples": 0}
    edge_src, edge_dst, edge_rel = edges
    model.eval()
    n = len(triplets)
    head = torch.from_numpy(triplets[:, 0]).long().to(device)
    tail = torch.from_numpy(triplets[:, 1]).long().to(device)
    labels = triplets[:, 2]
    preds = np.zeros(n, dtype=np.int64)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        logits = model(head[s:e], tail[s:e], edge_src, edge_dst, edge_rel)
        # logits shape: (B, n_classes)
        preds[s:e] = logits.argmax(dim=-1).cpu().numpy()
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "n_samples": int(n),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_setting(
    setting: str,
    n_epoch: int,
    n_batch: int,
    test_batch_size: int,
    epoch_per_test: int,
    n_dim: int,
    length: int,
    lr: float,
    weight_decay: float,
    shuffle_ratio: float,
    seed: int,
    device: str,
    chunk_size: int = 100_000,
    use_checkpoint: bool = True,
) -> dict:
    """Train + eval one S0 / S1_<seed> / S2_<seed> setting end-to-end.

    Returns a dict with final test metrics + per-eval-epoch trajectory +
    config snapshot.
    """
    # PyTorch / numpy seeds (paper-internal split seed is encoded in setting name)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Data
    print(f"[repro-em] loading vocab + setting={setting} ...", flush=True)
    t0 = time.time()
    vocab = load_vocab()
    s = load_setting(setting)
    morgan_mat = build_global_morgan_matrix(vocab)
    print(
        f"[repro-em] vocab: n_drug={vocab['n_drug']} n_entity={vocab['n_entity']} "
        f"n_relation={vocab['n_relation']} (load {time.time()-t0:.1f}s)",
        flush=True,
    )
    print(
        f"[repro-em] {setting}: train_ddi={len(s['train_ddi'])} valid_ddi={len(s['valid_ddi'])} "
        f"test_ddi={len(s['test_ddi'])}",
        flush=True,
    )
    print(
        f"[repro-em] {setting}: train_kg={len(s['train_kg'])} valid_kg(cum)={len(s['valid_kg'])} "
        f"test_kg(cum)={len(s['test_kg'])}",
        flush=True,
    )

    n_ent = vocab["n_entity"]
    n_base_rel = vocab["n_relation"]  # 109 = 86 DDI + 23 KG

    # 2. Build static vKG / tKG edge lists (used at val / test time)
    print("[repro-em] building static vKG + tKG edges ...", flush=True)
    t0 = time.time()
    # vKG = train_ddi + train_kg + valid_kg_delta  (paper convention)
    vkg_triplets = np.concatenate([s["train_ddi"], s["valid_kg"]], axis=0)
    tkg_triplets = np.concatenate(
        [s["train_ddi"], s["valid_ddi"], s["test_kg"]], axis=0
    )
    vkg_edges = build_kg_edges(vkg_triplets, n_ent, n_base_rel, device)
    tkg_edges = build_kg_edges(tkg_triplets, n_ent, n_base_rel, device)
    # Compute full KG entity set (train + valid + test KG) to feed
    # shuffle_train's ``ddi_in_kg`` — matches paper's
    # ``process_files_kg`` which iterates all 3 splits.
    full_kg_ent = set(np.unique(s["test_kg"][:, :2]).tolist()) if len(s["test_kg"]) else set()
    print(
        f"[repro-em] vKG edges={vkg_edges[0].numel()} tKG edges={tkg_edges[0].numel()} "
        f"(build {time.time()-t0:.1f}s)",
        flush=True,
    )

    # 3. Model + optimizer
    model = EmerGNN_MC(
        n_ent=n_ent,
        n_base_rel=n_base_rel,
        n_classes=N_DDI_CLASSES,  # paper hardcodes 86
        n_dim=n_dim,
        length=length,
        feat="M",
        morgan_features=morgan_mat,
        morgan_feat_dim=morgan_mat.shape[1],
    ).to(device)
    model.set_chunk_size(chunk_size)
    model.set_use_checkpoint(use_checkpoint)
    print(
        f"[repro-em] chunk_size={chunk_size} use_checkpoint={use_checkpoint}",
        flush=True,
    )
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Paper base_model.py: ReduceLROnPlateau(self.optimizer, mode='max') with
    # PyTorch defaults (factor=0.1, patience=10).
    scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=10)
    print(
        f"[repro-em] model: EmerGNN_MC(n_ent={n_ent}, n_base_rel={n_base_rel}, "
        f"n_classes={N_DDI_CLASSES}, n_dim={n_dim}, L={length}, feat=M)",
        flush=True,
    )

    # 4. Train loop with per-epoch shuffle_train
    best_val_f1 = -1.0
    best_state: dict | None = None
    best_epoch = -1
    history = []

    for epoch in range(1, n_epoch + 1):
        ep_start = time.time()
        # 4a. Per-epoch fact / target split
        epoch_kg, train_targets = shuffle_train(
            s["train_ddi"],
            s["train_kg"],
            setting,
            ratio=shuffle_ratio,
            rng=rng,
            extra_kg_ent=full_kg_ent,  # match paper's process_files_kg union
        )
        # 4b. Build this epoch's KG edges
        train_edges = build_kg_edges(epoch_kg, n_ent, n_base_rel, device)
        # 4c. Batch targets in sequential order (paper does NOT extra-shuffle
        # after shuffle_train; utils.batch_by_size slices in order). The
        # per-epoch randomness comes from shuffle_train itself.
        n_targets = len(train_targets)
        if n_targets == 0:
            print(f"[repro-em] epoch {epoch}: 0 train targets after shuffle_train; skip", flush=True)
            continue
        head_all = torch.from_numpy(train_targets[:, 0]).long().to(device)
        tail_all = torch.from_numpy(train_targets[:, 1]).long().to(device)
        rel_all = torch.from_numpy(train_targets[:, 2]).long().to(device)

        model.train()
        running_loss = 0.0
        n_steps = 0
        for s_idx in range(0, n_targets, n_batch):
            e_idx = min(n_targets, s_idx + n_batch)
            h_b = head_all[s_idx:e_idx]
            t_b = tail_all[s_idx:e_idx]
            r_b = rel_all[s_idx:e_idx]
            opt.zero_grad(set_to_none=True)
            logits = model(h_b, t_b, *train_edges)  # (B, 86)
            # Paper base_model.py: loss = -p_score + max_n + logsumexp(...);
            # loss = loss.sum() then backward. Mathematically identical to
            # F.cross_entropy(..., reduction='sum'), up to gradient scaling
            # (sum vs mean changes effective lr by batch_size factor).
            loss = F.cross_entropy(logits, r_b, reduction="sum")
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) / max(len(r_b), 1)
            n_steps += 1

        ep_loss = running_loss / max(n_steps, 1)
        ep_time = time.time() - ep_start

        # 4d. Periodic validation
        if epoch % epoch_per_test == 0 or epoch == n_epoch:
            val_metrics = evaluate_split(
                model, s["valid_ddi"], vkg_edges, test_batch_size, device
            )
            print(
                f"[repro-em] ep {epoch}/{n_epoch} loss={ep_loss:.4f} "
                f"val_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_kappa={val_metrics['cohen_kappa']:.4f} time={ep_time:.1f}s",
                flush=True,
            )
            scheduler.step(val_metrics["macro_f1"])
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            history.append(
                {
                    "epoch": epoch,
                    "loss": ep_loss,
                    "val": val_metrics,
                    "time_s": ep_time,
                }
            )
        else:
            print(
                f"[repro-em] ep {epoch}/{n_epoch} loss={ep_loss:.4f} time={ep_time:.1f}s",
                flush=True,
            )
            history.append({"epoch": epoch, "loss": ep_loss, "time_s": ep_time})

    # 5. Load best, final test
    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[repro-em] loaded best checkpoint (ep {best_epoch}, val_f1={best_val_f1:.4f})",
            flush=True,
        )

    test_metrics = evaluate_split(
        model, s["test_ddi"], tkg_edges, test_batch_size, device
    )
    print(
        f"[repro-em] FINAL test ({setting}): macro_f1={test_metrics['macro_f1']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} kappa={test_metrics['cohen_kappa']:.4f} "
        f"(n={test_metrics['n_samples']})",
        flush=True,
    )

    return {
        "setting": setting,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test": test_metrics,
        "history": history,
        "config": {
            "n_epoch": n_epoch,
            "n_batch": n_batch,
            "test_batch_size": test_batch_size,
            "epoch_per_test": epoch_per_test,
            "n_dim": n_dim,
            "length": length,
            "lr": lr,
            "weight_decay": weight_decay,
            "shuffle_ratio": shuffle_ratio,
            "seed": seed,
            "n_classes": N_DDI_CLASSES,
            "n_ent": n_ent,
            "n_base_rel": n_base_rel,
        },
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--setting",
        type=str,
        required=True,
        help="One of S0, S1_1, S1_12, S1_123, S1_1234, S1_12345, S2_1, ...",
    )
    p.add_argument("--tag", type=str, default=None)
    # Paper hyperparams: NOT argparse defaults in evaluate.py — those are
    # OVERWRITTEN inside run_model() for S1/S2 (lines 56-62 of upstream
    # DrugBank/evaluate.py). The actual values used to produce Table 1 are
    # the hardcoded overrides below; argparse defaults (lr=0.03, n_batch=512,
    # n_dim=128, lamb=7e-4) are misleading.
    p.add_argument("--n-epoch", type=int, default=100)
    p.add_argument("--n-batch", type=int, default=32)         # paper S1/S2 override
    p.add_argument("--test-batch-size", type=int, default=16)
    p.add_argument("--epoch-per-test", type=int, default=5)
    p.add_argument("--n-dim", type=int, default=64)            # paper S1/S2 override
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.001)          # paper S1/S2 override
    p.add_argument("--weight-decay", type=float, default=1e-8)  # paper S1/S2 lamb override
    p.add_argument("--shuffle-ratio", type=float, default=0.8)
    # Paper evaluate.py:run_model(0): random/np/torch.manual_seed(0).
    # The 5 paper "seeds" (1, 12, ..., 12345) are encoded in setting folder
    # names (S1_1, S1_12, ...), NOT in this RNG seed.
    p.add_argument("--seed", type=int, default=0, help="Python/torch RNG seed (paper uses 0)")
    # Speed knobs: bigger chunk_size → less Python loop overhead, more memory;
    # no-checkpoint → skip gradient checkpointing (~2x faster, more memory).
    p.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Edge chunk size for message passing (raise to speed up if VRAM allows)",
    )
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing (faster, more memory)",
    )
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.tag is None:
        args.tag = f"{args.setting}_e{args.n_epoch}"

    with RunLogger(
        script="emergnn_reproduce", tag=args.tag, seed=args.seed
    ) as rl:
        print(f"[run] config: {vars(args)}", flush=True)
        result = train_one_setting(
            setting=args.setting,
            n_epoch=args.n_epoch,
            n_batch=args.n_batch,
            test_batch_size=args.test_batch_size,
            epoch_per_test=args.epoch_per_test,
            n_dim=args.n_dim,
            length=args.length,
            lr=args.lr,
            weight_decay=args.weight_decay,
            shuffle_ratio=args.shuffle_ratio,
            seed=args.seed,
            device=args.device,
            chunk_size=args.chunk_size,
            use_checkpoint=not args.no_checkpoint,
        )
        out_path = rl.run_dir / "results.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"[run] saved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

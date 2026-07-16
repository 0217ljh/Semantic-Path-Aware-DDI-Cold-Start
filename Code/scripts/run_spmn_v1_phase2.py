"""SPMN v1 — Phase 2: neural coarse head (KG-only, binary).

Trains :class:`SPMNCoarseHead` — learned attention pooling over each type-tau
hyper-edge's members + K-channel routing, fused with the exact Phase-1
structural features — on the SAME 800-drug seed42 split v1.6 used. Target: beat
the Phase-1 GBDT (S2 AUC 0.705, the R13-A1 baseline) and close on / pass v1.6
(0.779).

No EmerGNN backbone, no absolute positional encoding, no fragments (Phase 3),
no pretraining (Phase 4). Binary / symmetric.

Per-pair supports + symmetric structural-feature vectors are precomputed in one
pass over the merged KG and cached (ragged npz). Training reads the cache.

Run (WSL conda env project_1, from project root):
  python Code/scripts/run_spmn_v1_phase2.py --seed 42 --l-max 3 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError("project root not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.coarse_head import SPMNCoarseHead  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, build_pair_support,
)
from my_code.models.spmn_v1.struct_features import (  # noqa: E402
    compute_struct_features, symmetric_binary_dim, symmetric_binary_vector,
)
from my_code.utils.train_progress import TrainProgress  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
CACHE_DIR = ROOT / "Code/data/_cache"


def _binary_frame(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    p = pos[["drug_a_id", "drug_b_id"]].copy()
    p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy()
    n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _precompute_split(kg: MergedKG, frame: pd.DataFrame, l_max: int,
                      tag: str, support_and: bool = True) -> dict:
    """One pass: per-pair support (ragged) + symmetric struct vector + label."""
    n = len(frame)
    struct_dim = symmetric_binary_dim()
    struct = np.zeros((n, struct_dim), dtype=np.float32)
    y = frame["label"].to_numpy().astype(np.float32)
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    med_chunks: list[np.ndarray] = []
    typ_chunks: list[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        ai = kg.id_to_idx.get(a_ids[i])
        bi = kg.id_to_idx.get(b_ids[i])
        if ai is None or bi is None or ai == bi:
            continue
        sup = build_pair_support(kg, ai, bi, l_max=l_max, support_and=support_and)
        feat = compute_struct_features(kg, sup, with_copath=True)
        struct[i] = symmetric_binary_vector(feat)
        if sup.n_support:
            med_chunks.append(sup.global_idx.astype(np.int64))
            typ_chunks.append(sup.type_id.astype(np.int64))
            counts[i] = sup.n_support
        if (i + 1) % 20000 == 0:
            print(f"  [{tag}] {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    med = np.concatenate(med_chunks) if med_chunks else np.zeros(0, np.int64)
    typ = np.concatenate(typ_chunks) if typ_chunks else np.zeros(0, np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"  [{tag}] done {n} pairs in {time.time() - t0:.0f}s "
          f"(mediator slots={med.shape[0]})", flush=True)
    return {"struct": struct, "y": y, "med": med, "typ": typ, "offsets": offsets}


def _load_or_build(kg_factory, splits: dict, l_max: int, seed: int,
                   no_cache: bool, support_and: bool = True) -> dict:
    mode = "v3and" if support_and else "v2sum"
    cache = CACHE_DIR / f"spmn_v1_phase2_supports_{mode}_seed{seed}_lmax{l_max}.npz"
    if cache.is_file() and not no_cache:
        print(f"[phase2] support cache HIT: {cache}", flush=True)
        z = np.load(cache)
        out = {}
        for name in splits:
            out[name] = {k: z[f"{name}__{k}"] for k in
                         ("struct", "y", "med", "typ", "offsets")}
        return out
    print(f"[phase2] building supports (support_and={support_and}) ...", flush=True)
    kg = kg_factory()
    out = {name: _precompute_split(kg, frame, l_max, name, support_and=support_and)
           for name, frame in splits.items()}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    flat = {f"{name}__{k}": v for name, d in out.items() for k, v in d.items()}
    np.savez_compressed(cache, **flat)
    print(f"[phase2] support cache saved: {cache}", flush=True)
    return out


def _gather_batch(split: dict, idx: np.ndarray, device: torch.device):
    """Build flattened mediator tensors + struct + labels for a minibatch."""
    offsets = split["offsets"]
    med, typ = split["med"], split["typ"]
    med_list, typ_list, pair_list = [], [], []
    for local, p in enumerate(idx):
        s, e = offsets[p], offsets[p + 1]
        if e > s:
            med_list.append(med[s:e])
            typ_list.append(typ[s:e])
            pair_list.append(np.full(e - s, local, dtype=np.int64))
    if med_list:
        med_t = torch.as_tensor(np.concatenate(med_list), device=device)
        typ_t = torch.as_tensor(np.concatenate(typ_list), device=device)
        pair_t = torch.as_tensor(np.concatenate(pair_list), device=device)
    else:
        med_t = torch.zeros(0, dtype=torch.long, device=device)
        typ_t = torch.zeros(0, dtype=torch.long, device=device)
        pair_t = torch.zeros(0, dtype=torch.long, device=device)
    struct_t = torch.as_tensor(split["struct"][idx], device=device)
    y_t = torch.as_tensor(split["y"][idx], device=device)
    return med_t, pair_t, typ_t, struct_t, y_t


@torch.no_grad()
def _evaluate(model, split: dict, device, batch: int) -> dict:
    model.eval()
    n = len(split["y"])
    probs = np.zeros(n, dtype=np.float64)
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        med_t, pair_t, typ_t, struct_t, _ = _gather_batch(split, idx, device)
        logit = model(med_t, pair_t, typ_t, struct_t, len(idx))
        probs[idx] = torch.sigmoid(logit).cpu().numpy()
    y = split["y"]
    return {"auc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--log-step-every", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--support-sum", action="store_true",
                    help="use the (buggy) sum-budget support instead of the "
                         "corrected AND intersection A_tau^(l)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pkl = PKL_DIR / f"seed{args.seed}.pkl"
    if not pkl.is_file():
        raise FileNotFoundError(pkl)
    print(f"[phase2] dataset: {pkl}  device: {device}", flush=True)
    ds = PairDataset.from_pkl(str(pkl))
    splits = {
        "train": _binary_frame(ds.splits.train, ds.get_train_negatives()),
        "val_s2": _binary_frame(ds.splits.val_s2, ds.get_negatives("val_s2")),
        "test_s2": _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2")),
    }
    print(f"[phase2] train={len(splits['train'])} "
          f"val_s2={len(splits['val_s2'])} test_s2={len(splits['test_s2'])}",
          flush=True)

    data = _load_or_build(lambda: MergedKG.from_parquet(), splits,
                          args.l_max, args.seed, args.no_cache,
                          support_and=not args.support_sum)

    # Standardize structural features on TRAIN stats only (raw counts reach
    # 10s-100s and would swamp the ~O(1) routed channels). Applied to all
    # splits with the train mean/std.
    mu = data["train"]["struct"].mean(axis=0, keepdims=True)
    sigma = data["train"]["struct"].std(axis=0, keepdims=True) + 1e-6
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sigma).astype(np.float32)

    model = SPMNCoarseHead(
        n_entities=178029, n_types=N_TYPES, struct_dim=symmetric_binary_dim(),
        d=args.d, hidden=args.hidden, dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train = data["train"]
    n_train = len(train["y"])
    n_steps = (n_train + args.batch - 1) // args.batch
    prog = TrainProgress(args.epochs, log_step_every=args.log_step_every,
                         total_steps_per_epoch=n_steps, prefix="[spmn-v1-p2] ")

    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            med_t, pair_t, typ_t, struct_t, y_t = _gather_batch(train, idx, device)
            logit = model(med_t, pair_t, typ_t, struct_t, len(idx))
            loss = loss_fn(logit, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            prog.step(loss.item())
        val = _evaluate(model, data["val_s2"], device, args.batch)
        prog.log_eval(val, scope="epoch")
        prog.epoch_end(extra={"val_auc": val["auc"]})
        if val["auc"] > best_auc:
            best_auc = val["auc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, data["val_s2"], device, args.batch)
    test = _evaluate(model, data["test_s2"], device, args.batch)

    print("\n=== SPMN v1 Phase-2 (neural coarse head, KG-only) ===", flush=True)
    print(f"d={args.d} hidden={args.hidden} epochs={args.epochs} "
          f"l_max={args.l_max}", flush=True)
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f}",
          flush=True)
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f}",
          flush=True)
    print("  references: Phase-1 GBDT 0.705 (A1) | v1.6 0.779", flush=True)

    res_dir = ROOT / "Code/runs/spmn_v1_phase2"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"seed{args.seed}_lmax{args.l_max}_d{args.d}.json"
    res_path.write_text(json.dumps({
        "seed": args.seed, "l_max": args.l_max, "d": args.d,
        "hidden": args.hidden, "epochs": args.epochs, "batch": args.batch,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "val_s2": val, "test_s2": test,
        "reference": {"phase1_gbdt": 0.705, "v1_6": 0.779},
    }, indent=2))
    print(f"[phase2] results -> {res_path}", flush=True)


if __name__ == "__main__":
    main()

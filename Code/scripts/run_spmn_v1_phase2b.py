"""SPMN v1 — Phase 2b: unified incidence head (affinity routing + a/b separation).

Tests whether v1.6's routing form — per-drug MARGINAL affinity gating the joint
hyper-edge channels (Step 1) + a/b-separated readout (Step 3) — closes the gap
to v1.6 (0.779). Reuses the Phase-2 v2aa support cache (mediators + struct) and
adds per-drug affinity computed from the merged KG. KG-only, binary, no
fragments / pretraining yet.

Run: python Code/scripts/run_spmn_v1_phase2b.py --seed 42 --l-max 3 --epochs 80 \
        --weight-decay 1e-2 --dropout 0.5 --d 32
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
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.incidence_head import SPMNIncidenceHead  # noqa: E402
from my_code.models.spmn_v1.retrieval import MergedKG, N_TYPES  # noqa: E402
from my_code.models.spmn_v1.struct_features import symmetric_binary_dim  # noqa: E402
from my_code.utils.train_progress import TrainProgress  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
CACHE_DIR = ROOT / "Code/data/_cache"


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _affinity_for_frame(frame, aff, dim):
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    z = np.zeros(dim, dtype=np.float32)
    Aa = np.stack([aff.get(d, z) for d in a_ids]).astype(np.float32)
    Ab = np.stack([aff.get(d, z) for d in b_ids]).astype(np.float32)
    return Aa, Ab


def _gather_batch(split, idx, device):
    offsets, med, typ = split["offsets"], split["med"], split["typ"]
    med_list, typ_list, pair_list = [], [], []
    for local, p in enumerate(idx):
        s, e = offsets[p], offsets[p + 1]
        if e > s:
            med_list.append(med[s:e]); typ_list.append(typ[s:e])
            pair_list.append(np.full(e - s, local, dtype=np.int64))
    if med_list:
        med_t = torch.as_tensor(np.concatenate(med_list), device=device)
        typ_t = torch.as_tensor(np.concatenate(typ_list), device=device)
        pair_t = torch.as_tensor(np.concatenate(pair_list), device=device)
    else:
        med_t = torch.zeros(0, dtype=torch.long, device=device)
        typ_t = torch.zeros(0, dtype=torch.long, device=device)
        pair_t = torch.zeros(0, dtype=torch.long, device=device)
    aff_a = torch.as_tensor(split["aff_a"][idx], device=device)
    aff_b = torch.as_tensor(split["aff_b"][idx], device=device)
    struct = torch.as_tensor(split["struct"][idx], device=device)
    y = torch.as_tensor(split["y"][idx], device=device)
    return med_t, pair_t, typ_t, aff_a, aff_b, struct, y


@torch.no_grad()
def _evaluate(model, split, device, batch):
    model.eval()
    n = len(split["y"])
    probs = np.zeros(n)
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        med_t, pair_t, typ_t, aff_a, aff_b, struct, _ = _gather_batch(split, idx, device)
        logit = model(med_t, pair_t, typ_t, aff_a, aff_b, struct, len(idx))
        probs[idx] = torch.sigmoid(logit).cpu().numpy()
    y = split["y"]
    return {"auc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--log-step-every", type=int, default=100)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pkl = PKL_DIR / f"seed{args.seed}.pkl"
    ds = PairDataset.from_pkl(str(pkl))
    frames = {
        "train": _binary_frame(ds.splits.train, ds.get_train_negatives()),
        "val_s2": _binary_frame(ds.splits.val_s2, ds.get_negatives("val_s2")),
        "test_s2": _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2")),
    }

    # Correct AND support (= A_tau^(l) intersection). Built by run_spmn_v1_phase2.py.
    cache = CACHE_DIR / f"spmn_v1_phase2_supports_v3and_seed{args.seed}_lmax{args.l_max}.npz"
    if not cache.is_file():
        raise FileNotFoundError(f"{cache} (run run_spmn_v1_phase2.py first to build it)")
    z = np.load(cache)
    data = {name: {k: z[f"{name}__{k}"] for k in
                   ("struct", "y", "med", "typ", "offsets")} for name in frames}

    # per-drug marginal affinity from the merged KG (cheap: 799 cached BFS)
    print("[phase2b] computing per-drug affinity ...", flush=True)
    kg = MergedKG.from_parquet()
    depth = max(0, args.l_max - 1)
    drug_ids = set()
    for fr in frames.values():
        drug_ids |= set(fr["drug_a_id"].astype(str)) | set(fr["drug_b_id"].astype(str))
    aff = {d: kg.marginal_affinity(kg.id_to_idx[d], depth)
           for d in drug_ids if d in kg.id_to_idx}
    for name, fr in frames.items():
        Aa, Ab = _affinity_for_frame(fr, aff, N_TYPES)
        data[name]["aff_a"] = Aa
        data[name]["aff_b"] = Ab

    # standardize struct on train stats (raw counts swamp routed channels)
    mu = data["train"]["struct"].mean(0, keepdims=True)
    sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sd).astype(np.float32)

    model = SPMNIncidenceHead(
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
                         total_steps_per_epoch=n_steps, prefix="[spmn-v1-p2b] ")

    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train(); prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            med_t, pair_t, typ_t, aff_a, aff_b, struct, y = _gather_batch(train, idx, device)
            logit = model(med_t, pair_t, typ_t, aff_a, aff_b, struct, len(idx))
            loss = loss_fn(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            prog.step(loss.item())
        val = _evaluate(model, data["val_s2"], device, args.batch)
        prog.log_eval(val, scope="epoch"); prog.epoch_end(extra={"val_auc": val["auc"]})
        if val["auc"] > best_auc:
            best_auc = val["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, data["val_s2"], device, args.batch)
    test = _evaluate(model, data["test_s2"], device, args.batch)

    print("\n=== SPMN v1 Phase-2b (incidence head: affinity routing + a/b sep) ===")
    print(f"d={args.d} wd={args.weight_decay} dropout={args.dropout} epochs={args.epochs}")
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f}")
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f}")
    print("  references: Phase-2 simple pool 0.734 | v1.6 0.779")

    res_dir = ROOT / "Code/runs/spmn_v1_phase2b"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"seed{args.seed}_lmax{args.l_max}_d{args.d}.json").write_text(json.dumps({
        "seed": args.seed, "l_max": args.l_max, "d": args.d, "epochs": args.epochs,
        "weight_decay": args.weight_decay, "dropout": args.dropout,
        "val_s2": val, "test_s2": test,
        "reference": {"phase2_simple": 0.734, "v1_6": 0.779},
    }, indent=2))
    print(f"[phase2b] results saved")


if __name__ == "__main__":
    main()

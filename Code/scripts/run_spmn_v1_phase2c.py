"""SPMN v1 — Phase 2c: relation-enriched head (the clean Phase-2 design).

AND support + attention pool + RELATION-enriched mediators (phi(m) includes
rel(a->m), rel(m->b)) + explicit struct features. Affinity routing / a/b
separation dropped (shown redundant). Target: beat the no-relation simple pool
(0.772) and v1.6 (env-comparable 0.7757).

Run: python Code/scripts/run_spmn_v1_phase2c.py --seed 42 --l-max 3 --epochs 80 \
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
from my_code.models.spmn_v1.rel_head import SPMNRelHead  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_REL_BUCKETS, N_TYPES, build_pair_support,
)
from my_code.models.spmn_v1.struct_features import (  # noqa: E402
    compute_struct_features, symmetric_binary_dim, symmetric_binary_vector,
)
from my_code.utils.train_progress import TrainProgress  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
CACHE_DIR = ROOT / "Code/data/_cache"


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _precompute(kg, frame, l_max, tag):
    n = len(frame)
    struct = np.zeros((n, symmetric_binary_dim()), dtype=np.float32)
    y = frame["label"].to_numpy().astype(np.float32)
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    med_c, typ_c, rela_c, relb_c = [], [], [], []
    counts = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        ai = kg.id_to_idx.get(a_ids[i]); bi = kg.id_to_idx.get(b_ids[i])
        if ai is None or bi is None or ai == bi:
            continue
        sup = build_pair_support(kg, ai, bi, l_max=l_max, support_and=True)
        struct[i] = symmetric_binary_vector(compute_struct_features(kg, sup, with_copath=True))
        if sup.n_support:
            ra, rb = kg.support_relations(sup)
            med_c.append(sup.global_idx.astype(np.int64))
            typ_c.append(sup.type_id.astype(np.int64))
            rela_c.append(ra); relb_c.append(rb)
            counts[i] = sup.n_support
        if (i + 1) % 20000 == 0:
            print(f"  [{tag}] {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    print(f"  [{tag}] done {n} in {time.time()-t0:.0f}s", flush=True)
    return {"struct": struct, "y": y, "med": cat(med_c), "typ": cat(typ_c),
            "rela": cat(rela_c), "relb": cat(relb_c), "offsets": offsets}


def _gather(split, idx, device):
    off, med, typ, rela, relb = (split["offsets"], split["med"], split["typ"],
                                 split["rela"], split["relb"])
    mL, tL, raL, rbL, pL = [], [], [], [], []
    for local, p in enumerate(idx):
        s, e = off[p], off[p + 1]
        if e > s:
            mL.append(med[s:e]); tL.append(typ[s:e])
            raL.append(rela[s:e]); rbL.append(relb[s:e])
            pL.append(np.full(e - s, local, dtype=np.int64))
    t = lambda L: torch.as_tensor(np.concatenate(L), device=device) if L \
        else torch.zeros(0, dtype=torch.long, device=device)
    struct = torch.as_tensor(split["struct"][idx], device=device)
    y = torch.as_tensor(split["y"][idx], device=device)
    return t(mL), t(pL), t(tL), t(raL), t(rbL), struct, y


@torch.no_grad()
def _evaluate(model, split, device, batch):
    model.eval()
    n = len(split["y"]); probs = np.zeros(n)
    for s in range(0, n, batch):
        idx = np.arange(s, min(s + batch, n))
        med, pair, typ, ra, rb, struct, _ = _gather(split, idx, device)
        probs[idx] = torch.sigmoid(model(med, pair, typ, ra, rb, struct, len(idx))).cpu().numpy()
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
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    frames = {"train": _binary_frame(ds.splits.train, ds.get_train_negatives()),
              "val_s2": _binary_frame(ds.splits.val_s2, ds.get_negatives("val_s2")),
              "test_s2": _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))}

    cache = CACHE_DIR / f"spmn_v1_phase2_supports_v4rel_seed{args.seed}_lmax{args.l_max}.npz"
    keys = ("struct", "y", "med", "typ", "rela", "relb", "offsets")
    if cache.is_file() and not args.no_cache:
        print(f"[phase2c] cache HIT: {cache}", flush=True)
        z = np.load(cache)
        data = {name: {k: z[f"{name}__{k}"] for k in keys} for name in frames}
    else:
        print("[phase2c] building supports+relations ...", flush=True)
        kg = MergedKG.from_parquet()
        data = {name: _precompute(kg, fr, args.l_max, name) for name, fr in frames.items()}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{f"{name}__{k}": v for name, d in data.items()
                                      for k, v in d.items()})
        print(f"[phase2c] cache saved: {cache}", flush=True)

    mu = data["train"]["struct"].mean(0, keepdims=True)
    sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sd).astype(np.float32)

    model = SPMNRelHead(n_entities=178029, n_types=N_TYPES,
                        n_rel_buckets=N_REL_BUCKETS, struct_dim=symmetric_binary_dim(),
                        d=args.d, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train = data["train"]; n_train = len(train["y"])
    n_steps = (n_train + args.batch - 1) // args.batch
    prog = TrainProgress(args.epochs, log_step_every=100,
                         total_steps_per_epoch=n_steps, prefix="[spmn-v1-p2c] ")
    best_auc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train(); prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            med, pair, typ, ra, rb, struct, y = _gather(train, idx, device)
            loss = loss_fn(model(med, pair, typ, ra, rb, struct, len(idx)), y)
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
    print("\n=== SPMN v1 Phase-2c (relation-enriched head) ===")
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f}")
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f}")
    print("  refs: no-rel pool 0.7720 | v1.6 repro 0.7757")
    res_dir = ROOT / "Code/runs/spmn_v1_phase2c"; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"seed{args.seed}_lmax{args.l_max}_d{args.d}.json").write_text(json.dumps({
        "seed": args.seed, "l_max": args.l_max, "d": args.d, "epochs": args.epochs,
        "val_s2": val, "test_s2": test,
        "reference": {"no_rel_pool": 0.7720, "v1_6_repro": 0.7757}}, indent=2))


if __name__ == "__main__":
    main()

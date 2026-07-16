"""SPMN v1 — Phase 3: multi-modal (KG + molecular FragAlign), finetune-only.

Tests whether the molecule↔KG bridge gives gain over the KG-only Phase-2
(test_s2 AUC 0.7794). This is the finetune-only version (no pretraining yet):
g_frag + bridge are trained end-to-end on DDI. If gain is weak/absent, the next
step is the masked mechanism-completion PRETRAINING that grounds w(v) (codex:
DDI alone weakly grounds the bridge). Per the lead: molecular MUST be added;
iterate methods until it gives some gain.

Correctness-first per-pair forward (slow; optimise after gain confirmed).

Run: python Code/scripts/run_spmn_v1_phase3.py --seed 42 --epochs 30
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score, f1_score, roc_auc_score,
)

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.frag_encoder import build_drug_fragment_graph_cache  # noqa: E402
from my_code.models.spmn_v1.frag_gnn import FragGNN  # noqa: E402
from my_code.models.spmn_v1.mm_head import SPMNMolHead  # noqa: E402
from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_REL_BUCKETS, N_TYPES, build_pair_support,
)
from my_code.models.spmn_v1.struct_features import (  # noqa: E402
    compute_copath_pairs, compute_struct_features, symmetric_binary_dim,
    symmetric_binary_vector,
)
from my_code.utils.train_progress import TrainProgress  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
SMILES_CSV = PKL_DIR / "drug_smiles__seed42.csv"
CACHE_DIR = ROOT / "Code/data/_cache"
D_FRAG = 64


def _binary_frame(pos, neg):
    p = pos[["drug_a_id", "drug_b_id"]].copy(); p["label"] = 1
    n = neg[["drug_a_id", "drug_b_id"]].copy(); n["label"] = 0
    return pd.concat([p, n], ignore_index=True)


def _precompute_pairs(kg, frame, l_supp, l_copath, tag):
    """Per-pair ragged arrays: support (med/typ/rela/relb), struct, copath
    (iu/iv), kappa, y. Drug ids kept for fragment lookup."""
    n = len(frame)
    struct = np.zeros((n, symmetric_binary_dim()), dtype=np.float32)
    y = frame["label"].to_numpy().astype(np.float32)
    a_ids = frame["drug_a_id"].astype(str).to_numpy()
    b_ids = frame["drug_b_id"].astype(str).to_numpy()
    med_c, typ_c, rela_c, relb_c, iu_c, iv_c = [], [], [], [], [], []
    s_cnt = np.zeros(n, dtype=np.int64)
    p_cnt = np.zeros(n, dtype=np.int64)
    kappa = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        ai = kg.id_to_idx.get(a_ids[i]); bi = kg.id_to_idx.get(b_ids[i])
        if ai is None or bi is None or ai == bi:
            continue
        sup = build_pair_support(kg, ai, bi, l_max=l_supp, support_and=True)
        struct[i] = symmetric_binary_vector(compute_struct_features(kg, sup, with_copath=True))
        if sup.n_support:
            ra, rb = kg.support_relations(sup)
            med_c.append(sup.global_idx.astype(np.int64)); typ_c.append(sup.type_id.astype(np.int64))
            rela_c.append(ra); relb_c.append(rb); s_cnt[i] = sup.n_support
            kappa[i] = 1.0
            # co-path over the SAME support, larger walk budget l_copath (endpoints
            # stay in the support so they carry bridge weights). Local idx into sup.
            iu_cp, iv_cp = compute_copath_pairs(kg, sup, l_copath=l_copath)
            iu_c.append(iu_cp); iv_c.append(iv_cp); p_cnt[i] = int(iu_cp.size)
        if (i + 1) % 2000 == 0 or (i + 1) == n:
            el = time.time() - t0
            eta = el / (i + 1) * (n - (i + 1))
            print(f"  [{tag}] {i+1}/{n}  {100*(i+1)/n:.0f}%  {el:.0f}s  eta {eta:.0f}s",
                  flush=True)
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    s_off = np.zeros(n + 1, dtype=np.int64); np.cumsum(s_cnt, out=s_off[1:])
    p_off = np.zeros(n + 1, dtype=np.int64); np.cumsum(p_cnt, out=p_off[1:])
    print(f"  [{tag}] done {n} in {time.time()-t0:.0f}s", flush=True)
    return {"struct": struct, "y": y, "med": cat(med_c), "typ": cat(typ_c),
            "rela": cat(rela_c), "relb": cat(relb_c), "s_off": s_off,
            "iu": cat(iu_c), "iv": cat(iv_c), "p_off": p_off, "kappa": kappa,
            "a_ids": a_ids, "b_ids": b_ids}


def _build_global_frag_graph(frag_cache, drug_ids, device):
    """Concatenate all drugs' fragment atom-subgraphs into ONE batched graph +
    a drug_id -> (frag_start, frag_end) map. One g_frag pass/step encodes all."""
    xs, eis, eas, atom_batch = [], [], [], []
    drug_range: dict[str, tuple[int, int]] = {}
    g_off = 0   # global fragment index
    a_off = 0   # global atom offset
    for did in drug_ids:
        frags = frag_cache.get(did, [])
        start = g_off
        for fg in frags:
            xs.append(fg.x)
            if fg.edge_index.shape[1] > 0:
                eis.append(fg.edge_index + a_off); eas.append(fg.edge_attr)
            atom_batch.append(np.full(fg.x.shape[0], g_off, dtype=np.int64))
            a_off += fg.x.shape[0]; g_off += 1
        drug_range[did] = (start, g_off)
    from my_code.models.spmn_v1.frag_encoder import ATOM_FEAT_DIM, BOND_FEAT_DIM
    x = torch.as_tensor(np.concatenate(xs), device=device) if xs else torch.zeros(0, ATOM_FEAT_DIM, device=device)
    edge_index = torch.as_tensor(np.concatenate(eis, axis=1), device=device) if eis else torch.zeros(2, 0, dtype=torch.long, device=device)
    edge_attr = torch.as_tensor(np.concatenate(eas, axis=0), device=device) if eas else torch.zeros(0, BOND_FEAT_DIM, device=device)
    batch = torch.as_tensor(np.concatenate(atom_batch), device=device) if atom_batch else torch.zeros(0, dtype=torch.long, device=device)
    return (x, edge_index, edge_attr, batch, g_off), drug_range


def _to_gpu(split, device):
    """Move the flattened split arrays to GPU tensors once (vectorised gather)."""
    g = {}
    for k in ("med", "typ", "rela", "relb", "iu", "iv"):
        g[k] = torch.as_tensor(split[k].astype(np.int64), device=device)
    g["struct"] = torch.as_tensor(split["struct"], device=device)
    g["kappa"] = torch.as_tensor(split["kappa"], device=device)
    g["y"] = torch.as_tensor(split["y"], device=device)
    return g


def _gather_batch(split, gpu, idx, z_all, drug_range, device):
    """Build flattened batched tensors for forward_batch. idx: (B,) pair indices."""
    s_off, p_off = split["s_off"], split["p_off"]
    a_ids, b_ids = split["a_ids"], split["b_ids"]
    B = len(idx)
    # support positions in the flattened split array + batch-local offsets
    sup_pos, p_pos = [], []
    E_batch, P_batch = np.zeros(B, np.int64), np.zeros(B, np.int64)
    for bi, i in enumerate(idx):
        sup_pos.append(np.arange(s_off[i], s_off[i + 1]))
        p_pos.append(np.arange(p_off[i], p_off[i + 1]))
        E_batch[bi] = s_off[i + 1] - s_off[i]
        P_batch[bi] = p_off[i + 1] - p_off[i]
    sup_pos = np.concatenate(sup_pos) if len(sup_pos) else np.zeros(0, np.int64)
    p_pos = np.concatenate(p_pos) if len(p_pos) else np.zeros(0, np.int64)
    b_off = np.cumsum(E_batch) - E_batch                       # batch-flattened support offset
    sup_pos_t = torch.as_tensor(sup_pos, device=device)
    med = gpu["med"][sup_pos_t]; typ = gpu["typ"][sup_pos_t]
    rel_a = gpu["rela"][sup_pos_t]; rel_b = gpu["relb"][sup_pos_t]
    pair_idx = torch.as_tensor(np.repeat(np.arange(B), E_batch), device=device)
    # co-path: local idx -> batch-global by adding the pair's b_off
    cp_pair = np.repeat(np.arange(B), P_batch)
    p_pos_t = torch.as_tensor(p_pos, device=device)
    add = torch.as_tensor(b_off[cp_pair], device=device)
    iu_g = gpu["iu"][p_pos_t] + add if p_pos.size else torch.zeros(0, dtype=torch.long, device=device)
    iv_g = gpu["iv"][p_pos_t] + add if p_pos.size else torch.zeros(0, dtype=torch.long, device=device)
    # fragments per drug (slice the per-step z_all)
    fa_list, fb_list, fa_pair, fb_pair = [], [], [], []
    for bi, i in enumerate(idx):
        ra = drug_range.get(a_ids[i], (0, 0)); rb = drug_range.get(b_ids[i], (0, 0))
        za = z_all[ra[0]:ra[1]]; zb = z_all[rb[0]:rb[1]]
        fa_list.append(za); fb_list.append(zb)
        fa_pair.append(torch.full((za.size(0),), bi, dtype=torch.long, device=device))
        fb_pair.append(torch.full((zb.size(0),), bi, dtype=torch.long, device=device))
    df = z_all.size(1)
    fa_z = torch.cat(fa_list) if fa_list else torch.zeros(0, df, device=device)
    fb_z = torch.cat(fb_list) if fb_list else torch.zeros(0, df, device=device)
    fa_pair = torch.cat(fa_pair) if fa_pair else torch.zeros(0, dtype=torch.long, device=device)
    fb_pair = torch.cat(fb_pair) if fb_pair else torch.zeros(0, dtype=torch.long, device=device)
    idx_t = torch.as_tensor(idx, device=device)
    return dict(med=med, pair_idx=pair_idx, typ=typ, rel_a=rel_a, rel_b=rel_b, n_pairs=B,
                fa_z=fa_z, fa_pair=fa_pair, fb_z=fb_z, fb_pair=fb_pair,
                iu_g=iu_g, iv_g=iv_g, struct=gpu["struct"][idx_t], kappa=gpu["kappa"][idx_t])


def _evaluate(model, gfrag, glob_graph, drug_range, split, gpu, device, batch=2048):
    model.eval(); gfrag.eval()
    with torch.no_grad():
        z_all = gfrag(*glob_graph)
        n = len(split["y"]); probs = np.zeros(n)
        for s in range(0, n, batch):
            idx = np.arange(s, min(s + batch, n))
            b = _gather_batch(split, gpu, idx, z_all, drug_range, device)
            probs[idx] = torch.sigmoid(model.forward_batch(**b)).cpu().numpy()
    y = split["y"]
    return {"auc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs)),
            "f1": float(f1_score(y, (probs >= 0.5).astype(int), zero_division=0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-supp", type=int, default=3)
    ap.add_argument("--l-copath", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--ablate", choices=["none", "no_f", "no_bridge"], default="none",
                    help="no_f = drop fragment fallback (bridge-only); "
                         "no_bridge = drop bridge F2/F3 (fragment-fallback-only)")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--debug-n", type=int, default=0,
                    help="subsample each split to N pairs (pipeline smoke test)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    frames = {"train": _binary_frame(ds.splits.train, ds.get_train_negatives()),
              "val_s2": _binary_frame(ds.splits.val_s2, ds.get_negatives("val_s2")),
              "test_s2": _binary_frame(ds.splits.test_s2, ds.get_negatives("test_s2"))}
    if args.debug_n:
        frames = {nm: fr.sample(min(args.debug_n, len(fr)), random_state=args.seed).reset_index(drop=True)
                  for nm, fr in frames.items()}
        args.no_cache = True  # don't pollute the real cache with a subsample

    cache = CACHE_DIR / f"spmn_v1_phase3_v5mol_seed{args.seed}_ls{args.l_supp}_lc{args.l_copath}.npz"
    keys = ("struct", "y", "med", "typ", "rela", "relb", "s_off", "iu", "iv",
            "p_off", "kappa", "a_ids", "b_ids")
    if cache.is_file() and not args.no_cache:
        print(f"[phase3] cache HIT: {cache}", flush=True)
        z = np.load(cache, allow_pickle=True)
        data = {nm: {k: z[f"{nm}__{k}"] for k in keys} for nm in frames}
        kg = MergedKG.from_parquet()
    else:
        print("[phase3] precomputing pairs ...", flush=True)
        kg = MergedKG.from_parquet()
        data = {nm: _precompute_pairs(kg, fr, args.l_supp, args.l_copath, nm) for nm, fr in frames.items()}
        if not args.debug_n:   # don't overwrite the real cache with a subsample
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{f"{nm}__{k}": v for nm, d in data.items() for k, v in d.items()})
            print(f"[phase3] cache saved: {cache}", flush=True)

    # standardize struct on train
    mu = data["train"]["struct"].mean(0, keepdims=True); sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    for nm in data:
        data[nm]["struct"] = ((data[nm]["struct"] - mu) / sd).astype(np.float32)

    print("[phase3] building fragment graphs ...", flush=True)
    frag_cache = build_drug_fragment_graph_cache(SMILES_CSV)
    drug_ids = sorted({d for nm in frames for col in ("drug_a_id", "drug_b_id")
                       for d in frames[nm][col].astype(str)})
    glob_graph, drug_range = _build_global_frag_graph(frag_cache, drug_ids, device)
    print(f"[phase3] global frag graph: {glob_graph[4]} fragments", flush=True)

    gfrag = FragGNN(d=D_FRAG).to(device)
    model = SPMNMolHead(n_entities=kg.n_nodes, n_types=N_TYPES, n_rel_buckets=N_REL_BUCKETS,
                        struct_dim=symmetric_binary_dim(), d_frag=D_FRAG, d=args.d,
                        dropout=args.dropout,
                        use_frag_fallback=(args.ablate != "no_f"),
                        use_bridge=(args.ablate != "no_bridge")).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(gfrag.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    gpu = {nm: _to_gpu(data[nm], device) for nm in frames}
    train = data["train"]; n_train = len(train["y"])
    prog = TrainProgress(args.epochs, log_step_every=50,
                         total_steps_per_epoch=(n_train + args.batch - 1) // args.batch,
                         prefix="[spmn-v1-p3] ")
    best_auc, best = -1.0, None
    for epoch in range(args.epochs):
        model.train(); gfrag.train(); prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            z_all = gfrag(*glob_graph)            # encode all drugs' fragments this step
            b = _gather_batch(train, gpu["train"], idx, z_all, drug_range, device)
            logits = model.forward_batch(**b)
            loss = loss_fn(logits, gpu["train"]["y"][torch.as_tensor(idx, device=device)])
            opt.zero_grad(); loss.backward(); opt.step()
            prog.step(loss.item())
        val = _evaluate(model, gfrag, glob_graph, drug_range, data["val_s2"], gpu["val_s2"], device)
        prog.log_eval(val, scope="epoch"); prog.epoch_end(extra={"val_auc": val["auc"]})
        if val["auc"] > best_auc:
            best_auc = val["auc"]
            best = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in gfrag.state_dict().items()})
    if best is not None:
        model.load_state_dict(best[0]); gfrag.load_state_dict(best[1])
    val = _evaluate(model, gfrag, glob_graph, drug_range, data["val_s2"], gpu["val_s2"], device)
    test = _evaluate(model, gfrag, glob_graph, drug_range, data["test_s2"], gpu["test_s2"], device)
    print("\n=== SPMN v1 Phase-3 (multi-modal, finetune-only) ===")
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f} F1@0.5={val['f1']:.4f}")
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f} F1@0.5={test['f1']:.4f}")
    print("  ref: KG-only phase2 0.7794 | v1.6 0.7757")
    res_dir = ROOT / "Code/runs/spmn_v1_phase3"; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"seed{args.seed}_ftonly.json").write_text(json.dumps(
        {"seed": args.seed, "val_s2": val, "test_s2": test,
         "reference": {"phase2_kg_only": 0.7794, "v1_6": 0.7757}}, indent=2))


if __name__ == "__main__":
    main()

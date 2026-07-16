"""A1 pilot (EmerGNN / PATH paradigm, RSPMM backend): full-merged-KG variant of
train_emergnn_rank_pilot.py. The pure-PyTorch chunk backend in that sibling is
far too slow for the full KG (per-pair propagation over 14.2M edges); this uses
EmerGNN's torchdrug ``generalized_rspmm`` CUDA kernel (EmerGNN_RSPMM), which the
baseline runs at ~6.4 min/epoch on the full merged KG for ddi800 S2.

Design (codex-reviewed 2026-07-04, thread 019f2ef3):
- SAME knowledge-only structural seed as the chunk pilot: 13-d KG-structural
  entity features (type-onehot + log-degree) fed through feat='M', ZERO change to
  the faithful model, no molecular info, cold-start-safe, comparable to R-GCN.
  The path mechanism (relation-gated bidirectional propagation) is EmerGNN's; the
  seed swap makes it "EmerGNN architecture, knowledge-only seed" (a path-paradigm
  representative for the grid, NOT the faithful Morgan baseline). A Morgan-seed
  sensitivity run is a planned follow-up (feat channel), and het:CrC-masking is a
  further sensitivity — both keep the merged KG unmasked in this PRIMARY run.
- FULL merged KG (EMERGNN_KG_SCOPE=full), so it is directly comparable to the
  R-GCN node pilot's full-KG run in BOTH feature policy (structural) and KG scope
  — resolving the "R-GCN saw 64x more edges" unfairness of the drug_incident run.
- The rspmm model reproduces the ORIGINAL EmerGNN message flow (out[row] += rel *
  in[col]); the sparse KG is built from the cache's (h,t,r) triplets via
  build_sparse_kg_from_triplets (adds reverse + self-loop, all_rel=2*n_base_rel+1).
- Pre-scorer rep = enc_ht(head,tail,kg) = cat([head_hid, tail_hid]) (2*n_dim),
  BEFORE Wr. Saved with the standard pilot_Z.npz schema (Z_==raw_==embed, no
  bottleneck) so analyze_rank_sufficiency.py runs unchanged.
- Data protocol + structural-feature build + entity mapping are imported verbatim
  from train_emergnn_rank_pilot (identical to the chunk pilot).
- Training: mini-batch, step-budget with warm-val early stop (avoid undertraining).

Run (from project root, WSL conda env project_1):
  python Code/scripts/train_emergnn_rspmm_rank_pilot.py --fold fold0 --kg-scope full \
      --steps 30000 --batch-size 32 --eval-every 1000 --patience 5000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH  # noqa: E402
from baseline.emergnn.model_rspmm import EmerGNN_RSPMM  # noqa: E402
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets  # noqa: E402
from baseline.emergnn._shared import ensure_kg_setup_cache  # noqa: E402
# reuse the chunk pilot's helpers verbatim (identical data/feature/mapping logic)
from train_emergnn_rank_pilot import (  # noqa: E402
    _MiniTrain, build_structural_features, _pairs_to_ent)
from train_gcn_rank_pilot import load_data  # noqa: E402


@torch.no_grad()
def _pair_embed(model: EmerGNN_RSPMM, head: torch.Tensor, tail: torch.Tensor,
                kg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-scorer rep + logit for a batch (rspmm backend)."""
    embed = model.enc_ht(head, tail, kg)          # (B, 2*n_dim), feat='M'
    logit = model.Wr(embed).squeeze(-1)
    return embed, logit


@torch.no_grad()
def _eval_auc(model, kg, ia, ib, y, device, batch_size):
    model.eval()
    ps = []
    for s in range(0, len(y), batch_size):
        h = torch.as_tensor(ia[s:s + batch_size], device=device)
        t = torch.as_tensor(ib[s:s + batch_size], device=device)
        _, logit = _pair_embed(model, h, t, kg)
        ps.append(torch.sigmoid(logit).cpu().numpy())
    return roc_auc_score(y, np.concatenate(ps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--kg-scope", choices=["full", "drug_incident"], default="full",
                    help="EmerGNN KG scope (sets EMERGNN_KG_SCOPE for the cache build)")
    ap.add_argument("--steps", type=int, default=30000, help="total optimizer steps (mini-batch)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-dim", type=int, default=64)
    ap.add_argument("--length", type=int, default=3, help="EmerGNN message-passing depth L")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-8)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100,
                    help="lightweight per-step PACE log (train-only elapsed, no eval) every N steps")
    ap.add_argument("--patience", type=int, default=5000, help="warm-val early-stop patience in STEPS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg-nodes", default=DEFAULT_NODES_PATH)
    ap.add_argument("--kg-edges", default=DEFAULT_EDGES_PATH)
    ap.add_argument("--tag", default="emergnn_rspmm_full_s2")
    args = ap.parse_args()

    # scope selector must be set BEFORE ensure_kg_setup_cache reads it
    os.environ["EMERGNN_KG_SCOPE"] = args.kg_scope

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the rspmm backend (torchdrug generalized_rspmm) requires CUDA; no GPU visible. "
            "Run under the WSL project_1 env, or use train_emergnn_rank_pilot.py (chunk backend) "
            "for a CPU-capable (but far slower) path.")
    device = torch.device("cuda")
    print(f"[env] device={device} cuda=True kg_scope={args.kg_scope}", flush=True)

    # -- data (identical S2 protocol to the node pilots) ---------------------
    tr, warm_val, warm_test, cold_test = load_data(args.fold, args.warm_frac, args.seed)
    pool_train = pd.concat([tr, warm_val, warm_test], ignore_index=True)

    # -- EmerGNN KG via the baseline builder (reused) ------------------------
    merged_edges = (ROOT / args.kg_edges).resolve()
    cache = ensure_kg_setup_cache(
        _MiniTrain({"train": pool_train, "test": cold_test}),
        backbone_kg_source="merged", merged_kg_path=merged_edges)
    entity2id = cache["entity2id"]
    n_ent = int(cache["n_ent"])
    n_base_rel = int(cache["n_base_rel"])
    print(f"[kg] scope={args.kg_scope} n_ent={n_ent} n_base_rel={n_base_rel} "
          f"triplets={len(cache['kg_triplets'])}", flush=True)

    # sparse KG for the rspmm kernel (forward+reverse+self-loop, all_rel=2*n+1)
    kg = build_sparse_kg_from_triplets(cache["kg_triplets"], n_ent, n_base_rel, device=device)
    print(f"[kg] sparse KG built: shape={tuple(kg.shape)} nnz={kg._nnz()}", flush=True)

    # -- structural entity features (knowledge-only seed) --------------------
    x, logdeg, n_missing_type = build_structural_features(cache, args.kg_nodes)
    print(f"[feat] structural entity feats={x.shape} (type-onehot+logdeg); "
          f"{n_missing_type} entities missing merged-KG kind -> 'other'", flush=True)

    packs = {
        "train": _pairs_to_ent(tr, entity2id),
        "warm": _pairs_to_ent(warm_test, entity2id),
        "cold": _pairs_to_ent(cold_test, entity2id),
    }
    ia_tr, ib_tr, y_tr = _pairs_to_ent(tr, entity2id)
    yv_ia, yv_ib, yv = _pairs_to_ent(warm_val, entity2id)

    # -- model (EmerGNN_RSPMM, feat='M' fed KG-structural features) ----------
    model = EmerGNN_RSPMM(n_ent=n_ent, n_base_rel=n_base_rel, n_dim=args.n_dim,
                          length=args.length, feat="M",
                          morgan_features=x, morgan_feat_dim=N_TYPES + 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -- run dir -------------------------------------------------------------
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__train_emergnn_rspmm_rank_pilot__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"[cfg] backend=rspmm kg_scope={args.kg_scope} steps={args.steps} batch={args.batch_size} "
        f"n_dim={args.n_dim} L={args.length} lr={args.lr} wd={args.weight_decay} "
        f"patience={args.patience} n_ent={n_ent} n_base_rel={n_base_rel}")

    # -- train (mini-batch, step budget, warm-val early stop) ----------------
    n_train = len(y_tr)
    y_tr_t = torch.as_tensor(y_tr, device=device)
    order = rng.permutation(n_train)
    cursor = 0
    best_auc, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        if cursor >= n_train:
            order = rng.permutation(n_train)
            cursor = 0
        idx = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        h = torch.as_tensor(ia_tr[idx], device=device)
        t = torch.as_tensor(ib_tr[idx], device=device)
        yb = y_tr_t[idx]

        model.train()
        opt.zero_grad()
        logits = model(h, t, kg)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward()
        opt.step()

        # lightweight PACE log: pure-training per-step rate, independent of eval,
        # so you can immediately see if slowness is training or the (heavy) eval.
        if step % args.log_every == 0 and step % args.eval_every != 0:
            el = time.time() - t0
            log(f"[pace] step {step}/{args.steps} loss={loss.item():.4f} "
                f"train_elapsed={el:.0f}s ({el/step:.3f}s/step)")

        if step % args.eval_every == 0 or step == args.steps:
            va = _eval_auc(model, kg, yv_ia, yv_ib, yv, device, args.batch_size)
            log(f"[step {step}/{args.steps}] loss={loss.item():.4f} warm_val_auc={va:.4f} "
                f"elapsed={time.time()-t0:.0f}s")
            if va > best_auc:
                best_auc, bad = va, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += args.eval_every
                if bad >= args.patience:
                    log(f"[early-stop] no warm_val improvement for ~{args.patience} steps")
                    break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def extract(state, suffix: str) -> dict:
        model.load_state_dict(state)
        model.eval()
        arrs, aucs = {}, {}
        with torch.no_grad():
            for name, (ia, ib, y) in packs.items():
                embs, logits = [], []
                for s in range(0, len(y), args.batch_size):
                    h = torch.as_tensor(ia[s:s + args.batch_size], device=device)
                    t = torch.as_tensor(ib[s:s + args.batch_size], device=device)
                    emb, logit = _pair_embed(model, h, t, kg)
                    embs.append(emb.cpu().numpy().astype(np.float32))
                    logits.append(logit.cpu().numpy().astype(np.float32))
                emb = np.concatenate(embs); logit = np.concatenate(logits)
                arrs[f"Z_{name}"] = emb          # no bottleneck: Z == raw == embed
                arrs[f"raw_{name}"] = emb
                arrs[f"y_{name}"] = y.astype(np.float32)
                arrs[f"logit_{name}"] = logit
                arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
                auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
                aucs[name] = float(auc)
                log(f"[extract{suffix}] {name}: n={len(y)} head_auc={auc:.4f} embed={emb.shape}")
        np.savez_compressed(run_dir / f"pilot_Z{suffix}.npz", **arrs)
        return aucs

    best_aucs = extract(best_state if best_state is not None else last_state, "")
    last_aucs = extract(last_state, "_last")

    meta = {"run_id": run_id, "fold": args.fold, "seed": args.seed, "n_dim": args.n_dim,
            "length": args.length, "n_ent": n_ent, "n_base_rel": n_base_rel,
            "kg_scope": args.kg_scope, "backend": "rspmm", "best_warm_val_auc": best_auc,
            "feat": "M(structural-13d)", "setting": "label-inductive, graph-transductive"}
    results = {"run_id": run_id, "best_warm_val_auc": best_auc,
               "head_aucs_best": best_aucs, "head_aucs_last": last_aucs}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"[done] saved Z (best + last) + artifacts to {run_dir}")
    print(f"\nNEXT: python Code/scripts/analyze_rank_sufficiency.py "
          f"--z {run_dir/'pilot_Z.npz'} --key raw --title EmerGNN-full-DDI", flush=True)


if __name__ == "__main__":
    main()

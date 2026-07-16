"""A1 pilot (R-GCN variant): inductive relation-aware R-GCN-DDI on ddi800
(drugbank_latest_partial) S2, producing the pair representation for the warm/cold
rank-transfer analysis. Sibling of ``train_gcn_rank_pilot.py``; the plain GCN there
is relation-AGNOSTIC, this one is relation-AWARE (the node-paradigm representative
codex locked for the encoder x task grid).

Design (codex-reviewed 2026-07-04):
- SAME data protocol as the GCN pilot (imported verbatim from it): one model on
  S2-train, a warm holdout of SEEN-drug pairs, cold = S2-test unseen drugs;
  model selection on warm-val -> cold = natural transfer.
- Node features are IDENTICAL to the GCN pilot (KG type one-hot + standardized
  log-degree, NO trainable drug-ID embedding). The ONLY change vs the GCN pilot
  is relation-aware message passing, so any warm/cold rank difference is
  attributable to relation-awareness, not to features.
- Encoder = R-GCN (Schlichtkrull 2018) over the FULL merged KG. Same edge SET as
  the GCN pilot (de-duplicated, self-loops dropped, fully bidirectional); the
  ONLY difference is edges carry relation TYPE. Directed relations (e.g. db:target
  drug->protein) get a distinct INVERSE type so a drug can aggregate its targets
  via protein->drug; undirected relations reuse one type both ways.
  num_relations = R + (#directed relations). Basis decomposition (num_bases)
  shares weight bases across relations to keep params bounded and let rare
  relations borrow strength. NOTE basis constrains the rank of the RELATION
  operators {W_r}, not the DEFINITION of the representation rank that
  analyze_rank_sufficiency.py measures (though, being an encoder choice, it can
  still change the empirical rank of the representation it produces).
- Pair head + saved artifacts are byte-identical in schema to the GCN pilot's
  pilot_Z.npz (Z_/raw_/y_/logit_/deg_ per split), so analyze_rank_sufficiency.py
  runs unchanged on this run's output.
- Training: full-batch (1 R-GCN forward + all train pairs / step), AdamW, mixed
  precision for the 14.2M-directed-edge forward; STEP-BUDGET (each --epochs is one
  optimizer step; the GCN pilot's undertraining bug was 1 step/epoch, so budget
  the steps, ~1000+). Early stop on warm-val.

Run (from project root, WSL conda env project_1):
  python Code/scripts/train_rgcn_rank_pilot.py --fold fold0 --epochs 1500 --amp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

try:
    from torch_geometric.nn import RGCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("project root (with Code/data/KG) not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
# reuse the GCN pilot's pure data helpers verbatim (no re-implementation) so the
# two pilots share an identical data protocol; importing does not run its main().
from train_gcn_rank_pilot import load_data, _pairs_to_idx  # noqa: E402


# --------------------------------------------------------------------------- #
# Relation-aware edges
# --------------------------------------------------------------------------- #
def build_relation_edges(edges_path: str, id_to_idx: dict[str, int]):
    """Build (edge_index, edge_type) for the full merged KG, matching the GCN
    pilot's edge SET (de-duplicated, self-loops dropped, fully bidirectional) so
    the ONLY difference vs the GCN pilot is that edges carry relation TYPE.

    Steps:
      1. map endpoints to idx, drop edges with a missing endpoint AND self-loops
         (mirrors MergedKG.from_parquet's `loop = u != v` + CSR de-dup).
      2. relation vocabulary = sorted distinct labels of the SURVIVING rows
         (ids 0..R-1) — built post-filter so no dropped-only label inflates R.
      3. per-relation directedness from the parquet `directed` flag (each
         relation is homogeneous: directed_frac is 0 or 1). DIRECTED relations
         get a distinct INVERSE type (so e.g. db:target drug->protein also lets a
         drug aggregate its targets via the protein->drug inverse type).
         UNDIRECTED relations use the SAME type in both directions (no artificial
         orientation learned from arbitrary stored column order).
         num_relations = R + (#directed relations).
      4. add forward (s->d) + reverse (d->s) edges, then de-duplicate exact
         (row, col, type) triples (removes stored multiplicity and the
         double-count when an undirected relation already stored both orders).
    """
    e = pd.read_parquet(edges_path, columns=["src", "dst", "relation", "directed"])
    src = e["src"].astype(str).map(id_to_idx).to_numpy()
    dst = e["dst"].astype(str).map(id_to_idx).to_numpy()
    keep = ~(pd.isna(src) | pd.isna(dst))
    n_missing = int((~keep).sum())
    src_i = src[keep].astype(np.int64)
    dst_i = dst[keep].astype(np.int64)
    not_loop = src_i != dst_i
    n_selfloop = int((~not_loop).sum())
    src_i = src_i[not_loop]
    dst_i = dst_i[not_loop]
    rel = e["relation"].astype(str).to_numpy()[keep][not_loop]
    directed = e["directed"].to_numpy()[keep][not_loop].astype(bool)

    rel_names = sorted(pd.unique(rel).tolist())
    rel_to_id = {r: i for i, r in enumerate(rel_names)}
    R = len(rel_names)
    rid = np.fromiter((rel_to_id[r] for r in rel), dtype=np.int64, count=rel.shape[0])

    # per-relation directedness (homogeneous per relation; OR over its edges)
    rel_is_directed = (pd.DataFrame({"rid": rid, "d": directed})
                       .groupby("rid")["d"].max()
                       .reindex(range(R), fill_value=False).to_numpy().astype(bool))
    directed_rids = np.where(rel_is_directed)[0]
    inv_id = np.full(R, -1, dtype=np.int64)
    inv_id[directed_rids] = R + np.arange(directed_rids.shape[0], dtype=np.int64)
    num_relations = int(R + directed_rids.shape[0])

    # reverse edge type: distinct inverse for directed, same type for undirected
    rev_type = np.where(rel_is_directed[rid], inv_id[rid], rid)
    row = np.concatenate([src_i, dst_i])
    col = np.concatenate([dst_i, src_i])
    ety = np.concatenate([rid, rev_type])

    # de-duplicate exact (row, col, type) triples via an injective int64 key
    n_nodes = len(id_to_idx)
    key = (row * n_nodes + col) * num_relations + ety
    _, uniq = np.unique(key, return_index=True)
    row, col, ety = row[uniq], col[uniq], ety[uniq]

    edge_index = torch.from_numpy(np.vstack([row, col]).astype(np.int64))
    edge_type = torch.from_numpy(ety.astype(np.int64))
    meta = {"R": R, "num_relations": num_relations,
            "n_directed_relations": int(directed_rids.shape[0]),
            "n_missing_endpoint": n_missing, "n_selfloop": n_selfloop,
            "n_directed_edges": int(edge_index.shape[1]),
            "rel_is_directed": {rel_names[i]: bool(rel_is_directed[i]) for i in range(R)}}
    return edge_index, edge_type, num_relations, rel_names, meta


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class RGCNPairModel(nn.Module):
    """Inductive R-GCN over the KG + a pair head with a bottleneck Z. Mirrors
    GCNPairModel's head exactly; only the convs are relation-aware."""

    def __init__(self, in_dim: int, hidden: int, bottleneck: int, dropout: float,
                 num_relations: int, num_bases: int, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList([
            RGCNConv(in_dim if i == 0 else hidden, hidden,
                     num_relations=num_relations, num_bases=num_bases)
            for i in range(num_layers)])
        self.dropout = dropout
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden, bottleneck), nn.ReLU(), nn.Dropout(dropout),
        )  # output = Z (bottleneck-d)
        self.out = nn.Linear(bottleneck, 1)

    def encode(self, x, edge_index, edge_type):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_type)
            if i < len(self.convs) - 1:   # ReLU+dropout between layers, not after last
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h  # (n_nodes, hidden)

    def pair_z(self, h, ia, ib):
        hu, hv = h[ia], h[ib]
        feat = torch.cat([hu * hv, (hu - hv).abs()], dim=-1)
        return self.pair_mlp(feat)  # (n_pairs, bottleneck) = Z

    def forward(self, x, edge_index, edge_type, ia, ib):
        h = self.encode(x, edge_index, edge_type)
        z = self.pair_z(h, ia, ib)
        return self.out(z).squeeze(-1), z


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_auc(model, x, edge_index, edge_type, ia, ib, y, device):
    model.eval()
    h = model.encode(x, edge_index, edge_type)
    logits = model.out(model.pair_z(h, torch.as_tensor(ia, device=device),
                                    torch.as_tensor(ib, device=device))).squeeze(-1)
    p = torch.sigmoid(logits).cpu().numpy()
    return roc_auc_score(y, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=1500,
                    help="number of optimizer STEPS (full-batch: 1 step/epoch)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-bases", type=int, default=16,
                    help="R-GCN basis-decomposition bases shared across the 118 relations")
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=80,
                    help="warm-val early-stop patience in STEPS (step-budget, not epochs)")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="run warm-val eval every N steps (R-GCN eval is a full forward)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed-precision training forward (recommended for the full KG)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg-nodes", default=DEFAULT_NODES_PATH)
    ap.add_argument("--kg-edges", default=DEFAULT_EDGES_PATH)
    ap.add_argument("--tag", default="rgcn_ddi800_s2")
    args = ap.parse_args()

    if not HAS_PYG:
        raise ImportError("torch_geometric required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()}", flush=True)

    # -- KG -> graph tensors (features IDENTICAL to the GCN pilot) ------------
    kg = MergedKG.from_parquet(args.kg_nodes, args.kg_edges)
    print(f"[kg] source nodes={args.kg_nodes} edges={args.kg_edges}", flush=True)
    n = kg.n_nodes
    deg = kg.degree.astype(np.float64)
    logdeg = np.log1p(deg)
    x = np.zeros((n, N_TYPES + 1), dtype=np.float32)
    x[np.arange(n), kg.type_id.astype(np.int64)] = 1.0
    x[:, N_TYPES] = ((logdeg - logdeg.mean()) / (logdeg.std() + 1e-8)).astype(np.float32)
    x = torch.from_numpy(x).to(device)

    # relation-aware edges (the one substantive difference vs the GCN pilot)
    edge_index, edge_type, num_relations, rel_names, rel_meta = build_relation_edges(
        args.kg_edges, kg.id_to_idx)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    R = len(rel_names)
    print(f"[kg] nodes={n} base_relations={R} directed_rel={rel_meta['n_directed_relations']} "
          f"num_relations={num_relations} directed_edges={edge_index.shape[1]} "
          f"(dropped: {rel_meta['n_missing_endpoint']} missing-endpoint, "
          f"{rel_meta['n_selfloop']} self-loop) feat_dim={x.shape[1]}", flush=True)

    # -- data (identical protocol to the GCN pilot) --------------------------
    tr, warm_val, warm_test, cold_test = load_data(args.fold, args.warm_frac, args.seed)
    id2i = kg.id_to_idx
    ia_tr, ib_tr, y_tr = _pairs_to_idx(tr, id2i)
    yv_ia, yv_ib, yv = _pairs_to_idx(warm_val, id2i)
    packs = {
        "train": _pairs_to_idx(tr, id2i),
        "warm": _pairs_to_idx(warm_test, id2i),
        "cold": _pairs_to_idx(cold_test, id2i),
    }
    ia_tr_t = torch.as_tensor(ia_tr, device=device)
    ib_tr_t = torch.as_tensor(ib_tr, device=device)
    y_tr_t = torch.as_tensor(y_tr, device=device)

    model = RGCNPairModel(x.shape[1], args.hidden, args.bottleneck, args.dropout,
                          num_relations=num_relations, num_bases=args.num_bases,
                          num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -- run dir -------------------------------------------------------------
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__train_rgcn_rank_pilot__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    (run_dir / "relations.json").write_text(json.dumps(
        {"base_relations": rel_names, **rel_meta,
         "note": "forward type = index in base_relations; directed relations get a "
                 "distinct inverse type (R + directed_index); undirected relations "
                 "reuse the same type in both directions"}, indent=2))

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # -- train (step budget; each epoch = 1 full-batch optimizer step) -------
    best_auc, best_state, bad = -1.0, None, 0
    for step in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(x, edge_index, edge_type, ia_tr_t, ib_tr_t)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_t)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if step % args.eval_every == 0 or step == args.epochs:
            va = _eval_auc(model, x, edge_index, edge_type, yv_ia, yv_ib, yv, device)
            log(f"[step {step}/{args.epochs}] loss={loss.item():.4f} warm_val_auc={va:.4f} "
                f"time={time.time()-t0:.2f}s")
            if va > best_auc:
                best_auc, bad = va, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
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
            h = model.encode(x, edge_index, edge_type)
            for name, (ia, ib, y) in packs.items():
                iat = torch.as_tensor(ia, device=device)
                ibt = torch.as_tensor(ib, device=device)
                hu, hv = h[iat], h[ibt]
                raw = torch.cat([hu * hv, (hu - hv).abs()], dim=-1)  # pre-MLP pair feats
                z = model.pair_mlp(raw)
                logit = model.out(z).squeeze(-1)
                arrs[f"Z_{name}"] = z.cpu().numpy().astype(np.float32)
                arrs[f"raw_{name}"] = raw.cpu().numpy().astype(np.float32)
                arrs[f"y_{name}"] = y.astype(np.float32)
                arrs[f"logit_{name}"] = logit.cpu().numpy().astype(np.float32)
                arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
                auc = roc_auc_score(y, torch.sigmoid(logit).cpu().numpy())
                aucs[name] = float(auc)
                log(f"[extract{suffix}] {name}: n={len(y)} head_auc={auc:.4f} "
                    f"Z={arrs[f'Z_{name}'].shape}")
        np.savez_compressed(run_dir / f"pilot_Z{suffix}.npz", **arrs)
        return aucs

    best_aucs = extract(best_state if best_state is not None else last_state, "")
    last_aucs = extract(last_state, "_last")

    meta = {"run_id": run_id, "fold": args.fold, "seed": args.seed, "hidden": args.hidden,
            "num_layers": args.num_layers, "num_bases": args.num_bases,
            "num_relations": num_relations, "bottleneck": args.bottleneck,
            "best_warm_val_auc": best_auc, "amp": bool(args.amp),
            "setting": "label-inductive, graph-transductive"}
    results = {"run_id": run_id, "best_warm_val_auc": best_auc,
               "head_aucs_best": best_aucs, "head_aucs_last": last_aucs}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"[done] saved Z (best + last) + artifacts to {run_dir}")
    print(f"\nNEXT: python Code/scripts/analyze_rank_sufficiency.py "
          f"--z {run_dir/'pilot_Z.npz'} --key raw", flush=True)


if __name__ == "__main__":
    main()

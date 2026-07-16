"""A1 pilot: inductive GCN-DDI on ddi800 (drugbank_latest_partial) S2, producing the
pair-bottleneck representation Z for the warm/cold rank-transfer analysis.

Design (codex-reviewed 2026-07-04, thread 019f2bd4):
- ONE model trained on S2-train (480 train-role drugs). A fraction of S2-train pairs is
  held out as warm-val / warm-test (SEEN-drug pairs). cold-test = S2-test (test-role
  drugs, unseen). Model selection / early stop on warm-val -> cold = natural transfer.
- Node paradigm = inductive GCN over the merged KG. Node features = KG node-type one-hot
  + log-degree (NO trainable drug-ID embedding). Unseen drugs get reps from their KG
  neighborhood.
- Setting label = "label-inductive, graph-transductive": the merged KG is DDI-free
  (mask1), so no DDI edge is ever in message passing; test drugs' NON-DDI edges DO stay
  in the training graph (needed to represent unseen drugs).
- Pair head: z_uv = MLP([H_u ⊙ H_v ; |H_u − H_v|]) -> 256-d bottleneck Z -> linear logit.
  Z is what analyze_rank_sufficiency.py rank-truncates.
- Saved: Z_{train,warm,cold} + labels + per-pair drug log-degrees (for the degree-only
  control) + full-rank head logits (for the probe-vs-head sanity check).

Run (from project root, WSL conda env project_1):
  python Code/scripts/train_gcn_rank_pilot.py --fold fold0 --epochs 60
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
    from torch_geometric.nn import GCNConv
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

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)

DDI800 = ROOT / "Code/data/ddi_unified/binary_cls/drugbank_latest_partial"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _canon(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize each pair so (a,b) and (b,a) are identical (a<=b), then drop dup
    pairs. Prevents a pair from splitting across train/warm-test."""
    a = df["drug_a_id"].astype(str)
    b = df["drug_b_id"].astype(str)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    out = df.copy()
    out["drug_a_id"] = lo
    out["drug_b_id"] = hi
    nlab = out.groupby(["drug_a_id", "drug_b_id"])["y_bin"].nunique()
    if (nlab > 1).any():
        raise ValueError(f"{int((nlab > 1).sum())} canonical pairs carry conflicting y_bin")
    return out.drop_duplicates(subset=["drug_a_id", "drug_b_id"]).reset_index(drop=True)


def _split_warm(train: pd.DataFrame, warm_frac: float, seed: int):
    """Hold out `warm_frac` of the (canonicalized) S2-train pairs as warm-test and the
    same amount as warm-val; the rest is the training set. All are SEEN-drug pairs."""
    rng = np.random.default_rng(seed)
    n = len(train)
    perm = rng.permutation(n)
    k = int(round(warm_frac * n))
    warm_test_idx = perm[:k]
    warm_val_idx = perm[k:2 * k]
    tr_idx = perm[2 * k:]
    return (train.iloc[tr_idx].reset_index(drop=True),
            train.iloc[warm_val_idx].reset_index(drop=True),
            train.iloc[warm_test_idx].reset_index(drop=True))


def _pairs_to_idx(df: pd.DataFrame, id_to_idx: dict[str, int]):
    """Map a pair frame to (idx_a, idx_b, y). FAIL HARD if any drug is missing from the
    KG (all ddi800 drugs are verified present; a silent drop could turn differential
    KG-coverage into a fake warm/cold divergence)."""
    a = df["drug_a_id"].astype(str).map(id_to_idx)
    b = df["drug_b_id"].astype(str).map(id_to_idx)
    bad = ~(a.notna() & b.notna())
    if int(bad.sum()):
        miss = pd.unique(pd.concat([
            df.loc[a.isna(), "drug_a_id"], df.loc[b.isna(), "drug_b_id"]]).astype(str))
        raise ValueError(f"{int(bad.sum())} pairs reference drugs not in merged KG "
                         f"(e.g. {list(miss[:5])}); aborting to avoid a coverage artifact")
    return a.to_numpy(np.int64), b.to_numpy(np.int64), df["y_bin"].to_numpy(np.float32)


def load_data(fold: str, warm_frac: float, seed: int):
    d = DDI800 / "inductive" / "S2"
    train = _canon(pd.read_parquet(d / fold / "train.parquet"))
    cold_test = _canon(pd.read_parquet(d / fold / "test.parquet"))
    tr, warm_val, warm_test = _split_warm(train, warm_frac, seed)
    print(f"[data] S2 fold={fold}: train_all={len(train)} -> train'={len(tr)} "
          f"warm_val={len(warm_val)} warm_test={len(warm_test)}; cold_test={len(cold_test)}",
          flush=True)
    return tr, warm_val, warm_test, cold_test


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class GCNPairModel(nn.Module):
    """Inductive GCN over the KG + a pair head with a 256-d bottleneck Z."""

    def __init__(self, in_dim: int, hidden: int, bottleneck: int, dropout: float,
                 num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        self.convs = nn.ModuleList([GCNConv(in_dim if i == 0 else hidden, hidden)
                                    for i in range(num_layers)])
        self.dropout = dropout
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hidden, bottleneck), nn.ReLU(), nn.Dropout(dropout),
        )  # output = Z (bottleneck-d)
        self.out = nn.Linear(bottleneck, 1)

    def encode(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:   # ReLU+dropout between layers, not after the last
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h  # (n_nodes, hidden)

    def pair_z(self, h, ia, ib):
        hu, hv = h[ia], h[ib]
        feat = torch.cat([hu * hv, (hu - hv).abs()], dim=-1)
        return self.pair_mlp(feat)  # (n_pairs, bottleneck) = Z

    def forward(self, x, edge_index, ia, ib):
        h = self.encode(x, edge_index)
        z = self.pair_z(h, ia, ib)
        return self.out(z).squeeze(-1), z


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _eval_auc(model, x, edge_index, ia, ib, y, device):
    model.eval()
    h = model.encode(x, edge_index)
    logits = model.out(model.pair_z(h, torch.as_tensor(ia, device=device),
                                    torch.as_tensor(ib, device=device))).squeeze(-1)
    p = torch.sigmoid(logits).cpu().numpy()
    return roc_auc_score(y, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kg-nodes", default=DEFAULT_NODES_PATH,
                    help="merged-KG nodes parquet (default = complete _merged_kg)")
    ap.add_argument("--kg-edges", default=DEFAULT_EDGES_PATH,
                    help="merged-KG edges parquet (default = complete _merged_kg, DDI-masked)")
    ap.add_argument("--tag", default="gcn_ddi800_s2")
    args = ap.parse_args()

    if not HAS_PYG:
        raise ImportError("torch_geometric required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} cuda={torch.cuda.is_available()}", flush=True)

    # -- KG -> graph tensors -------------------------------------------------
    kg = MergedKG.from_parquet(args.kg_nodes, args.kg_edges)
    print(f"[kg] source nodes={args.kg_nodes} edges={args.kg_edges}", flush=True)
    n = kg.n_nodes
    deg = kg.degree.astype(np.float64)
    logdeg = np.log1p(deg)
    # node features: type one-hot (N_TYPES) + standardized log-degree
    x = np.zeros((n, N_TYPES + 1), dtype=np.float32)
    x[np.arange(n), kg.type_id.astype(np.int64)] = 1.0
    x[:, N_TYPES] = ((logdeg - logdeg.mean()) / (logdeg.std() + 1e-8)).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    # edge_index from CSR (already undirected, de-duplicated)
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(kg.indptr))
    col = kg.indices.astype(np.int64)
    edge_index = torch.from_numpy(np.vstack([row, col])).to(device)
    print(f"[kg] nodes={n} edges(directed)={col.shape[0]} feat_dim={x.shape[1]}", flush=True)

    # -- data ----------------------------------------------------------------
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

    model = GCNPairModel(x.shape[1], args.hidden, args.bottleneck, args.dropout,
                         num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -- run dir -------------------------------------------------------------
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__train_gcn_rank_pilot__{args.tag}__seed{args.seed}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    # -- train (full-batch: 1 GCN forward + all train pairs / step) ----------
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        logits, _ = model(x, edge_index, ia_tr_t, ib_tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr_t)
        loss.backward()
        opt.step()
        va = _eval_auc(model, x, edge_index, yv_ia, yv_ib, yv, device)
        log(f"[ep {ep}/{args.epochs}] loss={loss.item():.4f} warm_val_auc={va:.4f} "
            f"time={time.time()-t0:.1f}s")
        if va > best_auc:
            best_auc, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"[early-stop] no warm_val improvement for {args.patience} epochs")
                break

    # capture last-epoch state BEFORE loading the warm-val-best checkpoint, so we can
    # extract Z from both and verify the divergence is not a checkpoint-selection effect.
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def extract(state, suffix: str) -> dict:
        model.load_state_dict(state)
        model.eval()
        arrs, aucs = {}, {}
        with torch.no_grad():
            h = model.encode(x, edge_index)
            for name, (ia, ib, y) in packs.items():
                iat = torch.as_tensor(ia, device=device)
                ibt = torch.as_tensor(ib, device=device)
                hu, hv = h[iat], h[ibt]
                raw = torch.cat([hu * hv, (hu - hv).abs()], dim=-1)  # pre-MLP pair feats (pre-collapse)
                z = model.pair_mlp(raw)
                logit = model.out(z).squeeze(-1)
                arrs[f"Z_{name}"] = z.cpu().numpy().astype(np.float32)
                arrs[f"raw_{name}"] = raw.cpu().numpy().astype(np.float32)  # "max info" representation
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
            "bottleneck": args.bottleneck, "best_warm_val_auc": best_auc,
            "setting": "label-inductive, graph-transductive"}
    results = {"run_id": run_id, "best_warm_val_auc": best_auc,
               "head_aucs_best": best_aucs, "head_aucs_last": last_aucs}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"[done] saved Z (best + last) + artifacts to {run_dir}")
    print(f"\nNEXT: python Code/scripts/analyze_rank_sufficiency.py --z {run_dir/'pilot_Z.npz'}",
          flush=True)


if __name__ == "__main__":
    main()

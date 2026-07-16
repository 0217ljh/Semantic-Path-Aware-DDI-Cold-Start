"""E2 — Depth-vs-AUC + over-smoothing diagnostics on ColdDDI S2.

Supports i2 (meeting-node anchor + over-smoothing). Primary claims:
  (a) Vanilla GCN test_s2 AUC peaks at K=2 or 3, drops at K∈{6, 8}
  (b) At K≥6, embedding collapse: mean cosine ↑, Dirichlet energy ↓, dim var ↓
  (c) PairNorm attenuates the AUC drop / collapse trajectory
  (d) MLP-control (no message passing) does NOT show collapse trajectory

Pinned init: node-type one-hot (NOT dependent on E3).

Compressed for speed (per "experiments should be fast and simple"):
  - K ∈ {1, 2, 4, 6, 8}
  - vanilla: 2 seeds; PairNorm: 1 seed; MLP-control: 1 seed at K=2 and K=8
  - 15 epochs each
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import GCNConv

# ---------------------------------------------------------------------------
def _find_project_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")


PROJECT_ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NODES = PROJECT_ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet"
EDGES = PROJECT_ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
SPLITS = PROJECT_ROOT / "Code/data/KG/drugbank/splits/seed42"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 128
HIDDEN = 128
LR = 0.005
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 15
DEPTHS = [1, 2, 4, 6, 8]


# ---------------------------------------------------------------------------
def build_edge_index(edges, id2idx):
    df = pd.DataFrame({
        "s": edges["src"].map(id2idx),
        "d": edges["dst"].map(id2idx),
        "directed": edges["directed"].values,
    }).dropna(subset=["s", "d"])
    src = df["s"].astype(np.int64).values
    dst = df["d"].astype(np.int64).values
    directed = df["directed"].values.astype(bool)
    edge_a = np.concatenate([src, dst[~directed]])
    edge_b = np.concatenate([dst, src[~directed]])
    return torch.tensor(np.stack([edge_a, edge_b], axis=0), dtype=torch.long)


def load_pairs(pos_path, neg_path, id2idx):
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]].copy()
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    pos["lab"] = 1
    neg["lab"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    df["a"] = df["drug_a_id"].map(id2idx)
    df["b"] = df["drug_b_id"].map(id2idx)
    df = df.dropna(subset=["a", "b"])
    pair = torch.tensor(df[["a", "b"]].values.astype(np.int64), dtype=torch.long)
    y = torch.tensor(df["lab"].values.astype(np.float32), dtype=torch.float32)
    return pair, y


# ---------------------------------------------------------------------------
class PairNorm(nn.Module):
    """Pairwise distance normalization (Zhao & Akoglu 2020), standard PN-SI form.

    Centers embeddings (x - mean) then rescales so the mean L2 norm per node
    equals `scale`. This prevents collapse without exploding magnitudes.
    """
    def __init__(self, scale: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        # mean L2 norm of each row
        mean_norm = (x.pow(2).sum(dim=1).mean().clamp(min=self.eps)).sqrt()
        return self.scale * x / mean_norm


class DepthGCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int, norm: str = "none"):
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.norm = norm
        self.convs = nn.ModuleList()
        for i in range(depth):
            d_in = in_dim if i == 0 else hidden
            d_out = out_dim if i == depth - 1 else hidden
            self.convs.append(GCNConv(d_in, d_out, cached=True))
        if norm == "pairnorm":
            self.pairnorms = nn.ModuleList([PairNorm() for _ in range(depth - 1)])
        else:
            self.pairnorms = None

    def forward(self, x, edge_index, return_all_layers: bool = False):
        hs = []
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.depth - 1:
                h = F.relu(h)
                if self.norm == "pairnorm" and self.pairnorms is not None:
                    h = self.pairnorms[i](h)
            hs.append(h)
        return (h, hs) if return_all_layers else h


class MLPControl(nn.Module):
    """Parameter-matched MLP — no message passing."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            d_in = in_dim if i == 0 else hidden
            d_out = out_dim if i == depth - 1 else hidden
            self.layers.append(nn.Linear(d_in, d_out))

    def forward(self, x, edge_index=None):
        h = x
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h


# ---------------------------------------------------------------------------
def collapse_metrics(emb: torch.Tensor, edge_index_sample: torch.Tensor) -> dict:
    """Compute cosine sim, Dirichlet energy, mean per-dim variance on `emb`."""
    with torch.no_grad():
        # mean pairwise cosine sim over 2000 random pairs
        n = emb.shape[0]
        idx_a = torch.randint(0, n, (2000,), device=emb.device)
        idx_b = torch.randint(0, n, (2000,), device=emb.device)
        a = F.normalize(emb[idx_a], dim=-1)
        b = F.normalize(emb[idx_b], dim=-1)
        cos = (a * b).sum(dim=-1).mean().item()

        # Dirichlet energy: avg ||h_i - h_j||² over sampled edges
        ei = edge_index_sample
        diff = emb[ei[0]] - emb[ei[1]]
        dirichlet = (diff.pow(2).sum(dim=-1)).mean().item()

        # per-dim variance averaged
        dim_var = emb.var(dim=0).mean().item()
    return {"cosine_sim": float(cos), "dirichlet": float(dirichlet), "dim_var": float(dim_var)}


def train_one(model: nn.Module, X_t, ei, train_pair, train_y, test_pair, test_y_np, max_epochs: int) -> tuple[float, dict, list[dict]]:
    bs = 16384
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = train_pair.shape[0]
    best_auc = 0.0
    val_aucs = []
    # sample edges for Dirichlet energy once
    n_edges = ei.shape[1]
    sample_e_idx = torch.randint(0, n_edges, (20000,), device=ei.device)
    ei_sample = ei[:, sample_e_idx]

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=X_t.device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            emb = model(X_t, ei)
            a = emb[train_pair[idx, 0]]
            b = emb[train_pair[idx, 1]]
            logits = (a * b).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(logits, train_y[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * idx.shape[0]
        model.eval()
        with torch.no_grad():
            emb = model(X_t, ei)
            a = emb[test_pair[:, 0]]
            b = emb[test_pair[:, 1]]
            scores = torch.sigmoid((a * b).sum(dim=-1)).cpu().numpy()
        auc = roc_auc_score(test_y_np, scores)
        val_aucs.append(auc)
        if auc > best_auc:
            best_auc = auc
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      epoch {epoch+1:2d}/{max_epochs}  loss={tot/n:.4f}  AUC={auc:.4f}  best={best_auc:.4f}")

    # Final collapse metrics
    model.eval()
    with torch.no_grad():
        final_emb = model(X_t, ei)
        metrics = collapse_metrics(final_emb, ei_sample)
    return best_auc, metrics, val_aucs


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[E2] device={DEVICE}")
    nodes = pd.read_parquet(NODES).reset_index(drop=True)
    edges = pd.read_parquet(EDGES)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"])}
    print(f"[E2] nodes={len(nodes)}  edges={len(edges)}")

    ei = build_edge_index(edges, id2idx).to(DEVICE)
    print(f"[E2] edge_index: {ei.shape}")
    train_pair, train_y = load_pairs(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet", id2idx)
    test_pair, test_y = load_pairs(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet", id2idx)
    train_pair = train_pair.to(DEVICE)
    train_y = train_y.to(DEVICE)
    test_pair = test_pair.to(DEVICE)
    test_y_np = test_y.numpy()
    print(f"[E2] train={len(train_y)}  test_s2={len(test_y_np)}")

    # Pinned init: node-type one-hot
    kinds = nodes["kind"].fillna("Unknown").values
    unique = sorted(set(kinds))
    kind_idx = {k: i for i, k in enumerate(unique)}
    type_oh = np.zeros((len(nodes), len(unique)), dtype=np.float32)
    for i, k in enumerate(kinds):
        type_oh[i, kind_idx[k]] = 1.0
    X_t = torch.tensor(type_oh, dtype=torch.float32, device=DEVICE)
    in_dim = X_t.shape[1]
    print(f"[E2] init dim (node-type one-hot): {in_dim}")

    results: dict = {"vanilla": {}, "pairnorm": {}, "mlp_control": {}}

    # --- Vanilla GCN, all depths, 2 seeds
    for k in DEPTHS:
        aucs_for_k = []
        metrics_for_k = []
        for s in [42, 43]:
            print(f"\n[E2] vanilla GCN K={k} seed={s}")
            torch.manual_seed(s)
            np.random.seed(s)
            model = DepthGCN(in_dim, HIDDEN, D, depth=k, norm="none").to(DEVICE)
            t0 = time.time()
            auc, m, val_aucs = train_one(model, X_t, ei, train_pair, train_y, test_pair, test_y_np, MAX_EPOCHS)
            print(f"    AUC={auc:.4f}  cosine={m['cosine_sim']:.4f}  dirichlet={m['dirichlet']:.2f}  dim_var={m['dim_var']:.4f}  time={time.time()-t0:.1f}s")
            aucs_for_k.append(auc)
            metrics_for_k.append(m)
            del model
            torch.cuda.empty_cache()
        results["vanilla"][k] = {
            "aucs": aucs_for_k,
            "mean_auc": float(np.mean(aucs_for_k)),
            "metrics_seed0": metrics_for_k[0],
        }

    # --- PairNorm GCN, all depths, 1 seed
    for k in DEPTHS:
        print(f"\n[E2] PairNorm GCN K={k} seed=42")
        torch.manual_seed(42)
        np.random.seed(42)
        model = DepthGCN(in_dim, HIDDEN, D, depth=k, norm="pairnorm").to(DEVICE)
        t0 = time.time()
        auc, m, val_aucs = train_one(model, X_t, ei, train_pair, train_y, test_pair, test_y_np, MAX_EPOCHS)
        print(f"    AUC={auc:.4f}  cosine={m['cosine_sim']:.4f}  dirichlet={m['dirichlet']:.2f}  dim_var={m['dim_var']:.4f}  time={time.time()-t0:.1f}s")
        results["pairnorm"][k] = {"aucs": [auc], "mean_auc": float(auc), "metrics_seed0": m}
        del model
        torch.cuda.empty_cache()

    # --- MLP-control at K=2 and K=8 only
    for k in [2, 8]:
        print(f"\n[E2] MLP-control K={k} seed=42")
        torch.manual_seed(42)
        np.random.seed(42)
        model = MLPControl(in_dim, HIDDEN, D, depth=k).to(DEVICE)
        t0 = time.time()
        auc, m, val_aucs = train_one(model, X_t, ei, train_pair, train_y, test_pair, test_y_np, MAX_EPOCHS)
        print(f"    AUC={auc:.4f}  cosine={m['cosine_sim']:.4f}  dirichlet={m['dirichlet']:.2f}  dim_var={m['dim_var']:.4f}  time={time.time()-t0:.1f}s")
        results["mlp_control"][k] = {"aucs": [auc], "mean_auc": float(auc), "metrics_seed0": m}
        del model
        torch.cuda.empty_cache()

    # --- Save
    (OUT_DIR / "depth_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[E2] saved → depth_sweep.json")

    # --- Summary
    print("\n[E2] SUMMARY (vanilla):")
    for k in DEPTHS:
        v = results["vanilla"][k]
        print(f"  K={k}  AUC={v['mean_auc']:.4f} (seeds={v['aucs']})  cosine={v['metrics_seed0']['cosine_sim']:.4f}  dirichlet={v['metrics_seed0']['dirichlet']:.2f}")
    print("\n[E2] SUMMARY (pairnorm):")
    for k in DEPTHS:
        v = results["pairnorm"][k]
        print(f"  K={k}  AUC={v['mean_auc']:.4f}  cosine={v['metrics_seed0']['cosine_sim']:.4f}  dirichlet={v['metrics_seed0']['dirichlet']:.2f}")
    print("\n[E2] SUMMARY (mlp_control):")
    for k, v in results["mlp_control"].items():
        print(f"  K={k}  AUC={v['mean_auc']:.4f}  cosine={v['metrics_seed0']['cosine_sim']:.4f}  dirichlet={v['metrics_seed0']['dirichlet']:.2f}")


if __name__ == "__main__":
    main()

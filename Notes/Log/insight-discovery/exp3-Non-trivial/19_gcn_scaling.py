"""exp3 step 19 — GCN training scaling (final AAAI bet).

Tests β candidate: does S2 (cold-start) AUC become NON-monotonic with training
data scale, while S0 (warm-start) remains monotonic? This would contradict
"more data is better" intuition and reveal a cold-start specific scaling pathology.

Setup:
- GCN K=2 with PubMedBERT init (E3 design)
- Fractions: {0.1, 0.3, 0.5, 0.8, 1.0}
- 2-3 seeds depending on time
- Subsample training pairs randomly (preserving balance)
- Eval on test_s0 AND test_s2

Success criterion (codex): S2 stable non-monotonic + S0 monotonic.
If fails: stop exp3 exploration, accept current ceiling.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.nn import GCNConv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_project_root():
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_project_root()
OUT_DIR = Path(__file__).parent
CACHE = ROOT / "Notes/Log/insight-discovery/exp2-0513-test-insight/_cache_features"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {DEVICE}")

D = 128
HIDDEN = 128
LR = 0.005
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 15  # reduced for scaling sweep (E3 used 25)
BS = 16384


class GCN2(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, out_dim, cached=True)
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def pair_logits(emb, pair_idx):
    a = emb[pair_idx[:, 0]]
    b = emb[pair_idx[:, 1]]
    return (a * b).sum(dim=-1)


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
    ei = np.stack([edge_a, edge_b], axis=0)
    return torch.tensor(ei, dtype=torch.long)


def load_pairs(pos_path, neg_path, id2idx):
    pos = pd.read_parquet(pos_path)[["drug_a_id", "drug_b_id"]]
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]]
    pos["lab"] = 1
    neg["lab"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    df["a"] = df["drug_a_id"].map(id2idx)
    df["b"] = df["drug_b_id"].map(id2idx)
    df = df.dropna(subset=["a", "b"])
    pair_idx = torch.tensor(df[["a", "b"]].values.astype(np.int64), dtype=torch.long)
    y = torch.tensor(df["lab"].values.astype(np.float32), dtype=torch.float32)
    return pair_idx, y


def train_eval(X_init, edge_index, tr_pair, tr_y, te_s0_pair, te_s0_y, te_s2_pair, te_s2_y, seed):
    torch.manual_seed(seed)
    X = X_init.astype(np.float32)
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ei = edge_index.to(DEVICE)
    tr_pair = tr_pair.to(DEVICE)
    tr_y = tr_y.to(DEVICE)
    te_s0_pair = te_s0_pair.to(DEVICE)
    te_s2_pair = te_s2_pair.to(DEVICE)
    te_s0_y_np = te_s0_y.cpu().numpy()
    te_s2_y_np = te_s2_y.cpu().numpy()

    model = GCN2(in_dim=X.shape[1], hidden=HIDDEN, out_dim=D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = tr_pair.shape[0]
    best_s0 = 0.0
    best_s2 = 0.0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            emb = model(X_t, ei)
            logits = pair_logits(emb, tr_pair[idx])
            loss = F.binary_cross_entropy_with_logits(logits, tr_y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            emb = model(X_t, ei)
            p_s0 = torch.sigmoid(pair_logits(emb, te_s0_pair)).cpu().numpy()
            p_s2 = torch.sigmoid(pair_logits(emb, te_s2_pair)).cpu().numpy()
        auc_s0 = roc_auc_score(te_s0_y_np, p_s0)
        auc_s2 = roc_auc_score(te_s2_y_np, p_s2)
        if auc_s0 > best_s0:
            best_s0 = auc_s0
        if auc_s2 > best_s2:
            best_s2 = auc_s2
        print(f"      ep{epoch+1:2d}: S0={auc_s0:.4f} S2={auc_s2:.4f} (best S0={best_s0:.4f} S2={best_s2:.4f})")
    return best_s0, best_s2


def main():
    t0 = time.time()
    print("[load] nodes/edges ...")
    nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet").reset_index(drop=True)
    edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
    id2idx = {nid: i for i, nid in enumerate(nodes["id"])}
    print(f"  nodes={len(nodes)} edges={len(edges)}")

    print("[build] edge_index ...")
    ei = build_edge_index(edges, id2idx)
    print(f"  ei: {ei.shape}")

    print("[load] PubMedBERT init + project to 128d via PCA ...")
    X_full = np.load(CACHE / "pubmedbert_real.npz")["X"]
    from sklearn.decomposition import PCA
    X_init = PCA(n_components=D, random_state=0).fit_transform(X_full).astype(np.float32)
    print(f"  X_init: {X_init.shape}")

    fractions = [0.10, 0.30, 0.50, 0.80, 1.00]
    seeds = [42, 43]  # 2 seeds for compute budget

    results = []
    for seed in seeds:
        print(f"\n========== seed {seed} ==========")
        SP = ROOT / f"Code/data/KG/drugbank/splits/seed{seed}"
        tr_pair, tr_y = load_pairs(SP / "train.parquet", SP / "train_negatives/epoch_0.parquet", id2idx)
        te_s0_pair, te_s0_y = load_pairs(SP / "test_s0.parquet", SP / "negatives/test_s0.parquet", id2idx)
        te_s2_pair, te_s2_y = load_pairs(SP / "test_s2.parquet", SP / "negatives/test_s2.parquet", id2idx)
        n_train = tr_pair.shape[0]
        print(f"  train: {n_train}, S0: {te_s0_pair.shape[0]}, S2: {te_s2_pair.shape[0]}")

        rng = np.random.default_rng(seed * 1000 + 7)
        perm = rng.permutation(n_train)
        tr_pair_full = tr_pair[perm]
        tr_y_full = tr_y[perm]

        for frac in fractions:
            n_use = max(int(n_train * frac), 100)
            tp = tr_pair_full[:n_use]
            ty = tr_y_full[:n_use]
            t_run = time.time()
            print(f"\n  ── seed={seed} frac={frac:.2f} n_tr={n_use} ──")
            auc_s0, auc_s2 = train_eval(X_init, ei, tp, ty, te_s0_pair, te_s0_y, te_s2_pair, te_s2_y, seed)
            results.append({"seed": seed, "frac": frac, "n_tr": n_use, "auc_s0": auc_s0, "auc_s2": auc_s2})
            print(f"     done in {time.time()-t_run:.1f}s | best S0={auc_s0:.4f}  S2={auc_s2:.4f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT_DIR / "gcn_scaling_results.csv", index=False)

    print("\n=== Aggregate (mean across seeds) ===")
    agg = rdf.groupby("frac").agg(
        n_tr_mean=("n_tr", "mean"),
        s0_mean=("auc_s0", "mean"), s0_std=("auc_s0", "std"),
        s2_mean=("auc_s2", "mean"), s2_std=("auc_s2", "std"),
    )
    print(agg.round(4))

    s0_vals = agg["s0_mean"].values
    s2_vals = agg["s2_mean"].values
    s0_diffs = np.diff(s0_vals)
    s2_diffs = np.diff(s2_vals)
    print(f"\nS0 diffs: {s0_diffs}")
    print(f"S2 diffs: {s2_diffs}")
    print(f"S2 has DECREASE step? {(s2_diffs < -0.005).any()}  (using 0.005 threshold for meaningful)")
    print(f"S0 monotonic up? {(s0_diffs >= -0.001).all()}")
    print(f"\nS0 AUC: max={s0_vals.max():.4f}, end={s0_vals[-1]:.4f}, drop={s0_vals.max()-s0_vals[-1]:+.4f}")
    print(f"S2 AUC: max={s2_vals.max():.4f}, end={s2_vals[-1]:.4f}, drop={s2_vals.max()-s2_vals[-1]:+.4f}")

    json.dump({"agg": agg.to_dict(), "results": results}, open(OUT_DIR / "gcn_scaling.json", "w"), indent=2, default=str)
    print(f"\nelapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

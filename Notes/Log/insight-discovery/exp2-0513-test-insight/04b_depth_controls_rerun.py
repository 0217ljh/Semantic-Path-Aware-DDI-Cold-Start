"""E2 controls re-run with fixes per codex round 2.

- PairNorm: corrected to PN-SI standard formula (no sqrt(N) magnification)
- MLP-control: now uses random Gaussian PER-NODE features (so each drug has
  a unique input; previous type-onehot version was degenerate)

Re-runs only the control variants. Vanilla results remain valid from previous run.
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

sys.path.insert(0, str(Path(__file__).parent))
# Import helpers from main E2 script
from importlib import import_module
e2_mod = import_module("04_depth_sweep")
build_edge_index = e2_mod.build_edge_index
load_pairs = e2_mod.load_pairs
DepthGCN = e2_mod.DepthGCN
MLPControl = e2_mod.MLPControl
collapse_metrics = e2_mod.collapse_metrics
train_one = e2_mod.train_one
DEVICE = e2_mod.DEVICE
D = e2_mod.D
HIDDEN = e2_mod.HIDDEN
LR = e2_mod.LR
WEIGHT_DECAY = e2_mod.WEIGHT_DECAY
MAX_EPOCHS = e2_mod.MAX_EPOCHS
DEPTHS = e2_mod.DEPTHS
NODES = e2_mod.NODES
EDGES = e2_mod.EDGES
SPLITS = e2_mod.SPLITS

OUT_DIR = Path(__file__).parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print(f"[E2b] device={DEVICE}")
    nodes = pd.read_parquet(NODES).reset_index(drop=True)
    edges = pd.read_parquet(EDGES)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"])}
    print(f"[E2b] nodes={len(nodes)}  edges={len(edges)}")

    ei = build_edge_index(edges, id2idx).to(DEVICE)
    train_pair, train_y = load_pairs(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet", id2idx)
    test_pair, test_y = load_pairs(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet", id2idx)
    train_pair = train_pair.to(DEVICE)
    train_y = train_y.to(DEVICE)
    test_pair = test_pair.to(DEVICE)
    test_y_np = test_y.numpy()

    # ---------- (1) PairNorm rerun (now with corrected PN-SI scaling) ----------
    # Init: same node-type one-hot as vanilla (so direct comparison is valid)
    kinds = nodes["kind"].fillna("Unknown").values
    unique = sorted(set(kinds))
    kind_idx = {k: i for i, k in enumerate(unique)}
    type_oh = np.zeros((len(nodes), len(unique)), dtype=np.float32)
    for i, k in enumerate(kinds):
        type_oh[i, kind_idx[k]] = 1.0
    X_t = torch.tensor(type_oh, dtype=torch.float32, device=DEVICE)
    in_dim = X_t.shape[1]

    pn_results = {}
    for k in DEPTHS:
        print(f"\n[E2b] PairNorm (corrected) K={k} seed=42")
        torch.manual_seed(42)
        np.random.seed(42)
        model = DepthGCN(in_dim, HIDDEN, D, depth=k, norm="pairnorm").to(DEVICE)
        t0 = time.time()
        auc, m, _ = train_one(model, X_t, ei, train_pair, train_y, test_pair, test_y_np, MAX_EPOCHS)
        print(f"  AUC={auc:.4f}  cosine={m['cosine_sim']:.4f}  dirichlet={m['dirichlet']:.2f}  dim_var={m['dim_var']:.4f}  time={time.time()-t0:.1f}s")
        pn_results[k] = {"auc": float(auc), **m}
        del model
        torch.cuda.empty_cache()

    # ---------- (2) MLP-control with random Gaussian per-node (unique per drug) ----------
    rng = np.random.default_rng(42)
    rand_init = rng.standard_normal((len(nodes), in_dim)).astype(np.float32) / np.sqrt(in_dim)
    X_rand = torch.tensor(rand_init, dtype=torch.float32, device=DEVICE)

    mlp_results = {}
    for k in [2, 8]:
        print(f"\n[E2b] MLP-control (random Gaussian init) K={k} seed=42")
        torch.manual_seed(42)
        np.random.seed(42)
        model = MLPControl(in_dim, HIDDEN, D, depth=k).to(DEVICE)
        t0 = time.time()
        auc, m, _ = train_one(model, X_rand, ei, train_pair, train_y, test_pair, test_y_np, MAX_EPOCHS)
        print(f"  AUC={auc:.4f}  cosine={m['cosine_sim']:.4f}  dirichlet={m['dirichlet']:.2f}  dim_var={m['dim_var']:.4f}  time={time.time()-t0:.1f}s")
        mlp_results[k] = {"auc": float(auc), **m}
        del model
        torch.cuda.empty_cache()

    # ---------- (3) Vanilla GCN with random Gaussian per-node init at K=8 only ----------
    #   sanity check that vanilla collapse persists even with diverse init
    print(f"\n[E2b] Vanilla GCN K=8 with random Gaussian init (sanity)")
    torch.manual_seed(42)
    np.random.seed(42)
    model = DepthGCN(in_dim, HIDDEN, D, depth=8, norm="none").to(DEVICE)
    t0 = time.time()
    auc, m, _ = train_one(model, X_rand, ei, train_pair, train_y, test_pair, test_y_np, MAX_EPOCHS)
    print(f"  AUC={auc:.4f}  cosine={m['cosine_sim']:.4f}  dirichlet={m['dirichlet']:.2f}  dim_var={m['dim_var']:.4f}  time={time.time()-t0:.1f}s")
    vanilla_rand_k8 = {"auc": float(auc), **m}
    del model
    torch.cuda.empty_cache()

    # ---------- Save ----------
    out = {
        "pairnorm_fixed": pn_results,
        "mlp_control_random_init": mlp_results,
        "vanilla_K8_random_init_sanity": vanilla_rand_k8,
    }
    (OUT_DIR / "depth_controls_rerun.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[E2b] saved → depth_controls_rerun.json")

    print("\n[E2b] SUMMARY:")
    print("PairNorm fixed:")
    for k, v in pn_results.items():
        print(f"  K={k}  AUC={v['auc']:.4f}  cosine={v['cosine_sim']:.4f}  dirichlet={v['dirichlet']:.2f}  dim_var={v['dim_var']:.4f}")
    print("MLP-control (random init):")
    for k, v in mlp_results.items():
        print(f"  K={k}  AUC={v['auc']:.4f}  cosine={v['cosine_sim']:.4f}  dirichlet={v['dirichlet']:.2f}")
    print(f"Vanilla K=8 with random init sanity: AUC={vanilla_rand_k8['auc']:.4f}  dirichlet={vanilla_rand_k8['dirichlet']:.2f}  dim_var={vanilla_rand_k8['dim_var']:.4f}")


if __name__ == "__main__":
    main()

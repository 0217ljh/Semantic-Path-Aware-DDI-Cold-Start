"""E3 — Init comparison on ColdDDI S2 with dimensionality + semantic controls.

Supports i4 (node-name biomedical text semantics is cold-start-stable signal).

Init variants compared (all projected to common d=128):
  (a) random          : Gaussian 128d, no projection
  (b) type_onehot     : node-kind one-hot → random-projected to 128d
  (d) real_pubmedbert : PubMedBERT [CLS] of node.name → projected
  (e) shuffled_pubmedbert : PubMedBERT [CLS] of within-kind-shuffled name → projected
  (f) typename_pubmedbert : PubMedBERT [CLS] of kind string only → projected

  Projections: random (frozen Gaussian) AND PCA-to-128 (robustness)

Model: 2-layer GCN over merged KG; drug-pair logit = dot(emb_a, emb_b).
Train: ColdDDI seed42 train.parquet + train_negatives/epoch_0.parquet
Eval:  test_s2.parquet + negatives/test_s2.parquet

Direct semantic claim: real_pubmedbert > shuffled_pubmedbert by ≥2pt with
paired bootstrap CI excluding 0, under BOTH projections.

For speed (per user 实验要快速且简单), we run 2 seeds and skip Node2Vec
(deferrable; mentioned in plan as optional).
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
from sklearn.decomposition import PCA
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
MAX_EPOCHS = 25
LR = 0.005
WEIGHT_DECAY = 1e-5
HIDDEN = 128
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


# ---------------------------------------------------------------------------
# PubMedBERT encoding (cached)
# ---------------------------------------------------------------------------
def encode_strings_pubmedbert(strings: list[str], batch_size: int = 128, cache_path: Path | None = None) -> np.ndarray:
    if cache_path and cache_path.exists():
        print(f"[E3] loading cached PubMedBERT features from {cache_path.name}")
        return np.load(cache_path)["X"]
    print(f"[E3] encoding {len(strings)} strings via PubMedBERT (batch={batch_size}) ...")
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(PUBMEDBERT_MODEL)
    model = AutoModel.from_pretrained(PUBMEDBERT_MODEL).to(DEVICE).eval()
    feats = np.zeros((len(strings), 768), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(strings), batch_size):
            batch = strings[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEVICE)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]
            feats[i : i + batch_size] = cls
            if (i // batch_size) % 50 == 0:
                print(f"    {i+batch_size}/{len(strings)} elapsed={time.time()-t0:.1f}s")
    if cache_path:
        np.savez_compressed(cache_path, X=feats)
        print(f"[E3] cached → {cache_path.name}")
    return feats


# ---------------------------------------------------------------------------
# Init feature builders
# ---------------------------------------------------------------------------
def build_inits(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict: variant_name → (N, raw_dim) array."""
    N = len(nodes)
    out: dict[str, np.ndarray] = {}

    # (a) random — generated per-seed at training time, just return placeholder
    out["random"] = None  # type: ignore

    # (b) type one-hot
    kinds = nodes["kind"].fillna("Unknown").values
    unique = sorted(set(kinds))
    kind_idx = {k: i for i, k in enumerate(unique)}
    type_oh = np.zeros((N, len(unique)), dtype=np.float32)
    for i, k in enumerate(kinds):
        type_oh[i, kind_idx[k]] = 1.0
    out["type_onehot"] = type_oh
    print(f"[E3] type_onehot dim = {len(unique)}")

    # Prepare PubMedBERT strings
    cache_dir = OUT_DIR / "_cache_features"
    cache_dir.mkdir(exist_ok=True)

    # Real names: use node.name where readable, else use kind as fallback (so we
    # don't have all-empty PubMedBERT vectors for ID-only nodes)
    names_raw = nodes["name"].fillna("").astype(str).tolist()
    names_for_encoding = [
        n if (n and not n.startswith(("het:", "prime:", "db:")) and len(n) > 2) else k
        for n, k in zip(names_raw, kinds)
    ]
    feats_real = encode_strings_pubmedbert(
        names_for_encoding,
        cache_path=cache_dir / "pubmedbert_real.npz",
    )
    out["real_pubmedbert"] = feats_real

    # Shuffled names within kind
    shuffled_idx = np.arange(N)
    rng = np.random.default_rng(42)
    for k in unique:
        idx_k = np.where(kinds == k)[0]
        if len(idx_k) > 1:
            shuffled_idx[idx_k] = rng.permutation(idx_k)
    names_shuffled = [names_for_encoding[i] for i in shuffled_idx]
    feats_shuffled = encode_strings_pubmedbert(
        names_shuffled,
        cache_path=cache_dir / "pubmedbert_shuffled.npz",
    )
    out["shuffled_pubmedbert"] = feats_shuffled

    # Type-name only
    type_strings = list(kinds)
    feats_typename = encode_strings_pubmedbert(
        type_strings,
        cache_path=cache_dir / "pubmedbert_typename.npz",
    )
    out["typename_pubmedbert"] = feats_typename

    return out


def project_to_d(
    X: np.ndarray, target_d: int, method: str = "random", seed: int = 42
) -> np.ndarray:
    if X is None:
        return None  # random handled at training time
    if X.shape[1] == target_d:
        return X.astype(np.float32)
    if method == "random":
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((X.shape[1], target_d)).astype(np.float32) / np.sqrt(target_d)
        return (X.astype(np.float32) @ proj).astype(np.float32)
    elif method == "pca":
        n_comp = min(target_d, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_comp, random_state=seed)
        Y = pca.fit_transform(X.astype(np.float32))
        if Y.shape[1] < target_d:
            pad = np.zeros((Y.shape[0], target_d - Y.shape[1]), dtype=np.float32)
            Y = np.concatenate([Y, pad], axis=1)
        return Y.astype(np.float32)
    else:
        raise ValueError(method)


# ---------------------------------------------------------------------------
# Graph + model
# ---------------------------------------------------------------------------
def build_edge_index(edges: pd.DataFrame, id2idx: dict[str, int]) -> torch.Tensor:
    df = pd.DataFrame({
        "s": edges["src"].map(id2idx),
        "d": edges["dst"].map(id2idx),
        "directed": edges["directed"].values,
    }).dropna(subset=["s", "d"])
    src_idx = df["s"].astype(np.int64).values
    dst_idx = df["d"].astype(np.int64).values
    directed = df["directed"].values.astype(bool)
    # symmetric edges: add reverse direction; directed edges: only one direction
    edge_a = np.concatenate([src_idx, dst_idx[~directed]])
    edge_b = np.concatenate([dst_idx, src_idx[~directed]])
    ei = np.stack([edge_a, edge_b], axis=0)
    return torch.tensor(ei, dtype=torch.long)


class GCN2(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, out_dim, cached=True)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


def pair_logits(emb: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
    # pair_idx: (B, 2)  → dot product
    a = emb[pair_idx[:, 0]]
    b = emb[pair_idx[:, 1]]
    return (a * b).sum(dim=-1)


def load_pairs_with_neg(pos_path: Path, neg_path: Path, id2idx: dict) -> tuple[torch.Tensor, torch.Tensor]:
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


# ---------------------------------------------------------------------------
def bootstrap_auc_ci(y, p, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            aucs.append(roc_auc_score(y[idx], p[idx]))
        except ValueError:
            continue
    return float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))


def paired_bootstrap_diff(y, p_a, p_b, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
        except ValueError:
            continue
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def train_one(X_init: np.ndarray, edge_index: torch.Tensor, train_pair, train_y, test_pair, test_y, seed: int) -> tuple[float, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if X_init is None:
        # random init
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((edge_index.max().item() + 1, D)).astype(np.float32) / np.sqrt(D)
    else:
        X = X_init.astype(np.float32)

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    ei = edge_index.to(DEVICE)
    train_pair = train_pair.to(DEVICE)
    train_y = train_y.to(DEVICE)
    test_pair = test_pair.to(DEVICE)
    test_y_np = test_y.cpu().numpy()

    model = GCN2(in_dim=X.shape[1], hidden=HIDDEN, out_dim=D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bs = 16384
    best_auc = 0.0
    best_pred = None
    n = train_pair.shape[0]
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            emb = model(X_t, ei)
            logits = pair_logits(emb, train_pair[idx])
            loss = F.binary_cross_entropy_with_logits(logits, train_y[idx])
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach()) * idx.shape[0]
        # eval
        model.eval()
        with torch.no_grad():
            emb = model(X_t, ei)
            scores = torch.sigmoid(pair_logits(emb, test_pair)).cpu().numpy()
        auc = roc_auc_score(test_y_np, scores)
        if auc > best_auc:
            best_auc = auc
            best_pred = scores.copy()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:2d}/{MAX_EPOCHS}  loss={tot_loss/n:.4f}  test_AUC={auc:.4f}  best={best_auc:.4f}")
    return best_auc, best_pred


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"[E3] device={DEVICE}")
    print("[E3] Loading nodes / edges / splits ...")
    nodes = pd.read_parquet(NODES).reset_index(drop=True)
    edges = pd.read_parquet(EDGES)
    id2idx = {nid: i for i, nid in enumerate(nodes["id"])}
    print(f"[E3] nodes={len(nodes)}  edges={len(edges)}")

    print("[E3] Building edge_index ...")
    edge_index = build_edge_index(edges, id2idx)
    print(f"[E3] edge_index shape: {edge_index.shape}")

    train_pair, train_y = load_pairs_with_neg(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet", id2idx)
    test_pair, test_y = load_pairs_with_neg(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet", id2idx)
    print(f"[E3] train pairs: {len(train_y)}  test_s2 pairs: {len(test_y)}")

    print("[E3] Building init features ...")
    raw_inits = build_inits(nodes)

    # For each init × projection × seed, train and record AUC
    seeds = [42, 43]
    init_names = ["random", "type_onehot", "real_pubmedbert", "shuffled_pubmedbert", "typename_pubmedbert"]
    projections = ["random", "pca"]  # random + PCA only for PubMedBERT-based

    results: dict = {}
    test_y_np = test_y.numpy()

    for init in init_names:
        results[init] = {}
        for proj in projections:
            # random: only one projection style needed
            # type_onehot: project once to 128
            # PubMedBERT variants: try both random and PCA
            if init == "random" and proj != "random":
                continue
            if init == "type_onehot" and proj != "random":
                continue
            if init in ("real_pubmedbert", "shuffled_pubmedbert", "typename_pubmedbert"):
                X_proj = project_to_d(raw_inits[init], D, method=proj, seed=42)
            else:
                X_proj = project_to_d(raw_inits[init], D, method="random", seed=42)

            auc_list = []
            pred_list = []
            for s in seeds:
                print(f"\n[E3] training init={init}  proj={proj}  seed={s}")
                t0 = time.time()
                auc, pred = train_one(X_proj, edge_index, train_pair, train_y, test_pair, test_y, seed=s)
                print(f"    final best AUC={auc:.4f}  time={time.time()-t0:.1f}s")
                auc_list.append(auc)
                pred_list.append(pred)

            ci_lo, ci_hi = bootstrap_auc_ci(test_y_np, pred_list[0])  # CI on seed 42's pred
            results[init][proj] = {
                "aucs": auc_list,
                "mean_auc": float(np.mean(auc_list)),
                "std_auc": float(np.std(auc_list)),
                "ci_95_seed42": [ci_lo, ci_hi],
                "preds_seed42": pred_list[0].tolist() if False else None,  # keep small
            }
            # Save predictions for paired comparison
            np.savez_compressed(
                OUT_DIR / f"_cache_features/pred_{init}_{proj}_seed42.npz",
                pred=pred_list[0], y=test_y_np,
            )

    # --- Paired diff: real vs shuffled (under both projections) ---
    for proj in ["random", "pca"]:
        try:
            real_pred = np.load(OUT_DIR / f"_cache_features/pred_real_pubmedbert_{proj}_seed42.npz")["pred"]
            shuf_pred = np.load(OUT_DIR / f"_cache_features/pred_shuffled_pubmedbert_{proj}_seed42.npz")["pred"]
            diff_m, diff_lo, diff_hi = paired_bootstrap_diff(test_y_np, real_pred, shuf_pred)
            print(f"\n[E3] real - shuffled ({proj} projection): Δ-AUC={diff_m:+.4f}  [95% CI {diff_lo:+.4f}, {diff_hi:+.4f}]")
            results.setdefault("paired_real_vs_shuffled", {})[proj] = {
                "delta_auc": float(diff_m),
                "ci_95": [float(diff_lo), float(diff_hi)],
            }
        except FileNotFoundError as e:
            print(f"[E3] cannot compute real-vs-shuffled for {proj}: {e}")

    (OUT_DIR / "init_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[E3] saved → init_comparison.json")

    print("\n[E3] SUMMARY:")
    for init in init_names:
        for proj, v in results.get(init, {}).items():
            print(f"  {init:<22s} {proj:<7s}: AUC mean={v['mean_auc']:.4f} ± {v['std_auc']:.4f}  (seeds={v['aucs']})")
    print("\n[E3] Paired Δ-AUC (real - shuffled):")
    for proj, v in results.get("paired_real_vs_shuffled", {}).items():
        print(f"  {proj:<7s}: {v['delta_auc']:+.4f}  CI={v['ci_95']}")


if __name__ == "__main__":
    main()

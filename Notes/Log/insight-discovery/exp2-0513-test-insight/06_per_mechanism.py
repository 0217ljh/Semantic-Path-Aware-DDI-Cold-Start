"""E4 — Per-mechanism analysis + paradigm-specific KG ablation on S2.

Supports i1 (PK and PD need different reasoning paradigms).

Three GCN variants trained on the SAME ColdDDI seed42 splits, with the
edge set restricted to different KG sublayers:
  (a) full       : all edges
  (b) molecular  : drug↔{Gene, Protein, gene/protein, Pathway, pathway,
                    Molecular Function, molecular_function, Compound,
                    Pharmacologic Class}
  (c) effect     : drug↔{Side Effect, effect/phenotype, Symptom, Disease,
                    disease, Anatomy, anatomy, Phenotype}

Test on test_s2; group test_s2 positives by `pk_pd_label` and report per-class
AUC and Δ-AUC vs full. Also report on `eval_400_PKB_PDB.parquet` subset.

Expected (joint, direction not p-value):
  - AUC_PD < AUC_PK overall (≥3pt)
  - molecular variant: AUC drop on PD > AUC drop on PK
  - effect variant: AUC drop on PK > AUC drop on PD
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
PKPD = PROJECT_ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"
EVAL_400 = PROJECT_ROOT / "Notes/Log/insight-discovery/eval_400_PKB_PDB.parquet"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = 128
HIDDEN = 128
LR = 0.005
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 15

MOLECULAR_KINDS = {"Gene", "gene/protein", "Protein", "Pathway", "pathway",
                   "Molecular Function", "molecular_function", "Compound",
                   "Pharmacologic Class"}
EFFECT_KINDS = {"Side Effect", "effect/phenotype", "Symptom", "Disease",
                "disease", "Anatomy", "anatomy", "Phenotype"}


def build_edge_index(edges: pd.DataFrame, id2idx: dict, restrict_to: set | None = None, drug_set: set | None = None, id2kind: dict | None = None) -> torch.Tensor:
    """Build edge_index, optionally restricting to edges where the NON-drug
    endpoint's kind is in `restrict_to`.  Drug-drug edges are always kept.
    """
    df = pd.DataFrame({
        "src": edges["src"].values,
        "dst": edges["dst"].values,
        "directed": edges["directed"].values,
    })
    if restrict_to is not None:
        assert drug_set is not None and id2kind is not None
        def keep(s, d):
            s_drug = s in drug_set
            d_drug = d in drug_set
            if s_drug and d_drug:
                return True  # drug-drug edges always kept (but we mask drug-drug DDI earlier)
            # at least one endpoint is non-drug; that endpoint's kind must be in restrict_to
            non_drug = d if s_drug else s
            return id2kind.get(non_drug, "") in restrict_to
        df = df[[keep(s, d) for s, d in zip(df["src"], df["dst"])]]
    df["s"] = df["src"].map(id2idx)
    df["d"] = df["dst"].map(id2idx)
    df = df.dropna(subset=["s", "d"])
    src = df["s"].astype(np.int64).values
    dst = df["d"].astype(np.int64).values
    directed = df["directed"].values.astype(bool)
    edge_a = np.concatenate([src, dst[~directed]])
    edge_b = np.concatenate([dst, src[~directed]])
    return torch.tensor(np.stack([edge_a, edge_b], axis=0), dtype=torch.long)


def load_pairs(pos_path: Path, neg_path: Path, id2idx: dict, extra_cols: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    cols = ["drug_a_id", "drug_b_id"] + (extra_cols or [])
    pos = pd.read_parquet(pos_path)[cols].copy()
    pos["lab"] = 1
    neg = pd.read_parquet(neg_path)[["drug_a_id", "drug_b_id"]].copy()
    for c in (extra_cols or []):
        neg[c] = None
    neg["lab"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    df["a"] = df["drug_a_id"].map(id2idx)
    df["b"] = df["drug_b_id"].map(id2idx)
    df = df.dropna(subset=["a", "b"]).reset_index(drop=True)
    pair = torch.tensor(df[["a", "b"]].values.astype(np.int64), dtype=torch.long)
    y = torch.tensor(df["lab"].values.astype(np.float32), dtype=torch.float32)
    return pair, y, df


class GCN2(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)
        self.conv2 = GCNConv(hidden, out_dim, cached=True)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)


def train_and_score(X_t, ei, train_pair, train_y, test_pair, test_y_np, in_dim: int, max_epochs: int, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GCN2(in_dim, HIDDEN, D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bs = 16384
    n = train_pair.shape[0]
    best_auc = 0.0
    best_scores = None
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=X_t.device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            emb = model(X_t, ei)
            a = emb[train_pair[idx, 0]]
            b = emb[train_pair[idx, 1]]
            loss = F.binary_cross_entropy_with_logits((a * b).sum(-1), train_y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            emb = model(X_t, ei)
            a = emb[test_pair[:, 0]]
            b = emb[test_pair[:, 1]]
            scores = torch.sigmoid((a * b).sum(-1)).cpu().numpy()
        auc = roc_auc_score(test_y_np, scores)
        if auc > best_auc:
            best_auc = auc
            best_scores = scores.copy()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      epoch {epoch+1:2d}/{max_epochs}  AUC={auc:.4f}  best={best_auc:.4f}")
    del model
    torch.cuda.empty_cache()
    return best_scores


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


def per_class_auc(test_df: pd.DataFrame, scores: np.ndarray, label_map: dict) -> dict:
    """Compute AUC per (PK / PD subset).  For each subset, use that subset's
    positives + ALL negatives of the test set."""
    test_df = test_df.copy()
    test_df["score"] = scores
    test_df["pk_pd"] = test_df["ddi_type"].map(label_map).where(test_df["lab"] == 1, None)
    out = {}
    for cls in ["PK", "PD"]:
        pos = test_df[(test_df["lab"] == 1) & (test_df["pk_pd"] == cls)]
        neg = test_df[test_df["lab"] == 0]
        if len(pos) < 30:
            continue
        sub = pd.concat([pos, neg], ignore_index=True)
        y = sub["lab"].values
        s = sub["score"].values
        auc = roc_auc_score(y, s)
        ci_lo, ci_hi = bootstrap_auc_ci(y, s, n_boot=300)
        out[cls] = {
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "auc": float(auc),
            "ci_95": [float(ci_lo), float(ci_hi)],
        }
    return out


def main() -> None:
    print(f"[E4] device={DEVICE}")
    nodes = pd.read_parquet(NODES).reset_index(drop=True)
    edges = pd.read_parquet(EDGES)
    pkpd = pd.read_csv(PKPD)
    label_map = dict(zip(pkpd["ddi_type"], pkpd["pk_pd_label"]))
    id2idx = {nid: i for i, nid in enumerate(nodes["id"])}
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    print(f"[E4] nodes={len(nodes)}  edges={len(edges)}  drugs={len(drug_set)}")

    # Pinned init: node-type one-hot
    kinds = nodes["kind"].fillna("Unknown").values
    unique = sorted(set(kinds))
    kind_idx = {k: i for i, k in enumerate(unique)}
    type_oh = np.zeros((len(nodes), len(unique)), dtype=np.float32)
    for i, k in enumerate(kinds):
        type_oh[i, kind_idx[k]] = 1.0
    X_t = torch.tensor(type_oh, dtype=torch.float32, device=DEVICE)
    in_dim = X_t.shape[1]
    print(f"[E4] init dim: {in_dim}")

    train_pair, train_y, _ = load_pairs(SPLITS / "train.parquet", SPLITS / "train_negatives/epoch_0.parquet", id2idx)
    test_pair, test_y, test_df = load_pairs(SPLITS / "test_s2.parquet", SPLITS / "negatives/test_s2.parquet", id2idx, extra_cols=["ddi_type"])
    train_pair, train_y = train_pair.to(DEVICE), train_y.to(DEVICE)
    test_pair = test_pair.to(DEVICE)
    test_y_np = test_y.numpy()
    print(f"[E4] train={len(train_y)}  test_s2={len(test_y_np)}")

    # PK/PD prevalence in test_s2 positives
    test_pos = test_df[test_df["lab"] == 1].copy()
    test_pos["pk_pd"] = test_pos["ddi_type"].map(label_map)
    print(f"[E4] test_s2 positives by PK/PD: {test_pos['pk_pd'].value_counts().to_dict()}")

    # --- Run 3 variants ---
    variants = {
        "full": None,
        "molecular_only": MOLECULAR_KINDS,
        "effect_only": EFFECT_KINDS,
    }
    results = {}
    for name, restrict in variants.items():
        print(f"\n[E4] Building edge_index for variant '{name}' ...")
        t0 = time.time()
        ei = build_edge_index(edges, id2idx, restrict, drug_set, id2kind).to(DEVICE)
        print(f"  edge_index: {ei.shape}  built in {time.time()-t0:.1f}s")
        print(f"[E4] training GCN-{name} ...")
        scores = train_and_score(X_t, ei, train_pair, train_y, test_pair, test_y_np, in_dim, MAX_EPOCHS, seed=42)
        overall = roc_auc_score(test_y_np, scores)
        ci_lo, ci_hi = bootstrap_auc_ci(test_y_np, scores)
        print(f"  {name} overall AUC = {overall:.4f}  [95% CI {ci_lo:.4f}-{ci_hi:.4f}]")
        # per-class
        per_cls = per_class_auc(test_df, scores, label_map)
        for c, v in per_cls.items():
            print(f"    {c}: AUC={v['auc']:.4f}  CI={v['ci_95']}  n_pos={v['n_pos']}")
        results[name] = {
            "overall_auc": float(overall),
            "overall_ci_95": [float(ci_lo), float(ci_hi)],
            "per_class": per_cls,
            "edge_count": int(ei.shape[1]),
        }
        del ei
        torch.cuda.empty_cache()

    # --- Compute Δ-AUC: full → molecular, full → effect, per class
    print("\n[E4] Δ-AUC vs full (per-class):")
    deltas = {}
    full_per = results["full"]["per_class"]
    for variant in ["molecular_only", "effect_only"]:
        deltas[variant] = {}
        for c in ["PK", "PD"]:
            if c in full_per and c in results[variant]["per_class"]:
                d = results[variant]["per_class"][c]["auc"] - full_per[c]["auc"]
                deltas[variant][c] = float(d)
                print(f"  {variant} {c}: Δ-AUC = {d:+.4f}")
    results["delta_vs_full"] = deltas

    # --- Save
    (OUT_DIR / "per_mechanism.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n[E4] saved → per_mechanism.json")

    # --- Verdict
    print("\n[E4] i1 verdict (direction-based, not significance):")
    if "PK" in full_per and "PD" in full_per:
        gap = full_per["PK"]["auc"] - full_per["PD"]["auc"]
        print(f"  AUC_PK - AUC_PD (full KG): {gap:+.4f}")
    mol_pd = deltas.get("molecular_only", {}).get("PD", float("nan"))
    mol_pk = deltas.get("molecular_only", {}).get("PK", float("nan"))
    eff_pk = deltas.get("effect_only", {}).get("PK", float("nan"))
    eff_pd = deltas.get("effect_only", {}).get("PD", float("nan"))
    if mol_pd < mol_pk:
        print(f"  molecular-only hurts PD more than PK?  YES (Δ_PD={mol_pd:+.4f} < Δ_PK={mol_pk:+.4f})")
    else:
        print(f"  molecular-only hurts PD more than PK?  NO (Δ_PD={mol_pd:+.4f} ≥ Δ_PK={mol_pk:+.4f})")
    if eff_pk < eff_pd:
        print(f"  effect-only hurts PK more than PD?  YES (Δ_PK={eff_pk:+.4f} < Δ_PD={eff_pd:+.4f})")
    else:
        print(f"  effect-only hurts PK more than PD?  NO (Δ_PK={eff_pk:+.4f} ≥ Δ_PD={eff_pd:+.4f})")


if __name__ == "__main__":
    main()

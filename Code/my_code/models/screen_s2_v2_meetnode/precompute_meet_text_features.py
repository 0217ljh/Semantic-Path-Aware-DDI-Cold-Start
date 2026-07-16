"""Stage 2 (i4): PubMedBERT text features for shared mediators.

Per codex round 15 (PASS w/ FIX):
- For each pair, gather the SHARED non-drug mediators M (same set as Stage 1).
- Mean-pool their PubMedBERT node-NAME embeddings (768d) -> 1 vector per pair.
- Project 768->64 via PCA FIT ON TRAIN-PAIR mediator-means only (not random proj),
  then transform all pairs.
- Cache t_0..t_63 per canonical pair.

Variants (separate caches, selected by --emb-tag):
  real     : d_name_only__pubmedbert.pt        (real node-name semantics)
  shuffled : e_shuffled_text__pubmedbert.pt    (codex semantic-vs-capacity control)

Output: Code/data/_cache/meet_text_{emb_tag}_drugbank_seed42_kgonly_v1.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FILE = Path(__file__).resolve()
sys.path.insert(0, str(_FILE.parent))
from precompute_meet_features import (  # noqa: E402
    PROJECT_ROOT, NODES, EDGES, SPLITS, build_neighbor_sets, canonical_pair,
)

EMB_DIR = PROJECT_ROOT / "Code/data/KG/_merged_kg/_cache/screen1_tag_init"
EMB_FILES = {
    "real": EMB_DIR / "d_name_only__pubmedbert.pt",
    "shuffled": EMB_DIR / "e_shuffled_text__pubmedbert.pt",
}
PROJ_DIM = 64


def load_embeddings(emb_tag: str) -> tuple[dict[str, int], np.ndarray]:
    p = EMB_FILES[emb_tag]
    obj = torch.load(p, map_location="cpu", weights_only=False)
    node_ids = obj["node_ids"]
    emb = obj["embeddings"].numpy().astype(np.float32)  # [178029, 768]
    nid2row = {str(nid): i for i, nid in enumerate(node_ids)}
    print(f"[txt-pre] loaded {emb_tag} embeddings: {emb.shape}", flush=True)
    return nid2row, emb


def build_shared_mediator_lists(edges, id2kind, drug_set):
    """Like Stage 1 but keep the actual shared NODE IDS (not just counts)."""
    n1, n2 = build_neighbor_sets(edges, id2kind, drug_set)
    # collapse (node,kind) sets to node-id sets per drug for 1hop+2hop union
    n_all: dict[str, set[str]] = defaultdict(set)
    for d in drug_set:
        s = set()
        for nid, _ in n1.get(d, ()):
            s.add(nid)
        for nid, _ in n2.get(d, ()):
            s.add(nid)
        if s:
            n_all[d] = s
    return n_all


def pooled_vectors(pairs, n_all, nid2row, emb) -> np.ndarray:
    """Per pair: mean of shared-mediator name-embeddings (768d). Empty -> zeros."""
    D = emb.shape[1]
    out = np.zeros((len(pairs), D), dtype=np.float32)
    for i, (a, b) in enumerate(pairs):
        sa = n_all.get(a)
        sb = n_all.get(b)
        if not sa or not sb:
            continue
        shared = sa & sb
        if not shared:
            continue
        rows = [nid2row[m] for m in shared if m in nid2row]
        if rows:
            out[i] = emb[rows].mean(axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-tag", choices=["real", "shuffled"], default="real")
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "Code/data/_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"meet_text_{args.emb_tag}_drugbank_seed42_kgonly_v1.parquet"
    if out_path.exists():
        print(f"[txt-pre] cache EXISTS at {out_path}")
        return

    nid2row, emb = load_embeddings(args.emb_tag)

    print("[txt-pre] Loading nodes/edges ...", flush=True)
    nodes = pd.read_parquet(NODES)
    edges = pd.read_parquet(EDGES)
    id2kind = dict(zip(nodes["id"], nodes["kind"]))
    drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
    n_all = build_shared_mediator_lists(edges, id2kind, drug_set)

    # all canonical pairs
    all_drugs: set[str] = set()
    for sub in ["train.parquet", "val_s2.parquet", "test_s2.parquet"]:
        df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
        all_drugs |= set(df["drug_a_id"].astype(str)) | set(df["drug_b_id"].astype(str))
    for sub in ["negatives/val_s2.parquet", "negatives/test_s2.parquet",
                "train_negatives/epoch_0.parquet"]:
        try:
            df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
            all_drugs |= set(df["drug_a_id"].astype(str)) | set(df["drug_b_id"].astype(str))
        except Exception:
            pass
    all_drugs = sorted(all_drugs)
    n = len(all_drugs)
    pairs = [(all_drugs[i], all_drugs[j]) for i in range(n) for j in range(i, n)]
    print(f"[txt-pre] {len(pairs)} canonical pairs; pooling 768d ...", flush=True)

    t0 = time.time()
    pooled = pooled_vectors(pairs, n_all, nid2row, emb)
    print(f"[txt-pre] pooled shape={pooled.shape} time={time.time()-t0:.1f}s", flush=True)

    # PCA fit on TRAIN-pair pooled means only (codex r15 FIX)
    pair2idx = {canonical_pair(a, b): i for i, (a, b) in enumerate(pairs)}
    train_idx = []
    for sub in ["train.parquet", "train_negatives/epoch_0.parquet"]:
        try:
            df = pd.read_parquet(SPLITS / sub)[["drug_a_id", "drug_b_id"]]
            for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
                k = canonical_pair(a, b)
                if k in pair2idx:
                    train_idx.append(pair2idx[k])
        except Exception:
            pass
    train_idx = np.unique(np.array(train_idx, dtype=np.int64))
    print(f"[txt-pre] PCA fit on {len(train_idx)} train pairs", flush=True)
    pca = PCA(n_components=PROJ_DIM, random_state=42)
    pca.fit(pooled[train_idx])
    proj = pca.transform(pooled).astype(np.float32)  # [N, 64]
    print(f"[txt-pre] PCA explained var (top5): {pca.explained_variance_ratio_[:5]}", flush=True)
    print(f"[txt-pre] cumvar@64 = {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    cols = [f"t_{i}" for i in range(PROJ_DIM)]
    df = pd.DataFrame(proj, columns=cols)
    df.insert(0, "drug_a_id", [p[0] for p in pairs])
    df.insert(1, "drug_b_id", [p[1] for p in pairs])
    df.to_parquet(out_path, index=False)
    print(f"[txt-pre] wrote {len(df)} rows, {PROJ_DIM} text dims -> {out_path}")


if __name__ == "__main__":
    main()

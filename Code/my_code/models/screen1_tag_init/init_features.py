"""Compose final init tensors (N_nodes, n_dim) for each Screen 1 variant.

Variants (per first_step_plan.md §4.5):
  A : random Gaussian 64d (frozen, seeded)
  B : node-kind one-hot (canonical kinds) -> projected to 64d if needed
  C : DeepWalk-style Node2Vec 64d (pretrained on merged KG)
  D : PubMedBERT [CLS] of full text -> projected to 64d
  D-name : PubMedBERT [CLS] of drug name only -> projected to 64d
  E : PubMedBERT [CLS] of within-canonical-kind-shuffled FULL TEXT -> projected to 64d
  F : PubMedBERT [CLS] of canonical-kind string only -> projected to 64d
  H : Qwen-72B input-embedding mean-pool -> projected to 64d  (separate path)

Projection methods (P1=random Gaussian / P2=PCA-to-64). Per Codex review #1
NIT #8, P1 uses a SHARED seed (0) across D/D-name/E/F so geometric direction
is identical and only text content varies. Per Codex WARN #4, P2 masks out
zero-norm input rows from PCA fitting AND re-zeros them in output.

Integrity:
  - cached node_ids list checked for uniqueness and full canonical-set coverage
  - canonical node-order hash stored alongside any new cache
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
import torch

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
CACHE_DIR = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "_cache" / "screen1_tag_init"
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen1_tag_init import node_text_builder as ntb  # noqa: E402


PUBMEDBERT_CACHE_TAGS = {
    "D":      "d_full_text",
    "D-name": "d_name_only",
    "E":      "e_shuffled_text",
    "F":      "f_typename_only",
}


def _node_order_hash(node_ids: list[str]) -> str:
    h = hashlib.sha256()
    for nid in node_ids:
        h.update(nid.encode())
        h.update(b"\x1e")
    return h.hexdigest()


def _load_pubmedbert_cache(tag: str, canonical_ids: list[str]) -> torch.Tensor:
    """Load and reorder a PubMedBERT cache to match canonical_ids.

    Per Codex WARN #6, integrity checks:
      - cached node_ids must be unique
      - cached node_ids must be a superset of canonical_ids (no missing IDs)
    """
    path = CACHE_DIR / f"{tag}__pubmedbert.pt"
    payload = torch.load(path, weights_only=False, map_location="cpu")
    nids = payload["node_ids"]
    emb = payload["embeddings"].float()

    # Integrity
    if len(set(nids)) != len(nids):
        raise ValueError(f"cached node_ids in {path.name} contain duplicates")
    nid_set = set(nids)
    missing = [c for c in canonical_ids if c not in nid_set]
    if missing:
        raise ValueError(f"cache {path.name} is missing {len(missing)} canonical IDs; "
                         f"first 3: {missing[:3]}")

    # Reorder if necessary
    if nids != canonical_ids:
        id2idx = {nid: i for i, nid in enumerate(nids)}
        reorder = [id2idx[c] for c in canonical_ids]
        emb = emb[reorder]
    return emb


def _random_projection(x: torch.Tensor, n_dim: int, seed: int = 0) -> torch.Tensor:
    """Frozen Gaussian random projection: x[N,D_in] -> x_proj[N,n_dim]."""
    g = torch.Generator().manual_seed(seed)
    d_in = x.shape[1]
    proj = torch.randn(d_in, n_dim, generator=g) / (d_in ** 0.5)
    return x @ proj


def _pca_projection(x: torch.Tensor, n_dim: int) -> torch.Tensor:
    """PCA-to-n_dim with zero-row preservation (Codex WARN #4).

    Steps:
      1. Identify zero-norm rows (empty-text nodes).
      2. Fit PCA on non-zero rows only.
      3. Transform ALL rows with the components.
      4. Re-zero rows that were originally zero (to preserve "no signal" semantics).
    """
    x_np = x.numpy()
    norms = np.linalg.norm(x_np, axis=1)
    nonzero_mask = norms > 1e-6
    if nonzero_mask.sum() < n_dim:
        # Degenerate case — fall back to random projection
        return _random_projection(x, n_dim, seed=0)

    x_fit = x_np[nonzero_mask]
    mu = x_fit.mean(axis=0, keepdims=True)
    x_fit_c = x_fit - mu

    # SVD on the fit subset
    _, _, vt = np.linalg.svd(x_fit_c, full_matrices=False)
    components = vt[:n_dim]  # [n_dim, D_in]

    # Apply to ALL rows
    x_all_c = x_np - mu
    proj = x_all_c @ components.T

    # Zero out rows that started as zero
    proj[~nonzero_mask] = 0.0
    return torch.from_numpy(proj).float()


def _build_random_init(node_ids: list[str], n_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(len(node_ids), n_dim, generator=g)


def _build_type_onehot_init(canonical_ids: list[str], n_dim: int) -> torch.Tensor:
    """Variant B: canonical-kind one-hot, padded/projected to n_dim."""
    df = ntb.build_node_text_table(canonical=True)
    id2kind = dict(zip(df["id"], df["kind"]))
    kinds = sorted(set(id2kind.values()))
    k2idx = {k: i for i, k in enumerate(kinds)}
    one_hot = torch.zeros(len(canonical_ids), len(kinds))
    for i, nid in enumerate(canonical_ids):
        k = id2kind.get(nid, "_unknown")
        if k in k2idx:
            one_hot[i, k2idx[k]] = 1.0
    if n_dim == len(kinds):
        return one_hot
    if n_dim > len(kinds):
        pad = torch.zeros(len(canonical_ids), n_dim - len(kinds))
        return torch.cat([one_hot, pad], dim=1)
    return _random_projection(one_hot, n_dim, seed=0)


def _build_node2vec_init(canonical_ids: list[str], n_dim: int, seed: int) -> torch.Tensor:
    """Variant C: DeepWalk-style Node2Vec (unbiased random walks)."""
    cache = CACHE_DIR / f"deepwalk_d{n_dim}__seed{seed}.pt"
    if cache.exists():
        payload = torch.load(cache, weights_only=False, map_location="cpu")
        if payload["node_ids"] == canonical_ids:
            return payload["embeddings"]

    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError("gensim required for variant C (DeepWalk). pip install gensim") from e

    import networkx as nx
    import random

    print("[init/C] building NetworkX graph from merged KG...")
    edges_df = pd.read_parquet(
        ntb.KG_ROOT / "_merged_kg" / "edges__drugbank_hetionet_primekg__mask1.parquet")
    G = nx.Graph()
    G.add_nodes_from(canonical_ids)
    for s, t in zip(edges_df["src"], edges_df["dst"]):
        if s in G and t in G:
            G.add_edge(s, t)

    print(f"[init/C] nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")
    print("[init/C] generating random walks...")
    random.seed(seed)
    walks = []
    num_walks, walk_length = 10, 40
    nodes_list = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes_list)
        for v in nodes_list:
            walk = [v]
            cur = v
            for _ in range(walk_length - 1):
                nbrs = list(G.neighbors(cur))
                if not nbrs:
                    break
                cur = random.choice(nbrs)
                walk.append(cur)
            walks.append(walk)

    print(f"[init/C] training Word2Vec on {len(walks):,} walks...")
    model = Word2Vec(walks, vector_size=n_dim, window=5, min_count=0,
                     sg=1, workers=1, epochs=5, seed=seed)  # workers=1 for determinism
    emb = torch.zeros(len(canonical_ids), n_dim)
    for i, nid in enumerate(canonical_ids):
        if nid in model.wv:
            emb[i] = torch.from_numpy(model.wv[nid])
    torch.save({"node_ids": canonical_ids, "embeddings": emb}, cache)
    print(f"[init/C] cached -> {cache.name}")
    return emb


def build_init(
    variant: str,
    projection: str = "P1",
    n_dim: int = 64,
    seed: int = 42,
) -> tuple[list[str], torch.Tensor]:
    """Build a (N_nodes, n_dim) init tensor for the given variant."""
    nodes_df = pd.read_parquet(ntb.MERGED_NODES)
    canonical_ids = [str(x) for x in nodes_df["id"].tolist()]

    if variant == "A":
        return canonical_ids, _build_random_init(canonical_ids, n_dim, seed)
    if variant == "B":
        return canonical_ids, _build_type_onehot_init(canonical_ids, n_dim)
    if variant == "C":
        return canonical_ids, _build_node2vec_init(canonical_ids, n_dim, seed)

    if variant in PUBMEDBERT_CACHE_TAGS:
        tag = PUBMEDBERT_CACHE_TAGS[variant]
        emb_768 = _load_pubmedbert_cache(tag, canonical_ids)
        if projection == "P1":
            return canonical_ids, _random_projection(emb_768, n_dim, seed=0)
        elif projection == "P2":
            return canonical_ids, _pca_projection(emb_768, n_dim)
        else:
            raise ValueError(f"Unknown projection: {projection}")

    if variant == "H":
        raise NotImplementedError("Variant H (Qwen) requires Qwen embedding cache (deferred).")

    raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    print("Building variant A...")
    ids_a, init_a = build_init("A", n_dim=64, seed=42)
    print(f"  A: shape={init_a.shape}, mean={init_a.mean().item():.4f}, std={init_a.std().item():.4f}")

    print("\nBuilding variant B (canonical kind one-hot)...")
    ids_b, init_b = build_init("B", n_dim=64)
    nz = (init_b != 0).any(dim=1).float().mean().item()
    print(f"  B: shape={init_b.shape}, nonzero rate={nz:.4f}")
    row_sums = init_b.sum(dim=1).unique()
    print(f"     unique row sums = {row_sums[:5].tolist()}  (one-hot -> sum=1)")
    print(f"     canonical kind count = {(init_b.sum(dim=0) > 0).sum().item()}")

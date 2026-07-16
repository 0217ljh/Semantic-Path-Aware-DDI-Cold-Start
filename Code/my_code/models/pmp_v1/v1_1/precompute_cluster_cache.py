"""Precompute the cluster cache for PMP v1.1 (rewrite 2026-06-02).

Builds the offline artifacts that the v1.1 cluster-path + per-cluster
within-pool need at runtime.

Schema v2 (current). Builds three matrices over the merged KG.

  1. Drug-cluster affinity C in R^(n_drugs, n_clusters), float32.
     C[d, k] = log1p(|N_<=2(d) ∩ cluster_k|)
     where N_<=2(d) is the union of d's 1-hop and 2-hop non-drug neighbors
     (set union; a mediator that appears in both n1 and n2 is counted once).

  2. Cluster members dict cluster_members in dict[int -> list[str]].
     cluster_members[k] = sorted list of mediator_id_str for every non-drug
     KG node whose KIND_TO_GROUP maps to cluster k.
     Used for both cluster_embed initialization and within-cluster pool set
     intersection.

  3. Drug-cluster relation distribution drug_cluster_rel_dist in
     R^(n_drugs, n_clusters, n_rels_plus_special), float32.
     drug_cluster_rel_dist[d, k, r] = count of edges (d, m, r) where
     m in members(cluster_k) ∩ N_<=2(d), normalized so that the (d, k) slice
     sums to 1 (or stays all-zero when (d, k) has no edges).
     1-hop edges use their KG relation id; the 2-hop pseudo-relation uses
     rel_2hop_id (= n_rels). At runtime r_d_to_k = sum_r dist[d, k, r] *
     rel_embed[r], i.e., the expected relation embedding for the drug going
     into the cluster.

Cluster transition T is NO LONGER precomputed. v1.1 (2026-06-02 rewrite)
makes T learnable: T = softmax(W_T) with W_T a (n_clusters, n_clusters)
nn.Parameter. Therefore the cache does not include cluster_transition.

Output: Code/data/_cache/pmp_v1_1_cluster_cache_drugbank_seed42_kgonly_v2.pkl
        (v2 file name to avoid clashing with the deprecated v1 schema cache.)

Cache schema (single pickle):
  {
    "drug2id": dict[drug_id_str -> int],
    "n_drugs": int,
    "drug_cluster_affinity": (n_drugs, n_clusters) float32,
    "cluster_members": dict[int -> list[str]],
    "drug_cluster_rel_dist": (n_drugs, n_clusters, n_rels_plus_special) float32,
    "n_clusters": int,
    "type2id": dict[cluster_name -> int],
    "rel2id": dict[rel_str -> int],
    "n_rels": int,
    "rel_2hop_id": int,
    "n_rels_plus_special": int,
    "kind_order": list[str],
    "nodes_path": str,
    "edges_path": str,
    "schema_version": "pmp_v1_1_cluster_v2",
  }
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


_FILE = Path(__file__).resolve()
# v1_1/ lives under Code/my_code/models/pmp_v1/, so parents[5] = project root.
PROJECT_ROOT = _FILE.parents[5]

DEFAULT_NODES_PATH = (
    PROJECT_ROOT / "Code/data/KG/_merged_kg/"
    "nodes__drugbank_hetionet_primekg.parquet"
)
DEFAULT_EDGES_PATH = (
    PROJECT_ROOT / "Code/data/KG/_merged_kg/"
    "edges__drugbank_hetionet_primekg__mask1.parquet"
)
DEFAULT_CACHE_PATH = (
    PROJECT_ROOT / "Code/data/_cache/"
    "pmp_v1_1_cluster_cache_drugbank_seed42_kgonly_v2.pkl"
)


# Identical type grouping as PMP v1's precompute_pmp_cache.
KIND_GROUPS = {
    "protein_gene": ["Gene", "gene/protein", "Protein"],
    "pathway": ["Pathway", "pathway"],
    "side_effect": ["Side Effect", "effect/phenotype", "Symptom"],
    "disease": ["Disease", "disease"],
    "anatomy": ["Anatomy", "anatomy"],
    "compound": ["Compound"],
    "biological_process": ["Biological Process", "biological_process"],
    "molecular_function": ["Molecular Function", "molecular_function"],
    "cellular_component": ["Cellular Component", "cellular_component"],
    "pharmacologic_class": ["Pharmacologic Class"],
    "exposure": ["exposure"],
}
KIND_ORDER = list(KIND_GROUPS.keys()) + ["other"]
KIND_TO_GROUP: dict[str, str] = {}
for grp, kinds in KIND_GROUPS.items():
    for k in kinds:
        KIND_TO_GROUP[k] = grp


def _kind_to_type_id(kind: str, type2id: dict[str, int]) -> int:
    grp = KIND_TO_GROUP.get(kind, "other")
    return type2id[grp]


def build_cluster_cache(
    nodes_path: Path,
    edges_path: Path,
    out_path: Path,
) -> None:
    """Build the v1.1 cluster cache (schema v2) and write to out_path."""
    print(f"[pmp-v1.1-pre] loading nodes {nodes_path}", flush=True)
    nodes = pd.read_parquet(nodes_path)
    print(f"[pmp-v1.1-pre] loading edges {edges_path}", flush=True)
    edges = pd.read_parquet(edges_path)
    print(f"[pmp-v1.1-pre] nodes={len(nodes)} edges={len(edges)}", flush=True)

    id2kind = dict(zip(nodes["id"].astype(str), nodes["kind"].astype(str)))
    drug_set = set(
        nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"].astype(str)
    )
    print(f"[pmp-v1.1-pre] |drugs| = {len(drug_set)}", flush=True)

    type2id = {k: i for i, k in enumerate(KIND_ORDER)}
    n_clusters = len(type2id)
    rel_values = sorted(edges["relation"].astype(str).unique().tolist())
    rel2id = {r: i for i, r in enumerate(rel_values)}
    n_rels = len(rel2id)
    rel_2hop_id = n_rels
    n_rels_plus_special = n_rels + 1
    print(
        f"[pmp-v1.1-pre] n_clusters={n_clusters} n_rels={n_rels} "
        f"rel_2hop_id={rel_2hop_id}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Pass 1. Adjacency build.
    #   n1[d] = list of (mediator_id, rel_id) for 1-hop non-drug neighbors
    #   fwd_nondrug[m] = set of non-drug neighbors of mediator m
    # ------------------------------------------------------------------
    print("[pmp-v1.1-pre] pass 1: build 1-hop and non-drug adjacency ...", flush=True)
    n1: dict[str, list[tuple[str, int]]] = defaultdict(list)
    n1_set: dict[str, set[str]] = defaultdict(set)
    fwd_nondrug: dict[str, set[str]] = defaultdict(set)

    t0 = time.time()
    for i, (src, dst, rel, directed) in enumerate(zip(
        edges["src"].astype(str),
        edges["dst"].astype(str),
        edges["relation"].astype(str),
        edges["directed"].astype(bool),
    )):
        src_is_drug = src in drug_set
        dst_is_drug = dst in drug_set
        rel_id = rel2id[rel]

        if src_is_drug and not dst_is_drug:
            n1[src].append((dst, rel_id))
            n1_set[src].add(dst)
        if dst_is_drug and (not src_is_drug) and (not directed):
            n1[dst].append((src, rel_id))
            n1_set[dst].add(src)

        if (not src_is_drug) and (not dst_is_drug):
            fwd_nondrug[src].add(dst)
            if not directed:
                fwd_nondrug[dst].add(src)

        if (i + 1) % 1_000_000 == 0:
            print(
                f"[pmp-v1.1-pre]   processed {i + 1}/{len(edges)} edges  "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    print(
        f"[pmp-v1.1-pre] pass 1 done: drugs_with_n1="
        f"{sum(1 for d in drug_set if d in n1_set)}/{len(drug_set)} "
        f"time={time.time() - t0:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Pass 2. Build per-drug 2-hop set, then compute three artifacts.
    # ------------------------------------------------------------------
    print("[pmp-v1.1-pre] pass 2: build affinity + relation dist + cluster members ...", flush=True)
    drugs_sorted = sorted(drug_set)
    drug2id = {d: i for i, d in enumerate(drugs_sorted)}
    n_drugs = len(drugs_sorted)
    affinity = np.zeros((n_drugs, n_clusters), dtype=np.float32)
    # Note: drug_cluster_rel_dist is potentially (n_drugs, n_clusters,
    # n_rels_plus_special). At ~2000 drugs * 12 clusters * (~50+1 rels) =
    # ~1.2M floats = ~5 MB. Manageable as a dense matrix.
    drug_cluster_rel_dist = np.zeros(
        (n_drugs, n_clusters, n_rels_plus_special), dtype=np.float32
    )

    t1 = time.time()
    for di, drug in enumerate(drugs_sorted):
        d_idx = drug2id[drug]

        # Track per-cluster UNIQUE mediator IDs for this drug. Codex r1 fix.
        # Previously we computed affinity from sum of (cluster, rel) cells,
        # which multi-counts mediators that have multiple typed edges to the
        # same drug. The locked spec is log1p(|N_<=2(d) ∩ cluster_k|), which
        # is the cardinality of UNIQUE mediator IDs per cluster.
        cluster_unique_mediators: dict[int, set[str]] = {
            k: set() for k in range(n_clusters)
        }

        # 1-hop edges: (mediator, rel) contributes to rel_dist AND unique set
        n1_pairs = n1.get(drug, [])
        n1_med_set = n1_set.get(drug, set())
        for m, rel_id in n1_pairs:
            k = _kind_to_type_id(id2kind.get(m, ""), type2id)
            drug_cluster_rel_dist[d_idx, k, rel_id] += 1.0
            cluster_unique_mediators[k].add(m)

        # 2-hop nodes via drug -> m1 -> m2 (where m2 not drug, not in 1-hop)
        n2_med_set: set[str] = set()
        for m1 in n1_med_set:
            for m2 in fwd_nondrug.get(m1, ()):
                if m2 == drug or m2 in drug_set or m2 in n1_med_set:
                    continue
                n2_med_set.add(m2)
        # Each 2-hop mediator contributes to (cluster, rel_2hop_id) AND unique
        # set (added to the existing cluster set; set semantics handle dedup).
        for m2 in n2_med_set:
            k = _kind_to_type_id(id2kind.get(m2, ""), type2id)
            drug_cluster_rel_dist[d_idx, k, rel_2hop_id] += 1.0
            cluster_unique_mediators[k].add(m2)

        # Affinity = log1p(|N_<=2(d) ∩ cluster_k|), computed from unique sets.
        for k in range(n_clusters):
            affinity[d_idx, k] = np.log1p(float(len(cluster_unique_mediators[k])))

        if (di + 1) % 500 == 0:
            print(
                f"[pmp-v1.1-pre]   drug progress {di + 1}/{n_drugs}  "
                f"elapsed={time.time() - t1:.1f}s",
                flush=True,
            )

    # Normalize drug_cluster_rel_dist so each (d, k) slice sums to 1
    # (or stays all-zero for empty slices).
    row_sums = drug_cluster_rel_dist.sum(axis=2, keepdims=True)  # (n_drugs, n_clusters, 1)
    nonzero = row_sums > 0
    np.divide(
        drug_cluster_rel_dist, row_sums,
        out=drug_cluster_rel_dist, where=nonzero,
    )

    print(
        f"[pmp-v1.1-pre] pass 2 done: time={time.time() - t1:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Pass 3. cluster_members. For each KG node m that is not a drug, look up
    # its cluster k by KIND_TO_GROUP; append m to cluster_members[k].
    # ------------------------------------------------------------------
    print("[pmp-v1.1-pre] pass 3: build cluster_members ...", flush=True)
    cluster_members: dict[int, list[str]] = {k: [] for k in range(n_clusters)}
    t2 = time.time()
    for node_id, kind in id2kind.items():
        if node_id in drug_set:
            continue
        k = _kind_to_type_id(kind, type2id)
        cluster_members[k].append(node_id)
    # Sort each list for determinism
    for k in cluster_members:
        cluster_members[k] = sorted(cluster_members[k])
    member_counts = {k: len(v) for k, v in cluster_members.items()}
    print(
        f"[pmp-v1.1-pre] pass 3 done: cluster_members count per cluster="
        f"{member_counts} time={time.time() - t2:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Write cache.
    # ------------------------------------------------------------------
    payload = {
        "drug2id": drug2id,
        "n_drugs": n_drugs,
        "drug_cluster_affinity": affinity,
        "cluster_members": cluster_members,
        "drug_cluster_rel_dist": drug_cluster_rel_dist,
        "n_clusters": n_clusters,
        "type2id": type2id,
        "rel2id": rel2id,
        "n_rels": n_rels,
        "rel_2hop_id": rel_2hop_id,
        "n_rels_plus_special": n_rels_plus_special,
        "kind_order": KIND_ORDER,
        "nodes_path": str(nodes_path),
        "edges_path": str(edges_path),
        "schema_version": "pmp_v1_1_cluster_v2",
    }
    print(f"[pmp-v1.1-pre] writing -> {out_path}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[pmp-v1.1-pre] DONE. cache size: {out_path.stat().st_size / 1e6:.1f} MB",
        flush=True,
    )


def ensure_cluster_cache(
    cache_path: str | Path | None = None,
    nodes_path: str | Path | None = None,
    edges_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Detect-and-build helper. HIT -> return existing path. MISS -> build."""
    cache_p = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH
    if cache_p.is_file() and not force:
        print(
            f"[pmp-v1.1-cache] HIT: {cache_p} "
            f"({cache_p.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        return cache_p
    print(
        f"[pmp-v1.1-cache] "
        f"{'FORCE-REBUILD' if force else 'MISS'} at {cache_p}; building ...",
        flush=True,
    )
    nodes_p = Path(nodes_path) if nodes_path is not None else DEFAULT_NODES_PATH
    edges_p = Path(edges_path) if edges_path is not None else DEFAULT_EDGES_PATH
    if not nodes_p.is_file():
        raise FileNotFoundError(f"KG nodes parquet missing: {nodes_p}")
    if not edges_p.is_file():
        raise FileNotFoundError(f"KG edges parquet missing: {edges_p}")
    build_cluster_cache(nodes_p, edges_p, cache_p)
    print(f"[pmp-v1.1-cache] BUILT: {cache_p}", flush=True)
    return cache_p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=str, default=str(DEFAULT_NODES_PATH))
    p.add_argument("--edges", type=str, default=str(DEFAULT_EDGES_PATH))
    p.add_argument("--out", type=str, default=str(DEFAULT_CACHE_PATH))
    p.add_argument("--force", action="store_true",
                   help="rebuild even if cache exists")
    args = p.parse_args()
    ensure_cluster_cache(
        cache_path=args.out,
        nodes_path=args.nodes,
        edges_path=args.edges,
        force=args.force,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_cluster_cache",
    "ensure_cluster_cache",
    "DEFAULT_NODES_PATH",
    "DEFAULT_EDGES_PATH",
    "DEFAULT_CACHE_PATH",
    "KIND_GROUPS",
    "KIND_ORDER",
]

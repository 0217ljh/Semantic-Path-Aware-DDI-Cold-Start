"""E1b — PK/PD path endpoint asymmetry test on merged KG.

Pipeline:
  1. Load PKL (800-drug seed42), get train + val + test positive pairs
  2. Attach ddi_type via ddi_edges.csv join
  3. Map ddi_type -> PK/PD/Mixed via pkpd.parquet
  4. Load merged-KG nodes + edges parquet, build undirected adj
  5. For each PK/PD-labeled positive pair (sample if too many), find shortest
     paths ≤3 hops; classify intermediates by canonical layer
  6. Build 2 x 6 contingency (PK/PD x {Drug, Mol, Effect, Other, BioProcess, NoPath})
  7. Chi-square omnibus + per-class histograms

Output:
  Notes/Experiments/_results/screen1/2026-05-20__e1b_pkpd_endpoint.md
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

# --- paths
KG_ROOT = PROJECT_ROOT / "Code" / "data" / "KG"
MERGED_NODES = KG_ROOT / "_merged_kg" / "nodes__drugbank_hetionet_primekg.parquet"
MERGED_EDGES = KG_ROOT / "_merged_kg" / "edges__drugbank_hetionet_primekg__mask1.parquet"
DDI_EDGES_CSV = KG_ROOT / "drugbank" / "filtered" / "ddi_edges.csv"
PKPD_PARQUET = PROJECT_ROOT / "Code/data/private/outputs_full/annotations/pkpd.parquet"
PKL = PROJECT_ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"

RESULTS_DIR = PROJECT_ROOT / "Notes" / "Experiments" / "_results" / "screen1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- canonical layer mapping (per first_step_plan.md §"CRITICAL PATH" E1b)
LAYER_PK = {"Gene/Protein", "Pathway"}   # molecular layer
LAYER_PD = {"SideEffect", "Disease", "Anatomy", "Phenotype"}  # effect system
LAYER_BP = {"BiologicalProcess", "MolecularFunction", "CellularComponent"}
LAYER_OTHER = {"Drug", "PharmacologicClass", "Exposure", "_unknown"}


def _canonical_kind(raw_kind: str) -> str:
    """Replicate ntb._KIND_CANONICAL mapping inline (avoid heavy import)."""
    m = {
        "Drug": "Drug", "drug": "Drug", "Compound": "Drug",
        "Gene": "Gene/Protein", "gene": "Gene/Protein",
        "gene/protein": "Gene/Protein", "Protein": "Gene/Protein",
        "enzyme": "Gene/Protein", "transporter": "Gene/Protein",
        "carrier": "Gene/Protein", "target": "Gene/Protein",
        "Side Effect": "SideEffect", "side_effect": "SideEffect",
        "SideEffect": "SideEffect", "drug_effect": "SideEffect",
        "Disease": "Disease", "disease": "Disease",
        "Anatomy": "Anatomy", "anatomy": "Anatomy",
        "Pathway": "Pathway", "pathway": "Pathway",
        "Symptom": "Phenotype", "symptom": "Phenotype",
        "Phenotype": "Phenotype", "effect/phenotype": "Phenotype",
        "BiologicalProcess": "BiologicalProcess",
        "Biological Process": "BiologicalProcess",
        "biological_process": "BiologicalProcess",
        "MolecularFunction": "MolecularFunction",
        "Molecular Function": "MolecularFunction",
        "molecular_function": "MolecularFunction",
        "CellularComponent": "CellularComponent",
        "Cellular Component": "CellularComponent",
        "cellular_component": "CellularComponent",
        "PharmacologicClass": "PharmacologicClass",
        "pharmacologic_class": "PharmacologicClass",
        "exposure": "Exposure",
    }
    return m.get(raw_kind, raw_kind)


def _layer_bucket(canon_kind: str) -> str:
    if canon_kind in LAYER_PK:
        return "PK_layer"
    if canon_kind in LAYER_PD:
        return "PD_layer"
    if canon_kind in LAYER_BP:
        return "BioProcess"
    return "Other"


def _build_adj(nodes_df, edges_df) -> tuple[dict, dict]:
    """Return adj[node_id] = set(neighbor_ids), id2kind dict (canonical)."""
    id2kind = {}
    for nid, k in zip(nodes_df["id"], nodes_df["kind"].astype(str)):
        id2kind[str(nid)] = _canonical_kind(k)
    adj = defaultdict(set)
    for s, d in zip(edges_df["src"], edges_df["dst"]):
        s, d = str(s), str(d)
        adj[s].add(d)
        adj[d].add(s)  # treat as undirected
    return adj, id2kind


def _shortest_paths_bfs(adj, src, dst, max_hops=3, max_paths_per_pair=3):
    """BFS up to `max_hops` for shortest paths from src to dst.

    Returns list of paths (each is list of node_ids including src and dst).
    Bounded by max_paths_per_pair.
    """
    if src not in adj or dst not in adj:
        return []
    if src == dst:
        return [[src]]
    # BFS layer by layer
    parents = {src: None}
    frontier = {src}
    found_at_hop = None
    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for n in frontier:
            for nb in adj[n]:
                if nb not in parents:
                    parents[nb] = n
                    next_frontier.add(nb)
        if dst in next_frontier:
            found_at_hop = hop
            break
        frontier = next_frontier
        if not frontier:
            break

    if found_at_hop is None:
        return []

    # Reconstruct ONE shortest path (full enumeration of all shortest paths is
    # exponential — we sacrifice multiplicity for tractability on a 178K graph).
    path = [dst]
    cur = dst
    while parents.get(cur) is not None:
        cur = parents[cur]
        path.append(cur)
    path.reverse()
    return [path]


def main():
    print("=" * 72)
    print("E1b — PK/PD path endpoint asymmetry")
    print("=" * 72)

    # --- load pkpd labels
    print("\n[1/6] Loading pkpd.parquet...")
    pkpd = pd.read_parquet(PKPD_PARQUET)
    print(f"  pkpd labels: {pkpd['pk_pd_label'].value_counts().to_dict()}")
    pkpd_label_by_type = dict(zip(pkpd["ddi_type"], pkpd["pk_pd_label"]))

    # --- load PKL pairs (positive only)
    print("\n[2/6] Loading PKL train+val+test positives...")
    sys.path.insert(0, str(PROJECT_ROOT / "Code"))
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    pos_pairs = pd.concat([
        ds.splits.train[["drug_a_id", "drug_b_id"]].assign(_split="train"),
        ds.splits.val_s0[["drug_a_id", "drug_b_id"]].assign(_split="val_s0"),
        ds.splits.val_s1[["drug_a_id", "drug_b_id"]].assign(_split="val_s1"),
        ds.splits.val_s2[["drug_a_id", "drug_b_id"]].assign(_split="val_s2"),
        ds.splits.test_s0[["drug_a_id", "drug_b_id"]].assign(_split="test_s0"),
        ds.splits.test_s1[["drug_a_id", "drug_b_id"]].assign(_split="test_s1"),
        ds.splits.test_s2[["drug_a_id", "drug_b_id"]].assign(_split="test_s2"),
    ], ignore_index=True)
    print(f"  total positives: {len(pos_pairs):,}")

    # --- attach ddi_type
    print("\n[3/6] Attaching ddi_type from ddi_edges.csv...")
    edges = pd.read_csv(DDI_EDGES_CSV, usecols=["drug_a_id", "drug_b_id", "ddi_type"])
    key_to_type = {}
    for a, b, t in zip(edges["drug_a_id"], edges["drug_b_id"], edges["ddi_type"]):
        key_to_type[(str(a), str(b))] = str(t)
        key_to_type[(str(b), str(a))] = str(t)
    pos_pairs["ddi_type"] = pos_pairs.apply(
        lambda r: key_to_type.get((str(r["drug_a_id"]), str(r["drug_b_id"]))), axis=1)
    pos_pairs["pk_pd_label"] = pos_pairs["ddi_type"].map(pkpd_label_by_type)
    n_with_label = pos_pairs["pk_pd_label"].notna().sum()
    print(f"  pos pairs with pk_pd label: {n_with_label:,}/{len(pos_pairs):,} "
          f"({100*n_with_label/len(pos_pairs):.1f}%)")
    print(f"  label distribution: {pos_pairs['pk_pd_label'].value_counts().to_dict()}")

    # --- subsample to keep BFS tractable: ≤3000 per class
    rng = np.random.default_rng(42)
    sampled = []
    for cls in ("PK", "PD"):
        sub = pos_pairs[pos_pairs["pk_pd_label"] == cls]
        n_sample = min(3000, len(sub))
        idx = rng.choice(len(sub), size=n_sample, replace=False)
        sampled.append(sub.iloc[idx])
    sampled_df = pd.concat(sampled, ignore_index=True)
    print(f"\n[4/6] Sampled {len(sampled_df):,} pairs for path enumeration "
          f"(PK={len(sampled[0]):,} PD={len(sampled[1]):,})")

    # --- build KG adjacency
    print("\n[5/6] Loading merged KG (178K nodes)...")
    t0 = time.time()
    nodes_df = pd.read_parquet(MERGED_NODES)
    edges_df = pd.read_parquet(MERGED_EDGES)
    adj, id2kind = _build_adj(nodes_df, edges_df)
    print(f"  built adj for {len(adj):,} nodes ({len(edges_df):,} edges) in {time.time()-t0:.1f}s")

    # --- enumerate shortest paths
    print("\n[6/6] Enumerating shortest paths (≤3 hops, 1 path per pair)...")
    t0 = time.time()
    layer_counts = {"PK": Counter(), "PD": Counter()}
    intermediate_kind_hist = {"PK": Counter(), "PD": Counter()}
    n_no_path = {"PK": 0, "PD": 0}
    n_processed = 0
    for _, r in sampled_df.iterrows():
        n_processed += 1
        if n_processed % 500 == 0:
            print(f"  processed {n_processed}/{len(sampled_df)} pairs "
                  f"({time.time()-t0:.0f}s elapsed)")
        cls = r["pk_pd_label"]
        if cls not in ("PK", "PD"):
            continue
        a, b = str(r["drug_a_id"]), str(r["drug_b_id"])
        paths = _shortest_paths_bfs(adj, a, b, max_hops=3, max_paths_per_pair=1)
        if not paths:
            n_no_path[cls] += 1
            layer_counts[cls]["NoPath"] += 1
            continue
        # Take the single path's intermediates (exclude both endpoints)
        path = paths[0]
        if len(path) <= 2:
            layer_counts[cls]["Direct"] += 1
            continue
        for nid in path[1:-1]:
            kind = id2kind.get(nid, "_unknown")
            intermediate_kind_hist[cls][kind] += 1
            layer_counts[cls][_layer_bucket(kind)] += 1

    print(f"\n  no-path counts: {n_no_path}")

    # --- build contingency and chi-square
    print("\n=" * 72)
    print("Results")
    print("=" * 72)
    layers = ["PK_layer", "PD_layer", "BioProcess", "Other", "Direct", "NoPath"]
    contingency = np.zeros((2, len(layers)), dtype=np.int64)
    for i, cls in enumerate(("PK", "PD")):
        for j, layer in enumerate(layers):
            contingency[i, j] = layer_counts[cls].get(layer, 0)

    print(f"\n2x6 Contingency table (rows: PK/PD, cols: layer):")
    print(f"{'':<12}", end=""); print("".join(f"{c:>12}" for c in layers))
    for i, cls in enumerate(("PK", "PD")):
        print(f"{cls:<12}", end=""); print("".join(f"{contingency[i,j]:>12,}" for j in range(len(layers))))

    # Chi-square
    from scipy.stats import chi2_contingency
    # Drop "NoPath" and "Direct" columns + any all-zero column (chi-square requires nonzero expectations)
    keep_cols = [j for j, c in enumerate(layers)
                 if c not in ("Direct", "NoPath") and contingency[:, j].sum() > 0]
    table = contingency[:, keep_cols]
    try:
        chi2, p, dof, exp = chi2_contingency(table)
        print(f"\nChi-square omnibus (excl. Direct/NoPath + zero cols): "
              f"chi2={chi2:.2f}, dof={dof}, p={p:.2e}")
    except Exception as e:
        print(f"\nChi-square failed: {e}")
        chi2, p, dof = float("nan"), float("nan"), 0

    # Per-class layer share
    print(f"\nPer-class layer share (semantic layers only):")
    for i, cls in enumerate(("PK", "PD")):
        total = sum(contingency[i, keep_cols])
        if total == 0:
            continue
        print(f"  {cls}:", end="")
        for j in keep_cols:
            share = 100 * contingency[i, j] / total
            print(f"  {layers[j]}={share:.1f}%", end="")
        print()

    # Top 5 intermediate kinds per class
    print(f"\nTop 5 intermediate node kinds per class:")
    for cls in ("PK", "PD"):
        print(f"  {cls}: {intermediate_kind_hist[cls].most_common(5)}")

    # --- write report
    out_md = RESULTS_DIR / "2026-05-20__e1b_pkpd_endpoint.md"
    lines = [
        "# E1b PK/PD Path Endpoint Asymmetry — Motivation for i1",
        "",
        f"**Date**: 2026-05-20",
        f"**Script**: `Code/my_code/exp_e1b_pkpd_endpoint/run_e1b.py`",
        "",
        "## Claim",
        "PK-pair intermediates concentrate in molecular layer "
        "(Gene/Protein, Pathway); PD-pair intermediates concentrate in "
        "effect-system layer (SideEffect, Disease, Anatomy, Phenotype).",
        "",
        "## Inputs",
        f"- pkpd labels: 215 ddi_types ({pkpd['pk_pd_label'].value_counts().to_dict()})",
        f"- positive pairs sampled: {len(sampled_df):,} "
        f"(PK={len(sampled[0]):,} PD={len(sampled[1]):,})",
        f"- KG: merged DrugBank+Hetionet+PrimeKG ({len(nodes_df):,} nodes, {len(edges_df):,} edges)",
        f"- path enumeration: ≤3 hops, 1 shortest path per pair (tractability)",
        "",
        "## 2x6 contingency",
        "",
        f"| class | {' | '.join(layers)} |",
        "|" + "---|" * (len(layers) + 1),
    ]
    for i, cls in enumerate(("PK", "PD")):
        lines.append(f"| {cls} | " + " | ".join(f"{contingency[i,j]:,}" for j in range(len(layers))) + " |")
    lines += [
        "",
        f"**Chi-square omnibus (excl Direct/NoPath)**: chi2={chi2:.2f}, dof={dof}, **p={p:.2e}**",
        "",
        "## Per-class semantic-layer share",
        "",
    ]
    for i, cls in enumerate(("PK", "PD")):
        total = sum(contingency[i, keep_cols])
        if total > 0:
            shares = [f"{layers[j]}={100*contingency[i,j]/total:.1f}%" for j in keep_cols]
            lines.append(f"- **{cls}**: " + ", ".join(shares))
    lines += [
        "",
        "## Top 5 intermediate node kinds per class",
        "",
    ]
    for cls in ("PK", "PD"):
        top = intermediate_kind_hist[cls].most_common(5)
        lines.append(f"- **{cls}**: " + ", ".join(f"{k}={v:,}" for k, v in top))

    out_md.write_text("\n".join(lines))
    print(f"\nReport saved -> {out_md}")


if __name__ == "__main__":
    main()

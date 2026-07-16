"""Merged-KG builder for EmerGNN — reads the merged DrugBank+Hetionet+PrimeKG
parquet edge table and produces the same artifacts schema as
`build_kg_from_kb`, but with `n_rel = number of distinct relations in
merged KG` instead of the DrugBank-specific 5-bucket schema.

Output dict (matches `build_kg_from_kb`):
    entity2id:     Dict[str, int]
    id2entity:     Dict[int, str]
    entity_types:  np.ndarray[int]   (0 for drugs, 1 for non-drug entities)
    drug_ids:      List[str]         (drugs first, in vocab order)
    triplets:      np.ndarray (n_edges, 3)  -- (head, tail, rel)
    n_ent:         int
    n_rel:         int               (number of distinct relations)
    rel2id:        Dict[str, int]    (extra: for inspection)

Conventions (per codex constraints):
  1. Relation strings stripped of whitespace (light canonicalize); kept
     case/prefix-distinct (e.g., `db:target` ≠ `het:CbG`).
  2. Triplets dedup-ed at (head, tail, rel) level.
  3. Direction preserved: drug-incident edges emit (drug_head, tail, rel).
     For edges where dst is a drug but src is NOT, we emit (drug=dst, tail=src, rel)
     — drug always as head in the triplet table.
  4. Leakage blacklist: empty by default (audit confirmed no direct DDI
     labels in merged KG; per Option B het:CrC retained as in EmerGNN paper).
  5. Drug namespace alignment: `drug_ids` argument is canonical; any drug in
     `drug_ids` missing from merged KG just contributes no edges.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Drug kind labels in the merged KG nodes parquet.
_DRUG_KINDS: set[str] = {"Drug", "drug"}

# Default leakage blacklist (empty per Option B). If you later want to
# remove specific relations as DDI-leakage suspects, add them here.
_DEFAULT_BLOCKLIST: frozenset[str] = frozenset()


def build_kg_from_merged_parquet(
    edges_parquet_path: str | Path,
    drug_ids: Iterable[str],
    *,
    blocklist: Iterable[str] = (),
    verbose: bool = False,
) -> dict:
    """Build EmerGNN-compatible KG artifacts from merged KG parquet.

    Args:
        edges_parquet_path: path to
            `Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet`.
            Required columns: src, src_kind, dst, dst_kind, relation, directed.
        drug_ids: iterable of drug IDs that must be the FIRST entries in
            the entity vocab (order preserved by `sorted(set(...))`).
        blocklist: optional iterable of relation names to drop (in addition
            to the empty `_DEFAULT_BLOCKLIST`).
        verbose: print summary stats.

    Returns:
        Dict with EmerGNN-compatible KG artifacts (see module docstring).
    """
    edges = pd.read_parquet(edges_parquet_path)
    # Light canonicalize — strip whitespace; keep case (relations are
    # source-prefixed like `db:target` / `het:CbG` / `prime:drug_protein`,
    # so case/prefix is meaningful and must NOT be collapsed).
    edges["relation"] = edges["relation"].astype(str).str.strip()

    # Apply blocklist
    block = frozenset(_DEFAULT_BLOCKLIST) | frozenset(blocklist)
    if block:
        n_before = len(edges)
        edges = edges[~edges["relation"].isin(block)]
        if verbose:
            print(f"[merged_kg] dropped {n_before - len(edges)} edges via blocklist "
                  f"({sorted(block)})")

    # Canonical drug list (deterministic order: drug_ids sorted)
    drug_id_list = sorted({str(d) for d in drug_ids})
    drug_set = set(drug_id_list)

    # Drug-incident edges: src in drug_set OR dst in drug_set
    src_is_drug = edges["src"].isin(drug_set)
    dst_is_drug = edges["dst"].isin(drug_set)
    drug_incident = edges[src_is_drug | dst_is_drug].copy()
    if verbose:
        print(f"[merged_kg] total edges: {len(edges):,}; drug-incident: {len(drug_incident):,}")

    # Normalize: always (drug_head, tail, rel)
    #   - src is drug → (head=src, tail=dst)
    #   - dst is drug AND src is NOT drug → flip (head=dst, tail=src)
    #   - both drug (drug-drug, e.g., het:CrC) → keep original direction
    src_in_drug = drug_incident["src"].isin(drug_set)
    dst_in_drug = drug_incident["dst"].isin(drug_set)
    flip = (~src_in_drug) & dst_in_drug

    h = drug_incident["src"].where(~flip, drug_incident["dst"])
    t = drug_incident["dst"].where(~flip, drug_incident["src"])
    r = drug_incident["relation"]

    triplet_df = pd.DataFrame({"head": h.values, "tail": t.values, "rel": r.values})

    # Dedup at (head, tail, rel) level
    before_dedup = len(triplet_df)
    triplet_df = triplet_df.drop_duplicates(subset=["head", "tail", "rel"]).reset_index(drop=True)
    if verbose:
        print(f"[merged_kg] dedup'd: {before_dedup - len(triplet_df):,} duplicates removed")

    # Entity vocab: drugs FIRST in canonical sorted order, then non-drug
    # entities (also sorted for determinism).
    entity2id: dict[str, int] = {d: i for i, d in enumerate(drug_id_list)}
    n_drugs = len(drug_id_list)

    all_ents = set(triplet_df["head"]) | set(triplet_df["tail"])
    non_drug_ents = sorted(all_ents - drug_set)
    for ent in non_drug_ents:
        entity2id[ent] = len(entity2id)
    n_ent = len(entity2id)

    # Map IDs to indices
    h_idx = triplet_df["head"].map(entity2id).to_numpy(dtype=np.int64)
    t_idx = triplet_df["tail"].map(entity2id).to_numpy(dtype=np.int64)

    # Enumerate relations (sorted for determinism)
    rel_list = sorted(triplet_df["rel"].unique())
    rel2id = {r_: i for i, r_ in enumerate(rel_list)}
    n_rel = len(rel_list)
    r_idx = triplet_df["rel"].map(rel2id).to_numpy(dtype=np.int64)

    triplets = np.column_stack([h_idx, t_idx, r_idx]).astype(np.int64)

    # Entity types: 0 = drug, 1 = non-drug (coarse — EmerGNN model uses
    # this only as a passive tag; type-aware aggregation not differentiated
    # beyond drug/non-drug in current model code).
    entity_types = np.zeros(n_ent, dtype=np.int64)
    entity_types[n_drugs:] = 1

    if verbose:
        print(f"[merged_kg] n_drugs={n_drugs}, n_ent={n_ent}, n_rel={n_rel}, "
              f"n_triplets={len(triplets):,}")
        # Top-5 relations by edge count
        rel_counts = triplet_df["rel"].value_counts().head(5)
        print(f"[merged_kg] top relations: \n{rel_counts.to_string()}")

    return {
        "entity2id": entity2id,
        "id2entity": {v: k for k, v in entity2id.items()},
        "entity_types": entity_types,
        "drug_ids": drug_id_list,
        "triplets": triplets,
        "n_ent": n_ent,
        "n_rel": n_rel,
        "rel2id": rel2id,
    }


def build_full_kg_from_merged_parquet(
    edges_parquet_path: str | Path,
    drug_ids: Iterable[str],
    *,
    blocklist: Iterable[str] = (),
    verbose: bool = False,
) -> dict:
    """Build EmerGNN-compatible KG artifacts from the FULL merged KG.

    Identical to :func:`build_kg_from_merged_parquet` EXCEPT it does NOT drop
    non-drug<->non-drug edges — every edge in the merged parquet is kept, so the
    resulting KG is the whole biomedical graph (drugs, proteins, pathways,
    diseases, ...), not just the 1-hop drug neighborhood. Vocab still lists drugs
    FIRST (so drug node ids match the drug-incident builder), and the same
    drug-as-head normalization is applied to edges with exactly one drug endpoint;
    non-drug edges keep their original ``src -> dst`` direction (``build_sparse_adj``
    adds the reverse edge downstream regardless).

    Args / Returns: same schema as :func:`build_kg_from_merged_parquet`.
    """
    edges = pd.read_parquet(edges_parquet_path)
    edges["relation"] = edges["relation"].astype(str).str.strip()

    block = frozenset(_DEFAULT_BLOCKLIST) | frozenset(blocklist)
    if block:
        n_before = len(edges)
        edges = edges[~edges["relation"].isin(block)]
        if verbose:
            print(f"[merged_kg:full] dropped {n_before - len(edges)} edges via blocklist "
                  f"({sorted(block)})")

    drug_id_list = sorted({str(d) for d in drug_ids})
    drug_set = set(drug_id_list)

    # FULL KG: keep every edge (no drug-incident filter).
    all_edges = edges.copy()
    if verbose:
        n_di = int((all_edges["src"].isin(drug_set) | all_edges["dst"].isin(drug_set)).sum())
        print(f"[merged_kg:full] total edges: {len(all_edges):,}; "
              f"(drug-incident subset: {n_di:,})")

    # Normalize direction: flip only when dst is a drug and src is NOT
    # (so a single-drug edge has the drug as head, matching the drug-incident
    # builder). Non-drug<->non-drug and drug<->drug edges keep original direction.
    src_in_drug = all_edges["src"].isin(drug_set)
    dst_in_drug = all_edges["dst"].isin(drug_set)
    flip = (~src_in_drug) & dst_in_drug

    h = all_edges["src"].where(~flip, all_edges["dst"])
    t = all_edges["dst"].where(~flip, all_edges["src"])
    r = all_edges["relation"]

    triplet_df = pd.DataFrame({"head": h.values, "tail": t.values, "rel": r.values})

    before_dedup = len(triplet_df)
    triplet_df = triplet_df.drop_duplicates(subset=["head", "tail", "rel"]).reset_index(drop=True)
    if verbose:
        print(f"[merged_kg:full] dedup'd: {before_dedup - len(triplet_df):,} duplicates removed")

    # Entity vocab: drugs FIRST (canonical sorted), then non-drug entities.
    entity2id: dict[str, int] = {d: i for i, d in enumerate(drug_id_list)}
    n_drugs = len(drug_id_list)

    all_ents = set(triplet_df["head"]) | set(triplet_df["tail"])
    non_drug_ents = sorted(all_ents - drug_set)
    for ent in non_drug_ents:
        entity2id[ent] = len(entity2id)
    n_ent = len(entity2id)

    h_idx = triplet_df["head"].map(entity2id).to_numpy(dtype=np.int64)
    t_idx = triplet_df["tail"].map(entity2id).to_numpy(dtype=np.int64)

    rel_list = sorted(triplet_df["rel"].unique())
    rel2id = {r_: i for i, r_ in enumerate(rel_list)}
    n_rel = len(rel_list)
    r_idx = triplet_df["rel"].map(rel2id).to_numpy(dtype=np.int64)

    triplets = np.column_stack([h_idx, t_idx, r_idx]).astype(np.int64)

    entity_types = np.zeros(n_ent, dtype=np.int64)
    entity_types[n_drugs:] = 1

    if verbose:
        print(f"[merged_kg:full] n_drugs={n_drugs}, n_ent={n_ent}, n_rel={n_rel}, "
              f"n_triplets={len(triplets):,}")
        rel_counts = triplet_df["rel"].value_counts().head(5)
        print(f"[merged_kg:full] top relations: \n{rel_counts.to_string()}")

    return {
        "entity2id": entity2id,
        "id2entity": {v: k for k, v in entity2id.items()},
        "entity_types": entity_types,
        "drug_ids": drug_id_list,
        "triplets": triplets,
        "n_ent": n_ent,
        "n_rel": n_rel,
        "rel2id": rel2id,
    }

"""TIGER BKG (Background Knowledge Graph) builder.

Faithfully reproduces the ``Code-Released/baseline/TIGER/train_custom_bundle.py``
BKG construction semantics, but with the KG source pointed at our
merged DrugBank+Hetionet+PrimeKG parquet (default) instead of the
upstream ``dataset/drugbank/networks.txt``.

Node layout (matches upstream):
  * indices ``0 .. n_drugs - 1`` are drugs (``drug_to_idx``)
  * indices ``n_drugs .. n_drugs + n_entities - 1`` are non-drug entities

Edge layout:
  * Relation ``0`` is reserved for DDI-train edges and isolated-drug
    self-loops (upstream uses ``1`` for both; we use ``0`` and shift
    everything else by 1 to keep ``num_rel = K + 1`` clean).
  * Relations ``1 .. K`` enumerate distinct KG relations (sorted for
    determinism).

Each logical edge is stored ONCE; the subgraph builder
(:mod:`baseline.tiger.subgraph_features`) un-directs by
concatenating reversed edges with the same relation id, matching
``data_process.generate_node_subgraphs`` upstream.

Cold-start: g2 (unseen) drugs are passed in via ``g2_drug_ids``; DDI
training edges that touch any g2 drug are SKIPPED, mirroring the
upstream "BKG: DDI edges train-only, exclude any edge touching g2"
guard. Drug-entity edges from the KG itself are kept for ALL drugs so
g2 drugs still get an entity neighborhood at inference time.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Reserve relation id 0 for DDI + self-loop (combined, matches TIGER's
# upstream "rel 1 = DDI/self-loop" convention but renumbered to start
# at 0 so ``num_rel`` lines up with the embedding layer.)
RELID_DDI_SELFLOOP: int = 0


def _drug_incident_edges(
    edges_df: pd.DataFrame, drug_set: set[str]
) -> pd.DataFrame:
    """Return rows whose ``src`` or ``dst`` is a drug.

    Normalizes direction so that drug is always at ``head``:
      * src ∈ drug_set → (head=src, tail=dst)
      * dst ∈ drug_set and src ∉ drug_set → flip to (head=dst, tail=src)
      * both drugs (drug-drug edges) → keep original direction
    """
    src_in = edges_df["src"].isin(drug_set)
    dst_in = edges_df["dst"].isin(drug_set)
    incident = edges_df[src_in | dst_in].copy()
    if incident.empty:
        return incident

    src_in_i = incident["src"].isin(drug_set)
    dst_in_i = incident["dst"].isin(drug_set)
    flip = (~src_in_i) & dst_in_i

    incident["head"] = incident["src"].where(~flip, incident["dst"])
    incident["tail"] = incident["dst"].where(~flip, incident["src"])
    return incident[["head", "tail", "relation"]]


def _full_kg_edges(edges_df: pd.DataFrame, drug_set: set[str]) -> pd.DataFrame:
    """FULL merged-KG variant of :func:`_drug_incident_edges`: keep EVERY edge (NOT
    just drug-incident). Same drug-head normalization for single-drug edges; drug-drug
    and non-drug edges keep original src->dst. Standing decision 2026-07-01: all KG
    baselines use the full merged KG by default."""
    incident = edges_df.copy()
    if incident.empty:
        return incident[["head", "tail", "relation"]] if "head" in incident else incident
    src_in = incident["src"].isin(drug_set)
    dst_in = incident["dst"].isin(drug_set)
    flip = (~src_in) & dst_in
    incident["head"] = incident["src"].where(~flip, incident["dst"])
    incident["tail"] = incident["dst"].where(~flip, incident["src"])
    return incident[["head", "tail", "relation"]]


def build_bkg_from_merged_parquet(
    edges_parquet_path: str | Path,
    drug_ids: Iterable[str],
    train_ddi_pairs: pd.DataFrame | None,
    *,
    g2_drug_ids: Iterable[str] = (),
    blocklist: Iterable[str] = (),
    kg_scope: str = "full",
    verbose: bool = False,
) -> dict:
    """Build TIGER-compatible BKG from merged KG + DDI-train pairs.

    Args:
        edges_parquet_path: path to merged KG edges parquet (must
            contain ``src``, ``dst``, ``relation`` columns).
        drug_ids: full ordered drug list. Indices ``0..n_drugs-1`` are
            assigned in ``sorted(set(drug_ids))`` order.
        train_ddi_pairs: DataFrame with ``drug_a_id``, ``drug_b_id``
            columns — positive DDI edges from the TRAIN split only. May
            be ``None`` or empty.
        g2_drug_ids: iterable of cold-start (unseen) drug ids; DDI-train
            edges touching any of these are dropped.
        blocklist: relation names to exclude from the KG (e.g. suspect
            DDI-leakage rels).
        verbose: print summary stats.

    Returns:
        Dict with keys:
            drug_to_idx       — {drugbank_id: 0..n_drugs-1}
            n_drugs           — int
            n_entities        — int (non-drug nodes)
            n_total_nodes     — n_drugs + n_entities
            num_rel           — K + 1 (DDI/self-loop + K KG rels)
            edge_list         — [[u, v], ...] one direction each
            rel_list          — [r, ...] same length as edge_list
            rel2id            — {rel_str: 1..K} (DDI is id 0)
            ddi_edges_kept    — int (count of train DDI edges added)
            ddi_edges_skipped — int (count of train DDI edges dropped due to g2)
            g2_idx            — set[int] (indices of unseen drugs in BKG node space)
    """
    drug_list = sorted({str(d) for d in drug_ids})
    drug_to_idx: dict[str, int] = {d: i for i, d in enumerate(drug_list)}
    n_drugs = len(drug_list)
    drug_set = set(drug_list)

    g2_set = {str(d) for d in g2_drug_ids if str(d) in drug_to_idx}
    g2_idx_set: set[int] = {drug_to_idx[d] for d in g2_set}

    # ── 1. Load + filter KG edges ────────────────────────────────────────
    edges = pd.read_parquet(edges_parquet_path)
    edges["relation"] = edges["relation"].astype(str).str.strip()
    block = frozenset(blocklist)
    if block:
        edges = edges[~edges["relation"].isin(block)]
    if kg_scope == "full":
        incident = _full_kg_edges(edges, drug_set)
    elif kg_scope == "drug_incident":
        incident = _drug_incident_edges(edges, drug_set)
    else:
        raise ValueError(f"kg_scope must be 'full' or 'drug_incident'; got {kg_scope!r}")

    # Dedup at (head, tail, relation) level
    if not incident.empty:
        incident = incident.drop_duplicates(
            subset=["head", "tail", "relation"]
        ).reset_index(drop=True)

    # ── 2. Build entity vocab: drugs first, then non-drug entities ──────
    all_tails = set(incident["tail"]) if not incident.empty else set()
    all_heads = set(incident["head"]) if not incident.empty else set()
    non_drug_entities = sorted((all_tails | all_heads) - drug_set)
    entity_to_idx: dict[str, int] = dict(drug_to_idx)
    for ent in non_drug_entities:
        entity_to_idx[ent] = len(entity_to_idx)
    n_entities = len(non_drug_entities)
    n_total = n_drugs + n_entities

    # ── 3. Assign relation ids: 0 = DDI/self-loop, 1.. = KG rels ────────
    rel_list = sorted(incident["relation"].unique()) if not incident.empty else []
    rel2id: dict[str, int] = {r: i + 1 for i, r in enumerate(rel_list)}
    num_rel = len(rel2id) + 1  # +1 for relation 0 (DDI/self-loop)

    # ── 4. Emit edges ────────────────────────────────────────────────────
    edge_list: list[list[int]] = []
    rel_id_list: list[int] = []

    # 4a. KG edges (one direction each — subgraph builder undirects)
    if not incident.empty:
        for h, t, r in zip(incident["head"], incident["tail"], incident["relation"]):
            u = entity_to_idx[h]
            v = entity_to_idx[t]
            edge_list.append([u, v])
            rel_id_list.append(rel2id[r])

    # 4b. DDI-train edges (rel=0). Exclude any edge touching g2.
    ddi_kept = 0
    ddi_skipped = 0
    if train_ddi_pairs is not None and len(train_ddi_pairs) > 0:
        for a, b in zip(
            train_ddi_pairs["drug_a_id"].astype(str),
            train_ddi_pairs["drug_b_id"].astype(str),
        ):
            ia = drug_to_idx.get(a)
            ib = drug_to_idx.get(b)
            if ia is None or ib is None:
                continue
            if ia in g2_idx_set or ib in g2_idx_set:
                ddi_skipped += 1
                continue
            edge_list.append([ia, ib])
            rel_id_list.append(RELID_DDI_SELFLOOP)
            ddi_kept += 1

    # 4c. Self-loops on isolated drugs (no incident edges at all)
    touched_drugs: set[int] = set()
    for u, v in edge_list:
        if u < n_drugs:
            touched_drugs.add(u)
        if v < n_drugs:
            touched_drugs.add(v)
    self_loops_added = 0
    for d_idx in range(n_drugs):
        if d_idx not in touched_drugs:
            edge_list.append([d_idx, d_idx])
            rel_id_list.append(RELID_DDI_SELFLOOP)
            self_loops_added += 1

    # ── 5. Sanity counts ─────────────────────────────────────────────────
    if verbose:
        # 2-hop drug neighbor reachability for g2 drugs
        adj: dict[int, set[int]] = defaultdict(set)
        for (u, v) in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        if g2_idx_set:
            zero_2hop = 0
            for d in g2_idx_set:
                two_hop_drugs = set()
                for nbr in adj[d]:
                    for nbr2 in adj[nbr]:
                        if nbr2 < n_drugs and nbr2 != d:
                            two_hop_drugs.add(nbr2)
                if not two_hop_drugs:
                    zero_2hop += 1
            print(
                f"[tiger-bkg] BKG: n_drugs={n_drugs}, n_entities={n_entities}, "
                f"edges={len(edge_list)}, num_rel={num_rel}; "
                f"DDI kept/skipped={ddi_kept}/{ddi_skipped}, "
                f"isolated-drug self-loops={self_loops_added}; "
                f"g2 drugs with 0 two-hop drug neighbors: {zero_2hop}/{len(g2_idx_set)}"
            )
        else:
            print(
                f"[tiger-bkg] BKG: n_drugs={n_drugs}, n_entities={n_entities}, "
                f"edges={len(edge_list)}, num_rel={num_rel}; "
                f"DDI kept={ddi_kept}, isolated-drug self-loops={self_loops_added}"
            )

    return {
        "drug_to_idx": drug_to_idx,
        "n_drugs": n_drugs,
        "n_entities": n_entities,
        "n_total_nodes": n_total,
        "num_rel": num_rel,
        "edge_list": edge_list,
        "rel_list": rel_id_list,
        "rel2id": rel2id,
        "ddi_edges_kept": ddi_kept,
        "ddi_edges_skipped": ddi_skipped,
        "g2_idx": g2_idx_set,
    }

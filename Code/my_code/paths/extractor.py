"""2-hop path enumeration over the merged biomedical KG.

A 2-hop path between drug_A and drug_B is:

    drug_A ──(edge1)── intermediate ──(edge2)── drug_B

Each edge has:
    - relation        (e.g. 'het:CcSE', 'prime:drug_effect', 'db:target')
    - direction (from drug's perspective):
        * 'out'  : drug → intermediate (drug is the source)
        * 'in'   : intermediate → drug (drug is the target)
        * 'sym'  : symmetric relation (no preferred direction)

Path type signature (for grouping / counting):
    (intermediate_kind, relation_a, dir_a, relation_b, dir_b, source_kg_a, source_kg_b)

We canonicalize so that the edge pair (a, b) is order-independent within a pair
(since drug_A ↔ drug_B is symmetric for DDI).
"""
from __future__ import annotations

from collections import defaultdict, Counter
from typing import Iterable
import pandas as pd


# -----------------------------------------------------------------------------
# Neighbor indices
# -----------------------------------------------------------------------------
def build_neighbor_index(edges: pd.DataFrame) -> dict:
    """Build a drug-only neighbor index for fast 2-hop path enumeration.

    Returns a dict with:
        drug_to_neighbors[drug_id] = {
            neighbor_id: [ (relation, direction, source_kg), ... ]
        }
        intermediate_kind[node_id] = 'Gene' | 'Side Effect' | ...

    Only drugs (kind='Drug') are kept as keys, since we only enumerate paths
    anchored on drug-drug queries.
    """
    drug_to_neighbors: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    # iterate edges and add drug-incident ones in BOTH semantic directions
    # (so that drug_A's neighbor set captures every node it touches)
    n_rows = len(edges)
    for i, (src, src_k, dst, dst_k, rel, sk, directed) in enumerate(zip(
        edges["src"], edges["src_kind"], edges["dst"], edges["dst_kind"],
        edges["relation"], edges["source_kg"], edges["directed"]
    )):
        if src_k == "Drug":
            drug_to_neighbors[src][dst].append((rel, "out" if directed else "sym", sk))
        if dst_k == "Drug" and src != dst:
            drug_to_neighbors[dst][src].append((rel, "in" if directed else "sym", sk))

    # intermediate node kind lookup (from edges table — collect unique node→kind)
    inter_kind = {}
    for nid, k in zip(edges["src"], edges["src_kind"]):
        inter_kind.setdefault(nid, k)
    for nid, k in zip(edges["dst"], edges["dst_kind"]):
        inter_kind.setdefault(nid, k)

    return {
        "drug_to_neighbors": {d: dict(nbrs) for d, nbrs in drug_to_neighbors.items()},
        "node_kind": inter_kind,
    }


# -----------------------------------------------------------------------------
# 2-hop path enumeration for a single pair
# -----------------------------------------------------------------------------
def enumerate_2hop_paths(drug_a: str, drug_b: str, index: dict) -> list[dict]:
    """Return all 2-hop paths from drug_a to drug_b via a shared intermediate.

    Each returned dict:
        {
          'inter_id': str,
          'inter_kind': str,
          'rel_a': str, 'dir_a': str, 'kg_a': str,
          'rel_b': str, 'dir_b': str, 'kg_b': str,
        }

    Multi-edges produce multiple rows. Self-loops (a == b) are skipped.
    """
    nbrs = index["drug_to_neighbors"]
    if drug_a not in nbrs or drug_b not in nbrs:
        return []
    nbrs_a = nbrs[drug_a]
    nbrs_b = nbrs[drug_b]
    if drug_a == drug_b:
        return []

    shared = set(nbrs_a.keys()) & set(nbrs_b.keys()) - {drug_a, drug_b}
    if not shared:
        return []

    kinds = index["node_kind"]
    paths = []
    for inter_id in shared:
        inter_kind = kinds.get(inter_id, "Unknown")
        for rel_a, dir_a, kg_a in nbrs_a[inter_id]:
            for rel_b, dir_b, kg_b in nbrs_b[inter_id]:
                paths.append(dict(
                    inter_id=inter_id, inter_kind=inter_kind,
                    rel_a=rel_a, dir_a=dir_a, kg_a=kg_a,
                    rel_b=rel_b, dir_b=dir_b, kg_b=kg_b,
                ))
    return paths


# -----------------------------------------------------------------------------
# Path-type aggregation across many pairs
# -----------------------------------------------------------------------------
def summarize_path_types(pairs: Iterable[tuple[str, str]], index: dict,
                          canonicalize: bool = True) -> dict:
    """Enumerate path types over a list of pairs, return aggregated stats.

    Path-type signature: (inter_kind, rel_a, dir_a, rel_b, dir_b)
    Canonicalized if canonicalize=True (sorting the (rel, dir) pair so
    swapping drug_a/drug_b gives the same signature).

    Returns:
        {
          'type_counter':       Counter[ (inter_kind, rel_a, dir_a, rel_b, dir_b) ] → total path count
          'pair_type_counter':  Counter[ same key ] → number of pairs that have ≥1 such path
          'per_pair_total':     dict[ (a,b) ] → total path count (for pairs in KG)
          'pairs_with_paths':   int
          'pairs_in_kg':        int
        }
    """
    type_counter = Counter()
    pair_type_counter = Counter()
    per_pair_total = {}
    pairs_in_kg = 0
    pairs_with_paths = 0
    drug_nbrs = index["drug_to_neighbors"]

    for a, b in pairs:
        if a not in drug_nbrs or b not in drug_nbrs:
            continue
        pairs_in_kg += 1
        paths = enumerate_2hop_paths(a, b, index)
        per_pair_total[(a, b)] = len(paths)
        if paths:
            pairs_with_paths += 1
        seen_signatures = set()
        for p in paths:
            sig = _canonical_sig(p, canonicalize)
            type_counter[sig] += 1
            seen_signatures.add(sig)
        for sig in seen_signatures:
            pair_type_counter[sig] += 1

    return {
        "type_counter": type_counter,
        "pair_type_counter": pair_type_counter,
        "per_pair_total": per_pair_total,
        "pairs_in_kg": pairs_in_kg,
        "pairs_with_paths": pairs_with_paths,
    }


def _canonical_sig(p: dict, canonicalize: bool):
    side_a = (p["rel_a"], p["dir_a"])
    side_b = (p["rel_b"], p["dir_b"])
    if canonicalize and side_a > side_b:
        side_a, side_b = side_b, side_a
    return (p["inter_kind"], side_a[0], side_a[1], side_b[0], side_b[1])


def signature_to_str(sig: tuple) -> str:
    """Human-readable rendering of a path-type signature."""
    inter_kind, ra, da, rb, db = sig
    arrow_a = "→" if da == "out" else ("←" if da == "in" else "—")
    arrow_b = "→" if db == "out" else ("←" if db == "in" else "—")
    return f"Drug {arrow_a}[{ra}]— {inter_kind} —[{rb}]{arrow_b} Drug"


def signature_to_dict(sig: tuple) -> dict:
    """Structured rendering of a path-type signature."""
    inter_kind, ra, da, rb, db = sig
    return dict(
        intermediate=inter_kind,
        edge_a=ra, dir_a=da,
        edge_b=rb, dir_b=db,
        readable=signature_to_str(sig),
    )

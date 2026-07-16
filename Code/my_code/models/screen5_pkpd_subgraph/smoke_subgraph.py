"""CPU smoke test for Screen 5 subgraph_builder.

Validates that PK/PD split:
  - Drug-Gene edges go to PK only (not PD)
  - Drug-SideEffect edges go to PD only (not PK)
  - Drug-Drug edges go to both (with include_both mode)
  - Disease-Gene edges go to both (cross-layer)
  - PK/PD edge counts are non-trivial on real merged KG
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen5_pkpd_subgraph.subgraph_builder import (
    split_edges_pk_pd, LAYER_PK, LAYER_PD,
)


def test_toy_split():
    print("[test1] toy edge classification")
    edges = pd.DataFrame({
        "src": ["DB001", "DB002", "DB003", "het:Gene:1", "DB004", "DB005"],
        "dst": ["het:Gene:1", "het:SideEffect:A", "DB002", "het:Disease:X", "het:Pathway:P", "DB006"],
        # Source kinds: Drug, Drug, Drug, Gene/Protein, Drug, Drug
        # Dst kinds:    Gene/Protein, SideEffect, Drug, Disease, Pathway, Drug
    })
    nodes = pd.DataFrame({
        "id":  ["DB001", "DB002", "DB003", "DB004", "DB005", "DB006",
                "het:Gene:1", "het:SideEffect:A", "het:Disease:X", "het:Pathway:P"],
        "kind": ["Drug", "Drug", "Drug", "Drug", "Drug", "Drug",
                 "Gene", "Side Effect", "Disease", "Pathway"],
    })

    pk_edges, pd_edges = split_edges_pk_pd(edges, nodes, overlap_mode="include_both")
    pk_set = set(zip(pk_edges["src"], pk_edges["dst"]))
    pd_set = set(zip(pd_edges["src"], pd_edges["dst"]))

    # Drug-Gene edge -> PK only
    assert ("DB001", "het:Gene:1") in pk_set, "Drug-Gene should be in PK"
    assert ("DB001", "het:Gene:1") not in pd_set, "Drug-Gene should NOT be in PD"
    print("  Drug-Gene -> PK only ✓")

    # Drug-SideEffect edge -> PD only
    assert ("DB002", "het:SideEffect:A") not in pk_set, "Drug-SE should NOT be in PK"
    assert ("DB002", "het:SideEffect:A") in pd_set, "Drug-SE should be in PD"
    print("  Drug-SideEffect -> PD only ✓")

    # Drug-Drug edge with include_both -> both
    assert ("DB003", "DB002") in pk_set, "Drug-Drug include_both -> PK"
    assert ("DB003", "DB002") in pd_set, "Drug-Drug include_both -> PD"
    print("  Drug-Drug (include_both) -> both ✓")

    # Disease-Gene (cross-layer, both endpoints non-Drug) -> BOTH
    assert ("het:Gene:1", "het:Disease:X") in pk_set, "Disease-Gene -> PK (has Gene)"
    assert ("het:Gene:1", "het:Disease:X") in pd_set, "Disease-Gene -> PD (has Disease)"
    print("  Disease-Gene (cross-layer) -> both ✓")

    # Drug-Pathway -> PK (Pathway is in LAYER_PK)
    assert ("DB004", "het:Pathway:P") in pk_set, "Drug-Pathway -> PK"
    assert ("DB004", "het:Pathway:P") not in pd_set, "Drug-Pathway -> NOT PD"
    print("  Drug-Pathway -> PK only ✓")

    print("  PASS")


def test_strict_modes():
    print("\n[test2] strict_pk / strict_pd / drop_drugdrug modes")
    edges = pd.DataFrame({
        "src": ["DB001", "DB001", "DB002"],
        "dst": ["DB002", "het:Gene:1", "het:SideEffect:A"],
    })
    nodes = pd.DataFrame({
        "id":  ["DB001", "DB002", "het:Gene:1", "het:SideEffect:A"],
        "kind": ["Drug", "Drug", "Gene", "Side Effect"],
    })

    # strict_pk: drug-only goes to PK only
    pk, pd_ = split_edges_pk_pd(edges, nodes, overlap_mode="strict_pk")
    assert ("DB001", "DB002") in set(zip(pk["src"], pk["dst"]))
    assert ("DB001", "DB002") not in set(zip(pd_["src"], pd_["dst"]))
    print("  strict_pk: drug-drug -> PK only ✓")

    # strict_pd: drug-only goes to PD only
    pk, pd_ = split_edges_pk_pd(edges, nodes, overlap_mode="strict_pd")
    assert ("DB001", "DB002") not in set(zip(pk["src"], pk["dst"]))
    assert ("DB001", "DB002") in set(zip(pd_["src"], pd_["dst"]))
    print("  strict_pd: drug-drug -> PD only ✓")

    # drop_drugdrug: drug-only goes to neither
    pk, pd_ = split_edges_pk_pd(edges, nodes, overlap_mode="drop_drugdrug")
    assert ("DB001", "DB002") not in set(zip(pk["src"], pk["dst"]))
    assert ("DB001", "DB002") not in set(zip(pd_["src"], pd_["dst"]))
    print("  drop_drugdrug: drug-drug -> neither ✓")

    print("  PASS")


def test_real_kg_sizes():
    print("\n[test3] real merged-KG split sizes")
    nodes_path = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "nodes__drugbank_hetionet_primekg.parquet"
    edges_path = PROJECT_ROOT / "Code" / "data" / "KG" / "_merged_kg" / "edges__drugbank_hetionet_primekg__mask1.parquet"
    if not nodes_path.exists() or not edges_path.exists():
        print("  SKIP (merged KG files not present)")
        return
    nodes_df = pd.read_parquet(nodes_path)
    edges_df = pd.read_parquet(edges_path)
    print(f"  loaded {len(edges_df):,} edges, {len(nodes_df):,} nodes")
    pk, pd_ = split_edges_pk_pd(edges_df, nodes_df, overlap_mode="include_both")
    print(f"  PK subgraph: {len(pk):,} edges ({100*len(pk)/len(edges_df):.1f}%)")
    print(f"  PD subgraph: {len(pd_):,} edges ({100*len(pd_)/len(edges_df):.1f}%)")
    overlap = set(zip(pk["src"], pk["dst"])) & set(zip(pd_["src"], pd_["dst"]))
    print(f"  Overlap (in both): {len(overlap):,} edges")
    assert len(pk) > 1_000_000, "PK subgraph too small — fix needed"
    assert len(pd_) > 100_000, "PD subgraph too small — fix needed"
    print("  PASS")


if __name__ == "__main__":
    test_toy_split()
    test_strict_modes()
    test_real_kg_sizes()
    print("\n=========== Screen 5 subgraph smoke: ALL TESTS PASS ===========")

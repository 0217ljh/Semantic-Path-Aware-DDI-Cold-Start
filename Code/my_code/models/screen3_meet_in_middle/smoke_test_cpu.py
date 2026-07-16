"""CPU-only smoke test for Screen 3 (no GPU, no full training).

Validates:
  1. junction_finder builds correct shape and J1/J3 semantics on a toy adj
  2. EmerGNN_MIM forward pass produces (B,) logits with junction injection
  3. Gradient flows through junction aggregation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen3_meet_in_middle.junction_finder import build_junction_table
from my_code.models.screen3_meet_in_middle.emergnn_mim import EmerGNN_MIM


def test_junction_finder():
    print("[test1] junction_finder")
    # Toy adjacency: 8 nodes, with shared neighbors for some pairs
    adj = {
        0: {2, 3, 4},
        1: {3, 4, 5},   # 0-1 share neighbors 3, 4
        2: {0, 6},
        3: {0, 1, 6},
        4: {0, 1, 7},
        5: {1, 7},
        6: {2, 3},
        7: {4, 5},
    }
    id2kind = {0: "Drug", 1: "Drug", 2: "Gene/Protein", 3: "Gene/Protein",
               4: "SideEffect", 5: "Disease", 6: "Pathway", 7: "Anatomy"}

    # J1: shared 1-hop of (0,1) should be {3, 4}
    jids, jmask = build_junction_table([(0, 1)], adj, id2kind,
                                       junction_type="J1", max_junctions=8)
    valid = set(int(j) for j in jids[0] if j >= 0)
    assert valid == {3, 4}, f"J1 expected {{3,4}} got {valid}"
    print(f"  J1: {valid} ✓")

    # J3-PK: J1 ∩ {Gene/Protein, Pathway} = {3} (3 is Gene/Protein; 4 is SideEffect)
    jids, jmask = build_junction_table([(0, 1)], adj, id2kind,
                                       junction_type="J3-PK", max_junctions=8)
    valid = set(int(j) for j in jids[0] if j >= 0)
    assert valid == {3}, f"J3-PK expected {{3}} got {valid}"
    print(f"  J3-PK: {valid} ✓")

    # J3-PD: J1 ∩ {SideEffect, Disease, Anatomy, Phenotype} = {4}
    jids, jmask = build_junction_table([(0, 1)], adj, id2kind,
                                       junction_type="J3-PD", max_junctions=8)
    valid = set(int(j) for j in jids[0] if j >= 0)
    assert valid == {4}, f"J3-PD expected {{4}} got {valid}"
    print(f"  J3-PD: {valid} ✓")

    print("  PASS")


def test_emergnn_mim_forward():
    print("\n[test2] EmerGNN_MIM forward + grad")
    torch.manual_seed(0)
    n_ent, n_base_rel, n_dim = 16, 4, 8
    external_init = torch.randn(n_ent, n_dim) * 0.1
    model = EmerGNN_MIM(
        n_ent=n_ent, n_base_rel=n_base_rel, n_dim=n_dim, length=2,
        external_init=external_init, feat="X", use_mim=True,
    )

    # Toy graph: 16 entities, random edges
    n_edges = 40
    edge_src = torch.randint(0, n_ent, (n_edges,))
    edge_dst = torch.randint(0, n_ent, (n_edges,))
    edge_rel = torch.randint(0, 2 * n_base_rel + 1, (n_edges,))

    # Batch of 4 pairs
    B = 4
    head = torch.tensor([0, 1, 2, 3])
    tail = torch.tensor([4, 5, 6, 7])
    K = 3
    # First 3 entries have a junction; 4th entry has no junction (all -1, mask 0)
    junctions = torch.tensor([
        [8, 9, -1],
        [10, -1, -1],
        [11, 12, 13],
        [-1, -1, -1],
    ], dtype=torch.long)
    junction_mask = torch.tensor([
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
    ], dtype=torch.float32)

    logits = model(head, tail, edge_src, edge_dst, edge_rel,
                   junctions=junctions, junction_mask=junction_mask)
    assert logits.shape == (B,), f"logits shape {logits.shape} != ({B},)"
    print(f"  forward logits shape={tuple(logits.shape)}, values={logits.detach().tolist()}")

    # Gradient flow
    loss = logits.sum()
    loss.backward()
    # Check ent_kg got non-zero gradient
    grad_norm = model.ent_kg.weight.grad.norm().item()
    assert grad_norm > 0, "ent_kg got zero gradient — junction aggregation may not connect to compute graph"
    print(f"  ent_kg grad norm: {grad_norm:.4f} ✓")
    # Check Wr (which is 6*n_dim head, includes junction columns)
    wr_grad = model.Wr.weight.grad.norm().item()
    assert wr_grad > 0
    print(f"  Wr grad norm: {wr_grad:.4f} ✓")

    print("  PASS")


def test_compatibility_with_mim_off():
    print("\n[test3] EmerGNN_MIM with use_mim=False fallback to TAG")
    torch.manual_seed(0)
    n_ent, n_base_rel, n_dim = 16, 4, 8
    external_init = torch.randn(n_ent, n_dim) * 0.1
    model = EmerGNN_MIM(
        n_ent=n_ent, n_base_rel=n_base_rel, n_dim=n_dim, length=2,
        external_init=external_init, feat="X", use_mim=False,
    )
    # Wr should be 4*n_dim shape (TAG default)
    assert model.Wr.weight.shape == (1, 4 * n_dim), \
        f"use_mim=False expected Wr=(1,{4*n_dim}) got {tuple(model.Wr.weight.shape)}"
    print(f"  Wr shape with use_mim=False: {tuple(model.Wr.weight.shape)} ✓")
    print("  PASS")


if __name__ == "__main__":
    test_junction_finder()
    test_emergnn_mim_forward()
    test_compatibility_with_mim_off()
    print("\n=========== Screen 3 CPU smoke: ALL TESTS PASS ===========")

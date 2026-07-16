"""Smoke-test: confirm save() → load() → predict_proba round-trip works
for BOTH binary and multi-class EmerGNN baselines without shape errors.

Builds tiny synthetic state directly (no fit) — just enough to exercise
the save/load code paths added by the round-3 codex fixes.
"""
from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# DEPRECATED smoke test (moved into deprecate/ on 2026-05-20 as part of
# the binary baseline reorg). The `binary_cls/baseline.py` module now
# contains the multimode wrapper, not the per-mode trainer; this smoke
# imports the per-mode helper directly via `_per_mode` and aliases the
# name so the rest of the file body can stay byte-identical to its
# pre-reorg form. NOTE: parents depth changed (file moved one level
# deeper into deprecate/), so the sys.path bump is one extra `.parents`.
# Whether this smoke still meaningfully exercises the public binary
# baseline is unverified — kept only as historical record.
_CODE_ROOT = Path(__file__).resolve().parents[3]  # deprecate -> emergnn -> baseline -> Code
sys.path.insert(0, str(_CODE_ROOT))

from baseline.emergnn._per_mode import _PerModeEmerGNN as EmerGNNBaseline
from baseline.emergnn.model import EmerGNN
from baseline.emergnn.multi_cls.baseline import EmerGNNMulticlassBaseline
from baseline.emergnn.multi_cls.model import EmerGNN_MC
from baseline.emergnn.shuffle_utils import build_edge_lists_from_triplets

device = "cpu"


def _fake_graph(n_drugs: int = 4, n_kg_rel: int = 3, n_kg_edges: int = 5):
    n_ent = n_drugs + 2  # +2 extra non-drug entities
    rng = np.random.default_rng(0)
    triplets = np.stack(
        [
            rng.integers(0, n_ent, n_kg_edges),
            rng.integers(0, n_ent, n_kg_edges),
            rng.integers(0, n_kg_rel, n_kg_edges),
        ],
        axis=1,
    ).astype(np.int64)
    entity2id = {f"DB{i:05d}": i for i in range(n_drugs)}
    morgan = np.zeros((n_ent, 1024), dtype=np.float32)
    return entity2id, n_ent, triplets, morgan


def _build_eval_edges(n_ent, n_base_rel_with_ddi, train_ddi, kg_trip):
    triplets = np.concatenate([train_ddi, kg_trip], axis=0)
    s, d, r = build_edge_lists_from_triplets(triplets, n_ent, n_base_rel_with_ddi)
    return (
        torch.from_numpy(s).long(),
        torch.from_numpy(d).long(),
        torch.from_numpy(r).long(),
    )


def test_binary():
    print("=== binary round-trip ===")
    entity2id, n_ent, kg_trip, morgan = _fake_graph()
    n_kg_rel = 3
    n_base_rel_with_ddi = n_kg_rel + 1

    inst = EmerGNNBaseline(n_dim=8, length=2, feat="M", batch_size=4, n_epochs=1)
    inst.device = device
    inst._entity2id = entity2id
    inst._n_ent = n_ent
    inst._n_base_rel = n_kg_rel
    inst._n_base_rel_with_ddi = n_base_rel_with_ddi
    inst._kg_triplets = kg_trip
    # placeholder dense edge lists (mirrors _setup_graph)
    src, dst, rel = build_edge_lists_from_triplets(kg_trip, n_ent, n_kg_rel)
    inst._edge_src = torch.from_numpy(src).long()
    inst._edge_dst = torch.from_numpy(dst).long()
    inst._edge_rel = torch.from_numpy(rel).long()
    inst._model = EmerGNN(
        n_ent=n_ent,
        n_base_rel=n_base_rel_with_ddi,
        n_dim=8,
        length=2,
        feat="M",
        morgan_features=morgan,
    ).to(device)
    # fake train_ddi (a single edge) and eval edges
    train_ddi = np.array([[0, 1, n_kg_rel]], dtype=np.int64)
    inst._eval_edges = _build_eval_edges(n_ent, n_base_rel_with_ddi, train_ddi, kg_trip)

    import pandas as pd
    pairs = pd.DataFrame({"drug_a_id": ["DB00000", "DB00001"], "drug_b_id": ["DB00002", "DB00003"]})
    p1 = inst.predict_proba(pairs)
    print(f"  pre-save preds: {p1}")

    with tempfile.TemporaryDirectory() as td:
        inst.save(td)
        # verify graph.pkl has the new keys
        with (Path(td) / "graph.pkl").open("rb") as f:
            blob = pickle.load(f)
        assert "n_base_rel_with_ddi" in blob, "missing n_base_rel_with_ddi in saved graph"
        assert blob["n_base_rel_with_ddi"] == n_base_rel_with_ddi
        assert blob["eval_edges"] is not None, "missing eval_edges in saved graph"
        print(f"  graph.pkl keys: {sorted(blob.keys())}")

        loaded = EmerGNNBaseline.load(td)
        assert loaded._n_base_rel_with_ddi == n_base_rel_with_ddi
        assert loaded._eval_edges is not None
        p2 = loaded.predict_proba(pairs)
        print(f"  post-load preds: {p2}")
        assert np.allclose(p1, p2, atol=1e-5), f"preds differ: {p1} vs {p2}"
    print("  BINARY PASS")


def test_multi():
    print("=== multi round-trip ===")
    entity2id, n_ent, kg_trip, morgan = _fake_graph()
    n_kg_rel = 3
    n_classes = 5
    n_base_rel_with_ddi = n_kg_rel + n_classes

    inst = EmerGNNMulticlassBaseline(n_classes=n_classes, n_dim=8, length=2, feat="M", batch_size=4, n_epochs=1)
    inst.device = device
    inst._entity2id = entity2id
    inst._n_ent = n_ent
    inst._n_base_rel = n_kg_rel
    inst._n_base_rel_with_ddi = n_base_rel_with_ddi
    inst._kg_triplets = kg_trip
    inst._ddi_type_to_idx = {f"t{i}": i for i in range(n_classes)}
    inst._idx_to_ddi_type = [f"t{i}" for i in range(n_classes)]
    src, dst, rel = build_edge_lists_from_triplets(kg_trip, n_ent, n_kg_rel)
    inst._edge_src = torch.from_numpy(src).long()
    inst._edge_dst = torch.from_numpy(dst).long()
    inst._edge_rel = torch.from_numpy(rel).long()
    inst._model = EmerGNN_MC(
        n_ent=n_ent,
        n_base_rel=n_base_rel_with_ddi,
        n_classes=n_classes,
        n_dim=8,
        length=2,
        feat="M",
        morgan_features=morgan,
    ).to(device)
    train_ddi = np.array([[0, 1, n_kg_rel]], dtype=np.int64)
    inst._eval_edges = _build_eval_edges(n_ent, n_base_rel_with_ddi, train_ddi, kg_trip)

    import pandas as pd
    pairs = pd.DataFrame({"drug_a_id": ["DB00000", "DB00001"], "drug_b_id": ["DB00002", "DB00003"]})
    p1 = inst.predict_proba(pairs)
    print(f"  pre-save preds shape: {p1.shape}, sum={p1.sum():.4f}")

    with tempfile.TemporaryDirectory() as td:
        inst.save(td)
        with (Path(td) / "graph.pkl").open("rb") as f:
            blob = pickle.load(f)
        assert "n_base_rel_with_ddi" in blob
        assert blob["n_base_rel_with_ddi"] == n_base_rel_with_ddi
        assert blob["eval_edges"] is not None
        print(f"  graph.pkl keys: {sorted(blob.keys())}")

        loaded = EmerGNNMulticlassBaseline.load(td)
        assert loaded._n_base_rel_with_ddi == n_base_rel_with_ddi
        assert loaded._eval_edges is not None
        p2 = loaded.predict_proba(pairs)
        print(f"  post-load preds shape: {p2.shape}, sum={p2.sum():.4f}")
        assert np.allclose(p1, p2, atol=1e-5), f"preds differ: {p1} vs {p2}"
    print("  MULTI PASS")


if __name__ == "__main__":
    test_binary()
    test_multi()
    print("\nALL ROUND-TRIP SMOKES PASS")

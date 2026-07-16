"""Verify StructuralVariableCore (use_asym=False) reproduces SPMNRelHead.

Regression guard for the spmn_v2 migration: with asymmetric stratification and
the abs-diff embedding OFF, ``StandaloneHead`` over ``StructuralVariableCore``
must compute byte-for-byte the same logits as ``spmn_v1.rel_head.SPMNRelHead``
once we copy the (identically named) weights across. Runs under ``eval()`` so
dropout is off, on a synthetic batch that includes an empty-support pair to
exercise the zero-mediator guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.spmn_v1.rel_head import SPMNRelHead  # noqa: E402
from my_code.models.spmn_v1.retrieval import N_REL_BUCKETS, N_TYPES  # noqa: E402
from my_code.models.spmn_v1.struct_features import symmetric_binary_dim  # noqa: E402
from my_code.models.spmn_v2 import StandaloneHead, StructuralVariableCore, SupportBatch  # noqa: E402

NENT, K, NREL, D, HID, DROP = 1000, N_TYPES, N_REL_BUCKETS, 32, 128, 0.2


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    struct_dim = symmetric_binary_dim()

    rel = SPMNRelHead(n_entities=NENT, n_types=K, n_rel_buckets=NREL,
                      struct_dim=struct_dim, d=D, hidden=HID, dropout=DROP).to(dev).eval()
    core = StructuralVariableCore(n_entities=NENT, n_types=K, n_rel_buckets=NREL,
                                  struct_dim=struct_dim, d=D, hidden=HID, dropout=DROP,
                                  use_asym=False, use_absdiff_embed=False).to(dev).eval()
    std = StandaloneHead(core, hidden=HID, dropout=DROP).to(dev).eval()

    # Copy weights: core shares rel's submodule names; scorer == head_mlp.
    rel_sd = rel.state_dict()
    core.load_state_dict({k: v for k, v in rel_sd.items()
                          if not k.startswith("head_mlp.")}, strict=True)
    std.scorer.load_state_dict({k[len("head_mlp."):]: v for k, v in rel_sd.items()
                                if k.startswith("head_mlp.")}, strict=True)

    # Synthetic batch of B pairs; pair 3 is deliberately empty (no mediators).
    B = 6
    med_l, pair_l, typ_l, ra_l, rb_l, da_l, db_l = [], [], [], [], [], [], []
    for p in range(B):
        E = 0 if p == 3 else int(np.random.randint(1, 30))
        med_l.append(torch.randint(0, NENT, (E,)))
        pair_l.append(torch.full((E,), p, dtype=torch.long))
        typ_l.append(torch.randint(0, K, (E,)))
        ra_l.append(torch.randint(0, NREL, (E,)))
        rb_l.append(torch.randint(0, NREL, (E,)))
        da_l.append(torch.randint(1, 5, (E,)))
        db_l.append(torch.randint(1, 5, (E,)))
    cat = lambda L: torch.cat(L).to(dev)
    med, pair, typ, ra, rb, da, db = (cat(med_l), cat(pair_l), cat(typ_l),
                                      cat(ra_l), cat(rb_l), cat(da_l), cat(db_l))
    struct = torch.randn(B, struct_dim, device=dev)

    batch = SupportBatch(med_id=med, pair_idx=pair, type_idx=typ, rel_a=ra,
                         rel_b=rb, d_a=da, d_b=db, struct_feats=struct, n_pairs=B)

    with torch.no_grad():
        ref = rel(med, pair, typ, ra, rb, struct, B)
        got = std(batch)

    diff = (ref - got).abs().max().item()
    print("rel_head :", [round(x, 4) for x in ref.tolist()])
    print("v2 std   :", [round(x, 4) for x in got.tolist()])
    print(f"max abs diff = {diff:.2e}  ->  {'MATCH' if diff < 1e-5 else 'MISMATCH!!'}")
    sys.exit(0 if diff < 1e-5 else 1)


if __name__ == "__main__":
    main()

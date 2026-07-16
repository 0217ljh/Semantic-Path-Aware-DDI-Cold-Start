"""30s micro-bench to locate why train_emergnn_rspmm_rank_pilot.py is ~20x slower
per step (3.3s) than the baseline rspmm core (0.166s/step) on the SAME full KG,
model, and config. Replicates the pilot's EXACT KG + model setup and times one
EmerGNN_RSPMM forward. Prime suspect: kg.is_coalesced() — generalized_rspmm calls
sparse.coalesce() every forward, which re-sorts 14M nnz if the KG is not already
coalesced. Run:
  CUDA_HOME=/home/lakestar_ljh/miniconda3 EMERGNN_KG_SCOPE=full python -u Code/scripts/bench_rspmm_forward.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

os.environ.setdefault("EMERGNN_KG_SCOPE", "full")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("root")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH  # noqa: E402
from baseline.emergnn.model_rspmm import EmerGNN_RSPMM  # noqa: E402
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets  # noqa: E402
from baseline.emergnn._shared import ensure_kg_setup_cache  # noqa: E402
from train_emergnn_rank_pilot import _MiniTrain, build_structural_features  # noqa: E402
from train_gcn_rank_pilot import load_data  # noqa: E402


def _time_forwards(model, head, tail, kg, n=5):
    with torch.no_grad():
        for _ in range(2):  # warmup (JIT compile + cache)
            model.enc_ht(head, tail, kg)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            model.enc_ht(head, tail, kg)
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    device = torch.device("cuda")
    tr, warm_val, warm_test, cold = load_data("fold0", 0.12, 42)
    pool = pd.concat([tr, warm_val, warm_test], ignore_index=True)
    cache = ensure_kg_setup_cache(
        _MiniTrain({"train": pool, "test": cold}),
        backbone_kg_source="merged", merged_kg_path=(ROOT / DEFAULT_EDGES_PATH).resolve())
    n_ent = int(cache["n_ent"]); n_base_rel = int(cache["n_base_rel"])

    kg = build_sparse_kg_from_triplets(cache["kg_triplets"], n_ent, n_base_rel, device=device)
    print(f"[bench] n_ent={n_ent} n_base_rel={n_base_rel} kg nnz={kg._nnz()} "
          f"is_coalesced={kg.is_coalesced()}", flush=True)

    x, _, _ = build_structural_features(cache, DEFAULT_NODES_PATH)
    model = EmerGNN_RSPMM(n_ent=n_ent, n_base_rel=n_base_rel, n_dim=64, length=3,
                          feat="M", morgan_features=x, morgan_feat_dim=N_TYPES + 1).to(device)

    head = torch.randint(0, n_ent, (32,), device=device)
    tail = torch.randint(0, n_ent, (32,), device=device)

    t_fwd = _time_forwards(model, head, tail, kg)
    print(f"[bench] enc_ht (no_grad): {t_fwd:.3f} s/forward", flush=True)

    # fwd+bwd (one training step's forward path)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad()
        logit = model(head, tail, kg)
        logit.sum().backward()
        opt.step()
    torch.cuda.synchronize()
    print(f"[bench] fwd+bwd: {(time.time()-t0)/5:.3f} s/step", flush=True)

    # A/B: force a fresh coalesce and re-time (if this is much faster, the KG was
    # effectively un-coalesced and generalized_rspmm was re-sorting every call).
    kg_c = kg.coalesce()
    print(f"[bench] kg.coalesce().is_coalesced={kg_c.is_coalesced()}", flush=True)
    t_fwd_c = _time_forwards(model, head, tail, kg_c)
    print(f"[bench] enc_ht on re-coalesced kg: {t_fwd_c:.3f} s/forward", flush=True)


if __name__ == "__main__":
    main()

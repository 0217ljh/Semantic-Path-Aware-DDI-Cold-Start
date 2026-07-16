"""Prebuild the KnowDDI per-pair enclosing-subgraph disk cache, in PARALLEL.

KnowDDI's bottleneck is per-pair BFS subgraph extraction (~73 ms/pair -> ~35 min for the
~29k fixed eval pairs). Extraction is deterministic + pair-independent (rng seeded by
(seed,u,v)), so this prebuilds the cache with multiprocessing (fork) and the cache is
BYTE-IDENTICAL to on-the-fly extraction. All KnowDDI runs (backbone_only + adapter
frozen/joint x last/default x folds) then LOAD it (seconds) instead of re-extracting.

CUDA-free by construction (no model, CUDA_VISIBLE_DEVICES="" ) so fork is safe.

Example (once per dataset/fold; extracts cold_val + cold_test + train positives):
  python Code/scripts/prepare_knowddi_subgraphs.py --dataset drugbank_ryu --fold fold0 \
      --hop 2 --max-nodes-per-hop 100 --seed 42 --nproc 8
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""                      # FORCE no CUDA (extraction is CPU-only;
#                     setdefault is a no-op if the parent set a GPU -> fork-after-CUDA trap). codex 019f5f9e.
# Force each worker single-threaded: numpy/scipy BLAS/OpenMP otherwise spawn threads inside every
# forked worker -> 6 procs x N threads oversubscribe 8 cores -> parallelism collapsed to ~1.8x
# (measured). MUST be set BEFORE importing numpy/scipy.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np                                           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (               # noqa: E402
    MergedKG, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
from train_knowddi_rank_pilot import build_graph_tensors, extract_enclosing_subgraph  # noqa: E402
from specs import TaskSpec                                   # noqa: E402
from data.loader import load_rank_data                       # noqa: E402
from kg import knowddi_subgraph_cache as C                   # noqa: E402

# fork-inherited globals (set in main BEFORE the pool; workers read them, no IPC of the CSR)
_INC = None
_HOP = None
_MNPH = None
_SEED = None


def _extract_one(uv):
    u, v = int(uv[0]), int(uv[1])
    rng = np.random.default_rng(_SEED * 1_000_003 + u * 31 + v)   # EXACT per-pair rng (wrapper parity)
    nodes, labels = extract_enclosing_subgraph(u, v, _INC, _HOP, _MNPH, rng)
    return (u, v), nodes, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--max-nodes-per-hop", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    print(f"[prepare-knowddi] building bio incidence (KG)...", flush=True)
    kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
    id2i = kg.id_to_idx
    empty = np.zeros((0, 2), dtype=np.int64)
    _g, inc_bio, _layout = build_graph_tensors(kg, str(DEFAULT_EDGES_PATH), empty)
    n_nodes = kg.n_nodes

    # -- collect the FIXED pairs to prebuild: cold_val + cold_test + train positives ----
    task = TaskSpec.binary() if args.task == "binary" else TaskSpec.multiclass(2)  # K irrelevant here
    data = load_rank_data(args.dataset, task, args.fold)

    def to_idx(pairs):
        out = []
        for a, b in np.asarray(pairs)[:, :2]:
            ia, ib = id2i.get(str(a)), id2i.get(str(b))
            if ia is not None and ib is not None:
                out.append((int(ia), int(ib)))
        return out

    # eval (cold_val+cold_test, fixed) + train positives (seen_ddi) + the WHOLE train_neg pool
    # (binary negatives are RE-SAMPLED per epoch from this pool, so caching the pool makes every
    # epoch's sampled negs a HIT; train_neg is empty for multiclass -> no-op there).
    want = (to_idx(data.cold_val_pairs) + to_idx(data.cold_test_pairs)
            + to_idx(data.seen_ddi[:, :2]) + to_idx(data.train_neg))
    want = list(dict.fromkeys(want))                          # dedup, preserve order + orientation
    print(f"[prepare-knowddi] {len(want)} unique known pairs (cold_val {len(data.cold_val_pairs)} + "
          f"cold_test {len(data.cold_test_pairs)} + train_pos {len(data.seen_ddi)} + "
          f"train_neg {len(data.train_neg)})", flush=True)

    sig = C.signature(hop=args.hop, max_nodes_per_hop=args.max_nodes_per_hop, seed=args.seed,
                      n_nodes=n_nodes, inc=inc_bio)
    path = C.cache_path(sig)
    existing = C.load(path, sig) or {}
    todo = [uv for uv in want if uv not in existing]
    print(f"[prepare-knowddi] cache {path.name}: {len(existing)} cached, {len(todo)} to extract "
          f"(hop={args.hop} max_nodes={args.max_nodes_per_hop} seed={args.seed})", flush=True)
    if not todo:
        print("[prepare-knowddi] nothing to do (all cached).", flush=True)
        return

    global _INC, _HOP, _MNPH, _SEED
    _INC, _HOP, _MNPH, _SEED = inc_bio, args.hop, args.max_nodes_per_hop, args.seed

    t0 = time.time()
    new = dict(existing)
    ctx = mp.get_context("fork")                              # fork shares _INC COW (no CSR pickling)
    with ctx.Pool(args.nproc) as pool:
        done = 0
        for (u, v), nodes, labels in pool.imap_unordered(_extract_one, todo, chunksize=64):
            new[(u, v)] = (nodes, labels)
            done += 1
            if done % 2000 == 0:
                el = time.time() - t0
                print(f"[prepare-knowddi] {done}/{len(todo)} ({el:.0f}s, "
                      f"{el/done*1e3:.1f} ms/pair, eta {(len(todo)-done)*el/done:.0f}s)", flush=True)
            if done % 20000 == 0:                            # periodic checkpoint (survive a kill)
                C.save(path, new, sig)
                print(f"[prepare-knowddi] checkpoint saved ({len(new)} pairs) -> {path.name}", flush=True)

    C.save(path, new, sig)
    print(f"[prepare-knowddi] DONE {len(todo)} extracted in {time.time()-t0:.0f}s "
          f"({args.nproc} procs); cache now {len(new)} pairs -> {path}", flush=True)


if __name__ == "__main__":
    main()

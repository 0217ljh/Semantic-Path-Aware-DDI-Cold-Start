"""Minimal step-by-step memory diagnostic.
Each step: print BEFORE, run the op, print AFTER. Force flush every line.
"""
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def gb(b):
    return b / 1024**3


def p(stage):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[GPU] {stage:50s} alloc={a:6.2f}GB reserved={r:6.2f}GB peak={peak:6.2f}GB", flush=True)
    # also report CPU RAM
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # MB
        print(f"[CPU] {stage:50s} ru_maxrss={ru:6.2f}GB", flush=True)
    except Exception:
        pass


def step(name):
    print(f"\n=== STEP: {name} ===", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM total: {gb(torch.cuda.get_device_properties(0).total_memory):.2f}GB", flush=True)
    p("baseline")

    step("import data_loader")
    from data_loader import load_vocab, load_setting, build_global_morgan_matrix
    p("after import data_loader")

    step("load_vocab")
    vocab = load_vocab()
    p("after load_vocab")

    step("build_global_morgan_matrix")
    morgan_mat = build_global_morgan_matrix(vocab)
    print(f"  morgan shape={morgan_mat.shape} bytes={morgan_mat.nbytes/1024**2:.1f}MB", flush=True)
    p("after morgan_mat")

    step("load_setting S1_1")
    s = load_setting("S1_1")
    print(
        f"  train_ddi={len(s['train_ddi'])} valid_ddi={len(s['valid_ddi'])} test_ddi={len(s['test_ddi'])}",
        flush=True,
    )
    print(
        f"  train_kg={len(s['train_kg'])} valid_kg(cum)={len(s['valid_kg'])} test_kg(cum)={len(s['test_kg'])}",
        flush=True,
    )
    p("after load_setting")

    n_ent = vocab["n_entity"]
    n_base_rel = vocab["n_relation"]

    step("concat vkg_triplets")
    vkg_triplets = np.concatenate([s["train_ddi"], s["valid_kg"]], axis=0)
    print(f"  vkg_triplets shape={vkg_triplets.shape} bytes={vkg_triplets.nbytes/1024**2:.1f}MB", flush=True)
    p("after vkg concat")

    step("TEST A: build_sparse_adj on CPU (original code path)")
    from kg_builder import build_sparse_adj, edges_as_dense_lists
    adj_cpu = build_sparse_adj(vkg_triplets, n_ent=n_ent, n_rel=n_base_rel, device=None)
    print(f"  adj_cpu shape={adj_cpu.shape} nnz={adj_cpu._nnz()}", flush=True)
    p("after build_sparse_adj on CPU")

    step("TEST A: adj.to(cuda)  ← suspected memory bomb")
    adj_cuda = adj_cpu.to(device)
    p("after adj.to(cuda)")

    step("TEST A: edges_as_dense_lists")
    src, dst, rel = edges_as_dense_lists(adj_cuda)
    print(f"  src/dst/rel shape={src.shape}", flush=True)
    p("after edges_as_dense_lists")

    del adj_cpu, adj_cuda, src, dst, rel
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    p("after del + empty_cache")

    step("TEST B: direct edge-list construction (proposed fix)")
    heads = vkg_triplets[:, 0]
    tails = vkg_triplets[:, 1]
    rels = vkg_triplets[:, 2]
    fwd_r = rels + n_base_rel
    rev_r = rels
    idd = np.arange(n_ent, dtype=np.int64)
    idd_r = np.full(n_ent, 2 * n_base_rel, dtype=np.int64)
    all_src = np.concatenate([heads, tails, idd])
    all_dst = np.concatenate([tails, heads, idd])
    all_rel = np.concatenate([fwd_r, rev_r, idd_r])
    print(f"  direct edge list len={len(all_src)}", flush=True)
    p("after direct numpy concat")
    src_d = torch.from_numpy(all_src).long().to(device)
    dst_d = torch.from_numpy(all_dst).long().to(device)
    rel_d = torch.from_numpy(all_rel).long().to(device)
    p("after direct .to(cuda)")


if __name__ == "__main__":
    main()

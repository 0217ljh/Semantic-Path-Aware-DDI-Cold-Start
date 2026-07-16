"""Microbenchmark: chunk+checkpoint propagation vs sparse-matmul propagation.

Times a realistic EmerGNN propagation (L layers of the message-passing kernel we
are replacing) forward+backward, and reports wall time + peak CUDA memory for:
  (A) current path : per-chunk index_select*rel -> scatter, under torch.utils.checkpoint
  (B) new path     : per-relation torch.sparse.mm (rspmm_sparsemm)

Only the propagation kernel is benchmarked (the part torchdrug fused); the rest
of EmerGNN (attention over relations, linear, head) is identical between paths.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && python Code/scripts/bench_rspmm_speed.py"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.utils.checkpoint as _cp

_EMER = Path(__file__).resolve().parents[1] / "baseline" / "emergnn"
sys.path.insert(0, str(_EMER))

from rspmm_sparsemm import (  # noqa: E402
    build_relation_adjacency,
    generalized_rspmm_sparsemm,
)


def _chunk_compute_and_scatter(hiddens, rel_t, es, er, ed, n_ent):
    h_feat = hiddens.index_select(0, es)
    r_feat = rel_t.index_select(0, er)
    msg = h_feat * r_feat
    out = torch.zeros(n_ent, hiddens.size(1), hiddens.size(2),
                      device=hiddens.device, dtype=hiddens.dtype)
    out.index_add_(0, ed, msg)
    return out


def _propagate_chunk(hiddens0, rel_t_layers, src, rel, dst, n_ent, L, chunk, use_ckpt):
    hiddens = hiddens0
    E = src.size(0)
    for l in range(L):
        rel_t = rel_t_layers[l]
        new_h = torch.zeros_like(hiddens)
        for start in range(0, E, chunk):
            stop = min(start + chunk, E)
            es, er, ed = src[start:stop], rel[start:stop], dst[start:stop]
            if use_ckpt:
                contrib = _cp.checkpoint(_chunk_compute_and_scatter,
                                         hiddens, rel_t, es, er, ed, n_ent,
                                         use_reentrant=False)
            else:
                contrib = _chunk_compute_and_scatter(hiddens, rel_t, es, er, ed, n_ent)
            new_h = new_h + contrib
        hiddens = torch.relu(new_h)  # stand-in for act(linear(.)) — same both paths
    return hiddens


def _propagate_sparse(hiddens0, rel_t_layers, adj, L):
    hiddens = hiddens0
    for l in range(L):
        new_h = generalized_rspmm_sparsemm(adj, rel_t_layers[l], hiddens)
        hiddens = torch.relu(new_h)
    return hiddens


def _propagate_csr(hiddens0, rel_t_layers, adj_csr, L):
    """Same as _propagate_sparse but with CSR adjacency (cuSPARSE spmm backend)."""
    n_ent, batch, dim = hiddens0.shape
    hiddens = hiddens0
    for l in range(L):
        rel_t = rel_t_layers[l]
        h2 = hiddens.reshape(n_ent, batch * dim)
        out = torch.zeros_like(hiddens)
        for r, a_r in enumerate(adj_csr):
            if a_r is None:
                continue
            agg = torch.sparse.mm(a_r, h2).reshape(n_ent, batch, dim)
            out = out + agg * rel_t[r].unsqueeze(0)
        hiddens = torch.relu(out)
    return hiddens


def _time_path(fn, n_iters, device):
    # warmup
    for _ in range(2):
        loss = fn().sum()
        loss.backward()
    torch.cuda.synchronize() if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        loss = fn().sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters
    peak = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
    return dt, peak


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] no CUDA — benchmark is only meaningful on GPU (run under WSL project_1).")
    n_ent, n_edges, all_rel = 21000, 307050, 11
    batch, dim, L = 32, 64, 3
    chunk = 100_000
    n_iters = 5
    g = torch.Generator(device=device).manual_seed(0)

    src = torch.randint(0, n_ent, (n_edges,), generator=g, device=device)
    dst = torch.randint(0, n_ent, (n_edges,), generator=g, device=device)
    rel = torch.randint(0, all_rel, (n_edges,), generator=g, device=device)

    hiddens0 = torch.randn(n_ent, batch, dim, generator=g, device=device)
    rel_t_layers = [torch.randn(all_rel, batch, dim, generator=g, device=device,
                                requires_grad=True) for _ in range(L)]
    adj = build_relation_adjacency(src, dst, rel, n_ent, all_rel, dtype=torch.float32)

    print(f"shapes: N={n_ent} E={n_edges} R={all_rel} B={batch} D={dim} L={L} "
          f"chunk={chunk} iters={n_iters} device={device}")

    dt_a, pk_a = _time_path(
        lambda: _propagate_chunk(hiddens0, rel_t_layers, src, rel, dst, n_ent, L,
                                 chunk, use_ckpt=True), n_iters, device)
    print(f"(A) chunk + checkpoint   : {dt_a*1e3:8.1f} ms/iter   peak {pk_a:8.0f} MB")

    dt_a2, pk_a2 = _time_path(
        lambda: _propagate_chunk(hiddens0, rel_t_layers, src, rel, dst, n_ent, L,
                                 chunk, use_ckpt=False), n_iters, device)
    print(f"(A') chunk, NO checkpoint: {dt_a2*1e3:8.1f} ms/iter   peak {pk_a2:8.0f} MB")

    dt_b, pk_b = _time_path(
        lambda: _propagate_sparse(hiddens0, rel_t_layers, adj, L), n_iters, device)
    print(f"(B) sparse.mm COO per-rel: {dt_b*1e3:8.1f} ms/iter   peak {pk_b:8.0f} MB")

    adj_csr = [a.to_sparse_csr() if a is not None else None for a in adj]
    dt_c, pk_c = _time_path(
        lambda: _propagate_csr(hiddens0, rel_t_layers, adj_csr, L), n_iters, device)
    print(f"(C) sparse.mm CSR per-rel: {dt_c*1e3:8.1f} ms/iter   peak {pk_c:8.0f} MB")

    print(f"\nspeedup B(COO) vs A (checkpoint): {dt_a/dt_b:6.2f}x   mem {pk_a/pk_b:5.2f}x lower")
    print(f"speedup C(CSR) vs A (checkpoint): {dt_a/dt_c:6.2f}x   mem {pk_a/pk_c:5.2f}x lower")
    print(f"speedup C(CSR) vs A'(no ckpt)   : {dt_a2/dt_c:6.2f}x   mem {pk_a2/pk_c:5.2f}x lower")


if __name__ == "__main__":
    main()

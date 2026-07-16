"""Benchmark different chunk_size + checkpoint configs on ONE training step
to find the sweet spot for our 32GB GPU.

Each config: time 5 forward+backward steps with batch=32, report avg + peak mem.
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kg_builder import build_sparse_adj, edges_as_dense_lists
from model_mc import EmerGNN_MC
from data_loader import build_global_morgan_matrix, load_setting, load_vocab, shuffle_train


def gb(b):
    return b / 1024**3


def bench(model, opt, h, t, r, src, dst, rel, n_warmup=2, n_iter=5):
    """Time n_iter fwd+bwd steps, return avg sec/step + peak GB."""
    torch.cuda.reset_peak_memory_stats()
    # warmup
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        logits = model(h, t, src, dst, rel)
        loss = F.cross_entropy(logits, r, reduction="sum")
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        opt.zero_grad(set_to_none=True)
        logits = model(h, t, src, dst, rel)
        loss = F.cross_entropy(logits, r, reduction="sum")
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iter
    peak = torch.cuda.max_memory_allocated()
    return elapsed, peak


def main():
    device = "cuda"
    print(f"VRAM total: {gb(torch.cuda.get_device_properties(0).total_memory):.1f} GB")

    # one-time data setup
    vocab = load_vocab()
    morgan_mat = build_global_morgan_matrix(vocab)
    s = load_setting("S1_1")
    n_ent = vocab["n_entity"]
    n_base_rel = vocab["n_relation"]

    # use shuffle_train epoch_kg (~1.75M edges, smaller than vKG)
    rng = np.random.default_rng(0)
    epoch_kg, train_targets = shuffle_train(
        s["train_ddi"], s["train_kg"], "S1_1", rng=rng,
        extra_kg_ent=set(np.unique(s["test_kg"][:, :2]).tolist()),
    )
    adj = build_sparse_adj(epoch_kg, n_ent=n_ent, n_rel=n_base_rel, device=device)
    src, dst, rel = edges_as_dense_lists(adj)
    print(f"epoch_kg edges (× directions+selfloop) = {src.shape[0]:,}")

    h = torch.from_numpy(train_targets[:32, 0]).long().to(device)
    t = torch.from_numpy(train_targets[:32, 1]).long().to(device)
    r = torch.from_numpy(train_targets[:32, 2]).long().to(device)

    print()
    print(f"{'config':40s}  {'sec/step':>10s}  {'peak GB':>10s}  {'~ep min @1k steps':>18s}")
    print("-" * 90)

    configs = [
        (100_000, True),   # current default
        (500_000, True),
        (100_000, False),  # no checkpoint, current chunk
        (500_000, False),
        (1_000_000, False),
        (2_000_000, False),
    ]
    for chunk_size, use_ckpt in configs:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # fresh model so timings are comparable (no warmstart)
        model = EmerGNN_MC(
            n_ent=n_ent,
            n_base_rel=n_base_rel,
            n_classes=86,
            n_dim=64,
            length=3,
            feat="M",
            morgan_features=morgan_mat,
            morgan_feat_dim=morgan_mat.shape[1],
        ).to(device)
        model.set_chunk_size(chunk_size)
        model.set_use_checkpoint(use_ckpt)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        try:
            sec, peak = bench(model, opt, h, t, r, src, dst, rel)
            ep_min = (sec * 1000) / 60
            print(
                f"chunk={chunk_size:>10,} ckpt={str(use_ckpt):5s}      "
                f"{sec:>10.3f}  {gb(peak):>10.2f}  {ep_min:>15.1f}"
            )
        except torch.cuda.OutOfMemoryError as e:
            print(
                f"chunk={chunk_size:>10,} ckpt={str(use_ckpt):5s}      OOM: {str(e)[:60]}"
            )
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

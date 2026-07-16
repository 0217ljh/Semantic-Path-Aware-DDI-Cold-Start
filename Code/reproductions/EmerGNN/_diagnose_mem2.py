"""Stage-2 diagnostic: build tKG + model + ONE training fwd+bwd at paper batch=32.

Adds explicit print before every potentially expensive op so we can see EXACTLY
where the memory explosion happens.
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


def p(stage):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[GPU] {stage:55s} alloc={a:6.2f}GB peak={peak:6.2f}GB", flush=True)
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
        print(f"[CPU] {stage:55s} ru_maxrss={ru:6.2f}GB", flush=True)
    except Exception:
        pass


def step(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}", flush=True)
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB", flush=True)
    p("baseline")

    from data_loader import load_vocab, load_setting, build_global_morgan_matrix

    step("vocab + morgan + S1_1")
    vocab = load_vocab()
    morgan_mat = build_global_morgan_matrix(vocab)
    s = load_setting("S1_1")
    p("after data load")

    n_ent = vocab["n_entity"]
    n_base_rel = vocab["n_relation"]
    print(f"  n_ent={n_ent} n_base_rel={n_base_rel}", flush=True)

    step("build vKG edges (sparse_coo)")
    from kg_builder import build_sparse_adj, edges_as_dense_lists
    vkg_triplets = np.concatenate([s["train_ddi"], s["valid_kg"]], axis=0)
    adj_v = build_sparse_adj(vkg_triplets, n_ent=n_ent, n_rel=n_base_rel, device=device)
    src_v, dst_v, rel_v = edges_as_dense_lists(adj_v)
    print(f"  vKG edges: {src_v.shape}", flush=True)
    del adj_v
    p("after vKG edges")

    step("build tKG edges (sparse_coo)")
    tkg_triplets = np.concatenate([s["train_ddi"], s["valid_ddi"], s["test_kg"]], axis=0)
    adj_t = build_sparse_adj(tkg_triplets, n_ent=n_ent, n_rel=n_base_rel, device=device)
    src_t, dst_t, rel_t = edges_as_dense_lists(adj_t)
    print(f"  tKG edges: {src_t.shape}", flush=True)
    del adj_t
    p("after tKG edges")

    step("construct EmerGNN_MC model")
    from model_mc import EmerGNN_MC
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
    n_params = sum(p.numel() for p in model.parameters())
    n_buffers = sum(b.numel() for b in model.buffers())
    print(f"  model: n_params={n_params:,}  n_buffers={n_buffers:,}", flush=True)
    p("after model.to(cuda)")

    step("ONE forward pass at batch=32 with vKG (no_grad)")
    model.eval()
    h = torch.randint(0, vocab["n_drug"], (32,), device=device)
    t = torch.randint(0, vocab["n_drug"], (32,), device=device)
    with torch.no_grad():
        logits = model(h, t, src_v, dst_v, rel_v)
    print(f"  logits shape: {logits.shape}", flush=True)
    p("after no_grad forward (batch=32)")

    step("ONE forward+backward at batch=32 (TRAIN MODE, autograd)")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    r = torch.randint(0, 86, (32,), device=device)
    opt.zero_grad(set_to_none=True)
    print("  -- forward --", flush=True)
    p("  before training forward")
    logits = model(h, t, src_v, dst_v, rel_v)
    p("  after training forward")
    print("  -- loss + backward --", flush=True)
    loss = torch.nn.functional.cross_entropy(logits, r, reduction="sum")
    p("  after loss")
    loss.backward()
    p("  after backward")
    opt.step()
    p("after one full train step")


if __name__ == "__main__":
    main()

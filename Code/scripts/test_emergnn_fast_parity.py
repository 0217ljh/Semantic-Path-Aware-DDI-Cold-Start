"""Full-model parity: EmerGNN vs EmerGNNFast produce identical logits + grads.

Loads EmerGNNFast's weights from a fresh EmerGNN (same seed), runs the full
forward (both u->v and v->u propagation + score head) on a random KG + pair
batch, and checks the output logits and parameter gradients match to within
fp32 summation-order roundoff. This verifies the drop-in at the MODEL level,
not just the propagation kernel.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && python Code/scripts/test_emergnn_fast_parity.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))  # so `baseline.emergnn.*` imports resolve

from baseline.emergnn.model import EmerGNN            # noqa: E402
from baseline.emergnn.model_fast import EmerGNNFast    # noqa: E402


def _make_kg(n_ent, n_edges, all_rel, device, g):
    src = torch.randint(0, n_ent, (n_edges,), generator=g, device=device)
    dst = torch.randint(0, n_ent, (n_edges,), generator=g, device=device)
    rel = torch.randint(0, all_rel, (n_edges,), generator=g, device=device)
    return src, dst, rel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] no CUDA — run under WSL project_1 for the real fp32 GPU check.")

    n_ent, n_base_rel, n_dim, length = 4000, 5, 64, 3
    all_rel = 2 * n_base_rel + 1
    n_edges, batch = 60_000, 32

    torch.manual_seed(0)
    morgan = np.random.RandomState(0).randint(0, 2, size=(n_ent, 1024)).astype(np.float32)

    base = EmerGNN(n_ent, n_base_rel, n_dim=n_dim, length=length, feat="M",
                   morgan_features=morgan).to(device)
    fast = EmerGNNFast(n_ent, n_base_rel, n_dim=n_dim, length=length, feat="M",
                       morgan_features=morgan).to(device)
    fast.load_state_dict(base.state_dict())  # identical weights
    base.train()
    fast.train()

    g = torch.Generator(device=device).manual_seed(1)
    src, dst, rel = _make_kg(n_ent, n_edges, all_rel, device, g)
    head = torch.randint(0, n_ent, (batch,), generator=g, device=device)
    tail = torch.randint(0, n_ent, (batch,), generator=g, device=device)
    labels = torch.randint(0, 2, (batch,), generator=g, device=device).float()

    def run(model):
        model.zero_grad(set_to_none=True)
        logits = model(head, tail, src, dst, rel)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return logits.detach(), grads

    lb, gb = run(base)
    lf, gf = run(fast)

    def rel_err(a, b):
        d = b.abs().max().clamp(min=1e-30).item()
        return (a - b).abs().max().item() / d

    logit_abs = (lb - lf).abs().max().item()
    logit_rel = rel_err(lf, lb)
    print(f"logits: max|Δ|={logit_abs:.2e}  rel={logit_rel:.1e}")

    worst_name, worst_rel = None, 0.0
    for n in gb:
        r = rel_err(gf[n], gb[n])
        if r > worst_rel:
            worst_rel, worst_name = r, n
    print(f"grads : worst rel={worst_rel:.1e}  ({worst_name})  over {len(gb)} param tensors")

    tol = 1e-3  # fp32 over a 3-layer double-propagation network — roundoff only
    ok = (logit_rel < tol) and (worst_rel < tol)
    print(f"\n[{'PASS' if ok else 'FAIL'}] EmerGNNFast parity vs EmerGNN "
          f"(rel tol {tol:.0e})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

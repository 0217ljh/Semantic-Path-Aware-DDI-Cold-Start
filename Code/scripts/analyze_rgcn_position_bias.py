"""Toy probe: does an R-GCN have a *positional* bias for a mediating node on a
path between a pair?

Setup. A chain of ``L_path`` nodes ``v1 - v2 - ... - vL`` (undirected + self loops,
single relation). The "pair" of interest is the two endpoints ``(v1, vL)``. Exactly
one intermediate position ``p in {2..L-1}`` holds a *mediating* node of type ``M``;
every other intermediate is a *neutral* filler ``N``. Endpoints are type ``D`` (drug).

Question. Treating the mediating *type* as a position-agnostic prototype throws away
WHERE on the path the mediator sits. But a message-passing net (R-GCN) is NOT
position-agnostic. This script measures the implicit positional weighting two ways:

Probe A (architecture, no training) -- perturbation influence.
  For a random-init R-GCN, influence(p) = || readout(M@p) - readout(all-neutral) ||.
  Averaged over many random inits. This is the pure architectural sensitivity of the
  pair readout to a mediator at position p. Reported for two readouts:
    * "head"  : readout = h_{v1}            -> expect MONOTONIC decay (proximity bias).
    * "pair"  : readout = h_{v1} + h_{vL}   -> expect a SYMMETRIC (often U-shaped) curve;
                the middle position is farthest from both endpoints.
  Also swept over the number of layers ``L_layers`` (a layer count below the distance
  to an endpoint is a hard cutoff -> zero influence from far positions).

Probe B (learned, optional --train) -- position-invariant task.
  Label y = 1 iff a mediator M is present (at a uniformly-random position); y = 0 iff
  all intermediates are neutral. The ground truth is position-INVARIANT, so any
  variation of the trained model's positive-logit across the mediator's position is a
  learned positional bias (harder to detect M at a disfavored position).

Run (CPU is fine; tiny):
  wsl bash -ic "conda activate project_1 && cd <root> && \
    python Code/scripts/analyze_rgcn_position_bias.py --path-len 5 --trials 300"
  add --train for probe B, --path-len 7/9 to see a longer decay curve.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# node types
D, M, N = 0, 1, 2   # drug-endpoint, mediator, neutral filler
N_TYPES = 3


def chain_adjacency(path_len: int) -> torch.Tensor:
    """Row-normalized neighbor aggregation matrix for an undirected chain with self
    loops (single relation). Returns (n, n) with rows summing to 1 (R-GCN 1/c_i norm)."""
    n = path_len
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, i] = 1.0                      # self loop
        if i > 0:
            A[i, i - 1] = 1.0
        if i < n - 1:
            A[i, i + 1] = 1.0
    A = A / A.sum(dim=1, keepdim=True)     # degree normalization
    return A


class RGCN(nn.Module):
    """Minimal single-relation R-GCN over a fixed chain: separate self vs neighbor
    weights (that split is exactly what makes it R-GCN, not plain GCN)."""

    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_TYPES, dim)
        self.self_w = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.rel_w = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.n_layers = n_layers

    def node_repr(self, types: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # types: (n,) long ; A: (n, n) normalized neighbor matrix (incl. self, but we
        # split self out below so remove the self-weight from A's off-diagonal use).
        h = self.embed(types)                          # (n, dim)
        # neighbor-only normalized matrix (A already includes self at 1/c_i; keep it as
        # the aggregation of ALL neighbors incl self for the rel branch, and add an
        # explicit self branch -> standard R-GCN self-loop-as-own-weight form).
        for l in range(self.n_layers):
            agg = A @ h                                # (n, dim) normalized neighbor+self agg
            h = F.relu(self.self_w[l](h) + self.rel_w[l](agg))
        return h


def sample_types(path_len: int, med_pos: int | None) -> torch.Tensor:
    """Endpoints=D; intermediate at med_pos (1-indexed position) = M; rest = N.
    med_pos None -> all intermediates neutral."""
    t = torch.full((path_len,), N, dtype=torch.long)
    t[0] = D
    t[path_len - 1] = D
    if med_pos is not None:
        t[med_pos - 1] = M
    return t


def readout(h: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "head":
        return h[0]
    if kind == "pair":
        return h[0] + h[-1]
    raise ValueError(kind)


@dataclass
class Cfg:
    path_len: int
    dim: int
    trials: int
    seed: int


@torch.no_grad()
def probe_influence(cfg: Cfg, n_layers: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, influence[pos]) averaged over random inits.
    influence(p) = || readout(M@p) - readout(all-neutral) ||_2."""
    A = chain_adjacency(cfg.path_len)
    positions = list(range(2, cfg.path_len))            # intermediate positions 2..L-1
    inf = np.zeros((cfg.trials, len(positions)))
    t_neutral = sample_types(cfg.path_len, None)
    types_by_pos = [sample_types(cfg.path_len, p) for p in positions]
    for tr in range(cfg.trials):
        torch.manual_seed(cfg.seed + tr)
        net = RGCN(cfg.dim, n_layers)
        base = readout(net.node_repr(t_neutral, A), kind)
        for j, tp in enumerate(types_by_pos):
            r = readout(net.node_repr(tp, A), kind)
            inf[tr, j] = torch.norm(r - base).item()
    return np.array(positions), inf.mean(0)


def run_probe_a(cfg: Cfg, layer_list: list[int]) -> None:
    print(f"\n=== PROBE A: architectural influence of a mediator vs path position "
          f"(path_len={cfg.path_len}, dim={cfg.dim}, {cfg.trials} random inits) ===")
    print("influence(p) = ||readout(M@p) - readout(all-neutral)||, mean over inits; "
          "each row normalized to its own max (shape matters, not scale).\n")
    for kind in ("head", "pair"):
        print(f"-- readout = {kind}  ({'h_v1' if kind=='head' else 'h_v1 + h_vL'}) --")
        for L in layer_list:
            pos, inf = probe_influence(cfg, L, kind)
            norm = inf / (inf.max() + 1e-12)
            cells = "  ".join(f"p{p}:{v:.2f}" for p, v in zip(pos, norm))
            print(f"   L={L} layers | {cells}   (raw max {inf.max():.3g})")
        print()
    print("Read: 'head' should decay monotonically with distance from v1 (proximity "
          "bias). 'pair' should be ~symmetric (p2~=pL-1); a U-shape means the MIDDLE "
          "mediator moves the pair representation LEAST. L below an endpoint's hop "
          "distance zeros out far positions (hard receptive-field cutoff).")


def run_probe_b(cfg: Cfg, n_layers: int, epochs: int) -> None:
    """Train on the position-invariant label 'mediator present', then report the
    trained positive-logit vs the mediator's position (flat = no learned bias)."""
    print(f"\n=== PROBE B: learned positional bias (position-invariant label) "
          f"L={n_layers} layers ===")
    torch.manual_seed(cfg.seed)
    A = chain_adjacency(cfg.path_len)
    net = RGCN(cfg.dim, n_layers)
    clf = nn.Linear(cfg.dim, 1)
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=1e-2)
    positions = list(range(2, cfg.path_len))

    def make_batch(bs: int):
        types, ys = [], []
        for _ in range(bs):
            if torch.rand(1).item() < 0.5:
                p = int(np.random.choice(positions)); y = 1.0
            else:
                p = None; y = 0.0
            types.append(sample_types(cfg.path_len, p)); ys.append(y)
        return torch.stack(types), torch.tensor(ys)

    net.train()
    for ep in range(epochs):
        tps, ys = make_batch(256)
        logits = torch.stack([clf(readout(net.node_repr(t, A), "pair")) for t in tps]).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"   [ep {ep+1}/{epochs}] loss={loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        print("   trained positive-logit by mediator position (higher = more confident; "
              "flat across positions = position-invariant learning):")
        for p in positions:
            tp = sample_types(cfg.path_len, p)
            lg = clf(readout(net.node_repr(tp, A), "pair")).item()
            print(f"     p{p}: logit={lg:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path-len", type=int, default=5, help="number of nodes on the chain (>=4)")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--trials", type=int, default=300, help="random inits for probe A")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="layer counts for probe A (default: 2..path_len)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train", action="store_true", help="also run probe B (training)")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    assert args.path_len >= 4, "need >=4 nodes so there is >=2 intermediate positions"

    cfg = Cfg(path_len=args.path_len, dim=args.dim, trials=args.trials, seed=args.seed)
    layer_list = args.layers or list(range(2, args.path_len + 1))
    run_probe_a(cfg, layer_list)
    if args.train:
        run_probe_b(cfg, n_layers=max(layer_list), epochs=args.epochs)


if __name__ == "__main__":
    main()

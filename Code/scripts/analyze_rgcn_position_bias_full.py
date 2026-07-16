"""Full R-GCN positional-bias study (2x2 tasks + multi-path joint prediction + depth sweep).

Companion to ``analyze_rgcn_position_bias.py`` (the quick random-init perturbation probe).
Per the codex-reviewed design:

  * DECODABILITY by position -- the primary operationalization of "utilization":
    a linear probe recovers "was the mediator M at position p?" from the readout
    actually used (node = h_v1, pair = h_v1 + h_vL), on a frozen random-init encoder.
  * 2x2 TASK CELLS: readout {node, pair} x label {position-INVARIANT, position-DEPENDENT},
    each reporting task accuracy *by mediator position*. The invariant cell uses a
    matched distractor token X so it cannot be solved by "some unusual token exists".
  * MULTI-PATH joint prediction: a theta graph (two anchors joined by K parallel paths);
    label = AND of a mediator being active on every path, slots EDGE-adjacent or CENTER.
  * DEPTH SWEEP as a primary axis + an OVERSMOOTHING metric.

Compute: all graphs share one (n,n) normalized adjacency A0, so a batch of B graphs is
a single batched matmul ``einsum('ij,bjd->bid', A0, h)`` -- O(B*n*d) memory, NO
block-diagonal blowup. Runs on GPU if available.

Run:
  python Code/scripts/analyze_rgcn_position_bias_full.py --part all --path-len 7
  python Code/scripts/analyze_rgcn_position_bias_full.py --part multipath --theta-inner 5
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D, M, N, X = 0, 1, 2, 3          # endpoint(drug), mediator, neutral, distractor
N_TYPES = 4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# graphs (return a single normalized (n,n) adjacency shared by every sample)
# --------------------------------------------------------------------------- #
def chain_norm_adj(path_len: int) -> torch.Tensor:
    n = path_len
    A = torch.zeros(n, n)
    for i in range(n):
        A[i, i] = 1.0
        if i > 0:
            A[i, i - 1] = 1.0
        if i < n - 1:
            A[i, i + 1] = 1.0
    return A / A.sum(1, keepdim=True)


def theta_norm_adj(n_paths: int, inner_len: int):
    """Two anchors (s=0, t=1) joined by ``n_paths`` internal chains of ``inner_len``
    nodes. Returns (A_norm, s, t, path_nodes) with path_nodes[k] the node indices along
    path k from the s-side to the t-side."""
    s, t = 0, 1
    nodes = 2
    path_nodes, edges = [], []
    for _k in range(n_paths):
        ch = list(range(nodes, nodes + inner_len))
        nodes += inner_len
        path_nodes.append(ch)
        edges.append((s, ch[0]))
        for a, b in zip(ch[:-1], ch[1:]):
            edges.append((a, b))
        edges.append((ch[-1], t))
    A = torch.zeros(nodes, nodes)
    for i in range(nodes):
        A[i, i] = 1.0
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    return A / A.sum(1, keepdim=True), s, t, path_nodes


# --------------------------------------------------------------------------- #
# model  (batched over graphs sharing A0; einsum, no block-diagonal)
# --------------------------------------------------------------------------- #
class RGCN(nn.Module):
    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_TYPES, dim)
        self.self_w = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.rel_w = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, types_bn: torch.Tensor, A0: torch.Tensor) -> torch.Tensor:
        # types_bn (B, n) long ; A0 (n, n). Returns (B, n, dim).
        h = self.embed(types_bn)
        for l in range(self.n_layers):
            agg = torch.einsum("ij,bjd->bid", A0, h)
            h = F.relu(self.self_w[l](h) + self.rel_w[l](agg))
        return h


def readout(h: torch.Tensor, s_idx: int, t_idx: int, kind: str) -> torch.Tensor:
    hs = h[:, s_idx]                 # (B, dim)
    if kind == "node":
        return hs
    return hs + h[:, t_idx]          # symmetric pair readout


# --------------------------------------------------------------------------- #
# chain sample generators  (types: (B, n) long)
# --------------------------------------------------------------------------- #
def _blank_chain(path_len: int, B: int) -> torch.Tensor:
    t = torch.full((B, path_len), N, dtype=torch.long)
    t[:, 0] = D
    t[:, -1] = D
    return t


def gen_invariant_distractor(path_len, B, rng):
    """Label = 'is a mediator M present'. Both classes carry a distractor X at a
    matched random position. Returns (types, y, m_pos) (m_pos 1-indexed or -1)."""
    inter = list(range(1, path_len - 1))
    t = _blank_chain(path_len, B)
    y = np.zeros(B, dtype=np.float32)
    m_pos = np.full(B, -1, dtype=np.int64)
    for b in range(B):
        xi = int(rng.choice(inter))
        t[b, xi] = X
        if rng.random() < 0.5:
            mi = int(rng.choice([i for i in inter if i != xi]))
            t[b, mi] = M
            y[b] = 1.0
            m_pos[b] = mi + 1
    return t, torch.tensor(y), m_pos


def gen_center_vs_edge(path_len, B, band, rng):
    """Position-DEPENDENT: one M at position p; label = 1 iff |p-center| <= band."""
    inter = list(range(1, path_len - 1))
    center = (path_len - 1) / 2.0
    central = [i for i in inter if abs(i - center) <= band]
    outer = [i for i in inter if abs(i - center) > band]
    t = _blank_chain(path_len, B)
    y = np.zeros(B, dtype=np.float32)
    m_pos = np.zeros(B, dtype=np.int64)
    for b in range(B):
        if rng.random() < 0.5 and central:
            mi = int(rng.choice(central)); y[b] = 1.0
        else:
            mi = int(rng.choice(outer if outer else central))
        t[b, mi] = M
        m_pos[b] = mi + 1
    return t, torch.tensor(y), m_pos


def gen_center_value(path_len, B, band, rng):
    """Position-DEPENDENT (decode-required): the CENTER always holds a mediator of one
    of two identities M(=1)/X(=3); every other intermediate is neutral N. Label = which
    identity sits in the center. The center is the ONLY non-neutral intermediate, so the
    label can be recovered ONLY by reading the center's identity -- reachable -> correct,
    unreachable -> chance (0.5). (An earlier version added an anti-correlated outer decoy
    to 'balance counts'; that leaked the label to a reachable outer node, so it is
    removed.) Returns (types, y, m_pos=center position)."""
    inter = list(range(1, path_len - 1))
    center = (path_len - 1) / 2.0
    central = [i for i in inter if abs(i - center) <= band]
    t = _blank_chain(path_len, B)
    y = np.zeros(B, dtype=np.float32)
    m_pos = np.zeros(B, dtype=np.int64)
    ident = (M, X)
    for b in range(B):
        ci = int(rng.choice(central))
        lab = int(rng.random() < 0.5)
        t[b, ci] = ident[lab]
        y[b] = float(lab)
        m_pos[b] = ci + 1
    return t, torch.tensor(y), m_pos


def gen_presence_at_p(path_len, B, p_idx, rng):
    t = _blank_chain(path_len, B)
    y = np.zeros(B, dtype=np.float32)
    for b in range(B):
        if rng.random() < 0.5:
            t[b, p_idx] = M; y[b] = 1.0
    return t, torch.tensor(y)


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
def train_encoder(gen_fn, path_len, s_idx, t_idx, kind, n_layers, dim, epochs, batch,
                  rng, seed):
    torch.manual_seed(seed)
    A0 = chain_norm_adj(path_len).to(DEV)
    net = RGCN(dim, n_layers).to(DEV)
    clf = nn.Linear(dim, 1).to(DEV)
    opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=5e-3)
    net.train()
    for _ep in range(epochs):
        types, y, _ = gen_fn(path_len, batch, rng)
        h = net(types.to(DEV), A0)
        r = readout(h, s_idx, t_idx, kind)
        loss = F.binary_cross_entropy_with_logits(clf(r).squeeze(-1), y.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    return net, clf, A0


@torch.no_grad()
def acc_by_position(net, clf, A0, gen_fn, path_len, s_idx, t_idx, kind, rng, n_eval=6000):
    types, y, m_pos = gen_fn(path_len, n_eval, rng)
    h = net(types.to(DEV), A0)
    r = readout(h, s_idx, t_idx, kind)
    pred = (torch.sigmoid(clf(r).squeeze(-1)) >= 0.5).cpu().numpy()
    yv = y.numpy()
    return {p: float((pred[m_pos == p] == yv[m_pos == p]).mean())
            for p in sorted(set(m_pos.tolist())) if p >= 0}


@torch.no_grad()
def decodability_by_position(net, A0, path_len, s_idx, t_idx, kind, rng, n_probe=3000):
    inter = list(range(1, path_len - 1))
    res = {}
    for p_idx in inter:
        types, y = gen_presence_at_p(path_len, n_probe, p_idx, rng)
        r = readout(net(types.to(DEV), A0), s_idx, t_idx, kind).cpu().numpy()
        yv = y.numpy(); ntr = n_probe // 2
        try:
            clf = LogisticRegression(max_iter=500).fit(r[:ntr], yv[:ntr])
            auc = roc_auc_score(yv[ntr:], clf.decision_function(r[ntr:]))
        except Exception:
            auc = float("nan")
        res[p_idx + 1] = auc
    return res


# --------------------------------------------------------------------------- #
# multi-path (theta) joint prediction
# --------------------------------------------------------------------------- #
def gen_theta_and(A_meta, inner_len, B, slot, rng):
    _A, s, t, path_nodes = A_meta
    n_nodes = _A.shape[0]
    K = len(path_nodes)
    slot_local = 0 if slot == "edge" else inner_len // 2
    types = torch.full((B, n_nodes), N, dtype=torch.long)
    types[:, s] = D; types[:, t] = D
    y = np.zeros(B, dtype=np.float32)
    for b in range(B):
        if rng.random() < 0.5:
            active = [True] * K; y[b] = 1.0
        else:
            active = [rng.random() < 0.5 for _ in range(K)]
            if all(active):
                active[int(rng.integers(K))] = False
        for k in range(K):
            types[b, path_nodes[k][slot_local]] = M if active[k] else N
    return types, torch.tensor(y)


def run_multipath(n_paths, inner_len, dim, layer_grid, epochs, batch, seed):
    print(f"\n=== MULTI-PATH joint prediction (theta: {n_paths} parallel paths, "
          f"inner_len={inner_len}); label = AND(mediator active on every path) ===")
    print(f"[device={DEV}] test accuracy by slot placement x depth (chance=0.5). "
          f"s-t hop of a slot: edge=1, center~={inner_len//2+1}.\n")
    A_meta = theta_norm_adj(n_paths, inner_len)
    A0 = A_meta[0].to(DEV); s, t = A_meta[1], A_meta[2]
    for slot in ("edge", "center"):
        row = []
        for L in layer_grid:
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            net = RGCN(dim, L).to(DEV); clf = nn.Linear(dim, 1).to(DEV)
            opt = torch.optim.Adam(list(net.parameters()) + list(clf.parameters()), lr=5e-3)
            net.train()
            for _ep in range(epochs):
                types, y = gen_theta_and(A_meta, inner_len, batch, slot, rng)
                r = readout(net(types.to(DEV), A0), s, t, "pair")
                loss = F.binary_cross_entropy_with_logits(clf(r).squeeze(-1), y.to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                types, y = gen_theta_and(A_meta, inner_len, 6000, slot, rng)
                r = readout(net(types.to(DEV), A0), s, t, "pair")
                pred = (torch.sigmoid(clf(r).squeeze(-1)) >= 0.5).cpu()
                row.append((pred == y).float().mean().item())
        print(f"   slot={slot:6s} | " + "  ".join(f"L{L}:{a:.2f}" for L, a in zip(layer_grid, row)))
    print("\nRead: if center under-use compounds under a joint(AND) read, 'center' "
          "accuracy lags 'edge', worst at shallow L (slot outside receptive field), "
          "improving as L covers it, then possibly degrading if over-deep (oversmoothing).")


# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_oversmooth(path_len, dim, layer_grid, seed, trials=200):
    print(f"\n=== OVERSMOOTHING vs depth (path_len={path_len}, {trials} random inits) ===")
    print("center M-vs-N gap = mean ||readout_pair(M@center) - readout_pair(all-N)||; "
          "node-cos-sim -> 1 = collapsed.\n")
    A0 = chain_norm_adj(path_len).to(DEV)
    ci = (path_len - 1) // 2
    tn = _blank_chain(path_len, 1)
    tm = tn.clone(); tm[0, ci] = M
    for L in layer_grid:
        gaps, sims = [], []
        for tr in range(trials):
            torch.manual_seed(seed + tr)
            net = RGCN(dim, L).to(DEV)
            hn = net(tn.to(DEV), A0)[0]; hm = net(tm.to(DEV), A0)[0]
            gaps.append(torch.norm((hm[0] + hm[-1]) - (hn[0] + hn[-1])).item())
            hh = F.normalize(hn, dim=1); sim = hh @ hh.t(); n = path_len
            sims.append(((sim.sum() - n) / (n * (n - 1))).item())
        print(f"   L={L:2d} | center M-vs-N gap={np.mean(gaps):.4f}  "
              f"node-cos-sim={np.mean(sims):.3f}")
    print("\nRead: gap rising then FALLING with depth = sweet spot; node-cos-sim -> 1 "
          "marks the oversmoothing regime where adding layers hurts all positions.")


def run_decode(path_len, dim, layer_grid, seed):
    print(f"\n=== DECODABILITY by position (frozen random-init; probe AUROC of 'M@p'; "
          f"path_len={path_len}, device={DEV}) ===  0.5=undecodable\n")
    for kind in ("node", "pair"):
        print(f"-- readout={kind} --")
        for L in layer_grid:
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            net = RGCN(dim, L).to(DEV)
            A0 = chain_norm_adj(path_len).to(DEV)
            res = decodability_by_position(net, A0, path_len, 0, path_len - 1, kind, rng)
            print(f"   L={L} | " + "  ".join(f"p{p}:{a:.2f}" for p, a in sorted(res.items())))
        print()


def run_tasks(path_len, dim, epochs, batch, seed, task_layers):
    need = -(-(path_len - 1) // 2)
    print(f"\n=== 2x2 TASK CELLS: trained accuracy by mediator position x DEPTH "
          f"(path_len={path_len}, device={DEV}) ===")
    print(f"pair-readout full coverage needs L >= {need}. chance=0.5. band=0 (exact center).\n")
    s_idx, t_idx = 0, path_len - 1
    b = 0
    cells = [
        ("node", "invariant", gen_invariant_distractor),
        ("pair", "invariant", gen_invariant_distractor),
        ("node", "center_value", lambda pl, bb, r: gen_center_value(pl, bb, b, r)),
        ("pair", "center_value", lambda pl, bb, r: gen_center_value(pl, bb, b, r)),
    ]
    for kind, label, gen in cells:
        print(f"-- readout={kind:4s} label={label:12s} --")
        for L in task_layers:
            rng = np.random.default_rng(seed)
            net, clf, A0 = train_encoder(gen, path_len, s_idx, t_idx, kind, L, dim,
                                         epochs, batch, rng, seed)
            acc = acc_by_position(net, clf, A0, gen, path_len, s_idx, t_idx, kind, rng)
            print(f"   L={L} | " + "  ".join(f"p{p}:{a:.2f}" for p, a in sorted(acc.items())))
        print()
    print("Read: 'invariant' (label position-independent) should be FLAT; a dip = that "
          "position can't be used (cutoff at small L). 'center_value' REQUIRES reading "
          "the center's identity (decoy balances global counts), so it is NOT solvable "
          "by 'center = the unreachable position': center accuracy -> chance until L "
          "covers the center, and may fall again when over-deep (oversmoothing).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["decode", "tasks", "multipath", "oversmooth", "all"],
                    default="all")
    ap.add_argument("--path-len", type=int, default=7)
    ap.add_argument("--dim", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--theta-paths", type=int, default=3)
    ap.add_argument("--theta-inner", type=int, default=5)
    args = ap.parse_args()
    grid = args.layers or [2, 3, 4, 6, 8]

    if args.part in ("decode", "all"):
        run_decode(args.path_len, args.dim, grid, args.seed)
    if args.part in ("tasks", "all"):
        run_tasks(args.path_len, args.dim, args.epochs, args.batch, args.seed,
                  args.layers or [2, 3, 4, 7])
    if args.part in ("multipath", "all"):
        run_multipath(args.theta_paths, args.theta_inner, args.dim, grid, args.epochs,
                      args.batch, args.seed)
    if args.part in ("oversmooth", "all"):
        run_oversmooth(args.path_len, args.dim, grid if args.layers else [2, 3, 4, 6, 8, 12, 16],
                       args.seed)


if __name__ == "__main__":
    main()

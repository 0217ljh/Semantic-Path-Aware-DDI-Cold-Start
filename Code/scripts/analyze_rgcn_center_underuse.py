"""Decisive test: does a TRAINED R-GCN still under-preserve a CENTER mediator vs an
EDGE mediator once the center is within the receptive field -- and does it compound
with multi-path joint (AND) contribution?  (Design agreed via a 2-round debate with
codex, thread 019f3f84.)

Key design decisions (why this is the fair test, not the earlier noiseless cutoff):
  * SLOT-SYMMETRIC training: each path's mediator, when present, sits at a uniformly
    random slot TYPE in {edge, center} (edge -> left/right uniform). The label is the
    multi-path AND of "mediator present", which is POSITION-INVARIANT. So training has
    ZERO incentive to prefer edge or center -> any residual slot bias is the trained
    architecture's own, not task targeting. (This corrected codex's first proposal of
    training separate edge/center models, which would confound architecture w/ task.)
  * Decisive metric = dAUROC = AUROC_edge - AUROC_center of a LINEAR probe that recovers
    a tagged path's mediator-presence bit from the FROZEN pair representation
    r = h_s + h_t (BEFORE the classifier head -- head nonlinearity would contaminate).
  * K = path multiplicity is the HEADLINE axis (multi-path compounding); hidden width is
    the MECHANISM-DIAGNOSTIC axis (gap shrinks with width => compression/oversquashing).
  * Depth pinned at minimal-covering (L=3) and +1 (L=4) only, to NOT conflate
    oversquashing (K/width) with oversmoothing (depth).
  * Collapse guard: if BOTH edge & center probe AUROC <= 0.6, the cell is a global
    compression failure, not slot-specific evidence -> reported but excluded.
  * Phase 2 (secondary): freeze the trained encoder, add eval-time noise at input vs
    readout, robustness by slot -> adjudicates the feature-vs-readout-noise question on
    the TRAINED representation.

CPU-only (forced). Batched via a shared (n,n) adjacency + einsum (no block-diagonal).

Run:
  python Code/scripts/analyze_rgcn_center_underuse.py                 # full matrix
  python Code/scripts/analyze_rgcn_center_underuse.py --seeds 3 --epochs 300 --noise
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D, M, N = 0, 1, 2                      # endpoint(drug), mediator, neutral
N_TYPES = 3
DEV = torch.device("cpu")             # forced CPU per user
INNER = 5                             # path inner length -> center is 3 hops from each anchor


def theta_adj(n_paths: int, inner_len: int):
    s, t = 0, 1
    nodes, path_nodes, edges = 2, [], []
    for _k in range(n_paths):
        ch = list(range(nodes, nodes + inner_len)); nodes += inner_len
        path_nodes.append(ch)
        edges.append((s, ch[0]))
        for a, b in zip(ch[:-1], ch[1:]):
            edges.append((a, b))
        edges.append((ch[-1], t))
    A = torch.zeros(nodes, nodes)
    for i in range(nodes):
        A[i, i] = 1.0
    for a, b in edges:
        A[a, b] = 1.0; A[b, a] = 1.0
    return A / A.sum(1, keepdim=True), s, t, path_nodes


class RGCN(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.embed = nn.Embedding(N_TYPES, dim)
        self.self_w = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.rel_w = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, types_bn, A0, feat_noise=0.0):
        h = self.embed(types_bn)
        if feat_noise:
            h = h + feat_noise * torch.randn_like(h)
        for l in range(self.n_layers):
            h = F.relu(self.self_w[l](h) + self.rel_w[l](torch.einsum("ij,bjd->bid", A0, h)))
        return h


def rep(h, s, t):                      # frozen pair representation r = h_s + h_t
    return h[:, s] + h[:, t]


_EDGE = (0, INNER - 1)
_CENTER = INNER // 2


def _place(types, b, node_list, local):
    types[b, node_list[local]] = M


def gen_train(meta, K, B, rng):
    """Slot-symmetric AND: each present path's mediator at a uniform slot-type
    {edge(->L/R), center}. Label = AND of presence. 50/50 balanced."""
    A, s, t, pn = meta
    types = torch.full((B, A.shape[0]), N, dtype=torch.long)
    types[:, s] = D; types[:, t] = D
    y = np.zeros(B, dtype=np.float32)
    for b in range(B):
        if rng.random() < 0.5:
            present = [True] * K; y[b] = 1.0
        else:
            present = [rng.random() < 0.5 for _ in range(K)]
            if all(present):
                present[int(rng.integers(K))] = False
        for k in range(K):
            if present[k]:
                local = int(rng.choice(_EDGE)) if rng.random() < 0.5 else _CENTER
                _place(types, b, pn[k], local)
    return types, torch.tensor(y)


def gen_probe(meta, K, slot_type, B, rng):
    """Tagged path 0 mediator at slot_type, presence bit = probe target; other paths
    are nuisance (random presence + random slot)."""
    A, s, t, pn = meta
    types = torch.full((B, A.shape[0]), N, dtype=torch.long)
    types[:, s] = D; types[:, t] = D
    target = np.zeros(B, dtype=np.float32)
    for b in range(B):
        if rng.random() < 0.5:
            local = int(rng.choice(_EDGE)) if slot_type == "edge" else _CENTER
            _place(types, b, pn[0], local)
            target[b] = 1.0
        for k in range(1, K):
            if rng.random() < 0.5:
                local = int(rng.choice(_EDGE)) if rng.random() < 0.5 else _CENTER
                _place(types, b, pn[k], local)
    return types, target


def train_model(K, L, dim, seed, epochs, batch):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    meta = theta_adj(K, INNER); A0 = meta[0].to(DEV); s, t = meta[1], meta[2]
    net = RGCN(dim, L).to(DEV); head = nn.Linear(dim, 1).to(DEV)
    opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=5e-3)
    net.train()
    for _ep in range(epochs):
        types, y = gen_train(meta, K, batch, rng)
        r = rep(net(types.to(DEV), A0), s, t)
        loss = F.binary_cross_entropy_with_logits(head(r).squeeze(-1), y.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        types, y = gen_train(meta, K, 4000, rng)
        pred = (torch.sigmoid(head(rep(net(types.to(DEV), A0), s, t)).squeeze(-1)) >= 0.5)
        val_acc = (pred.cpu() == (y >= 0.5)).float().mean().item()
    return net, head, meta, A0, val_acc


@torch.no_grad()
def probe_auroc(net, head, meta, A0, K, slot_type, rng, n=4000, feat_noise=0.0,
                read_noise=0.0):
    s, t = meta[1], meta[2]
    types, target = gen_probe(meta, K, slot_type, n, rng)
    r = rep(net(types.to(DEV), A0, feat_noise=feat_noise), s, t)
    if read_noise:
        r = r + read_noise * torch.randn_like(r)
    r = r.cpu().numpy()
    ntr = n // 2
    if len(set(target[:ntr].tolist())) < 2 or len(set(target[ntr:].tolist())) < 2:
        return float("nan"), float("nan")
    clf = LogisticRegression(max_iter=500).fit(r[:ntr], target[:ntr])
    auc = roc_auc_score(target[ntr:], clf.decision_function(r[ntr:]))
    # marginal logit drop: same tagged slot present vs absent, others balanced
    return auc, _marginal_logit(net, head, meta, A0, K, slot_type, rng)


@torch.no_grad()
def _marginal_logit(net, head, meta, A0, K, slot_type, rng, n=2000):
    s, t, pn = meta[1], meta[2], meta[3]
    A = meta[0]

    def batch(present0):
        types = torch.full((n, A.shape[0]), N, dtype=torch.long)
        types[:, s] = D; types[:, t] = D
        for b in range(n):
            if present0:
                local = int(rng.choice(_EDGE)) if slot_type == "edge" else _CENTER
                _place(types, b, pn[0], local)
            for k in range(1, K):
                if rng.random() < 0.5:
                    local = int(rng.choice(_EDGE)) if rng.random() < 0.5 else _CENTER
                    _place(types, b, pn[k], local)
        return head(rep(net(types.to(DEV), A0), s, t)).squeeze(-1).mean().item()

    return batch(True) - batch(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--Ls", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--dims", type=int, nargs="+", default=[8, 64])
    ap.add_argument("--noise", action="store_true", help="run Phase-2 noise robustness (L=3)")
    args = ap.parse_args()

    print(f"[device={DEV}] slot-symmetric AND; INNER={INNER} (center 3 hops from anchors); "
          f"seeds={args.seeds} epochs={args.epochs}")
    print("DECISIVE: dAUROC = AUROC_edge - AUROC_center (linear probe on frozen r=h_s+h_t). "
          "collapse if BOTH AUROC<=0.60.\n")

    # cache trained models for Phase-2 reuse
    store = {}
    print(f"{'L':>2} {'d':>3} {'K':>2} | {'val_acc':>7} | {'AUROC_edge':>10} {'AUROC_ctr':>10} "
          f"{'dAUROC':>8} | {'dlogit_e':>8} {'dlogit_c':>8} | flag")
    for L in args.Ls:
        for dim in args.dims:
            for K in args.Ks:
                ae, ac, va, de, dc = [], [], [], [], []
                for sd in range(args.seeds):
                    net, head, meta, A0, vacc = train_model(K, L, dim, sd, args.epochs, args.batch)
                    rng = np.random.default_rng(1000 + sd)
                    aue, dle = probe_auroc(net, head, meta, A0, K, "edge", rng)
                    auc, dlc = probe_auroc(net, head, meta, A0, K, "center", rng)
                    ae.append(aue); ac.append(auc); va.append(vacc); de.append(dle); dc.append(dlc)
                    if L == 3:
                        store[(dim, K, sd)] = (net, head, meta, A0)
                ae, ac = np.array(ae), np.array(ac)
                da = ae - ac
                collapse = (np.nanmean(ae) <= 0.60 and np.nanmean(ac) <= 0.60)
                flag = "COLLAPSE" if collapse else ("center<edge" if np.nanmean(da) > 0.03 else "~equal")
                print(f"{L:>2} {dim:>3} {K:>2} | {np.mean(va):>7.3f} | "
                      f"{np.nanmean(ae):>10.3f} {np.nanmean(ac):>10.3f} "
                      f"{np.nanmean(da):>+8.3f} | {np.mean(de):>+8.3f} {np.mean(dc):>+8.3f} | {flag}")
        print()

    print("PLOT DATA (x=K, y=dAUROC mean+-sem; lines=width; panels=L):")
    for L in args.Ls:
        for dim in args.dims:
            row = []
            for K in args.Ks:
                ae, ac = [], []
                for sd in range(args.seeds):
                    net, head, meta, A0, _ = train_model(K, L, dim, sd, args.epochs, args.batch)
                    rng = np.random.default_rng(1000 + sd)
                    aue, _ = probe_auroc(net, head, meta, A0, K, "edge", rng)
                    auc, _ = probe_auroc(net, head, meta, A0, K, "center", rng)
                    ae.append(aue); ac.append(auc)
                da = np.array(ae) - np.array(ac)
                row.append(f"K{K}:{np.nanmean(da):+.3f}±{np.nanstd(da)/max(1,np.sqrt(len(da))):.3f}")
            print(f"  L={L} d={dim:>2} | " + "  ".join(row))

    if args.noise:
        print("\nPHASE 2 (secondary): noise robustness on FROZEN trained encoder (L=3). "
              "probe AUROC vs sigma, by slot & noise-site.")
        for dim in args.dims:
            for K in args.Ks:
                for site in ("input", "readout"):
                    line = []
                    for sig in (0.0, 0.25, 0.5, 1.0, 2.0):
                        aue, auc = [], []
                        for sd in range(args.seeds):
                            key = (dim, K, sd)
                            if key not in store:
                                continue
                            net, head, meta, A0 = store[key]
                            rng = np.random.default_rng(2000 + sd)
                            fn = sig if site == "input" else 0.0
                            rn = sig if site == "readout" else 0.0
                            e, _ = probe_auroc(net, head, meta, A0, K, "edge", rng, feat_noise=fn, read_noise=rn)
                            c, _ = probe_auroc(net, head, meta, A0, K, "center", rng, feat_noise=fn, read_noise=rn)
                            aue.append(e); auc.append(c)
                        line.append(f"s{sig}:e{np.nanmean(aue):.2f}/c{np.nanmean(auc):.2f}")
                    print(f"  d={dim:>2} K={K} {site:>7} | " + "  ".join(line))


if __name__ == "__main__":
    main()

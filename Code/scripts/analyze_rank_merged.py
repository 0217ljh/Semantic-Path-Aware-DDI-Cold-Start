"""Merged analysis figure (NO retrain): per backbone SVD rank band, show BOTH the fixed CONTENT
(two lines: KG-structure vs DDI-mechanism alignment) AND the adapter-induced USAGE SHIFT (bubble
size = scorer usage, before=gray vs after=red), placed on the mechanism line.
Reading: the low-rank core is a KG-structure region DECOUPLED from mechanism (green<0); the frozen
scorer's usage (big gray bubble) sits exactly there -> misses mechanism. The adapter shrinks the
low-rank bubble and grows the high-rank bubbles (red) -> usage moves onto the mechanism-rich ranks.

Usage: python Code/scripts/analyze_rank_merged.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "scripts"))
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, pair_labels,  # noqa: E402
                                 cmnc_stratified)
from analyze_f1_function_subspace import head_weight          # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_scorer_effective import head_grad                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--cmnc-k", type=int, default=20); ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs)
    p_bb = comp._encode_backbone(te, proto.fact_context(data)).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)
    Xc = p_bb - p_bb.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False); V = Vt.T

    # usage per direction: backbone (linear head) vs design-R (mean head-gradient)
    w = head_weight(comp); w = w if w.ndim == 1 else w.mean(0)
    ub = (w @ V) ** 2 * S ** 2; ub = ub / ub.sum()
    gbar = head_grad(comp, p_prime).astype(np.float64).mean(0)
    varpp = ((p_prime - p_prime.mean(0)) @ V).var(0)
    udr = (gbar @ V) ** 2 * varpp; udr = udr / udr.sum()

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Yh = Ymbe[has]; Th = topo[has]; Bbb = Xc[has]
    tvar = Th[:, 0] + Th[:, 1]; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))
    tq = np.quantile(tvar, [.25, .5, .75]); nsh = Th[:, 2]; sq = np.quantile(nsh, [.25, .5, .75])
    Ytopo = np.concatenate([np.eye(4, dtype=np.int8)[np.digitize(tvar, tq)],
                            np.eye(4, dtype=np.int8)[np.digitize(nsh, sq)]], axis=1)
    g0 = np.zeros(len(Yh), dtype=np.int64)

    bands = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 40), (40, 80), (80, len(S))]
    lab = [f"{a+1}-{b}" for a, b in bands]
    st, me, uba, uda = [], [], [], []
    for a, b in bands:
        cols = list(range(a, b))
        st.append(cmnc_stratified(Bbb @ V[:, cols], Ytopo, g0, args.cmnc_k, args.n_perm)[0])
        me.append(cmnc_stratified(Bbb @ V[:, cols], Yh, strata, args.cmnc_k, args.n_perm)[0])
        uba.append(ub[a:b].sum()); uda.append(udr[a:b].sum())
    st = np.array(st); me = np.array(me); uba = np.array(uba); uda = np.array(uda)
    print("[merged] band | struct mech | usage bb -> dr")
    for i, l in enumerate(lab):
        print(f"    [{l:>7}] {st[i]:+.3f} {me[i]:+.3f} | {uba[i]:.3f} -> {uda[i]:.3f}")

    x = np.arange(len(bands))
    fig, axL = plt.subplots(figsize=(9.4, 5.8))
    axR = axL.twinx()
    axL.axhline(0, color="gray", lw=0.8, ls=":")
    axL.axvspan(-0.5, 0.5, color="#f2c94c", alpha=0.16, zorder=0)
    # LEFT axis: fixed CONTENT alignment (cMNC)
    lS, = axL.plot(x, st, "-o", color="#e08214", lw=2.0, ms=8, mec="black", mew=0.4,
                   label="KG structure alignment (left)")
    lM, = axL.plot(x, me, "-s", color="#2ca02c", lw=2.0, ms=8, mec="black", mew=0.4,
                   label="DDI mechanism alignment (left)")
    axL.set_ylabel("content alignment  (cMNC;  <0 = decoupled)", color="#333")
    # RIGHT axis: scorer USAGE before->after as vertical DUMBBELLS (aligned on x; no cross-band lines)
    axR.set_yscale("log")
    for i in range(len(bands)):
        axR.plot([x[i], x[i]], [uba[i], uda[i]], color="#d62728", lw=1.6, alpha=0.8, zorder=2)
    lB = axR.scatter(x, uba, s=70, facecolors="#9aa0a6", edgecolors="black", linewidths=0.5,
                     zorder=3, label="scorer usage BEFORE (right)")
    lA = axR.scatter(x, uda, s=85, marker="D", facecolors="#d62728", edgecolors="black",
                     linewidths=0.5, zorder=3, label="scorer usage AFTER (right)")
    axR.set_ylabel("scorer usage  (fraction of score variance, log)", color="#8a1a1a")
    axR.tick_params(axis="y", labelcolor="#8a1a1a")
    axL.set_xticks(x); axL.set_xticklabels(lab)
    axL.set_xlabel("backbone SVD rank band  (low -> high)")
    axL.set_title("Low ranks = KG structure DECOUPLED from mechanism (green<0), yet the frozen scorer\n"
                  "spends usage there (gray). The adapter moves usage onto the mechanism-rich high ranks (red).",
                  fontsize=9.5)
    axL.legend(handles=[lS, lM, lB, lA], fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "rank_merged.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[merged] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

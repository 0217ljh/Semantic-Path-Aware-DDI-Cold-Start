"""Figure A (single logic): the method-induced RANK SHIFT. We quantify how the design-R scorer
redistributes its usage across the frozen backbone's SVD ranks: usage at the low ranks (1-3, the
dominant/most-explanatory directions) is significantly WEAKENED, and at the non-low ranks it is
CONSISTENTLY ENHANCED. Only two quantities: rank band (x) and scorer usage (y), before vs after.

No cumulative-variance axis, no mechanism-alignment axis (those belong to Figure B). Model-fixed.

Usage: python Code/scripts/analyze_rank_shift.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
from analyze_f1_precheck import rebuild_composer              # noqa: E402
from analyze_f1_function_subspace import head_weight          # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_scorer_effective import head_grad                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs)
    p_bb = comp._encode_backbone(te, proto.fact_context(data)).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)

    Xc = p_bb - p_bb.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False); V = Vt.T
    w = head_weight(comp); w = w if w.ndim == 1 else w.mean(0)
    ub = (w @ V) ** 2 * S ** 2; ub = ub / ub.sum()           # backbone usage per dir
    gbar = head_grad(comp, p_prime).astype(np.float64).mean(0)
    varpp = ((p_prime - p_prime.mean(0)) @ V).var(0)
    udr = (gbar @ V) ** 2 * varpp; udr = udr / udr.sum()      # design-R usage per dir

    bands = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 40), (40, 80), (80, len(S))]
    lab = [f"{a+1}-{b}" for a, b in bands]
    ubb = np.array([ub[a:b].sum() for a, b in bands])
    udd = np.array([udr[a:b].sum() for a, b in bands])
    print("[rankshift] band | backbone_usage -> designR_usage (ratio)")
    for i, (a, b) in enumerate(bands):
        print(f"    [{lab[i]:>7}]  {ubb[i]:.4f} -> {udd[i]:.4f}  (x{udd[i]/max(ubb[i],1e-9):.1f})")

    x = np.arange(len(bands))
    fig, axx = plt.subplots(figsize=(7.6, 5.0))
    # before -> after arrows per band
    for i in range(len(bands)):
        up = udd[i] > ubb[i]
        axx.annotate("", xy=(x[i], udd[i]), xytext=(x[i], ubb[i]),
                     arrowprops=dict(arrowstyle="-|>", color=("#c0392b" if up else "#2c3e50"),
                                     lw=1.6, alpha=0.8))
    axx.scatter(x, ubb, s=90, c="#9aa0a6", edgecolors="black", linewidths=0.6, zorder=3,
                label="backbone scorer usage (before)")
    axx.scatter(x, udd, s=90, c="#c0392b", edgecolors="black", linewidths=0.6, zorder=3, marker="D",
                label="design-R scorer usage (after)")
    axx.set_yscale("log")
    axx.set_xticks(x); axx.set_xticklabels(lab)
    axx.set_xlabel("backbone SVD rank band  (low rank -> high rank)")
    axx.set_ylabel("scorer usage  (fraction of score variance, log)")
    axx.set_title("Method-induced rank shift: usage weakens at the low ranks (1-3)\n"
                  "and is consistently enhanced at the non-low ranks", fontsize=10)
    # background regions labeling what each rank range corresponds to (substantiated in Fig B)
    ylo, yhi = axx.get_ylim()
    axx.axvspan(-0.5, 0.5, color="#f2c94c", alpha=0.16, zorder=0)     # low-rank core
    axx.axvspan(0.5, len(bands) - 0.5, color="#56b4e9", alpha=0.08, zorder=0)  # mid/high ranks
    axx.text(0.0, yhi * 0.55, "KG basic structure\n(topology, low rank)", ha="center", va="top",
             fontsize=8, color="#8a6d00")
    axx.text(3.6, yhi * 0.55, "DDI mechanism  (mid / high rank)", ha="center", va="top",
             fontsize=8.5, color="#1f6aa5")
    axx.annotate(f"core: {ubb[0]:.2f} -> {udd[0]:.2f}", xy=(0, udd[0]), xytext=(0.55, udd[0] * 0.35),
                 fontsize=8, color="#2c3e50", arrowprops=dict(arrowstyle="->", lw=0.8))
    axx.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "rank_shift.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[rankshift] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

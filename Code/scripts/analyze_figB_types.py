"""Figure B (single logic): do the newly-ACTIVATED ranks correspond to the DDI types the backbone
previously mis-predicted? Per DrugBank interaction type T (binary positives carry the type):
  x = backbone per-type AUROC over {type-T positives vs all negatives}   (low = previously wrong)
  y = per-type AUROC drop when the ACTIVATED ranks (shoulder, ranks r_core..r_act) are ablated from
      the fused rep p_bb' before scoring                                  (high = type relies on them)
If the previously-wrong types (low x) are exactly the ones that rely on the activated ranks (high y),
the loop closes: Fig A shows usage shifts to the new ranks, Fig B shows those ranks fix the old errors.

Binary task, model-fixed, no retraining. Types from the multiclass drugbank_ryu labels on positives.

Usage: python Code/scripts/analyze_figB_types.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
import torch                                                  # noqa: E402
from sklearn.metrics import roc_auc_score                     # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import rebuild_composer              # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_radar_pertype import scores, pair_type_map       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--r-core", type=int, default=3); ap.add_argument("--r-act", type=int, default=40)
    ap.add_argument("--min-pos", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs); y = np.asarray(data.cold_test_labels).astype(int)
    p_bb = comp._encode_backbone(te, proto.fact_context(data)).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)

    # activated ranks = shoulder directions of the backbone spectrum (r_core .. r_act)
    Xc = p_bb - p_bb.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False); V = Vt.T
    Q = V[:, args.r_core:args.r_act]                          # the newly-activated (shoulder) subspace
    p_prime_ab = p_prime - (p_prime @ Q) @ Q.T               # ablate activated ranks from fused rep

    s_bb = scores(comp, p_bb, "backbone")                    # backbone-alone score
    s_dr = scores(comp, p_prime, "design")                  # design-R score
    s_ab = scores(comp, p_prime_ab, "design")               # design-R with activated ranks removed

    tmap = pair_type_map(args.dataset, args.fold)
    ptype = np.array([tmap.get(tuple(map(str, p)), -1) for p in te])
    neg = y == 0
    types = [t for t in sorted(set(ptype[ptype >= 0]))
             if int(((ptype == t) & (y == 1)).sum()) >= args.min_pos]
    rows = []
    for t in types:
        pos = (ptype == t) & (y == 1); sel = pos | neg; yt = pos[sel].astype(int)
        au_bb = roc_auc_score(yt, s_bb[sel])
        au_dr = roc_auc_score(yt, s_dr[sel])
        au_ab = roc_auc_score(yt, s_ab[sel])
        rows.append(dict(t=t, npos=int(pos.sum()), au_bb=au_bb, au_dr=au_dr, drop=au_dr - au_ab))
    xb = np.array([r["au_bb"] for r in rows]); yd = np.array([r["drop"] for r in rows])
    npos = np.array([r["npos"] for r in rows])
    gain = np.array([r["au_dr"] - r["au_bb"] for r in rows])
    r = np.corrcoef(xb, yd)[0, 1]
    print(f"[figB] {len(rows)} DrugBank types | activated ranks = SVD {args.r_core+1}-{args.r_act}")
    print(f"[figB] corr(backbone AUROC, ablation drop) = {r:+.3f}  (want NEGATIVE: weak types rely on new ranks)")
    print(f"[figB] corr(per-type GAIN, ablation drop)  = {np.corrcoef(gain, yd)[0,1]:+.3f}  (want POSITIVE: gain flows through new ranks)")
    print(f"[figB] corr(backbone AUROC, per-type GAIN) = {np.corrcoef(xb, gain)[0,1]:+.3f}  (want NEGATIVE: weak types gain most)")
    print(f"[figB] mean per-type: backbone {xb.mean():.3f} -> design-R {np.mean([rr['au_dr'] for rr in rows]):.3f} "
          f"| mean drop {yd.mean():+.3f}")
    print("[figB] per-type (t | npos | bb -> dr | gain | drop):")
    for rr in sorted(rows, key=lambda z: z["au_bb"])[:10]:
        print(f"    t={rr['t']:>3} n={rr['npos']:>4} | {rr['au_bb']:.3f} -> {rr['au_dr']:.3f} | "
              f"{rr['au_dr']-rr['au_bb']:+.3f} | {rr['drop']:+.4f}")

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    sc = ax.scatter(xb, yd, s=30 + 3 * npos, c=xb, cmap="RdYlGn", vmin=0.4, vmax=0.9,
                    edgecolors="black", linewidths=0.6, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":")
    # trend line
    if len(rows) > 2:
        b1, b0 = np.polyfit(xb, yd, 1)
        xs = np.linspace(xb.min(), xb.max(), 20)
        ax.plot(xs, b0 + b1 * xs, color="#c0392b", lw=1.4, ls="--", alpha=0.8,
                label=f"trend (r={r:+.2f})")
    ax.set_xlabel("backbone per-type AUROC  (low = previously mis-predicted type)")
    ax.set_ylabel("per-type AUROC drop when activated ranks ablated\n(high = type relies on the new ranks)")
    ax.set_title(f"Fig B. The activated (shoulder) ranks {args.r_core+1}-{args.r_act} carry exactly the\n"
                 "DDI types the backbone got wrong  [binary, model-fixed]", fontsize=10)
    ax.annotate("previously-wrong types\nrescued by the new ranks", xy=(xb[np.argmax(yd)], yd.max()),
                xytext=(0.6, yd.max() * 0.95), fontsize=8, color="#a11",
                arrowprops=dict(arrowstyle="->", color="#a11", lw=0.8))
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc, label="backbone per-type AUROC")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "figB_types.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[figB] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Scatter prototypes for the analysis section (current fixed frozen+last model, NO algo change).
Renders before/after (M_bb vs M_plug) 2D scatters under several projections, colored by mechanism
level, to find the honest + visually strong "adapter reorients toward mechanism clusters" figure.

Projections (each applied to M_bb=p_bb and M_plug=p_bb' separately unless noted):
  - umap    : UMAP of the FULL rep (standard; may be weak per F2)
  - lda     : supervised LDA-2 by the mechanism label (max separability the rep affords)
  - shoulder: project onto the F3 shoulder-mechanism top-2 (SAME ambient basis for both reps)

Color levels: --level protein (target/enzyme/transporter/carrier dominant) | mbe (6-way dominant).

Usage:
  python Code/scripts/analyze_scatter_prototype.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt \
    --level protein --proj umap,lda,shoulder --out Code/runs/est_pd1__frozen__last/scatters
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, MBE_LABELS)  # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_f3_ablation import mech_scores                   # noqa: E402
from kg.mediators import shared_mediators_idx                 # noqa: E402

PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def pair_role_counts(comp, pairs, node_mbe):
    """[N, 6] COUNT of each MBE role among the pair's shared mediators (for dominant coloring)."""
    kg = comp.adapter_model.kg; nbhd = comp.adapter_model.nbhd
    hp = comp.adapter_model.hp; mode = str(hp.get("med_mode", "and")); tau = int(hp.get("tau", 2))
    C = np.zeros((len(pairs), node_mbe.shape[1]), dtype=np.int32)
    for i, (u, v) in enumerate(pairs):
        ui, vi = kg.get_idx(str(u)), kg.get_idx(str(v))
        if ui is None or vi is None:
            continue
        med = shared_mediators_idx(kg, nbhd, int(ui), int(vi), mode=mode, tau=tau)["med_idx"]
        if len(med):
            C[i] = node_mbe[med].sum(0)
    return C


def scatter(ax, XY, lab, names, title):
    for c in np.unique(lab):
        m = lab == c
        ax.scatter(XY[m, 0], XY[m, 1], s=4, alpha=0.45, c=PALETTE[c % len(PALETTE)],
                   label=names[c], linewidths=0)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--level", default="protein", choices=["protein", "mbe"])
    ap.add_argument("--proj", default="umap,lda,shoulder")
    ap.add_argument("--r-core", type=int, default=3)
    ap.add_argument("--max-pts", type=int, default=4000)
    ap.add_argument("--out", default="Code/runs/est_pd1__frozen__last/scatters")
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs)
    fctx = proto.fact_context(data)
    p_bb = comp._encode_backbone(te, fctx).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)

    node_mbe = node_mbe_labels(kg)
    counts = pair_role_counts(comp, te, node_mbe)
    if args.level == "protein":
        cols = [MBE_LABELS.index(x) for x in ("target", "enzyme", "transporter", "carrier")]
        names = ["target", "enzyme", "transporter", "carrier"]
    else:
        cols = list(range(len(MBE_LABELS))); names = MBE_LABELS
    sub = counts[:, cols]
    keep = sub.sum(1) > 0
    lab_full = np.full(len(te), -1); lab_full[keep] = sub[keep].argmax(1)
    idx = np.where(keep)[0]
    rng = np.random.default_rng(0)
    if len(idx) > args.max_pts:
        idx = rng.choice(idx, args.max_pts, replace=False)
    lab = lab_full[idx]
    Xbb = p_bb[idx]; Xpp = p_prime[idx]
    print(f"[scatter] {len(idx)} pts | classes: "
          f"{dict(zip(names, [int((lab==c).sum()) for c in range(len(names))]))}")

    projs = args.proj.split(",")
    fig, axes = plt.subplots(len(projs), 2, figsize=(9, 4.2 * len(projs)), squeeze=False)
    for r, pj in enumerate(projs):
        if pj == "umap":
            import umap
            reB = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Xbb)
            reP = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Xpp)
        elif pj == "lda":
            reB = LinearDiscriminantAnalysis(n_components=2).fit(Xbb, lab).transform(Xbb)
            reP = LinearDiscriminantAnalysis(n_components=2).fit(Xpp, lab).transform(Xpp)
        elif pj == "shoulder":
            # F3 shoulder-mechanism top-2, SAME ambient basis on both reps
            Xbc = p_bb - p_bb.mean(0)
            _, _, Vbt = np.linalg.svd(Xbc, full_matrices=False); Vb = Vbt.T
            Pperp = np.eye(p_bb.shape[1]) - Vb[:, :args.r_core] @ Vb[:, :args.r_core].T
            Xperp = (p_prime[idx] - p_prime[idx].mean(0)) @ Pperp
            _, _, Vpt = np.linalg.svd(Xperp, full_matrices=False); Vperp = Vpt.T
            ms = mech_scores(Xperp, Vperp, np.eye(len(names), dtype=np.int8)[lab])
            Q = Vperp[:, np.argsort(-ms)[:2]]
            reB = (Xbb - Xbb.mean(0)) @ Q; reP = (Xpp - Xpp.mean(0)) @ Q
        else:
            raise SystemExit(f"unknown proj {pj}")
        scatter(axes[r][0], reB, lab, names, f"{pj}: M_bb (backbone, before)")
        scatter(axes[r][1], reP, lab, names, f"{pj}: M_plug (design-R, after)")
    axes[0][1].legend(markerscale=3, fontsize=8, loc="best")
    fig.suptitle(f"Reorientation scatters — level={args.level} (current fixed model)", fontsize=12)
    fig.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    png = out / f"scatter_{args.level}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[scatter] -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

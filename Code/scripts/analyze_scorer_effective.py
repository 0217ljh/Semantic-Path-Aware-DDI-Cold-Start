"""Scorer-effective scatter (path B, NO algorithm change). Visualizes F1/F3 honestly:
  LEFT  = backbone DOMINANT-VARIANCE plane (top-2 PCA of p_bb) -> what the backbone rep is
          organized by (topology, per F1) -> mechanism labels intermixed.
  RIGHT = design-R SCORING plane (top-2 PCA of the design-R head's per-pair gradient wrt p_bb')
          -> the directions the trained scorer is actually sensitive to -> mechanism separated (F3).
Not a naive full-rep UMAP (F2 shows that is flat); this shows what each model EFFECTIVELY uses.
Colored by mechanism level. Reports silhouette in each plane as a quant proxy for visible change.

Usage: python Code/scripts/analyze_scorer_effective.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt --level protein
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
from sklearn.metrics import silhouette_score                  # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import rebuild_composer, node_mbe_labels, MBE_LABELS  # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_scatter_prototype import scatter                 # noqa: E402
from analyze_scatter_sweep import build_levels                # noqa: E402


def head_grad(comp, p_prime, bs=2048):
    """Per-pair gradient of the design-R logit wrt p_bb' [N, d] (scorer sensitivity cloud)."""
    comp._train_mode(False)
    dev = comp.device
    g = np.zeros_like(p_prime)
    for s in range(0, len(p_prime), bs):
        x = torch.tensor(p_prime[s:s + bs], dtype=torch.float32, device=dev, requires_grad=True)
        out = comp.head(x)
        out[:, 0].sum().backward()                            # logit_i depends only on x_i -> per-pair grad
        g[s:s + bs] = x.grad.detach().cpu().numpy()
    return g


def pca2(X, basis=None):
    Xc = X - X.mean(0)
    if basis is None:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        basis = Vt[:2].T
    return Xc @ basis, basis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--level", default="protein", choices=["protein", "mbe", "pkpd", "posneg"])
    ap.add_argument("--max-pts", type=int, default=4000)
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
    grad = head_grad(comp, p_prime).astype(np.float64)        # scorer sensitivity cloud

    levels = build_levels(comp, te, np.asarray(data.cold_test_labels), node_mbe_labels(kg))
    lab_full, cls = levels[args.level]
    keep = np.where(lab_full >= 0)[0]
    rng = np.random.default_rng(0)
    if len(keep) > args.max_pts:
        keep = rng.choice(keep, args.max_pts, replace=False)
    lab = lab_full[keep]

    # LEFT: backbone dominant-variance plane (top-2 PCA of p_bb)
    XY_bb, _ = pca2(p_bb[keep])
    # RIGHT: design-R scoring plane = top-2 PCA of the head-gradient cloud, applied to p_bb'
    _, gbasis = pca2(grad[keep])
    XY_sc = (p_prime[keep] - p_prime[keep].mean(0)) @ gbasis

    sil_bb = silhouette_score(XY_bb, lab) if len(np.unique(lab)) > 1 else float("nan")
    sil_sc = silhouette_score(XY_sc, lab) if len(np.unique(lab)) > 1 else float("nan")
    print(f"[scorer-eff] level={args.level} n={len(lab)} | silhouette: "
          f"backbone-dominant-plane={sil_bb:.3f}  design-R-scoring-plane={sil_sc:.3f}  "
          f"delta={sil_sc-sil_bb:+.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.4))
    scatter(ax[0], XY_bb, lab, cls, "backbone dominant-variance plane (before)")
    scatter(ax[1], XY_sc, lab, cls, "design-R scoring plane (after)")
    ax[1].legend(markerscale=3, fontsize=8)
    fig.suptitle(f"Scorer-effective view by {args.level} — {Path(args.ckpt).parent.parent.name}")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters"
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"scorer_effective_{args.level}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"[scorer-eff] -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

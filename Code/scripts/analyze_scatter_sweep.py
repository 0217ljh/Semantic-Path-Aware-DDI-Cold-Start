"""Sweep visualization angles on the CURRENT fixed model to find the biggest VISIBLE before/after
(M_bb -> M_plug) change. Separation proxy = single-label cMNC (topology-stratified null) per level;
renders LDA-2 / UMAP scatters for the best-improving level + the task pos/neg level.

Levels: protein (target/enzyme/transporter/carrier dominant) | mbe (6-way dominant) |
        pkpd (PD=target-dominant vs PK=enzyme/transporter/carrier-dominant) | posneg (task label).

Usage: python Code/scripts/analyze_scatter_sweep.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
from sklearn.metrics import silhouette_score                  # noqa: E402

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, cmnc_stratified,  # noqa: E402
                                 MBE_LABELS)
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_scatter_prototype import pair_role_counts, scatter, PALETTE  # noqa: E402


def build_levels(comp, te, labels_posneg, node_mbe):
    counts = pair_role_counts(comp, te, node_mbe)
    prot = [MBE_LABELS.index(x) for x in ("target", "enzyme", "transporter", "carrier")]
    out = {}
    # protein 4-way dominant
    sub = counts[:, prot]; keep = sub.sum(1) > 0
    lab = np.full(len(te), -1); lab[keep] = sub[keep].argmax(1)
    out["protein"] = (lab, ["target", "enzyme", "transporter", "carrier"])
    # mbe 6-way dominant
    keep6 = counts.sum(1) > 0; lab6 = np.full(len(te), -1); lab6[keep6] = counts[keep6].argmax(1)
    out["mbe"] = (lab6, MBE_LABELS)
    # PK/PD binary: PD = target-count > (enzyme+transporter+carrier); PK otherwise
    pd_c = counts[:, MBE_LABELS.index("target")]
    pk_c = sum(counts[:, MBE_LABELS.index(x)] for x in ("enzyme", "transporter", "carrier"))
    keep2 = (pd_c + pk_c) > 0
    lab2 = np.full(len(te), -1); lab2[keep2] = (pd_c[keep2] > pk_c[keep2]).astype(int)
    out["pkpd"] = (lab2, ["PK", "PD"])
    # task pos/neg
    out["posneg"] = (labels_posneg.astype(int), ["neg", "pos"])
    return out


def lda2(X, lab):
    n = len(np.unique(lab))
    Z = LinearDiscriminantAnalysis(n_components=min(2, n - 1)).fit(X, lab).transform(X)
    if Z.shape[1] == 1:                                        # 2-class: add a PCA-residual y-axis
        r = X - X.mean(0)
        d = LinearDiscriminantAnalysis(n_components=1).fit(X, lab).coef_[0]
        d = d / (np.linalg.norm(d) + 1e-9)
        res = r - np.outer(r @ d, d)
        _, _, Vt = np.linalg.svd(res, full_matrices=False)
        Z = np.column_stack([Z[:, 0], res @ Vt[0]])
    return Z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--ks", default="15,30"); ap.add_argument("--max-pts", type=int, default=4000)
    ap.add_argument("--n-perm", type=int, default=150)
    ap.add_argument("--out", default="Code/runs/est_pd1__frozen__last/scatters")
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs); yposneg = np.asarray(data.cold_test_labels)
    fctx = proto.fact_context(data)
    p_bb = comp._encode_backbone(te, fctx).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)

    node_mbe = node_mbe_labels(kg)
    levels = build_levels(comp, te, yposneg, node_mbe)
    ks = [int(x) for x in args.ks.split(",")]

    print("\n=== separation-improvement sweep (single-label cMNC, topology-stratified null) ===")
    print(f"{'level':>8} {'n':>6} {'classes':>8} | {'k':>4} {'bb':>7} {'plug':>7} {'delta':>7} | "
          f"{'LDA-sil bb':>10} {'plug':>6}")
    results = {}
    for name, (lab_full, cls) in levels.items():
        keep = lab_full >= 0
        Xb, Xp, lab = p_bb[keep], p_prime[keep], lab_full[keep]
        # topology strata from degree proxy on the kept subset (deg via KG)
        deg = np.diff(kg.u_indptr).astype(np.float64)
        du = np.array([deg[kg.get_idx(str(a))] if kg.get_idx(str(a)) is not None else 0
                       for a, b in te[keep]])
        dv = np.array([deg[kg.get_idx(str(b))] if kg.get_idx(str(b)) is not None else 0
                       for a, b in te[keep]])
        tvar = du + dv; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))
        Yoh = np.eye(len(cls), dtype=np.int8)[lab]
        best_delta = -9
        for k in ks:
            bb = cmnc_stratified(Xb, Yoh, strata, k, args.n_perm)[0]
            pp = cmnc_stratified(Xp, Yoh, strata, k, args.n_perm)[0]
            silb = silhouette_score(lda2(Xb, lab), lab) if len(np.unique(lab)) > 1 else float("nan")
            silp = silhouette_score(lda2(Xp, lab), lab) if len(np.unique(lab)) > 1 else float("nan")
            print(f"{name:>8} {len(lab):>6} {len(cls):>8} | {k:>4} {bb:>7.3f} {pp:>7.3f} "
                  f"{pp-bb:>+7.3f} | {silb:>10.3f} {silp:>6.3f}")
            best_delta = max(best_delta, pp - bb)
        results[name] = best_delta
    winner = max(results, key=results.get)
    print(f"\n[sweep] biggest cMNC improvement: level={winner} (delta={results[winner]:+.3f})")

    # render winner + posneg (LDA-2, M_bb vs M_plug)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name in dict.fromkeys([winner, "posneg", "pkpd"]):
        lab_full, cls = levels[name]
        keep = np.where(lab_full >= 0)[0]
        if len(keep) > args.max_pts:
            keep = rng.choice(keep, args.max_pts, replace=False)
        lab = lab_full[keep]; Xb, Xp = p_bb[keep], p_prime[keep]
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
        scatter(ax[0], lda2(Xb, lab), lab, cls, f"{name}: M_bb (before)")
        scatter(ax[1], lda2(Xp, lab), lab, cls, f"{name}: M_plug (after)")
        ax[1].legend(markerscale=3, fontsize=8)
        fig.suptitle(f"LDA-2 by {name} (current fixed model)")
        fig.tight_layout(); png = out / f"sweep_{name}.png"
        fig.savefig(png, dpi=130, bbox_inches="tight"); plt.close(fig)
        print(f"[sweep] rendered -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

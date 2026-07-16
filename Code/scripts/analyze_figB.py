"""Figure B (two messages, both about the high ranks the adapter activates; NO retrain):
  B1: the HIGH ranks are enriched with mechanism. x = backbone SVD rank band (same framing as Fig A),
      y = mechanism alignment (topology-stratified cMNC) of that band. The low-rank core (1-3) is
      mechanism-poor (negative); the non-low ranks are mechanism-rich (positive).
  B2: those high-rank / mechanism-rich directions carry the OVERALL cold-start gain (F3). Subspace
      ablation: x = mechanism content of the ablated subspace, y = cold-test AUROC drop from p_bb'
      (filled) vs from p_bb (hollow ring, ~0); mechanism subspaces vs matched/random nulls.
Specific per-mechanism detail is left to the case study (part 3), not forced at the noisy per-type level.

Usage: python Code/scripts/analyze_figB.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_f3_ablation import head_auroc, mech_scores       # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--r-core", type=int, default=3); ap.add_argument("--cmnc-k", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs); y = np.asarray(data.cold_test_labels).astype(int)
    p_bb = comp._encode_backbone(te, proto.fact_context(data)).detach().cpu().numpy().astype(np.float64)
    _, p_prime = correction_and_fused(comp, te, p_bb); p_prime = p_prime.astype(np.float64)
    Xc = p_bb - p_bb.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False); V = Vt.T

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Yh = Ymbe[has]; Th = topo[has]; Xbh = Xc[has]; Xph = (p_prime[has] - p_prime[has].mean(0))
    tvar = Th[:, 0] + Th[:, 1]; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))

    # ---- B1: per-band DUAL content = KG-structure(topology) alignment vs mechanism alignment.
    # topology labels = quantile bins of (deg_u+deg_v) and of #shared mediators (KG basic structure);
    # topology cMNC uses a GLOBAL null (raw topology grouping). mechanism cMNC uses a topology-
    # stratified null (mechanism BEYOND topology). Evidence: low rank = topology-high/mechanism-low,
    # high rank = topology-low/mechanism-high  -> proves the "rank region -> content" labels.
    tq = np.quantile(tvar, [.25, .5, .75]); Ytopo = np.eye(4, dtype=np.int8)[np.digitize(tvar, tq)]
    nsh = Th[:, 2]; sq = np.quantile(nsh, [.25, .5, .75]); Ytopo2 = np.eye(4, dtype=np.int8)[np.digitize(nsh, sq)]
    Ytopo = np.concatenate([Ytopo, Ytopo2], axis=1)          # degree + #shared structure labels
    g0 = np.zeros(len(Yh), dtype=np.int64)                   # global null for topology
    bands = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 40), (40, 80), (80, len(S))]
    lab = [f"{a+1}-{b}" for a, b in bands]
    cmb = np.array([cmnc_stratified(Xbh @ V[:, list(range(a, b))], Yh, strata, args.cmnc_k,
                                    args.n_perm)[0] for a, b in bands])           # mechanism
    ctopo = np.array([cmnc_stratified(Xbh @ V[:, list(range(a, b))], Ytopo, g0, args.cmnc_k,
                                      args.n_perm)[0] for a, b in bands])         # KG structure
    print("[figB] B1 per-band (KG-structure cMNC | mechanism cMNC):")
    for i, l in enumerate(lab):
        print(f"    [{l:>7}] structure {ctopo[i]:+.3f} | mechanism {cmb[i]:+.3f}")

    # ---- B2: F3 subspace ablation (fused rep, core-complement mechanism subspace)
    V_core = V[:, :args.r_core]; Pperp = np.eye(p_bb.shape[1]) - V_core @ V_core.T
    Xpp_perp = Xph @ Pperp
    _, Sp, Vpt = np.linalg.svd(Xpp_perp, full_matrices=False); Vperp = Vpt.T
    keep = Sp > 1e-6 * Sp[0]; Vperp = Vperp[:, keep]; Sp = Sp[keep]
    ms = mech_scores(Xpp_perp, Vperp, Yh); order = np.argsort(-ms); low = list(np.argsort(ms)[:len(ms)//2])
    base_dr = head_auroc(comp, p_prime, y, task, "design"); base_bb = head_auroc(comp, p_bb, y, task, "backbone")

    def drop(cols):
        Q = Vperp[:, cols]
        d_dr = base_dr - head_auroc(comp, p_prime - (p_prime @ Q) @ Q.T, y, task, "design")
        d_bb = base_bb - head_auroc(comp, p_bb - (p_bb @ Q) @ Q.T, y, task, "backbone")
        mc = cmnc_stratified(Xph @ Q, Yh, strata, args.cmnc_k, 80)[0]
        return mc, d_dr, d_bb

    rng = np.random.default_rng(0); pts = []
    for kab in [5, 10, 20, 30]:
        mc, ddr, dbb = drop(order[:kab]); pts.append(("mech", mc, ddr, dbb))
        pool = list(low); vm = []
        for j in order[:kab]:
            c = min(pool, key=lambda q: abs(Sp[q] - Sp[j])); vm.append(c); pool.remove(c)
        mc, ddr, dbb = drop(np.array(vm)); pts.append(("varmatch", mc, ddr, dbb))
        for _ in range(6):
            mc, ddr, dbb = drop(rng.choice(len(ms), kab, replace=False)); pts.append(("random", mc, ddr, dbb))

    # ---- render
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.2))
    x = np.arange(len(bands))
    ax[0].axhline(0, color="gray", lw=0.8, ls=":")
    ax[0].plot(x, ctopo, "-o", color="#e08214", lw=1.8, ms=9, mec="black", mew=0.5,
               label="KG structure alignment (topology)")
    ax[0].plot(x, cmb, "-s", color="#2ca02c", lw=1.8, ms=9, mec="black", mew=0.5,
               label="DDI mechanism alignment (shared-MBE)")
    # shade which content dominates each band
    dom_struct = ctopo > cmb
    for i in range(len(bands)):
        ax[0].axvspan(i - 0.5, i + 0.5, color=("#f2c94c" if dom_struct[i] else "#56b4e9"),
                      alpha=0.10, zorder=0)
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab)
    ax[0].set_xlabel("backbone SVD rank band  (low -> high)")
    ax[0].set_ylabel("alignment  (cMNC)")
    ax[0].set_title("B1. Low ranks encode KG structure, high ranks encode\nDDI mechanism (content of each rank region)",
                    fontsize=9.5)
    ax[0].legend(fontsize=8, loc="center right")

    KIND = {"random": ("#2ca02c", "s", "random null", 38, 0.30),
            "varmatch": ("#1f77b4", "D", "variance-matched null", 85, 0.85),
            "mech": ("#d62728", "o", "high-rank mechanism subspace", 100, 0.9)}
    for kind, (col, mk, l, sz, al) in KIND.items():
        m = [p for p in pts if p[0] == kind]
        ax[1].scatter([p[1] for p in m], [p[2] for p in m], s=sz, c=col, marker=mk, alpha=al,
                      edgecolors="black", linewidths=0.4, label=l)
    mm = [p for p in pts if p[0] == "mech"]
    ax[1].scatter([p[1] for p in mm], [p[3] for p in mm], s=100, facecolors="none",
                  edgecolors="#d62728", linewidths=1.3, marker="o", label="same dirs from p_bb (reverse ~0)")
    ax[1].axhline(0, color="gray", lw=0.8, ls=":")
    ax[1].set_xlabel("mechanism content of ablated subspace  (topology-stratified cMNC)")
    ax[1].set_ylabel("overall cold-test AUROC drop when ablated")
    ax[1].set_title("B2. The high-rank mechanism subspace carries the\noverall cold-start gain (F3)", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fig B. High ranks are mechanism-rich (B1) and carry the overall gain (B2)  "
                 "[binary, model-fixed]", fontsize=11)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "figB.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[figB] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

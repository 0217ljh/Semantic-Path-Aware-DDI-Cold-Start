"""Analysis figure parts 1+2 (NO retrain), using the ROBUST subspace-level computations (not the
per-direction ones that failed). Panels share the F1/F3 story:

Panel A (F1 misalignment, part 1): the backbone spectrum in variance BANDS (robust subspaces, not
single dirs). Per band: x = cumulative variance reached, y = mechanism cMNC (topology-stratified
null), bubble size = frozen scorer usage (score-variance) of that band. The dominant/most-scored
bands (big bubbles, low cumvar) sit at LOW/NEGATIVE mechanism -> the scored subspace is not
mechanism-aligned; mechanism lives in low-usage shoulder bands.

Panel B (F3 causal + part 2 alignment): subspace ablation scatter. Points = the backbone-ignored
shoulder-mechanism subspace (top-k mech dirs in the core complement, several k) and matched nulls.
x = mechanism content of the subspace, y = cold-test AUROC drop when ablated from the FUSED p_bb'
(hollow ring = same ablation from backbone p_bb, ~0). The gain flows through the mechanism subspace.

Usage: python Code/scripts/analyze_analysis_fig.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
from analyze_f3_ablation import head_auroc, mech_scores       # noqa: E402
from analyze_scorer_effective import head_grad                # noqa: E402


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
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    w = head_weight(comp); w = w if w.ndim == 1 else w.mean(0)
    svar = (w @ V) ** 2 * S ** 2; svar = svar / svar.sum()   # backbone scorer usage per dir
    # design-R scorer usage per BACKBONE dir: mean head-gradient g_bar; (g_bar.v)^2 * var(p_bb'@v)
    gbar = head_grad(comp, p_prime).astype(np.float64).mean(0)
    varpp = ((p_prime - p_prime.mean(0)) @ V).var(0)
    svar_dr = (gbar @ V) ** 2 * varpp; svar_dr = svar_dr / svar_dr.sum()

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Yh = Ymbe[has]; Th = topo[has]
    tvar = Th[:, 0] + Th[:, 1]; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))
    Xbh = Xc[has]; Xph = (p_prime[has] - p_prime[has].mean(0))

    # ---- Panel A: variance BANDS (robust subspaces)
    bands = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 40), (40, 80), (80, len(S))]
    rows = []
    for a, b in bands:
        cols = list(range(a, b))
        cm_bb = cmnc_stratified(Xbh @ V[:, cols], Yh, strata, args.cmnc_k, args.n_perm)[0]
        cm_pp = cmnc_stratified(Xph @ V[:, cols], Yh, strata, args.cmnc_k, args.n_perm)[0]
        rows.append(dict(a=a, b=b, cv=cumvar[b - 1], usage=svar[a:b].sum(),
                         usage_dr=svar_dr[a:b].sum(), cm_bb=cm_bb, cm_pp=cm_pp))
    print("[fig] bands (rank | cumvar | bb_usage -> dr_usage | mech_cMNC bb -> plug):")
    for r in rows:
        print(f"    [{r['a']:>3}-{r['b']:<3}] cv={r['cv']:.2f} usage {r['usage']:.3f}->{r['usage_dr']:.3f} "
              f"mech {r['cm_bb']:+.3f} -> {r['cm_pp']:+.3f}")

    # ---- Panel B: F3 subspace ablation. The causal subspace is the core-complement mechanism
    # subspace OF THE FUSED REP p_bb' (defined relative to the backbone core V_core). NOTE: defining
    # it in backbone coords makes the drop vanish -> the gain flows through the ADAPTER-CONSTRUCTED
    # mechanism subspace, not the backbone's own dirs. The reverse test (ablate same dirs from p_bb
    # -> ~0) shows the backbone does not use them. Framed honestly, not as backbone-emergent.
    V_core = V[:, :args.r_core]; Pperp = np.eye(p_bb.shape[1]) - V_core @ V_core.T
    Xpp_perp = Xph @ Pperp                                    # fused rep, core-complement
    _, Sp, Vpt = np.linalg.svd(Xpp_perp, full_matrices=False); Vperp = Vpt.T
    keep = Sp > 1e-6 * Sp[0]; Vperp = Vperp[:, keep]; Sp = Sp[keep]
    ms = mech_scores(Xpp_perp, Vperp, Yh)
    order = np.argsort(-ms); low = list(np.argsort(ms)[:len(ms) // 2])
    base_dr = head_auroc(comp, p_prime, y, task, "design")
    base_bb = head_auroc(comp, p_bb, y, task, "backbone")

    def ablate_drop(cols):
        Q = Vperp[:, cols]
        d_dr = base_dr - head_auroc(comp, p_prime - (p_prime @ Q) @ Q.T, y, task, "design")
        d_bb = base_bb - head_auroc(comp, p_bb - (p_bb @ Q) @ Q.T, y, task, "backbone")
        mc = cmnc_stratified(Xph @ Q, Yh, strata, args.cmnc_k, 80)[0]
        return mc, d_dr, d_bb

    rng = np.random.default_rng(0)
    pts = []
    for kab in [5, 10, 20, 30]:
        mc, ddr, dbb = ablate_drop(order[:kab])              # mechanism subspace
        pts.append(dict(kab=kab, kind="mech", mech=mc, d_dr=ddr, d_bb=dbb))
        pool = list(low); vm = []                            # variance-matched-low-mech null
        for j in order[:kab]:
            c = min(pool, key=lambda q: abs(Sp[q] - Sp[j])); vm.append(c); pool.remove(c)
        mc, ddr, dbb = ablate_drop(np.array(vm))
        pts.append(dict(kab=kab, kind="varmatch", mech=mc, d_dr=ddr, d_bb=dbb))
        for _ in range(6):                                   # random null replicate cloud
            mc, ddr, dbb = ablate_drop(rng.choice(len(ms), kab, replace=False))
            pts.append(dict(kab=kab, kind="random", mech=mc, d_dr=ddr, d_bb=dbb))
    print("[fig] F3 subspace (kind k | mech_cMNC | drop_dr | drop_bb):")
    for p in pts:
        print(f"    {p['kind']:>8} {p['kab']:>3} | {p['mech']:+.3f} | {p['d_dr']:+.4f} | {p['d_bb']:+.4f}")

    # ---- render
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.4))
    cv_lo = np.array([cumvar[r["a"] - 1] if r["a"] > 0 else 0.0 for r in rows])
    cv_hi = np.array([r["cv"] for r in rows]); cvm = (cv_lo + cv_hi) / 2
    ub = np.array([r["usage"] for r in rows]); udr = np.array([r["usage_dr"] for r in rows])
    cmb = np.array([r["cm_bb"] for r in rows])
    # y = backbone mechanism alignment per band (fixed reference: where mechanism lives). Two bubble
    # sets = scorer usage BEFORE (backbone) and AFTER (design-R); the scorer redistributes usage
    # from the mechanism-poor core toward the mechanism-rich shoulder.
    ax[0].axhline(0, color="gray", lw=0.8, ls=":")
    ax[0].hlines(cmb, cv_lo, cv_hi, color="#dddddd", lw=1.0, zorder=0)
    ax[0].scatter(cvm, cmb, s=50 + 2600 * ub, c="#9aa0a6", alpha=0.55, edgecolors="black",
                  linewidths=0.6, label="backbone scorer usage (before)")
    ax[0].scatter(cvm, cmb, s=50 + 2600 * udr, facecolors="none", edgecolors="#d62728",
                  linewidths=1.6, label="design-R scorer usage (after)")
    for r, x, yv in zip(rows, cvm, cmb):
        ax[0].annotate(f"[{r['a']+1}-{r['b']}]", (x, yv), textcoords="offset points",
                       xytext=(5, 7), fontsize=6.5, color="#555")
    ax[0].set_xlabel("backbone variance band  (fixed a-priori bins)")
    ax[0].set_ylabel("mechanism alignment of band  (topology-stratified cMNC)")
    ax[0].set_title("A. The adapter moves scorer usage off the mechanism-poor\ncore onto mechanism-rich bands (F1->F2)",
                    fontsize=9.5)
    ax[0].legend(fontsize=8, loc="lower center")
    ax[0].annotate("core: 97%->62% usage\n(mechanism-poor)", xy=(cvm[0], cmb[0] + 0.03),
                   xytext=(0.60, -0.15), fontsize=8, color="#333",
                   arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))
    ax[0].annotate("shoulder usage grows\n(mechanism-rich)", xy=(cvm[3], cmb[3]),
                   xytext=(0.30, 0.42), fontsize=8, color="#a11",
                   arrowprops=dict(arrowstyle="->", color="#a11", lw=0.8))
    # Panel B: filled = ablate from fused p_bb'; random null = low-alpha cloud
    KIND = {"random": ("#2ca02c", "s", "random null", 38, 0.30),
            "varmatch": ("#1f77b4", "D", "variance-matched null", 85, 0.85),
            "mech": ("#d62728", "o", "shoulder-mechanism dirs", 100, 0.9)}
    for kind, (col, mk, lab, sz, al) in KIND.items():
        m = [p for p in pts if p["kind"] == kind]
        ax[1].scatter([p["mech"] for p in m], [p["d_dr"] for p in m], s=sz, c=col, marker=mk,
                      alpha=al, edgecolors="black", linewidths=0.4, label=lab)
    mm = [p for p in pts if p["kind"] == "mech"]              # reverse: same dirs ablated from p_bb ~0
    ax[1].scatter([p["mech"] for p in mm], [p["d_bb"] for p in mm], s=100, facecolors="none",
                  edgecolors="#d62728", linewidths=1.3, marker="o", label="same dirs from p_bb (reverse ~0)")
    ax[1].axhline(0, color="gray", lw=0.8, ls=":")
    ax[1].set_xlabel("mechanism content of ablated subspace  (topology-stratified cMNC)")
    ax[1].set_ylabel("cold-test AUROC drop when ablated")
    ax[1].set_title("B. Cold-start gain is carried by mechanism-rich\nsubspaces the backbone ignores (F3)",
                    fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Backbone scorer under-reads mechanism (A); the gain flows through it (B)  "
                 "[model-fixed, no retraining]", fontsize=11)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "analysis_fig.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

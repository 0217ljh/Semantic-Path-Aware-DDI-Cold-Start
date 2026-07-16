"""Per-band content, BEFORE vs AFTER the adapter (NO retrain). For each backbone SVD rank band we
measure two alignments, on the backbone rep p_bb (before) AND on the fused rep p_bb' (after):
  KG-structure alignment = cMNC with topology labels (degree & #shared-mediator quantile bins), GLOBAL null.
  DDI-mechanism alignment = cMNC with shared-MBE labels (DDI-masked KG), TOPOLOGY-STRATIFIED null.
=> 4 lines: {before, after} x {KG, mechanism}. The bands are the backbone's own SVD directions, so
before/after project the SAME directions but on the two representations.

Usage: python Code/scripts/analyze_rank_content.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False); V = Vt.T   # backbone SVD directions

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Yh = Ymbe[has]; Th = topo[has]
    Bbb = Xc[has]; Bpp = (p_prime[has] - p_prime[has].mean(0))
    tvar = Th[:, 0] + Th[:, 1]; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))
    tq = np.quantile(tvar, [.25, .5, .75]); nsh = Th[:, 2]; sq = np.quantile(nsh, [.25, .5, .75])
    Ytopo = np.concatenate([np.eye(4, dtype=np.int8)[np.digitize(tvar, tq)],
                            np.eye(4, dtype=np.int8)[np.digitize(nsh, sq)]], axis=1)
    g0 = np.zeros(len(Yh), dtype=np.int64)

    bands = [(0, 3), (3, 8), (8, 15), (15, 22), (22, 40), (40, 80), (80, len(S))]
    lab = [f"{a+1}-{b}" for a, b in bands]
    rows = {"struct_bb": [], "mech_bb": [], "struct_pp": [], "mech_pp": []}
    for a, b in bands:
        cols = list(range(a, b))
        rows["struct_bb"].append(cmnc_stratified(Bbb @ V[:, cols], Ytopo, g0, args.cmnc_k, args.n_perm)[0])
        rows["mech_bb"].append(cmnc_stratified(Bbb @ V[:, cols], Yh, strata, args.cmnc_k, args.n_perm)[0])
        rows["struct_pp"].append(cmnc_stratified(Bpp @ V[:, cols], Ytopo, g0, args.cmnc_k, args.n_perm)[0])
        rows["mech_pp"].append(cmnc_stratified(Bpp @ V[:, cols], Yh, strata, args.cmnc_k, args.n_perm)[0])
    for kk in rows:
        rows[kk] = np.array(rows[kk])
    print("[content] band | struct_bb mech_bb | struct_pp mech_pp")
    for i, l in enumerate(lab):
        print(f"    [{l:>7}] {rows['struct_bb'][i]:+.3f} {rows['mech_bb'][i]:+.3f} | "
              f"{rows['struct_pp'][i]:+.3f} {rows['mech_pp'][i]:+.3f}")

    x = np.arange(len(bands))
    st = rows["struct_bb"]; me = rows["mech_bb"]        # fixed content property of the representation
    fig, axx = plt.subplots(figsize=(8.2, 5.4))
    axx.axhline(0, color="gray", lw=0.8, ls=":")
    axx.axvspan(-0.5, 0.5, color="#f2c94c", alpha=0.16, zorder=0)   # low-rank decoupled region
    axx.plot(x, st, "-o", color="#e08214", lw=2.0, ms=9, mec="black", mew=0.5,
             label="KG structure alignment (topology)")
    axx.plot(x, me, "-s", color="#2ca02c", lw=2.0, ms=9, mec="black", mew=0.5,
             label="DDI mechanism alignment (shared-MBE)")
    axx.set_xticks(x); axx.set_xticklabels(lab)
    axx.set_xlabel("backbone SVD rank band  (low -> high)")
    axx.set_ylabel("alignment  (cMNC)   [<0 = decoupled from that content]")
    axx.set_title("Low ranks are a KG-structure region DECOUPLED from DDI mechanism\n"
                  "(mechanism < 0); mechanism only appears in the higher ranks", fontsize=10)
    axx.annotate("low-rank core: high KG structure,\nbut mechanism DECOUPLED (cMNC<0)",
                 xy=(0, me[0]), xytext=(0.7, me[0] + 0.18), fontsize=8.5, color="#8a1a1a",
                 arrowprops=dict(arrowstyle="->", color="#8a1a1a", lw=0.9))
    axx.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "rank_content.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[content] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

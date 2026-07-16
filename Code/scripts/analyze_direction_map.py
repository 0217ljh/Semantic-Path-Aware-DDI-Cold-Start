"""Analysis figure (NO retrain): the causal direction-map for F1+F3. Two panels, SAME axes; each
point = one frozen-backbone SVD direction v_i.
  x = scorer usage of v_i        = (w.v_i)^2 * sigma_i^2 / sum   (frozen native head w, spectrum)
  y = mechanism content of v_i   = shared-MBE class-mean separation of the 1-D projection p_bb@v_i
Panel A (F1): color = role (top-3 core / shoulder 4-21 / tail-null). The scorer spends usage on
  mechanism-poor core directions and underweights mechanism-rich shoulder directions.
Panel B (F3): size = cold-test AUROC drop when v_i is ablated from the FUSED rep p_bb' (design-R
  head); hollow ring = same ablation from backbone p_bb (native head) -> ~0 (clean reverse).
  Causal dependence localizes to the low-usage / high-mechanism (ignored) directions.

All model-fixed. shared-MBE from a DDI-masked KG, independent of the adapter (no probe, no retrain).

Usage: python Code/scripts/analyze_direction_map.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
from analyze_f1_precheck import rebuild_composer, node_mbe_labels, pair_labels  # noqa: E402
from analyze_f1_function_subspace import head_weight          # noqa: E402
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_f3_ablation import head_auroc                    # noqa: E402


def mech_content_1d(coord, Y, Nz):
    """Mechanism content of a 1-D projection AFTER residualizing topology (degree/#shared) out of
    the projection, so a topology (degree) direction does not look mechanistic via the degree<->MBE
    correlation. = sum_l |mean(res|l=1)-mean(res|l=0)|/std over shared-MBE labels."""
    beta, *_ = np.linalg.lstsq(Nz, coord, rcond=None)
    res = coord - Nz @ beta                                  # degree/#shared regressed out
    sd = res.std() + 1e-9
    s = 0.0
    for l in range(Y.shape[1]):
        pos = Y[:, l] == 1
        if 5 <= pos.sum() <= len(Y) - 5:
            s += abs(res[pos].mean() - res[~pos].mean()) / sd
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu"); ap.add_argument("--fold", default="fold0")
    ap.add_argument("--n-dir", type=int, default=40)         # directions shown (by variance rank)
    ap.add_argument("--r-core", type=int, default=3); ap.add_argument("--r-sh", type=int, default=22)
    ap.add_argument("--n-null", type=int, default=8)         # tail matched-null directions
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

    # backbone spectrum + scorer usage
    Xc = p_bb - p_bb.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False); V = Vt.T
    w = head_weight(comp); w = w if w.ndim == 1 else w.mean(0)
    proj2 = (w @ V) ** 2
    usage = proj2 * S ** 2; usage = usage / usage.sum()

    # mechanism content per direction (shared-MBE, 1-D projection)
    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Yh = Ymbe[has]; Ch = Xc[has] @ V                          # projections of labeled pairs onto all dirs
    Th = topo[has]
    Nz = np.column_stack([np.log1p(Th[:, 0] + Th[:, 1]), np.log1p(Th[:, 2]), np.ones(len(Th))])
    base_dr = head_auroc(comp, p_prime, y, task, "design")
    base_bb = head_auroc(comp, p_bb, y, task, "backbone")
    print(f"[dirmap] base design={base_dr:.4f} backbone={base_bb:.4f}")

    # candidate direction set: top-n_dir by variance + n_null tail directions
    dirs = list(range(min(args.n_dir, len(S)))) + list(range(len(S) - args.n_null, len(S)))
    dirs = sorted(set(dirs))
    rows = []
    for i in dirs:
        mech = mech_content_1d(Ch[:, i], Yh, Nz)
        vi = V[:, i:i + 1]
        pp_ab = p_prime - (p_prime @ vi) @ vi.T
        bb_ab = p_bb - (p_bb @ vi) @ vi.T
        drop_dr = base_dr - head_auroc(comp, pp_ab, y, task, "design")
        drop_bb = base_bb - head_auroc(comp, bb_ab, y, task, "backbone")
        role = "core" if i < args.r_core else ("shoulder" if i < args.r_sh else "tail")
        rows.append(dict(i=i, usage=usage[i], mech=mech, drop_dr=drop_dr, drop_bb=drop_bb, role=role))
    print(f"[dirmap] {len(rows)} directions | mean shoulder drop_dr="
          f"{np.mean([r['drop_dr'] for r in rows if r['role']=='shoulder']):+.4f}")

    ux = np.array([r["usage"] for r in rows]); my = np.array([r["mech"] for r in rows])
    ddr = np.array([r["drop_dr"] for r in rows]); dbb = np.array([r["drop_bb"] for r in rows])
    role = np.array([r["role"] for r in rows])
    RCOL = {"core": "#d62728", "shoulder": "#1f77b4", "tail": "#bbbbbb"}
    xlim = [max(ux.min() * 0.5, 1e-6), 1.0]

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4))
    # Panel A: F1 misalignment (color=role)
    for rl in ["tail", "shoulder", "core"]:
        m = role == rl
        ax[0].scatter(ux[m], my[m], s=55, c=RCOL[rl], alpha=0.85, edgecolors="white",
                      linewidths=0.6, label=f"{rl} " + ("(1-3)" if rl == "core" else
                      "(4-21)" if rl == "shoulder" else "(tail null)"))
    ax[0].set_xscale("log"); ax[0].set_xlim(xlim)
    ax[0].set_xlabel("frozen scorer usage of direction  (log)")
    ax[0].set_ylabel("mechanism content of direction  (shared-MBE)")
    ax[0].set_title("A. Frozen scorer under-reads mechanism (F1)", fontsize=11)
    ax[0].legend(fontsize=8, loc="upper right")
    # Panel B: F3 causal (size=drop_dr, hollow ring=drop_bb)
    sz = 30 + 4000 * np.clip(ddr, 0, None)
    for rl in ["tail", "shoulder", "core"]:
        m = role == rl
        mk = "D" if rl == "tail" else "o"
        ax[1].scatter(ux[m], my[m], s=sz[m], c=RCOL[rl], alpha=0.7, marker=mk,
                      edgecolors="black", linewidths=0.6)
    # hollow rings sized by backbone reverse drop (should be ~0 -> tiny)
    szb = 30 + 4000 * np.clip(dbb, 0, None)
    ax[1].scatter(ux, my, s=szb, facecolors="none", edgecolors="gray", linewidths=1.0, alpha=0.7)
    ax[1].set_xscale("log"); ax[1].set_xlim(xlim)
    ax[1].set_xlabel("frozen scorer usage of direction  (log)")
    ax[1].set_ylabel("mechanism content of direction  (shared-MBE)")
    ax[1].set_title("B. Gain flows through ignored-mechanism dirs (F3)\n"
                    "size = AUROC drop ablated from p_bb' ; ring = from p_bb (~0)", fontsize=10)
    fig.suptitle("The adapter recovers mechanism through directions the frozen scorer ignores",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "scatters" / "direction_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[dirmap] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

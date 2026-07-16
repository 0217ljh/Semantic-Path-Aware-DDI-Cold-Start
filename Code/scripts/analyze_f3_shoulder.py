"""F3-refined (shoulder-only causal ablation), current fixed frozen+last model, NO algo change.
The first F3 was confounded: the mechanism selector picked topology-core directions (backbone uses
them -> dirty reverse). This restricts mechanism-direction selection to the ORTHOGONAL COMPLEMENT
of the backbone's topology core (r_core=3), then asks: does the backbone-IGNORED mechanism shoulder
causally carry the design-R gain?

  V_core = top-r_core right-sing-vecs of frozen p_bb ; P_perp = I - V_core V_core^T
  X'_perp = p_prime P_perp ; SVD -> complement basis ; select mechanism dirs inside it (shared-MBE only)
  ablate from p_prime -> DESIGN drop ; reverse-ablate same dirs from p_bb -> BACKBONE drop
  nulls: complement variance-matched-low-mech + complement random
  decomposition: also ablate the backbone CORE from p_prime (how much design-R reuses the core)

Narrative HOLDS on current model iff (codex): shoulder-mech design drop is material (>=~0.03-0.05),
>> complement nulls, reverse < ~25% of design drop, and core ablation does not dominate.
Else -> the orthogonal-projection retrain is narratively needed.

Usage: python Code/scripts/analyze_f3_shoulder.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "scripts"))
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, pair_labels,  # noqa: E402
                                 cmnc_stratified)
from analyze_f2_reorientation import correction_and_fused     # noqa: E402
from analyze_f3_ablation import head_auroc, mech_scores       # noqa: E402


def ablate_basis(reps, Q):
    """Project the orthonormal ambient basis Q [d,k] out of reps [N,d]."""
    return reps - (reps @ Q) @ Q.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--r-core", type=int, default=3)
    ap.add_argument("--ks-ablate", default="5,10,20")
    ap.add_argument("--n-rand", type=int, default=20)
    ap.add_argument("--cmnc-k", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs); y = np.asarray(data.cold_test_labels)
    fctx = proto.fact_context(data)
    p_bb = comp._encode_backbone(te, fctx).detach().cpu().numpy().astype(np.float64)
    c, p_prime = correction_and_fused(comp, te, p_bb)
    p_prime = p_prime.astype(np.float64)
    base_design = head_auroc(comp, p_prime, y, task, "design")
    base_bb = head_auroc(comp, p_bb, y, task, "backbone")
    gain = base_design - base_bb
    print(f"[f3sh] base design={base_design:.4f} backbone={base_bb:.4f} gain={gain:.4f}")

    # backbone topology core (from frozen p_bb) -> orthogonal complement projector
    Xbc = p_bb - p_bb.mean(0)
    _, _, Vbt = np.linalg.svd(Xbc, full_matrices=False); Vb = Vbt.T
    V_core = Vb[:, :args.r_core]                               # [d, r_core]
    P_perp = np.eye(p_bb.shape[1]) - V_core @ V_core.T

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Y = Ymbe[has]; T = topo[has]
    Xp = p_prime[has]
    Xp_perp = Xp @ P_perp                                      # p_prime inside the core-complement
    Xpp_c = Xp_perp - Xp_perp.mean(0)
    _, Sperp, Vpt = np.linalg.svd(Xpp_c, full_matrices=False); Vperp = Vpt.T
    keep = Sperp > 1e-6 * Sperp[0]                             # drop the r_core null directions
    Vperp = Vperp[:, keep]; Sperp = Sperp[keep]
    ms = mech_scores(Xpp_c, Vperp, Y)                          # mechanism score inside the complement
    order_mech = np.argsort(-ms); low_pool_all = list(np.argsort(ms)[:len(ms) // 2])
    tvar = T[:, 0] + T[:, 1]; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))
    rng = np.random.default_rng(0)

    # decomposition reference: ablate the backbone CORE from p_prime
    d_core = base_design - head_auroc(comp, ablate_basis(p_prime, V_core), y, task, "design")
    print(f"[f3sh] r_core={args.r_core} | design drop from ablating the backbone CORE = {d_core:+.4f} "
          f"({100*d_core/max(gain,1e-9):.0f}% of gain)")

    print(f"\n=== F3-shoulder: mechanism dirs INSIDE core-complement (backbone-ignored) ===")
    print(f"{'k':>4} | {'design_drop':>11} {'varmatch':>9} {'random':>8} | {'bb_reverse':>10} | "
          f"{'rev/des':>7} {'sub_cMNC':>8}")
    for kab in [int(x) for x in args.ks_ablate.split(",")]:
        mech_cols = order_mech[:kab]
        low_pool = list(low_pool_all)
        vm_cols = []
        for j in mech_cols:
            cand = min(low_pool, key=lambda q: abs(Sperp[q] - Sperp[j]))
            vm_cols.append(cand); low_pool.remove(cand)
        Q = Vperp[:, mech_cols]
        d_des = base_design - head_auroc(comp, ablate_basis(p_prime, Q), y, task, "design")
        d_vm = base_design - head_auroc(comp, ablate_basis(p_prime, Vperp[:, vm_cols]), y, task, "design")
        d_rand = np.mean([base_design - head_auroc(
            comp, ablate_basis(p_prime, Vperp[:, rng.choice(Vperp.shape[1], kab, replace=False)]),
            y, task, "design") for _ in range(args.n_rand)])
        d_rev = base_bb - head_auroc(comp, ablate_basis(p_bb, Q), y, task, "backbone")
        sub = cmnc_stratified(Xpp_c @ Q, Y, strata, args.cmnc_k, args.n_perm)[0]
        ratio = d_rev / d_des if abs(d_des) > 1e-6 else float("nan")
        print(f"{kab:>4} | {d_des:>+11.4f} {d_vm:>+9.4f} {d_rand:>+8.4f} | {d_rev:>+10.4f} | "
              f"{ratio:>7.2f} {sub:>8.3f}")

    print(f"\nHOLDS iff: design_drop material (>=~0.03-0.05) AND >> nulls AND rev/des < ~0.25 AND")
    print(f"core-ablation ({d_core:+.4f}) does NOT dominate the shoulder design_drop. Else -> the")
    print(f"orthogonal-projection retrain is narratively needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

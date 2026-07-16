"""F3 (causal ablation), on the CURRENT fixed frozen+last model (NO algorithm change). Is the
cold-start GAIN (backbone 0.68 -> design-R 0.79) causally carried by MECHANISM-aligned directions
of the scorer-visible rep? This is the real claim (codex): F1/F2 are geometric, F3 is causal.

Procedure:
  base   = compute_metrics(comp.head(p_prime), y)             # reproduce the 0.786 design-R AUROC
  select = top-k directions of p_prime by a shared-MBE mechanism score (class-mean separation;
           uses ONLY shared-MBE labels, NOT AUROC -> non-circular direction selection)
  ablate = comp.head(p_prime - P_sel p_prime)  -> AUROC drop
  nulls  = variance-matched-low-mechanism dirs, and random dirs (matched k) -> AUROC drop
  reverse= ablate the SAME mechanism dirs from p_bb, score via the BACKBONE native head -> should
           NOT drop (the backbone under-reads those dirs, per F1)

Attribution holds if: mechanism-dir ablation drop >> matched-null drops, AND the reverse (backbone)
drop is negligible. Then the gain is specifically carried by mechanism directions.

Usage:
  python Code/scripts/analyze_f3_ablation.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "scripts"))
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

import torch                                                  # noqa: E402
from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from model.metrics import compute_metrics                     # noqa: E402
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, pair_labels,  # noqa: E402
                                 cmnc_stratified)
from analyze_f2_reorientation import correction_and_fused     # noqa: E402


@torch.no_grad()
def head_auroc(comp, reps_np, y, task, which="design"):
    """AUROC of a scorer applied to reps [N,d]. which='design' -> comp.head; 'backbone' -> native out."""
    dev = comp.device
    x = torch.as_tensor(reps_np, dtype=torch.float32, device=dev)
    if which == "design":
        logits = comp.head(x)
    else:
        logits = comp.backbone.model.out(x)                   # native R-GCN linear head
    return compute_metrics(logits.detach().cpu().numpy(), y, task)["auroc"]


def mech_scores(Xc, V, Y):
    """Per-direction shared-MBE alignment (class-mean separation), uses labels only (no AUROC).
    score_j = sum_l |mean(coord_j | label l=1) - mean(| l=0)| / (std(coord_j)+eps)."""
    C = Xc @ V                                                # [N, d] coords in SVD basis
    sd = C.std(0) + 1e-9
    s = np.zeros(V.shape[1])
    for l in range(Y.shape[1]):
        pos = Y[:, l] == 1
        if pos.sum() < 5 or (~pos).sum() < 5:
            continue
        s += np.abs(C[pos].mean(0) - C[~pos].mean(0)) / sd
    return s


def ablate(reps, V, cols):
    """Project the columns V[:, cols] out of reps (reps - reps V_cols V_cols^T)."""
    P = V[:, cols]
    return reps - (reps @ P) @ P.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--ks-ablate", default="5,10,20")         # # directions ablated
    ap.add_argument("--n-rand", type=int, default=20)         # random-null repeats
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
    print(f"[f3] base design-R AUROC={base_design:.4f} (expect ~0.786) | "
          f"backbone AUROC={base_bb:.4f} (expect ~0.680)")

    # mechanism-direction selection on the scorer-visible rep p_prime (labels only)
    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Y = Ymbe[has]; T = topo[has]
    Xp = p_prime[has]; Xpc = Xp - Xp.mean(0)
    _, Sp, Vpt = np.linalg.svd(Xpc, full_matrices=False); Vp = Vpt.T
    ms = mech_scores(Xpc, Vp, Y)
    order_mech = np.argsort(-ms)                               # high mech first
    order_lowmech = np.argsort(ms)                            # low mech first (null pool)
    tvar = T[:, 0] + T[:, 1]; strata = np.digitize(tvar, np.quantile(tvar, [.2, .4, .6, .8]))
    rng = np.random.default_rng(0)

    print(f"\n=== F3 CAUSAL ABLATION (drop = base - ablated AUROC; on ALL {len(te)} eval pairs) ===")
    print(f"{'k':>4} | {'mech_drop':>9} {'varmatch_drop':>13} {'random_drop':>11} | "
          f"{'bb_reverse_drop':>15} | {'mech_subspace_cMNC':>18}")
    for kab in [int(x) for x in args.ks_ablate.split(",")]:
        mech_cols = order_mech[:kab]
        # variance-matched-low-mechanism null: nearest singular-value dir from the low-mech pool
        low_pool = list(order_lowmech[:len(ms) // 2])
        vm_cols = []
        for j in mech_cols:
            cand = min(low_pool, key=lambda q: abs(Sp[q] - Sp[j]))
            vm_cols.append(cand); low_pool.remove(cand)
        # ablate on p_prime -> design head
        d_mech = base_design - head_auroc(comp, ablate(p_prime, Vp, list(mech_cols)), y, task, "design")
        d_vm = base_design - head_auroc(comp, ablate(p_prime, Vp, vm_cols), y, task, "design")
        d_rand = np.mean([base_design - head_auroc(
            comp, ablate(p_prime, Vp, list(rng.choice(len(ms), kab, replace=False))), y, task, "design")
            for _ in range(args.n_rand)])
        # reverse: ablate the SAME mech dirs from p_bb -> backbone native head
        d_bb = base_bb - head_auroc(comp, ablate(p_bb, Vp, list(mech_cols)), y, task, "backbone")
        # validate the selected subspace really is mechanism-aligned
        sub_cmnc = cmnc_stratified(Xpc @ Vp[:, mech_cols], Y, strata, args.cmnc_k, args.n_perm)[0]
        print(f"{kab:>4} | {d_mech:>+9.4f} {d_vm:>+13.4f} {d_rand:>+11.4f} | "
              f"{d_bb:>+15.4f} | {sub_cmnc:>18.3f}")

    print("\nF3 reads: mech_drop >> varmatch_drop and random_drop => the cold-start gain is carried")
    print("by mechanism-aligned directions (not just any k dims). bb_reverse_drop ~0 => the backbone")
    print("does NOT use those directions (consistent with F1). mech_subspace_cMNC high => the ablated")
    print("directions really are the mechanism-aligned ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

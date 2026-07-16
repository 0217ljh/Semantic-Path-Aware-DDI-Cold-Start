"""F1 completion (on the CURRENT fixed frozen+last model, NO algorithm change): does the
frozen backbone's SCORING FUNCTION under-read the mechanism directions? Plus an artifact
check that the dominant-variance core is not merely a degree/count nuisance axis.

Rationale (codex, load-bearing): the pre-check showed the backbone's dominant-variance core is
topology-rich / mechanism-poor and mechanism sits in a lower-variance shoulder. A reviewer will
ask "if mechanism is in dirs 4..22, why can't the head use it?". The R-GCN head is LINEAR
(out = Linear(d,1)), so the score s = w . z decomposes EXACTLY over the backbone's own SVD
directions: Var-contribution of direction i = (w . v_i)^2 * sigma_i^2. If that concentrates in
the topology core and is tiny on the mechanism shoulder, the scorer provably under-reads
mechanism -> "spectral under-weighting" is a real limitation, not a non-issue.

This is a MEASUREMENT on the frozen backbone (+ its native head). No training, no algorithm change.

Usage:
  python Code/scripts/analyze_f1_function_subspace.py \
    --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "scripts"))          # reuse the reviewed pre-check helpers
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

import torch                                                  # noqa: E402
from specs import TaskSpec                                    # noqa: E402
from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, pair_labels,  # noqa: E402
                                 cmnc_stratified, MBE_LABELS)


def head_weight(comp) -> np.ndarray:
    """The frozen R-GCN native pre-scorer head weight w [d] (out = Linear(d,1) for binary)."""
    out = comp.backbone.model.out
    W = out.weight.detach().cpu().numpy()                     # [1, d] binary | [K, d] multiclass
    if W.shape[0] != 1:
        # multiclass: use the row-mean-centered energy across classes (per-direction usage)
        return W                                              # [K, d] (handled by caller)
    return W[0]                                               # [d]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--ks", default="15,30")
    ap.add_argument("--core-cut", type=float, default=0.5)    # dominant-core variance cutoff
    ap.add_argument("--shoulder-cut", type=float, default=0.8)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg

    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)
    te = np.asarray(data.cold_test_pairs)
    fctx = proto.fact_context(data)
    p_bb = comp._encode_backbone(te, fctx).detach().cpu().numpy().astype(np.float64)
    print(f"[fsub] {len(te)} cold-test pairs | p_bb {p_bb.shape}")

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    X = p_bb[has]; Y = Ymbe[has]; T = topo[has]

    # SVD of p_bb (backbone's own variance directions)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    r_core = int(np.searchsorted(cumvar, args.core_cut) + 1)
    r_sh = int(np.searchsorted(cumvar, args.shoulder_cut) + 1)
    print(f"[fsub] r_core(@{int(args.core_cut*100)}%)={r_core}  "
          f"r_shoulder(@{int(args.shoulder_cut*100)}%)={r_sh}  d={X.shape[1]}")

    # ---- (1) FUNCTION SUBSPACE: how the frozen native head spends score variance over ranks
    w = head_weight(comp)
    proj = (w @ V) if w.ndim == 1 else (w @ V)               # [d] or [K, d]: head weight in SVD basis
    proj2 = proj ** 2 if w.ndim == 1 else (proj ** 2).mean(0)   # (w.v_i)^2 (mean over classes if multi)
    score_var = proj2 * (S ** 2)                             # (w.v_i)^2 * sigma_i^2 = score-var per dir
    sv_frac = score_var / max(float(score_var.sum()), 1e-12)
    we_frac = proj2 / max(float(proj2.sum()), 1e-12)         # raw head-weight energy per direction
    core = float(sv_frac[:r_core].sum())
    shoulder = float(sv_frac[r_core:r_sh].sum())
    tail = float(sv_frac[r_sh:].sum())
    print("\n=== (1) FUNCTION SUBSPACE: native head score-variance attribution over backbone ranks ===")
    print(f"score-var  in CORE(1..{r_core})={core:.3f}  SHOULDER({r_core+1}..{r_sh})={shoulder:.3f}  "
          f"TAIL({r_sh+1}..)={tail:.3f}")
    print(f"head-weight energy in CORE={float(we_frac[:r_core].sum()):.3f}  "
          f"SHOULDER={float(we_frac[r_core:r_sh].sum()):.3f}  TAIL={float(we_frac[r_sh:].sum()):.3f}")
    print("F1-function reads: if CORE >> SHOULDER, the frozen scorer reads the topology core and")
    print("under-uses the mechanism shoulder -> under-weighting is a real scoring limitation.")

    # ---- (2) ARTIFACT CHECK: is the dominant core just a degree/count nuisance axis?
    print("\n=== (2) ARTIFACT: dominant-core coords vs topology nuisances (Pearson r) ===")
    core_coords = Xc @ V[:, :r_core]                          # [N, r_core]
    nus = {"deg_u": T[:, 0], "deg_v": T[:, 1], "deg_sum": T[:, 0] + T[:, 1], "n_shared": T[:, 2]}
    def _corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    for j in range(r_core):
        cj = core_coords[:, j]
        rs = {name: _corr(cj, val) for name, val in nus.items()}
        print(f"  core dir {j+1}: " + "  ".join(f"{n}={r:+.2f}" for n, r in rs.items()))

    # residualize the top nuisances out of p_bb, re-check core vs shoulder mechanism cMNC
    Nz = np.column_stack([np.log1p(nus["deg_sum"]), np.log1p(nus["n_shared"]), np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(Nz, Xc, rcond=None)
    Xres = Xc - Nz @ beta                                     # p_bb with degree/count regressed out
    _, Sr, Vtr = np.linalg.svd(Xres, full_matrices=False); Vr = Vtr.T
    tvar = T[:, 0] + T[:, 1]
    qs = np.quantile(tvar, [.2, .4, .6, .8]); strata = np.digitize(tvar, qs)
    ks = [min(int(x), len(X) - 1) for x in args.ks.split(",")]
    print("\n=== (2b) mechanism cMNC after residualizing degree/#shared (core vs shoulder) ===")
    print(f"{'k':>4} | {'core_mech':>9} {'shoulder_mech':>13} (raw p_bb) | "
          f"{'core_mech':>9} {'shoulder_mech':>13} (residualized)")
    for k in ks:
        cm = cmnc_stratified(Xc @ V[:, :r_core], Y, strata, k, args.n_perm)[0]
        sm = cmnc_stratified(Xc @ V[:, r_core:r_sh], Y, strata, k, args.n_perm)[0]
        cmr = cmnc_stratified(Xres @ Vr[:, :r_core], Y, strata, k, args.n_perm)[0]
        smr = cmnc_stratified(Xres @ Vr[:, r_core:r_sh], Y, strata, k, args.n_perm)[0]
        print(f"{k:>4} | {cm:>9.3f} {sm:>13.3f} | {cmr:>9.3f} {smr:>13.3f}")
    print("If core stays mechanism-poor and shoulder stays mechanism-rich after residualizing,")
    print("the finding is NOT a pure degree artifact (topology core is real structure).")

    out = Path(args.ckpt).parent / "f1_function_subspace.npz"
    np.savez(out, S=S, cumvar=cumvar, sv_frac=sv_frac, r_core=r_core, r_sh=r_sh)
    print(f"\n[cache] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

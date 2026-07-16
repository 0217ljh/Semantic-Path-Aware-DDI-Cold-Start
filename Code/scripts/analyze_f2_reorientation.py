"""F2 (reorientation) + S2 (orthogonal injection), on the CURRENT fixed frozen+last model
(NO algorithm change). Does the adapter's correction move the scorer-visible pair rep TOWARD
mechanism, and does that correction live in the mechanism shoulder / OUTSIDE the topology core?

Objects (all from the fixed ckpt, frozen backbone):
  M_bb   = p_bb                       (backbone pre-scorer pair rep)
  c      = gamma * W(z_uv)            (pre-LN additive design-R correction, pure M_A.M_B)
  M_plug = p_bb' = LN(p_bb + c)       (scorer-visible corrected rep)

F2: cMNC(M_plug, shared-MBE) > cMNC(M_bb, shared-MBE)  -> the fused rep is more mechanism-aligned.
S2: decompose c over the BACKBONE's own SVD directions (core 1..r_core = topology, shoulder
    r_core+1..r_sh, complement) -> is the correction energy OUTSIDE the topology core, and does
    the complement part carry mechanism? (non-tautological: geometry vs the backbone subspace.)

Measurement only. Topology-stratified null for all cMNC.

Usage:
  python Code/scripts/analyze_f2_reorientation.py --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
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
from analyze_f1_precheck import (rebuild_composer, node_mbe_labels, pair_labels,  # noqa: E402
                                 cmnc_stratified)


@torch.no_grad()
def correction_and_fused(comp, pairs, p_bb, bs=1024):
    """-> (c [N,d], p_prime [N,d]). c = gamma*W(z_uv); p_prime = LN(p_bb + c). Minibatched."""
    comp._train_mode(False)
    dev = comp.device
    c_out = np.zeros_like(p_bb)
    pp_out = np.zeros_like(p_bb)
    for s in range(0, len(pairs), bs):
        sel = np.arange(s, min(s + bs, len(pairs)))
        z_uv, _ = comp._z_uv(pairs[sel])                      # [b, z_dim]
        c = comp.gamma * comp.W(z_uv)                         # [b, d] pre-LN correction
        pbb = torch.as_tensor(p_bb[sel], dtype=torch.float32, device=dev)
        pp = comp.ln(pbb + c)
        c_out[sel] = c.detach().cpu().numpy()
        pp_out[sel] = pp.detach().cpu().numpy()
    return c_out, pp_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--ks", default="15,30")
    ap.add_argument("--core-cut", type=float, default=0.5)
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
    c, p_prime = correction_and_fused(comp, te, p_bb)
    c = c.astype(np.float64); p_prime = p_prime.astype(np.float64)
    print(f"[f2] {len(te)} pairs | p_bb {p_bb.shape} | ||c||/||p_bb|| median="
          f"{np.median(np.linalg.norm(c,axis=1)/(np.linalg.norm(p_bb,axis=1)+1e-9)):.3f}")

    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    Y = Ymbe[has]; T = topo[has]
    Xbb = p_bb[has]; Xpp = p_prime[has]; Xc_corr = c[has]
    tvar = T[:, 0] + T[:, 1]
    qs = np.quantile(tvar, [.2, .4, .6, .8]); strata = np.digitize(tvar, qs)
    ks = [min(int(x), len(Y) - 1) for x in args.ks.split(",")]

    # backbone SVD -> core / shoulder / complement subspaces
    Xbc = Xbb - Xbb.mean(0)
    _, S, Vt = np.linalg.svd(Xbc, full_matrices=False); V = Vt.T
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    r_core = int(np.searchsorted(cumvar, args.core_cut) + 1)
    r_sh = int(np.searchsorted(cumvar, args.shoulder_cut) + 1)

    # ---- F2: does the fused rep align MORE with mechanism than the backbone rep?
    # control M_ln0 = LN(p_bb) (correction zeroed) isolates the LayerNorm effect from the
    # correction's own contribution (codex): correction-specific gain = cMNC(M_plug)-cMNC(M_ln0).
    with torch.no_grad():
        Xln0 = comp.ln(torch.as_tensor(Xbb, dtype=torch.float32, device=comp.device)
                       ).detach().cpu().numpy().astype(np.float64)
    print("\n=== F2 REORIENTATION: cMNC(shared-MBE), topology-stratified null ===")
    print(f"{'k':>4} | {'M_bb':>7} {'M_ln0':>7} {'M_plug':>7} | "
          f"{'plug-bb':>8} {'plug-ln0(corr)':>14}")
    for k in ks:
        bb = cmnc_stratified(Xbb, Y, strata, k, args.n_perm)[0]
        ln0 = cmnc_stratified(Xln0, Y, strata, k, args.n_perm)[0]
        pp = cmnc_stratified(Xpp, Y, strata, k, args.n_perm)[0]
        print(f"{k:>4} | {bb:>7.3f} {ln0:>7.3f} {pp:>7.3f} | "
              f"{pp-bb:>+8.3f} {pp-ln0:>+14.3f}")

    # ---- S2: where does the correction c live vs the backbone subspace, and does it carry mech?
    def energy(M, cols):
        return float((np.linalg.norm(M @ V[:, cols], axis=1) ** 2).sum())
    tot = energy(Xc_corr, list(range(len(S))))
    e_core = energy(Xc_corr, list(range(r_core))) / max(tot, 1e-12)
    e_sh = energy(Xc_corr, list(range(r_core, r_sh))) / max(tot, 1e-12)
    e_comp = energy(Xc_corr, list(range(r_sh, len(S)))) / max(tot, 1e-12)
    print(f"\n=== S2 ORTHOGONAL INJECTION: correction c energy vs backbone subspace ===")
    print(f"c energy in CORE(1..{r_core})={e_core:.3f}  SHOULDER({r_core+1}..{r_sh})={e_sh:.3f}  "
          f"COMPLEMENT({r_sh+1}..)={e_comp:.3f}")
    # mechanism content of the correction, split parallel-to-core vs outside-core
    c_core = Xc_corr @ V[:, :r_core]
    c_out = Xc_corr @ V[:, r_core:]                           # complement of the topology core
    print(f"{'k':>4} | {'c_core_mech':>11} {'c_outside_mech':>14}")
    for k in ks:
        cc = cmnc_stratified(c_core, Y, strata, k, args.n_perm)[0]
        co = cmnc_stratified(c_out, Y, strata, k, args.n_perm)[0]
        print(f"{k:>4} | {cc:>11.3f} {co:>14.3f}")
    print("\nF2 reads: M_plug > M_bb => the fused rep is more mechanism-aligned (reorientation).")
    print("S2 reads: if c energy is OUTSIDE the topology core AND c_outside carries mechanism,")
    print("the correction writes mechanism into directions the backbone core ignores (non-tautological).")

    out = Path(args.ckpt).parent / "f2_reorientation.npz"
    np.savez(out, r_core=r_core, r_sh=r_sh, e_core=e_core, e_sh=e_sh, e_comp=e_comp)
    print(f"\n[cache] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""F1 go/no-go pre-check (design R, pair-level). Fail-fast gate for the STRONG claim.

Question: is the FROZEN backbone's dominant-variance pair subspace mechanism-poor? I.e. does
shared-MBE mechanism structure live OUTSIDE the top principal directions of p_bb (in tail
directions the backbone under-weights)? If the top-variance subspace is ALREADY strongly
mechanism-aligned, the strong claim is dead -> stop before any algorithm change / retrain.

Method (no probe, no extra training; clustering / model-free):
  - p_bb   = frozen R-GCN pre-scorer pair rep on cold-test pairs (from the fixed frozen+last ckpt).
  - S_mech = per-pair shared-MBE profile (multi-label over target/enzyme/transporter/carrier/
             pathway/sideeffect that the pair's SHARED mediators carry). Leak-free (KG has no DDI).
  - SVD(p_bb) -> rank slices by cumulative variance (top r@50%/80%, width-matched tail slice).
  - Primary metric: cMNC (chance-corrected multi-label kNN mechanism-neighborhood consistency)
    with a permutation null taken WITHIN topology strata (so a positive result is not topology).

GO if the top-variance slice is mechanism-poor (cMNC_top(mech) near null AND << tail slice), yet
still carries topology content (cMNC_top(topo) clearly > null). Else NO-GO (kill the strong claim).

Usage:
  python Code/scripts/analyze_f1_precheck.py \
    --ckpt Code/runs/est_pd1__frozen__last/model/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code" / "code-adapter"))

from specs import TaskSpec                                   # noqa: E402
from data.loader import load_rank_data                        # noqa: E402
from model_meta import Protocol                               # noqa: E402
from model.protocol import ColdStartProtocol                  # noqa: E402
from model.runner import _filter_to_known                     # noqa: E402
from adapter.composer import AdapterBackboneComposer          # noqa: E402
from kg.mediators import shared_mediators_idx                  # noqa: E402

# leak-free MBE label sources: drug->mediator RELATION (protein role) + node TYPE (pathway/se).
MBE_RELS = {"db:target": "target", "db:enzyme": "enzyme",
            "db:transporter": "transporter", "db:carrier": "carrier"}
MBE_TYPES = {"SideEffect": "sideeffect", "Pathway": "pathway"}
MBE_LABELS = ["target", "enzyme", "transporter", "carrier", "pathway", "sideeffect"]


def rebuild_composer(ckpt_path: str):
    b = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    task = (TaskSpec.binary() if b.get("task_kind") == "binary"
            else TaskSpec.multiclass(int(b["n_classes"])))
    m = AdapterBackboneComposer(); m.setup(task, b["hp"]); m.load_state_dict(b["state"])
    return m, task, b["hp"]


def node_mbe_labels(kg) -> np.ndarray:
    """[n_nodes, 6] multi-hot. A node is target/enzyme/transporter/carrier if SOME drug connects
    to it via that relation; pathway/sideeffect by node type. Leak-free (no DDI edge in KG)."""
    n = kg.n_nodes
    col = {l: i for i, l in enumerate(MBE_LABELS)}
    Y = np.zeros((n, len(MBE_LABELS)), dtype=np.int8)
    src = np.repeat(np.arange(n), np.diff(kg.d_indptr))       # source of each directed edge
    is_drug = kg.drug_mask[src]
    found = []
    for raw, lab in MBE_RELS.items():
        if raw not in kg.rel_names:                           # fail-loud: this is a claim gate
            print(f"[precheck] WARN MBE relation absent from KG: {raw}")
            continue
        rid = kg.rel_names.index(raw)
        sel = (kg.d_rel == rid) & is_drug                     # drug --raw--> dst
        Y[kg.d_dst[sel], col[lab]] = 1
        found.append(raw)
    for tn, lab in MBE_TYPES.items():
        if tn not in kg.type_names:
            print(f"[precheck] WARN MBE node type absent from KG: {tn}")
            continue
        t = kg.type_names.index(tn)
        Y[kg.type_id == t, col[lab]] = 1
        found.append(tn)
    if not found:
        raise SystemExit("[precheck] FATAL: no MBE relation/type present in KG — cannot label "
                         "mechanism; verdict would be meaningless")
    print(f"[precheck] MBE sources found: {found}")
    return Y


def pair_labels(comp, pairs: np.ndarray, node_mbe: np.ndarray):
    """Per-pair shared-MBE profile [N,6] + topology [N,3]=(deg_u,deg_v,#shared). Uses UNCAPPED
    kg.mediators.shared_mediators_idx (the pair's TRUE shared-mediator set) — NOT _build_batch,
    which caps at max_med and would bias the biological label (codex)."""
    kg = comp.adapter_model.kg
    nbhd = comp.adapter_model.nbhd
    hp = comp.adapter_model.hp
    mode = str(hp.get("med_mode", "and")); tau = int(hp.get("tau", 2))
    deg = np.diff(kg.u_indptr).astype(np.float64)
    L = node_mbe.shape[1]
    Ymbe = np.zeros((len(pairs), L), dtype=np.int8)
    topo = np.zeros((len(pairs), 3), dtype=np.float64)
    for i, (u, v) in enumerate(pairs):
        ui, vi = kg.get_idx(str(u)), kg.get_idx(str(v))
        if ui is None or vi is None:
            topo[i] = [np.nan, np.nan, 0]
            continue
        med = shared_mediators_idx(kg, nbhd, int(ui), int(vi), mode=mode, tau=tau)["med_idx"]
        if len(med):
            Ymbe[i] = node_mbe[med].max(0)
        topo[i] = [deg[ui], deg[vi], len(med)]
    return Ymbe, topo


def _knn(Q: np.ndarray, k: int) -> np.ndarray:
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    sim = Qn @ Qn.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]


def _overlap(Y: np.ndarray, nbr: np.ndarray) -> float:
    hit = (Y[:, None, :] * Y[nbr]).sum(-1) > 0                 # neighbor shares >=1 label
    return float(hit.mean())


def cmnc_stratified(Q, Y, strata, k, n_perm=200, seed=0):
    """cMNC with permutation null WITHIN strata (stratify by topology -> a positive result is
    mechanism structure NOT explained by topology; pass a single dummy stratum for a GLOBAL null).
    -> (cmnc, obs, null_mean, null_p95)."""
    nbr = _knn(Q, k)
    obs = _overlap(Y, nbr)
    rng = np.random.default_rng(seed)
    uniq = np.unique(strata)
    vals = np.empty(n_perm)
    for p in range(n_perm):
        Yp = Y.copy()
        for s in uniq:
            idx = np.where(strata == s)[0]
            if len(idx) > 1:
                Yp[idx] = Y[idx][rng.permutation(len(idx))]
        vals[p] = _overlap(Yp, nbr)
    p0 = float(vals.mean())
    p95 = float(np.quantile(vals, 0.95))
    return (obs - p0) / max(1e-9, 1.0 - p0), obs, p0, p95


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Code/runs/est_pd1__frozen__last/model/best.pt")
    ap.add_argument("--dataset", default="drugbank_ryu")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--ks", default="15,30")
    ap.add_argument("--var-cuts", default="0.5,0.8")
    ap.add_argument("--n-strata", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    comp, task, hp = rebuild_composer(args.ckpt)
    kg = comp.adapter_model.kg

    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, comp.known_drugs(), print)
    proto = ColdStartProtocol(Protocol.P1_FIXED, task)        # composer/backbone protocol
    te = np.asarray(data.cold_test_pairs)
    fctx = proto.fact_context(data)
    print(f"[precheck] {len(te)} cold-test pairs (known-drug, P1_FIXED)")

    # frozen backbone pair reps
    p_bb = comp._encode_backbone(te, fctx).detach().cpu().numpy().astype(np.float64)
    print(f"[precheck] p_bb {p_bb.shape}")

    # labels
    node_mbe = node_mbe_labels(kg)
    Ymbe, topo = pair_labels(comp, te, node_mbe)
    has = (Ymbe.sum(1) > 0) & np.isfinite(topo).all(1)
    print(f"[precheck] MBE-labeled pairs: {int(has.sum())}/{len(te)} "
          f"({100*has.mean():.1f}%) | per-label: {dict(zip(MBE_LABELS, Ymbe.sum(0).tolist()))}")
    X = p_bb[has]; Y = Ymbe[has]; T = topo[has]

    # topology strata (for the null) + topology pseudo-labels (for cMNC-vs-topo)
    tvar = T[:, 0] + T[:, 1]                                    # deg_u + deg_v
    qs = np.quantile(tvar, np.linspace(0, 1, args.n_strata + 1)[1:-1])
    strata = np.digitize(tvar, qs)
    tq = np.quantile(tvar, [.25, .5, .75])
    Ytopo = np.eye(4, dtype=np.int8)[np.digitize(tvar, tq)]

    # SVD -> variance rank slices
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T                                                    # [d, r] right singular vectors
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    cuts = [float(c) for c in args.var_cuts.split(",")]
    print(f"[precheck] d={X.shape[1]} rank={len(S)} | "
          + " ".join(f"r@{int(c*100)}%={int(np.searchsorted(cumvar, c)+1)}" for c in cuts))

    ks = [min(int(x), len(X) - 1) for x in args.ks.split(",")]  # guard k < N (codex)
    sizes = np.bincount(strata)                                # singleton-strata warning
    n_sing = int((sizes[strata] == 1).sum())
    if n_sing > 0.2 * len(strata):
        print(f"[precheck] WARN {n_sing}/{len(strata)} pairs in singleton strata (weak null)")
    strata_global = np.zeros(len(strata), dtype=np.int64)      # GLOBAL null for topology labels

    print("\n=== F1 rank-resolved cMNC (mech: topology-stratified null; topo: global null) ===")
    print(f"{'cut':>5} {'r':>4} {'k':>4} | {'TOP_mech':>8} {'TOP_topo':>9} | {'TAIL_mech':>9}")
    rows = []
    for c in cuts:
        r = int(np.searchsorted(cumvar, c) + 1)
        top = list(range(r)); tail = list(range(len(S) - r, len(S)))   # width-matched tail
        Xtop = Xc @ V[:, top]; Xtail = Xc @ V[:, tail]
        for k in ks:
            tm, tm_obs, _, tm_p95 = cmnc_stratified(Xtop, Y, strata, k, args.n_perm)
            tt, tt_obs, _, tt_p95 = cmnc_stratified(Xtop, Ytopo, strata_global, k, args.n_perm)
            lm, lm_obs, _, lm_p95 = cmnc_stratified(Xtail, Y, strata, k, args.n_perm)
            rows.append(dict(c=c, r=r, k=k, tm=tm, tt=tt, tt_obs=tt_obs, tt_p95=tt_p95,
                             lm=lm, lm_obs=lm_obs, lm_p95=lm_p95))
            print(f"{c:>5.2f} {r:>4} {k:>4} | {tm:>8.3f} {tt:>7.3f}"
                  f"({'sig' if tt_obs > tt_p95 else '  n'}) | {lm:>7.3f}"
                  f"({'sig' if lm_obs > lm_p95 else '  n'})")

    # preregistered verdict: GO only if EVERY primary slice is mechanism-poor at the top
    # (tm < 0.9 * best_tail_mech) AND the top slice significantly carries topology (obs>p95).
    best_tail_row = max(rows, key=lambda rr: rr["lm"])
    best_tail = best_tail_row["lm"]
    tail_has_mech = best_tail_row["lm_obs"] > best_tail_row["lm_p95"]
    all_go = all((row["tm"] < 0.9 * best_tail) and (row["tt_obs"] > row["tt_p95"]) for row in rows)

    print("\nF1 reads: TOP_mech should be << TAIL_mech (mechanism in the low-variance tail the")
    print("backbone under-weights) and TOP_topo significant (top subspace IS topology).")
    print(f"[precheck] best_tail_mech={best_tail:.3f} (tail carries mechanism: "
          f"{'YES' if tail_has_mech else 'NO'})")
    if not tail_has_mech:
        verdict = ("AMBIGUOUS: backbone looks mechanism-blind EVERYWHERE (tail has no mechanism "
                   "signal). Strong claim shifts to 'adapter ADDS absent mechanism' not 'reorients "
                   "underweighted mechanism' -> decide framing before retrain.")
    elif all_go:
        verdict = "GO (F1 alive: dominant subspace mechanism-poor; mechanism lives in the tail)"
    else:
        verdict = ("NO-GO (F1 dead: a dominant slice is already mechanism-aligned >=0.9*best_tail) "
                   "-> STOP the strong claim / do not retrain")
    print(f"\n>>> PRE-CHECK VERDICT: {verdict}")

    out = Path(args.ckpt).parent / "f1_precheck.npz"
    np.savez(out, p_bb=p_bb, Ymbe=Ymbe, topo=topo, has=has, S=S, cumvar=cumvar)
    print(f"[cache] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

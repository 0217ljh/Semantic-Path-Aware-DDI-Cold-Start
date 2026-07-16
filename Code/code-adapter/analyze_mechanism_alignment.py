"""Analysis phase (frozen framework) - Stage-A mediator-level cMNC: does the model's
low-rank subspace align with mechanism (S_mech) more than topology (S_topo)?
F1 (M_bb misorientation) + F2 (M_plug reorientation). Primary metric = chance-corrected
multi-label kNN mechanism neighborhood consistency (cMNC), MODEL-FREE.

Extracts M_bb=h_base (R-GCN reps) and M_plug=n_m (adapted) per mediator from a frozen
adapter+R-GCN run, caches them, then computes cMNC vs a mechanism label set and a topology
nuisance set. THIS FIRST VERSION uses NODE-TYPE labels = COARSE + CIRCULAR (adapter is typed
by node type) -> machinery validation only, NOT a paper claim. Non-circular ontologies
(drug-side MBE relation, meta-path, DrugBank) come in the sweep.

Usage: python Code/code-adapter/analyze_mechanism_alignment.py --adapter-ckpt <M_plug best.pt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(_ADAPTER))

from adapter.rank_model import AdapterRankModel          # noqa: E402
from data.loader import load_rank_data                   # noqa: E402
from model.protocol import ColdStartProtocol             # noqa: E402
from model.runner import _filter_to_known                # noqa: E402
from model_meta import Protocol                           # noqa: E402
from specs import TaskSpec                                # noqa: E402


def rebuild(ckpt_path: str):
    b = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    task = (TaskSpec.binary() if b.get("task_kind") == "binary"
            else TaskSpec.multiclass(int(b["n_classes"])))
    m = AdapterRankModel(); m.setup(task, b["hp"]); m.load_state_dict(b["state"])
    return m, task, b["hp"]


@torch.no_grad()
def extract_stage_a(model, pairs, context):
    """-> (med_ids [Nm], Z_bb [Nm,d], Z_plug [Nm,d]) mediator-mean reps over occurrences."""
    model.adapter.eval(); model.hbase_provider.eval()
    model.hbase_provider.encode(context)
    sums_bb, sums_pl, cnt = {}, {}, {}
    rng = np.random.default_rng(0)
    for sel in model._minibatches(len(pairs), rng, shuffle=False):
        batch = model._build_batch(pairs[sel])
        if batch.n_mediators == 0:
            continue
        st = model.adapter.mediator_states(batch)
        mid = batch.med_idx.cpu().numpy()
        hb = st["h_base"].cpu().numpy(); nm = st["n_m"].cpu().numpy()
        for i, m in enumerate(mid):
            m = int(m)
            if m not in sums_bb:
                sums_bb[m] = hb[i].copy(); sums_pl[m] = nm[i].copy(); cnt[m] = 1
            else:
                sums_bb[m] += hb[i]; sums_pl[m] += nm[i]; cnt[m] += 1
    ids = np.array(sorted(sums_bb), dtype=np.int64)
    Z_bb = np.stack([sums_bb[m] / cnt[m] for m in ids]).astype(np.float32)
    Z_pl = np.stack([sums_pl[m] / cnt[m] for m in ids]).astype(np.float32)
    return ids, Z_bb, Z_pl


def node_type_labels(kg, med_ids):
    """PLACEHOLDER S_mech: node-type one-hot (COARSE + CIRCULAR). [Nm, T] multi-hot (single here)."""
    T = len(kg.type_names)
    Y = np.zeros((len(med_ids), T), dtype=np.int8)
    Y[np.arange(len(med_ids)), kg.type_id[med_ids].astype(np.int64)] = 1
    return Y


def mbe_relation_labels(kg, med_ids):
    """NON-CIRCULAR S_mech (option c, fine MBE): a mediator's drug-side mechanistic relations
    (db:target/enzyme/transporter/carrier, finer than node type) + SideEffect/Pathway type.
    Multi-label: a protein can be target for some drugs AND enzyme for others. -> (Y [Nm,L], labels)."""
    labels = ["target", "enzyme", "transporter", "carrier", "sideeffect", "pathway"]
    col = {l: i for i, l in enumerate(labels)}
    rel_lab = {"db:target": "target", "db:enzyme": "enzyme",
               "db:transporter": "transporter", "db:carrier": "carrier"}
    n = kg.n_nodes
    src = np.repeat(np.arange(n), np.diff(kg.d_indptr))     # source of each directed edge
    is_drug = kg.drug_mask[src]
    med_bool = np.zeros(n, bool); med_bool[med_ids] = True
    row_of = np.full(n, -1, np.int64); row_of[med_ids] = np.arange(len(med_ids))
    Y = np.zeros((len(med_ids), len(labels)), dtype=np.int8)
    for raw, lab in rel_lab.items():
        if raw not in kg.rel_names:
            continue
        rid = kg.rel_names.index(raw)
        sel = (kg.d_rel == rid) & is_drug & med_bool[kg.d_dst]   # drug --raw--> mediator
        Y[row_of[kg.d_dst[sel]], col[lab]] = 1
    for tn, lab in (("SideEffect", "sideeffect"), ("Pathway", "pathway")):
        if tn in kg.type_names:
            t = kg.type_names.index(tn)
            Y[kg.type_id[med_ids].astype(np.int64) == t, col[lab]] = 1
    return Y, labels


def topo_features(kg, med_ids):
    """S_topo nuisance: [log_degree] (first pass). Continuous."""
    deg = np.diff(kg.u_indptr)[med_ids].astype(np.float64)
    X = np.log1p(deg)[:, None].astype(np.float32)
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _knn(Q, k):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    sim = Qn @ Qn.T
    np.fill_diagonal(sim, -np.inf)
    return np.argpartition(-sim, k, axis=1)[:, :k]         # [N, k] neighbor idx


def cmnc(Q, Y, k, n_perm=200, seed=0):
    """Chance-corrected multi-label kNN mechanism neighborhood consistency."""
    nbr = _knn(Q, k)
    def overlap_rate(Ylab):
        hit = (Ylab[:, None, :] * Ylab[nbr]).sum(-1) > 0   # [N,k] share>=1 label
        return hit.mean()
    obs = overlap_rate(Y)
    rng = np.random.default_rng(seed)
    perm = np.array([overlap_rate(Y[rng.permutation(len(Y))]) for _ in range(n_perm)])
    p0 = perm.mean()
    return float((obs - p0) / max(1e-9, 1 - p0)), float(obs), float(p0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-ckpt", required=True)
    ap.add_argument("--dataset", default="drugbank_ryu")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--ks", default="10,20,30")
    args = ap.parse_args()

    model, task, hp = rebuild(args.adapter_ckpt)
    data = load_rank_data(args.dataset, task, args.fold)
    data = _filter_to_known(data, model.known_drugs(), print)  # drop pairs/facts outside KG
    te = data.cold_test_pairs
    proto = ColdStartProtocol(Protocol.P2_EMERGING, task)
    fctx = proto.fact_context(data)
    proto.assert_pairs_absent(te, fctx)
    print(f"[extract] {len(te)} cold-test pairs (known-drug)")
    ids, Z_bb, Z_pl = extract_stage_a(model, te, fctx)
    print(f"[extract] {len(ids)} unique mediators | d={Z_bb.shape[1]}")

    X_topo = topo_features(model.kg, ids)
    q = np.quantile(X_topo[:, 0], [.25, .5, .75])          # degree-quantile nuisance pseudo-labels
    Y_topo = np.eye(4, dtype=np.int8)[np.digitize(X_topo[:, 0], q)]

    # NON-CIRCULAR primary: drug-side MBE relations, on the mechanism-labeled subset
    Y_mbe, mbe_labels = mbe_relation_labels(model.kg, ids)
    has = Y_mbe.sum(1) > 0
    print(f"\n[S_mech] MBE-labeled mediators: {int(has.sum())}/{len(ids)} ({100*has.mean():.1f}%) "
          f"| per-label counts: {dict(zip(mbe_labels, Y_mbe.sum(0).tolist()))}")
    Zbb_s, Zpl_s, Ymbe_s, Ytopo_s = Z_bb[has], Z_pl[has], Y_mbe[has], Y_topo[has]

    print("\n=== Stage-A cMNC : NON-CIRCULAR drug-side MBE labels (mechanism-labeled subset) ===")
    print(f"{'k':>4} | {'bb_MBE':>7} {'bb_topo':>7} | {'plug_MBE':>8} {'plug_topo':>9} | {'F2(plug-bb)':>11}")
    for k in [int(x) for x in args.ks.split(",")]:
        bm = cmnc(Zbb_s, Ymbe_s, k)[0]; bt = cmnc(Zbb_s, Ytopo_s, k)[0]
        pm = cmnc(Zpl_s, Ymbe_s, k)[0]; pt = cmnc(Zpl_s, Ytopo_s, k)[0]
        print(f"{k:>4} | {bm:>7.3f} {bt:>7.3f} | {pm:>8.3f} {pt:>9.3f} | {pm-bm:>+11.3f}")
    print("\nF1: bb_MBE vs bb_topo (misorientation if MBE alignment weak / <= topo).")
    print("F2: plug_MBE - bb_MBE > 0  => adapter REORIENTS toward mechanism.")

    # WITHIN-TYPE non-circular test: protein mediators, labeled by drug-side mechanism relation
    # (target/enzyme/transporter/carrier). Same node type -> tests within-type mechanism (NOT
    # node type = NOT circular). sideeffect/pathway excluded (they ARE node types = circular).
    prot_col = [mbe_labels.index(l) for l in ("target", "enzyme", "transporter", "carrier")]
    Yp = Y_mbe[:, prot_col]
    pm_has = Yp.sum(1) > 0
    print(f"\n[within-protein] protein-mechanism-labeled mediators: {int(pm_has.sum())} "
          f"| counts: {dict(zip(('target','enzyme','transporter','carrier'), Yp.sum(0).tolist()))}")
    if int(pm_has.sum()) >= 50:
        Zbb_p, Zpl_p, Yp_p = Z_bb[pm_has], Z_pl[pm_has], Yp[pm_has]
        # within-protein topo nuisance (degree quantiles among these proteins)
        dp = np.log1p(np.diff(model.kg.u_indptr)[ids[pm_has]].astype(np.float64))
        qp = np.quantile(dp, [.33, .66]); Ytp = np.eye(3, dtype=np.int8)[np.digitize(dp, qp)]
        print("\n=== Stage-A cMNC : WITHIN-PROTEIN mechanism (NON-CIRCULAR, node type constant) ===")
        print(f"{'k':>4} | {'bb_mech':>7} {'bb_topo':>7} | {'plug_mech':>9} {'plug_topo':>9} | {'F2':>7}")
        for k in [int(x) for x in args.ks.split(",")]:
            k = min(k, int(pm_has.sum()) - 1)
            bm = cmnc(Zbb_p, Yp_p, k)[0]; bt = cmnc(Zbb_p, Ytp, k)[0]
            pm = cmnc(Zpl_p, Yp_p, k)[0]; pt = cmnc(Zpl_p, Ytp, k)[0]
            print(f"{k:>4} | {bm:>7.3f} {bt:>7.3f} | {pm:>9.3f} {pt:>9.3f} | {pm-bm:>+7.3f}")
        print("This is the REAL non-circular F1/F2: does the subspace separate target vs enzyme")
        print("proteins (same node type)? F1 misorientation if bb_mech low; F2 if plug_mech > bb_mech.")
    # cache reps for the sweep
    out = Path(args.adapter_ckpt).parent / "analysis_cache"
    out.mkdir(exist_ok=True)
    np.savez(out / "stageA_reps.npz", med_ids=ids, Z_bb=Z_bb, Z_plug=Z_pl)
    print(f"\n[cache] Stage-A reps -> {out/'stageA_reps.npz'}")


if __name__ == "__main__":
    raise SystemExit(main())

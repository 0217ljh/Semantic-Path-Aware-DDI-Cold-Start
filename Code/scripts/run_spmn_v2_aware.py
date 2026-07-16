"""Aware adapter Step 1 — AND substrate + zero-init promiscuity/regime branch.

Reuses the EXACT cached AND supports built by ``run_spmn_v2_standalone.py`` (so the
mediator substrate is byte-identical to the naked-AND Step 0 reference) and adds the
decomposed scorer ``s = scorer(core) + b(v_ab)``, where ``v_ab`` is the 6 per-pair
density features (``spmn_v2.density``). The branch is zero-init, so ``--aware-step 0``
reproduces naked AND and ``--aware-step 1`` adds the promiscuity branch — one
isolatable ablation (codex Step 0 -> Step 1). Standalone mode (vs EmerGNN).

This is a new entry point: it imports the frames loader + batch gather from
``run_spmn_v2_standalone`` (no logic duplication) and does not modify any existing
file. Persists ``val_curve`` + ``best_epoch`` + per-pair test predictions per the
project logging convention.

Prereq: the AND support cache must already exist (run the Step 0 standalone once);
this script loads it read-only and errors with a clear message if absent.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

# Sibling-script reuse (frames loader, batch gather, shared paths) — importing the
# module runs its top-level sys.path setup but not main() (guarded by __main__).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_spmn_v2_standalone import (  # noqa: E402
    CACHE_DIR, ROOT, _gather, _load_frames,
)

sys.path.insert(0, str(ROOT / "Code"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from my_code.models.spmn_v1.retrieval import MergedKG, N_REL_BUCKETS, N_TYPES  # noqa: E402
from my_code.models.spmn_v1.struct_features import symmetric_binary_dim  # noqa: E402
from my_code.models.spmn_v2 import density  # noqa: E402
from my_code.models.spmn_v2 import SupportBatch  # noqa: E402
from my_code.models.spmn_v2.aware_core import AwareStructuralCore  # noqa: E402
from my_code.models.spmn_v2.aware_core_gated import GatedAwareStructuralCore  # noqa: E402
from my_code.models.spmn_v2.aware_bipartite import (  # noqa: E402
    BipartiteAwareCore, LabelAdversary)
from my_code.models.spmn_v2.aware_heads import DecomposedStandaloneHead  # noqa: E402
from my_code.models.spmn_v2.core import StructuralVariableCore  # noqa: E402
from my_code.utils.train_progress import TrainProgress  # noqa: E402

KEYS = ("struct", "y", "med", "typ", "rela", "relb", "da", "db", "offsets")


def _load_support_cache(support_mode, seed, l_max, k_per_type, n_max, with_copath):
    cache = CACHE_DIR / (f"spmn_v2_supports_{support_mode}_seed{seed}"
                         f"_lmax{l_max}_kpt{k_per_type}_nmax{n_max}"
                         f"_cp{int(with_copath)}.npz")
    if not cache.is_file():
        raise FileNotFoundError(
            f"AND support cache not found: {cache}\n  build it once via "
            f"run_spmn_v2_standalone.py with the SAME (support_mode, seed, l_max, "
            f"k_per_type, n_max) so the aware run uses an identical substrate.")
    print(f"[aware] support cache HIT: {cache}", flush=True)
    z = np.load(cache)
    names = sorted({k.split("__")[0] for k in z.files})
    return {name: {k: z[f"{name}__{k}"] for k in KEYS} for name in names}


def _build_pair_feats(frames, l_max, kg):
    """Per-split (n, 6) v_ab, standardised with TRAIN stats only."""
    drug_ids = []
    for fr in frames.values():
        drug_ids += fr["drug_a_id"].astype(str).tolist()
        drug_ids += fr["drug_b_id"].astype(str).tolist()
    print(f"[aware] computing per-drug reach stats (depth={l_max - 1}) for "
          f"{len(set(drug_ids))} drugs ...", flush=True)
    t0 = time.time()
    stats = density.compute_drug_reach_stats(kg, drug_ids, depth=l_max - 1)
    raw = {name: density.build_pair_feature_matrix(
        fr["drug_a_id"].astype(str).to_numpy(),
        fr["drug_b_id"].astype(str).to_numpy(), stats)
        for name, fr in frames.items()}
    mu = raw["train"].mean(0, keepdims=True)
    sd = raw["train"].std(0, keepdims=True) + 1e-6
    feats = {name: ((v - mu) / sd).astype(np.float32) for name, v in raw.items()}
    print(f"[aware] v_ab built ({time.time() - t0:.0f}s)", flush=True)
    return feats, stats


def _build_pair_regime(frames, data, kg, stats, d_hub=1000):
    """Per-split (n, N_Q_FEATS) q_ab regime features (Step 3), standardised with
    TRAIN stats only. Per-drug reach sizes come from ``stats``; per-pair shared-
    support size and hub count come from the cached support (offsets + med + KG
    degree). Aligned to frames row order (same as the support cache)."""
    deg = kg.degree
    raw = {}
    for name, fr in frames.items():
        off = data[name]["offsets"]; med = data[name]["med"]
        # belt-and-suspenders on the frames<->cache row alignment (the main-loop
        # label-equality guard already runs before this and ties the orders; this
        # also catches a pair-count mismatch). q_ab mixes frames-order (reach) with
        # cache-order (support counts), so they MUST be the same pair order.
        if len(fr) != len(off) - 1:
            raise SystemExit(
                f"[aware] q_ab: frames/{name} len {len(fr)} != cache pairs "
                f"{len(off) - 1}; rebuild the AND cache before running step 3.")
        a = fr["drug_a_id"].astype(str).to_numpy()
        b = fr["drug_b_id"].astype(str).to_numpy()
        reach_a = np.array([stats.get(x, (0, None))[0] for x in a], dtype=np.float64)
        reach_b = np.array([stats.get(x, (0, None))[0] for x in b], dtype=np.float64)
        hub = (deg[med] > d_hub).astype(np.int64)
        cum = np.concatenate([[0], np.cumsum(hub)])
        n_hub = (cum[off[1:]] - cum[off[:-1]]).astype(np.float64)
        n_shared = np.diff(off).astype(np.float64)
        raw[name] = density.build_pair_regime_matrix(reach_a, reach_b, n_shared, n_hub)
    mu = raw["train"].mean(0, keepdims=True)
    sd = raw["train"].std(0, keepdims=True) + 1e-6
    return {name: ((v - mu) / sd).astype(np.float32) for name, v in raw.items()}


def _prune_pool_redundant(split, frac, n_rel_buckets):
    """UPSTREAM redundancy prune (codex idea-2 round): drop the most signature-redundant
    pooled mediators per pair BEFORE pooling, deterministically (same at train/val/test ->
    no leakage). Tests whether redundant support hurts by ENTERING the pool at all (which
    pooling-time reweighting could not fix). Struct count features are left UNCHANGED, so
    this isolates the pool effect. Redundancy = within-pair (type, rel_a, rel_b) signature
    duplication (r_sig); we drop the top-`frac` most-duplicated mediators per pair while
    keeping >=1 per type. Returns a NEW split dict with rebuilt mediator arrays + offsets;
    `struct`/`y` are shared (unchanged)."""
    off = split["offsets"]; med = split["med"]; typ = split["typ"]
    rela = split["rela"]; relb = split["relb"]; da = split["da"]; db = split["db"]
    n = len(off) - 1; nrb = int(n_rel_buckets)
    keep_mask = np.ones(len(med), dtype=bool)
    for p in range(n):
        s, e = int(off[p]), int(off[p + 1])
        if e - s <= 2:
            continue
        t = typ[s:e].astype(np.int64)
        sig = (t * nrb + rela[s:e].astype(np.int64)) * nrb + relb[s:e].astype(np.int64)
        uniq, inv, cnt = np.unique(sig, return_inverse=True, return_counts=True)
        dup = cnt[inv].astype(np.float64)                 # signature multiplicity per mediator
        n_p = e - s
        n_drop = int(np.floor(frac * n_p))
        if n_drop <= 0:
            continue
        # drop the most-duplicated first (desc dup, stable by original order)
        order = np.lexsort((np.arange(n_p), -dup))         # most redundant first
        drop_local = []
        # per-type remaining counter to keep >=1 per type
        from collections import Counter
        remain = Counter(t.tolist())
        for li in order:
            if len(drop_local) >= n_drop:
                break
            ty = int(t[li])
            if remain[ty] <= 1:                            # never drop last of a type
                continue
            if dup[li] <= 1.0:                             # only drop genuinely duplicated
                continue
            drop_local.append(li); remain[ty] -= 1
        if drop_local:
            keep_mask[s + np.array(drop_local, dtype=np.int64)] = False
    # rebuild flat arrays + offsets
    new_counts = np.zeros(n, dtype=np.int64)
    seg_id = np.repeat(np.arange(n), np.diff(off))
    np.add.at(new_counts, seg_id[keep_mask], 1)
    new_off = np.zeros(n + 1, dtype=np.int64); np.cumsum(new_counts, out=new_off[1:])
    out = dict(split)
    out["med"] = med[keep_mask]; out["typ"] = typ[keep_mask]
    out["rela"] = rela[keep_mask]; out["relb"] = relb[keep_mask]
    out["da"] = da[keep_mask]; out["db"] = db[keep_mask]; out["offsets"] = new_off
    print(f"    [prune] kept {int(keep_mask.sum())}/{len(med)} mediators "
          f"({100*keep_mask.mean():.1f}%) frac={frac}", flush=True)
    return out


def _degrade_struct(raw, r, mu, sd):
    """Sparser-support VIEW (codex OOD augmentation): scale the uncapped count block
    of raw struct by per-pair keep-ratio r, recompute log1p, then standardize with
    TRAIN mu/sd. Exposes the scorer to the sparse test-count regime at train time."""
    nt = N_TYPES
    d = raw.copy()
    rr = r[:, None]
    d[:, 0:nt] = raw[:, 0:nt] * rr                       # s_tau
    d[:, nt:2 * nt] = np.log1p(raw[:, 0:nt] * rr)        # log1p(s_tau)
    d[:, 2 * nt:3 * nt] = raw[:, 2 * nt:3 * nt] * rr     # AA
    d[:, -2] = raw[:, -2] * r                            # n_support
    return ((d - mu) / sd).astype(np.float32)


def _degrade_full(split, idx, r_local, deg, mu, sd, rng, struct_dim, device,
                  topk=False, select="topk"):
    """Full sparser-support VIEW: subsample each pair's pooling mediators down to a
    keep-ratio r (>=1 kept) AND recompute the struct count block from the retained
    subset — degraded pool + counts = one sparse-support world. Standardize w/ TRAIN
    mu/sd. topk=True keeps per-(pair,type) top-k by corridor score -(d_a+d_b)-log1p(deg)
    (deterministic, preserves strongest evidence, lower variance); else Bernoulli."""
    off = split["offsets"]; med = split["med"]; typ = split["typ"]
    rela = split["rela"]; relb = split["relb"]; da = split["da"]; db = split["db"]
    nt = N_TYPES; n = len(idx)
    mL, tL, raL, rbL, daL, dbL, pL = [], [], [], [], [], [], []
    cnt = np.zeros((n, nt)); aa = np.zeros((n, nt)); nsup = np.zeros(n)
    for local, p in enumerate(idx):
        s, e = off[p], off[p + 1]
        if e <= s:
            continue
        n_seg = e - s
        if topk:
            seg_typ = typ[s:e].astype(np.int64)
            # corridor score (used by select='topk'/'bottomk'/'degree'/'dist')
            md = med[s:e]; dd = (da[s:e] + db[s:e]).astype(np.float64)
            lgd = np.log1p(deg[md].astype(np.float64))
            if select == "degree":
                score = -lgd
            elif select == "dist":
                score = -dd
            else:
                score = -dd - lgd                       # full corridor score
            k = np.zeros(n_seg, dtype=bool)
            for t in np.unique(seg_typ):
                pos = np.where(seg_typ == t)[0]
                kt = max(1, int(round(r_local[local] * len(pos))))
                if len(pos) <= kt:
                    k[pos] = True
                elif select == "matchrand":             # exact-k random (control)
                    k[pos[rng.choice(len(pos), kt, replace=False)]] = True
                elif select == "bottomk":               # keep WORST (anti-control)
                    k[pos[np.argsort(score[pos])[:kt]]] = True
                else:                                    # topk / degree / dist
                    k[pos[np.argsort(-score[pos])[:kt]]] = True
        else:
            k = rng.random(n_seg) < r_local[local]
            if not k.any():
                k[rng.integers(n_seg)] = True
        mk = med[s:e][k]; tk = typ[s:e][k].astype(np.int64)
        mL.append(mk); tL.append(tk); raL.append(rela[s:e][k]); rbL.append(relb[s:e][k])
        daL.append(da[s:e][k]); dbL.append(db[s:e][k])
        pL.append(np.full(len(mk), local, np.int64))
        np.add.at(cnt[local], tk, 1.0)
        np.add.at(aa[local], tk, 1.0 / np.log(np.clip(deg[mk].astype(np.float64), 2, None)))
        nsup[local] = len(mk)
    draw = np.zeros((n, struct_dim))
    draw[:, 0:nt] = cnt; draw[:, nt:2 * nt] = np.log1p(cnt); draw[:, 2 * nt:3 * nt] = aa
    draw[:, -2] = nsup; draw[:, -1] = (cnt > 0).sum(1)
    dstruct = ((draw - mu) / sd).astype(np.float32)
    t = lambda L: torch.as_tensor(np.concatenate(L), device=device) if L \
        else torch.zeros(0, dtype=torch.long, device=device)
    return SupportBatch(med_id=t(mL), pair_idx=t(pL), type_idx=t(tL), rel_a=t(raL),
                        rel_b=t(rbL), d_a=t(daL), d_b=t(dbL),
                        struct_feats=torch.as_tensor(dstruct, device=device), n_pairs=n)


@torch.no_grad()
def _predict(model, split, vab, qab, device, batch_size):
    model.eval()
    n = len(split["y"]); probs = np.zeros(n)
    for s in range(0, n, batch_size):
        idx = np.arange(s, min(s + batch_size, n))
        batch, _ = _gather(split, idx, device)
        if qab is not None:
            batch.pair_q = torch.as_tensor(qab[idx], device=device)
        pf = torch.as_tensor(vab[idx], device=device) if model.use_branch else None
        probs[idx] = torch.sigmoid(model(batch, pf)).cpu().numpy()
    return probs


def _evaluate(model, split, vab, qab, device, batch_size):
    probs = _predict(model, split, vab, qab, device, batch_size)
    y = split["y"]
    return {"auc": float(roc_auc_score(y, probs)),
            "auprc": float(average_precision_score(y, probs))}


def _branch_finetune(model, data, vab, device, batch_size, lr, wd, epochs, patience):
    """Phase 2 (codex): freeze core+scorer at the phase-1 best, then fit ONLY the
    additive branch on the FROZEN base logit. Single model (not an ensemble): the
    branch is part of the head, summed into the logit. The frozen tiny linear branch
    (logistic-residual on base_logit) realizes the mechanism-feature signal the
    collapse-capped phase-1 (best_epoch=1) could not. val_s2-selected; dropout off."""
    for p in model.core.parameters():
        p.requires_grad_(False)
    for p in model.scorer.parameters():
        p.requires_grad_(False)
    model.eval()  # deterministic, dropout-off backbone for base-logit precompute
    base, pf, yt = {}, {}, {}
    for name in ("train", "val_s2", "test_s2"):
        split = data[name]; n = len(split["y"]); bl = np.zeros(n)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                idx = np.arange(s, min(s + batch_size, n))
                b, _ = _gather(split, idx, device)
                bl[idx] = model.scorer(model.core(b)).squeeze(-1).cpu().numpy()
        base[name] = torch.as_tensor(bl, dtype=torch.float32, device=device)
        pf[name] = torch.as_tensor(vab[name], dtype=torch.float32, device=device)
        yt[name] = torch.as_tensor(split["y"], dtype=torch.float32, device=device)
    for p in model.branch.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.branch.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()

    def _auc(name):
        with torch.no_grad():
            logit = base[name] + model.branch(pf[name]).squeeze(-1)
            return roc_auc_score(data[name]["y"],
                                 torch.sigmoid(logit).cpu().numpy())

    best_auc, best_state, bad, ran = -1.0, None, 0, 0
    for ep in range(epochs):
        model.branch.train()
        opt.zero_grad()
        logit = base["train"] + model.branch(pf["train"]).squeeze(-1)
        loss_fn(logit, yt["train"]).backward(); opt.step()
        ran = ep + 1
        va = _auc("val_s2")
        if va > best_auc:
            best_auc = va; bad = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.branch.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.branch.load_state_dict(best_state)
    with torch.no_grad():
        tl = base["test_s2"] + model.branch(pf["test_s2"]).squeeze(-1)
        tp = torch.sigmoid(tl).cpu().numpy()
    return {"auc": float(roc_auc_score(data["test_s2"]["y"], tp)),
            "auprc": float(average_precision_score(data["test_s2"]["y"], tp)),
            "bf_best_val": float(best_auc), "bf_epochs_run": ran}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aware-step", type=int, choices=[0, 1, 2, 3], default=1,
                    help="0 = naked AND (branch off, reproduces Step 0); "
                         "1 = + zero-init promiscuity branch b(v_ab); "
                         "2 = + global monotone hub (degree) gate on w(m); "
                         "3 = + pair-conditioned hub gate lambda_tau(q_ab)")
    ap.add_argument("--gate-reg", type=float, default=1e-3,
                    help="weight of the degree-gate L2 + smoothness penalty (step>=2)")
    ap.add_argument("--support-mode", choices=["and", "sum"], default="and")
    ap.add_argument("--l-max", type=int, default=None)
    ap.add_argument("--k-per-type", type=int, default=64)
    ap.add_argument("--n-max", type=int, default=400)
    ap.add_argument("--with-copath", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--zero-count-feats", action="store_true",
                    help="DIAGNOSTIC: zero the uncapped density COUNT block of "
                         "struct_feats (s_tau/log/AA/n_support/n_active) to test if it "
                         "drives the best_epoch=1 collapse")
    ap.add_argument("--use-mech-feats", action="store_true",
                    help="append explicit DDI-mechanism-pair counts (log1p shared-"
                         "enzyme PK + shared-target PD) into the zero-init branch; "
                         "forces the branch on (the probe-validated +1pt lever)")
    ap.add_argument("--uncapped-mech", action="store_true",
                    help="use UNCAPPED richer mech evidence (8-d: enzyme/target "
                         "count, AA, non-hub count, normalized overlap) instead of the "
                         "capped 2-d counts (codex round-2: richer frozen-branch evidence)")
    ap.add_argument("--shuffle-mech", action="store_true",
                    help="CONTROL: row-shuffle the mech features (break pair<->mech "
                         "link); the gain should vanish if it's real mechanism signal")
    ap.add_argument("--branch-finetune", action="store_true",
                    help="phase 2: freeze core+scorer at phase-1 best, fit ONLY the "
                         "additive branch on the frozen base logit (realizes the "
                         "mech signal the best_epoch=1 collapse caps). single model.")
    ap.add_argument("--support-degrade", action="store_true",
                    help="phase-1 OOD augmentation: per-pair sparser-count VIEW "
                         "(scale struct count block by r~U(0.65,0.9)) + consistency "
                         "loss, to expose the scorer to the sparse test regime")
    ap.add_argument("--degrade-pool", action="store_true",
                    help="round-4: also subsample the POOLING mediators and recompute "
                         "the count block from the retained subset, so the degraded "
                         "pool+counts are one sparse world")
    ap.add_argument("--degrade-topk", action="store_true",
                    help="round-5: degraded-view pool subsample = per-(pair,type) top-k "
                         "by corridor score -(d_a+d_b)-log1p(deg) (deterministic, lower "
                         "variance) instead of random Bernoulli keep")
    ap.add_argument("--degrade-select", default="topk",
                    choices=["topk", "matchrand", "bottomk", "degree", "dist"],
                    help="exact-k selection rule (rounds 6-9 controls): topk=corridor "
                         "score; matchrand=random; bottomk=worst; degree/dist=ablate "
                         "the score components")
    ap.add_argument("--mech-only", default="both", choices=["both", "pk", "pd"],
                    help="round-15 mechanism ablation: restrict the mech branch to PK "
                         "(shared-enzyme) or PD (shared-target) only by zeroing the other "
                         "standardized column. 'both' = default full 2-d mech.")
    ap.add_argument("--deterministic", action="store_true",
                    help="round-11: enable deterministic cudnn/torch kernels for a "
                         "reproducibility pass (records drift vs non-deterministic runs)")
    ap.add_argument("--degrade-rlo", type=float, default=0.65,
                    help="round-9: lower bound of per-pair keep-ratio r~U(rlo,rhi) for the "
                         "degraded support view (default 0.65)")
    ap.add_argument("--degrade-rhi", type=float, default=0.9,
                    help="round-9: upper bound of keep-ratio r~U(rlo,rhi) (default 0.9)")
    ap.add_argument("--consistency-weight", type=float, default=0.05,
                    help="round-7/21: weight of the MSE(view_logit, stopgrad(orig_logit)) "
                         "teacher-student consistency term. Set 0.0 to ablate it (test "
                         "whether the gain is sparse-view exposure alone vs logit agreement)")
    ap.add_argument("--view-bce-weight", type=float, default=0.5,
                    help="weight of the BCE on the degraded VIEW logit (default 0.5)")
    ap.add_argument("--bipartite", action="store_true",
                    help="ACTUAL invariance-training general-graph bipartition: per-mediator "
                         "parallel/perp split vs pair prototype -> z_C(common)+z_D(distinct) "
                         "dual pools (full support in both) + GRL null on z_C + cross-view "
                         "invariance + orthogonality. Uses BipartiteAwareCore.")
    ap.add_argument("--bp-null-weight", type=float, default=0.1,
                    help="weight of the GRL label-nulling loss on z_C (I(z_C;y)->0)")
    ap.add_argument("--bp-inv-weight", type=float, default=0.1,
                    help="weight of cross-view invariance ||z_c^orig - z_c^view||^2 (both branches)")
    ap.add_argument("--bp-sep-weight", type=float, default=0.1,
                    help="weight of cos^2(z_C, z_D) orthogonality separation")
    ap.add_argument("--bp-grl-lambda", type=float, default=1.0,
                    help="gradient-reversal strength for the z_C adversary")
    ap.add_argument("--prune-pool-frac", type=float, default=0.0,
                    help="upstream redundancy prune: drop top-frac most signature-redundant "
                         "(within-pair, by type+rel_a+rel_b duplication) pooled mediators per "
                         "pair, deterministically across ALL splits (no leakage); struct "
                         "counts unchanged. 0 = off. Tests if redundant support hurts the pool.")
    ap.add_argument("--mediator-gate", action="store_true",
                    help="anchor (stable-predictive idea): learned structural-only "
                         "distinctiveness gate m_i injected as +log(m_i) into within-type "
                         "attention. Uses GatedAwareStructuralCore.")
    ap.add_argument("--gate-input", default="full", choices=["degree", "redund", "full"],
                    help="gate input ablation: degree=learned hub down-weight; "
                         "redund=r_sig+r_rep_loo (path redundancy); full=both. "
                         "degree==full => reduced to corridor; redund/full win => signal "
                         "beyond degree.")
    ap.add_argument("--gate-vrex-weight", type=float, default=0.0,
                    help="V-REx weight: variance of per-view BCE risks (orig + degraded "
                         "view); needs --support-degrade. 0 = off.")
    ap.add_argument("--gate-budget-weight", type=float, default=0.0,
                    help="budget weight: (mean_i m_i - gate_budget_tau)^2. 0 = off.")
    ap.add_argument("--gate-budget-tau", type=float, default=0.7,
                    help="target mean gate mass for the budget term (default 0.7)")
    ap.add_argument("--bf-lr", type=float, default=3e-2)
    ap.add_argument("--bf-wd", type=float, default=1e-4)
    ap.add_argument("--bf-epochs", type=int, default=300)
    ap.add_argument("--bf-patience", type=int, default=30)
    ap.add_argument("--decomposed-readout", action="store_true",
                    help="codex unified design: logit = MLP_small(z_comp) + beta.e_mech, "
                         "e_mech = UNCAPPED mech evidence (enzyme/target count+AA). "
                         "single-phase, separate optimizer groups (beta higher LR), small "
                         "scorer. one coherent two-statistic readout (composition + "
                         "evidence mass); replaces the capped-mech branch + bf hack.")
    ap.add_argument("--scorer-hidden", type=int, default=None,
                    help="scorer (composition) MLP hidden for the decomposed readout "
                         "(default --hidden; use 32/64 — the big scorer overfits)")
    ap.add_argument("--branch-lr", type=float, default=3e-2,
                    help="LR for the mech-evidence branch (separate optimizer group)")
    args = ap.parse_args()

    support_and = args.support_mode == "and"
    if not support_and:
        raise SystemExit("run_spmn_v2_aware is AND-only (Step 1 reuses the cached "
                         "AND substrate); --support-mode sum is not supported here.")
    l_max = args.l_max if args.l_max is not None else 3
    use_branch = args.aware_step >= 1

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.deterministic:
        # round-11 reproducibility pass: force deterministic kernels. Requires
        # CUBLAS_WORKSPACE_CONFIG=:4096:8 in the env for CUDA>=10.2 matmul.
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[aware] DETERMINISTIC mode on (cudnn.deterministic, "
              "use_deterministic_algorithms warn_only)", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = _load_frames(args.seed)
    data = _load_support_cache(args.support_mode, args.seed, l_max,
                               args.k_per_type, args.n_max, args.with_copath)

    if args.prune_pool_frac > 0:
        print(f"[aware] UPSTREAM redundancy prune frac={args.prune_pool_frac} "
              f"(all splits, deterministic)", flush=True)
        for name in list(data):
            data[name] = _prune_pool_redundant(data[name], args.prune_pool_frac,
                                               N_REL_BUCKETS)

    # standardise struct with TRAIN stats only (matches the standalone script).
    mu = data["train"]["struct"].mean(0, keepdims=True)
    sd = data["train"]["struct"].std(0, keepdims=True) + 1e-6
    raw_train_struct = data["train"]["struct"].copy() if args.support_degrade else None
    for name in data:
        data[name]["struct"] = ((data[name]["struct"] - mu) / sd).astype(np.float32)
    if args.zero_count_feats:
        nt = N_TYPES; dim = data["train"]["struct"].shape[1]
        count_idx = list(range(0, 3 * nt)) + [dim - 2, dim - 1]
        for name in data:
            data[name]["struct"][:, count_idx] = 0.0
        print(f"[aware] DIAGNOSTIC: zeroed {len(count_idx)} count-block struct cols "
              f"(s_tau/log/AA/n_support/n_active) — collapse localization", flush=True)

    print("[aware] loading merged KG ...", flush=True)
    kg = MergedKG.from_parquet()
    vab, reach_stats = _build_pair_feats(frames, l_max, kg)
    # alignment guard: v_ab is built in frames row order; the support cache stores y
    # in the SAME frames order it was built from. Asserting current-frames labels ==
    # cached y ties v_ab to the cached supports and catches a stale/reordered cache
    # vs current frames (the silent misattach bug codex flagged). Strong+cheap: 50/50
    # labels over thousands of rows make an order-preserving relabel implausible.
    for name in data:
        fy = frames[name]["label"].to_numpy().astype(np.float32)
        cy = data[name]["y"].astype(np.float32)
        if len(fy) != len(cy) or not np.array_equal(fy, cy):
            raise SystemExit(
                f"[aware] frame/cache MISALIGNMENT for {name}: current frames row "
                f"order does not match the support cache (labels differ). Rebuild the "
                f"AND cache via run_spmn_v2_standalone before running aware.")

    decomp = args.decomposed_readout
    use_mech = args.use_mech_feats and not decomp
    mech_dim = 0
    if decomp:
        print("[aware] decomposed readout: building UNCAPPED mech evidence "
              "(enzyme/target count+AA) ...", flush=True)
        em = {name: density.build_mech_evidence_matrix(
            kg, frames[name]["drug_a_id"].to_numpy(),
            frames[name]["drug_b_id"].to_numpy()) for name in data}
        emu = em["train"].mean(0, keepdims=True)
        ems = em["train"].std(0, keepdims=True) + 1e-6
        vab = {name: ((em[name] - emu) / ems).astype(np.float32) for name in data}
        use_branch = True
    if use_mech:
        if args.uncapped_mech:
            print("[aware] UNCAPPED richer mech evidence (8-d) ...", flush=True)
            mfull = {name: density.build_mech_evidence_matrix(
                kg, frames[name]["drug_a_id"].to_numpy(),
                frames[name]["drug_b_id"].to_numpy()) for name in data}
        else:
            # 2-feat: enzyme(PK) + target(PD), capped (transporter + interaction
            # tested and dropped — within noise).
            mfull = {name: np.log1p(density.build_mech_feature_matrix(
                data[name]["rela"], data[name]["relb"], data[name]["offsets"],
                len(data[name]["y"]))[:, :2]) for name in data}   # log1p [enz, tgt]
        if args.shuffle_mech:                          # control: kill pair<->mech link
            rng = np.random.default_rng(0)
            for name in mfull:
                mfull[name] = mfull[name][rng.permutation(len(mfull[name]))]
        mmu = mfull["train"].mean(0, keepdims=True)
        msd = mfull["train"].std(0, keepdims=True) + 1e-6
        mech_dim = mfull["train"].shape[1]
        for name in data:
            mz = ((mfull[name] - mmu) / msd).astype(np.float32)
            if args.mech_only != "both" and not args.uncapped_mech and mech_dim >= 2:
                # round-15 PK/PD branch ablation: zero the OTHER standardized mech
                # column (col0=enzyme/PK, col1=target/PD) so the branch sees one axis.
                drop = 1 if args.mech_only == "pk" else 0  # pk keeps col0, pd keeps col1
                mz[:, drop] = 0.0
            vab[name] = np.concatenate([vab[name], mz], axis=1)
        use_branch = True
        print(f"[aware] mech feats ON: +{mech_dim} dims (enzyme PK, target PD, "
              f"transporter, enzyme*target)"
              f"{' [SHUFFLED CONTROL]' if args.shuffle_mech else ''}", flush=True)

    use_pair_gate = (args.aware_step >= 3) and not decomp
    qab = _build_pair_regime(frames, data, kg, reach_stats) if use_pair_gate else None

    # Step 0 base config: clean naked AND (entity_embed on, asym/absdiff/dist off).
    core_kwargs = dict(
        n_entities=178029, n_types=N_TYPES, n_rel_buckets=N_REL_BUCKETS,
        struct_dim=symmetric_binary_dim(), d=args.d, hidden=args.hidden,
        dropout=args.dropout, use_entity_embed=True, use_asym=False,
        use_absdiff_embed=False, use_dist_attn=False, max_dist=l_max - 1)
    use_deg_gate = (args.aware_step >= 2) and not decomp
    bp_adv = None
    if args.bipartite:
        core = BipartiteAwareCore(**core_kwargs)
        bp_adv = LabelAdversary(core.n_types * core.n_chan * core.d).to(device)
        print(f"[aware] BIPARTITE core ON (z_C/z_D dual pool, out_dim={core.out_dim}; "
              f"null={args.bp_null_weight} inv={args.bp_inv_weight} sep={args.bp_sep_weight} "
              f"grl_lambda={args.bp_grl_lambda})", flush=True)
    elif args.mediator_gate:
        deg_log = np.log1p(kg.degree.astype(np.float64))
        tm = deg_log[data["train"]["med"]]
        gdeg_mu = float(tm.mean()) if tm.size else 0.0
        gdeg_sigma = float(tm.std()) if tm.size else 1.0
        core = GatedAwareStructuralCore(
            **core_kwargs, gate_input=args.gate_input,
            deg_log=deg_log, deg_mu=gdeg_mu, deg_sigma=gdeg_sigma)
        print(f"[aware] mediator-gate ON, input={args.gate_input} "
              f"(vrex={args.gate_vrex_weight} budget={args.gate_budget_weight}"
              f"@tau={args.gate_budget_tau})", flush=True)
    elif use_deg_gate:
        deg_mode = "pair" if use_pair_gate else "global"
        deg_log = np.log1p(kg.degree.astype(np.float64))
        tm = deg_log[data["train"]["med"]]
        deg_mu = float(tm.mean()) if tm.size else 0.0
        deg_sigma = float(tm.std()) if tm.size else 1.0
        core = AwareStructuralCore(
            **core_kwargs, use_deg_gate=True, deg_gate_mode=deg_mode,
            deg_log=deg_log, deg_mu=deg_mu, deg_sigma=deg_sigma,
            n_q_feats=(density.N_Q_FEATS if use_pair_gate else None))
        print(f"[aware] hub gate on, mode={deg_mode} (deg_log mu={deg_mu:.3f} "
              f"sigma={deg_sigma:.3f}, gate_reg={args.gate_reg})", flush=True)
    else:
        core = StructuralVariableCore(**core_kwargs)
    scorer_hidden = args.scorer_hidden if args.scorer_hidden is not None else args.hidden
    n_pair_feats = (density.N_MECH_EVIDENCE if decomp
                    else density.N_PAIR_FEATS + mech_dim)
    model = DecomposedStandaloneHead(
        core, n_pair_feats=n_pair_feats, hidden=scorer_hidden,
        dropout=args.dropout, use_branch=use_branch).to(device)
    if decomp:
        opt = torch.optim.AdamW([
            {"params": list(core.parameters()) + list(model.scorer.parameters()),
             "lr": args.lr},
            {"params": model.branch.parameters(), "lr": args.branch_lr},
        ], weight_decay=args.weight_decay)
        print(f"[aware] decomposed: scorer_hidden={scorer_hidden} branch_lr={args.branch_lr} "
              f"single-phase (no bf)", flush=True)
    else:
        params = list(model.parameters())
        if bp_adv is not None:
            params += list(bp_adv.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    tag = (f"aware{args.aware_step}_{args.support_mode}_seed{args.seed}"
           + ("_zerocount" if args.zero_count_feats else "")
           + ("_decomp" if args.decomposed_readout else "")
           + ("_mech" if args.use_mech_feats else "")
           + ("_unc" if args.uncapped_mech else "")
           + ("_sdeg" if args.support_degrade else "")
           + ("_dpool" if args.degrade_pool else "")
           + ("_topk" if args.degrade_topk else "")
           + (f"_{args.degrade_select}" if (args.degrade_topk and args.degrade_select != "topk") else "")
           + (f"_r{args.degrade_rlo:g}-{args.degrade_rhi:g}" if (args.support_degrade and (args.degrade_rlo, args.degrade_rhi) != (0.65, 0.9)) else "")
           + ("_nomse" if (args.support_degrade and args.consistency_weight == 0) else "")
           + (f"_cw{args.consistency_weight:g}" if (args.support_degrade and args.consistency_weight not in (0, 0.05)) else "")
           + (f"_vbw{args.view_bce_weight:g}" if (args.support_degrade and args.view_bce_weight != 0.5) else "")
           + ("_shuf" if args.shuffle_mech else "")
           + ("_det" if args.deterministic else "")
           + (f"_{args.mech_only}only" if args.mech_only != "both" else "")
           + (f"_prune{args.prune_pool_frac:g}" if args.prune_pool_frac > 0 else "")
           + ("_bipartite" if args.bipartite else "")
           + (f"_n{args.bp_null_weight:g}i{args.bp_inv_weight:g}s{args.bp_sep_weight:g}"
              if (args.bipartite and (args.bp_null_weight, args.bp_inv_weight,
                  args.bp_sep_weight) != (0.1, 0.1, 0.1)) else "")
           + (f"_gate{args.gate_input}" if args.mediator_gate else "")
           + (f"_vrex{args.gate_vrex_weight:g}" if (args.mediator_gate and args.gate_vrex_weight > 0) else "")
           + (f"_gbud{args.gate_budget_weight:g}" if (args.mediator_gate and args.gate_budget_weight > 0) else "")
           + ("_bf" if args.branch_finetune else ""))
    print(f"[aware] variant={tag} l_max={l_max} d={args.d} out_dim={core.out_dim} "
          f"use_branch={use_branch} n_pair_feats={density.N_PAIR_FEATS}", flush=True)

    train = data["train"]; n_train = len(train["y"])
    n_steps = (n_train + args.batch - 1) // args.batch
    prog = TrainProgress(args.epochs, log_step_every=100,
                         total_steps_per_epoch=n_steps, prefix=f"[aware {tag}] ")
    best_auc, best_state, best_epoch = -1.0, None, -1
    val_curve = []
    dgr = np.random.default_rng(args.seed) if args.support_degrade else None
    deg_table = kg.degree if args.support_degrade else None
    struct_dim = data["train"]["struct"].shape[1]
    for epoch in range(args.epochs):
        model.train(); prog.epoch_start(epoch)
        perm = np.random.permutation(n_train)
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            batch, y = _gather(train, idx, device)
            if use_pair_gate:
                batch.pair_q = torch.as_tensor(qab["train"][idx], device=device)
            pf = torch.as_tensor(vab["train"][idx], device=device) if use_branch else None
            if args.support_degrade:
                r = dgr.uniform(args.degrade_rlo, args.degrade_rhi,
                                size=len(idx)).astype(np.float32)
                if args.degrade_pool:
                    deg_batch = _degrade_full(train, idx, r, deg_table, mu, sd, dgr,
                                              struct_dim, device, topk=args.degrade_topk,
                                              select=args.degrade_select)
                else:
                    ds = _degrade_struct(raw_train_struct[idx], r, mu, sd)
                    deg_batch = dataclasses.replace(
                        batch, struct_feats=torch.as_tensor(ds, device=device))
                ol = model(batch, pf)
                if args.bipartite:
                    pC_o, pD_o = model.core._pooled_C, model.core._pooled_D
                dl = model(deg_batch, pf)
                r0 = loss_fn(ol, y); r1 = loss_fn(dl, y)
                loss = r0 + args.view_bce_weight * r1
                if args.consistency_weight > 0:
                    loss = loss + args.consistency_weight * nn.functional.mse_loss(
                        dl, ol.detach())
                if args.mediator_gate and args.gate_vrex_weight > 0:
                    # V-REx: variance of per-view BCE risks (orig + degraded view)
                    loss = loss + args.gate_vrex_weight * torch.stack([r0, r1]).var()
                if args.bipartite:
                    pC_v, pD_v = model.core._pooled_C, model.core._pooled_D
                    if args.bp_inv_weight > 0:   # cross-view invariance (both channels)
                        loss = loss + args.bp_inv_weight * (
                            (pC_o - pC_v).pow(2).mean() + (pD_o - pD_v).pow(2).mean())
                    if args.bp_null_weight > 0:  # GRL: make z_C label-uninformative
                        lam = args.bp_grl_lambda * min(
                            1.0, (epoch + 1) / max(1, int(0.15 * args.epochs)))
                        loss = loss + args.bp_null_weight * loss_fn(bp_adv(pC_o, lam), y)
                    if args.bp_sep_weight > 0:   # orthogonality cos^2(z_C, z_D)
                        cos = nn.functional.cosine_similarity(pC_o, pD_o, dim=-1, eps=1e-6)
                        loss = loss + args.bp_sep_weight * cos.pow(2).mean()
            else:
                loss = loss_fn(model(batch, pf), y)
            if use_deg_gate:
                loss = loss + args.gate_reg * model.core.gate_reg()
            if args.mediator_gate and args.gate_budget_weight > 0:
                loss = loss + args.gate_budget_weight * model.core.gate_budget_reg(
                    args.gate_budget_tau)
            opt.zero_grad(); loss.backward(); opt.step()
            if use_deg_gate:
                # project gate magnitude to >= 0 (suppression only; avoids the
                # dead-gate from a two-sided clamp). Keeps the exact zero start.
                model.core.gate_g.data.clamp_(min=0.0)
            prog.step(loss.item())
        val = _evaluate(model, data["val_s2"], vab["val_s2"],
                        qab["val_s2"] if qab else None, device, args.batch)
        prog.log_eval(val, scope="epoch"); prog.epoch_end(extra={"val_auc": val["auc"]})
        val_curve.append({"epoch": epoch + 1, "val_auc": val["auc"],
                          "val_auprc": val["auprc"]})
        if val["auc"] > best_auc:
            best_auc = val["auc"]; best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val = _evaluate(model, data["val_s2"], vab["val_s2"],
                    qab["val_s2"] if qab else None, device, args.batch)
    test = _evaluate(model, data["test_s2"], vab["test_s2"],
                     qab["test_s2"] if qab else None, device, args.batch)
    print("\n=== SPMN v2 aware (Step 1 decomposed scorer) ===")
    print(f"  variant: {tag}  l_max={l_max}  aware_step={args.aware_step}")
    print(f"  best val_s2: AUC={val['auc']:.4f} AUPRC={val['auprc']:.4f}")
    print(f"  test_s2:     AUC={test['auc']:.4f} AUPRC={test['auprc']:.4f}")
    print("  refs: naked AND seed42 0.7844 / 43 0.7472 / 44 0.7657 | EmerGNN ~0.7458")

    bf = None
    base_probs = None
    if args.branch_finetune and use_branch:
        phase1_test = dict(test)
        if args.save_predictions:
            # round-17: capture phase-1 BASE probs (zero-init branch contributes 0)
            # before bf fits the branch, so we can decompose the branch Δlogit.
            base_probs = _predict(model, data["test_s2"], vab["test_s2"],
                                  qab["test_s2"] if qab else None, device, args.batch)
        bf = _branch_finetune(model, data, vab, device, args.batch, args.bf_lr,
                              args.bf_wd, args.bf_epochs, args.bf_patience)
        print(f"  [branch-finetune] test AUC={bf['auc']:.4f} AUPRC={bf['auprc']:.4f}"
              f"  (phase1 test {phase1_test['auc']:.4f}; bf_val {bf['bf_best_val']:.4f}"
              f"; epochs {bf['bf_epochs_run']})", flush=True)
        test = {"auc": bf["auc"], "auprc": bf["auprc"]}

    res_dir = ROOT / "Code/runs/spmn_v2_standalone"; res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{tag}_lmax{l_max}_d{args.d}.json").write_text(json.dumps({
        "seed": args.seed, "variant": tag, "aware_step": args.aware_step,
        "support_mode": args.support_mode, "l_max": l_max, "use_branch": use_branch,
        "n_pair_feats": n_pair_feats, "use_mech": use_mech, "d": args.d,
        "epochs": args.epochs, "best_epoch": best_epoch, "val_s2": val,
        "test_s2": test, "branch_finetune": bf, "val_curve": val_curve}, indent=2))

    if args.save_predictions:
        probs = _predict(model, data["test_s2"], vab["test_s2"],
                         qab["test_s2"] if qab else None, device, args.batch)
        fr = frames["test_s2"]
        pred_path = res_dir / f"{tag}_test_s2_scores.npz"
        save_kw = dict(pair_a=fr["drug_a_id"].astype(str).to_numpy(),
                       pair_b=fr["drug_b_id"].astype(str).to_numpy(),
                       y_true=data["test_s2"]["y"].astype(np.float64),
                       y_score=probs.astype(np.float32))
        if base_probs is not None:
            save_kw["base_score"] = base_probs.astype(np.float32)
        np.savez(pred_path, **save_kw)
        print(f"  saved test_s2 predictions -> {pred_path}", flush=True)


if __name__ == "__main__":
    main()

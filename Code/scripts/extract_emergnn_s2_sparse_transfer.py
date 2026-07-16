"""EmerGNN S2 SPARSE-SOURCE transfer probe (codex recipe, 2026-07-06).

WHY: the dense fact-target source used in extract_emergnn_s2_rank.py gave an
INVALID (inverted, AUC~0.30) transfer curve, because the S2 model was TRAINED on
SPARSE emerging-emerging pairs and inverts on dense seen-seen pairs (its head on
that dense source scored AUC=0.169 = the S2 density shortcut). The deployable
transfer probe SOURCE must therefore be in the SAME sparse regime as cold.

RECIPE (codex-approved):
- Hold out K seen TRAIN drugs as pseudo-emerging P (several random draws).
- G_probe = bio KG + train DDI facts whose BOTH endpoints are NOT in P
  (so every p in P has DDI degree 0 in G_probe -> emerging, exactly like S2 and
  exactly like a cold unseen drug). This mirrors shuffle_train's S2 branch with
  removed=P instead of a random 20%.
- SOURCE = P x P pairs (BOTH emerging = sparse query, matching cold unseen x
  unseen): positives = known train DDIs among P; negatives = balanced sampled
  P x P non-edges.
- Extract enc_ht for SOURCE and COLD on the SAME G_probe (identical KG -> the
  transfer probe is apples-to-apples). Frozen ep18 checkpoint, NO retrain.
- Fit PCA+logreg on SOURCE only, apply to COLD, per PCA rank. Also degree_only /
  degree-residual (fit on source). Average over draws; report mean/std.

Supporting evidence only (codex: transfer is source-construction-sensitive; the
primary story stays within-cold + degree-residual + model-reproduction).

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && EMERGNN_BACKEND=rspmm \
    EMERGNN_KG_SCOPE=full PYTHONUNBUFFERED=1 python -u \
    Code/scripts/extract_emergnn_s2_sparse_transfer.py \
    --run-dir Code/runs/2026-07-06_02-16-48__extract_emergnn_s2_rank__ddi800_cold_s2_fold0 \
    --ckpt checkpoint-ep018__n00018 --k 150 --draws 5"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets  # noqa: E402
from extract_emergnn_fact_protocol_rank import (  # noqa: E402
    _embed_logit_on_kg, _load_trainable, MERGED_EDGES)
from extract_emergnn_s2_rank import _S2Core  # noqa: E402
from extract_emergnn_multickpt_rank import _pairs  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64, 128]


def _transfer(Zsrc, ysrc, Zc, yc, ranks):
    """PCA+logreg fit on SOURCE only, applied to COLD, per rank. Returns AUC list."""
    mu, sd = Zsrc.mean(0, keepdims=True), Zsrc.std(0, keepdims=True) + 1e-8
    Ss, Sc = (Zsrc - mu) / sd, (Zc - mu) / sd
    ctr = Ss.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ss - ctr, full_matrices=False)
    ysrc = ysrc.astype(int)
    out = []
    for r in ranks:
        rr = min(r, Vt.shape[0])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit((Ss - ctr) @ Vt[:rr].T, ysrc)
        out.append(float(roc_auc_score(yc, clf.predict_proba((Sc - ctr) @ Vt[:rr].T)[:, 1])))
    return out


def _residualize(Zsrc, dsrc, Zc, dc):
    """Regress Z on [deg, 1] using SOURCE, subtract from both source and cold."""
    Xs = np.hstack([dsrc, np.ones((len(dsrc), 1))])
    Xc = np.hstack([dc, np.ones((len(dc), 1))])
    B, _, _, _ = np.linalg.lstsq(Xs, Zsrc, rcond=None)
    return Zsrc - Xs @ B, Zc - Xc @ B


def _build_g_probe(train_ddi_int, kg_tri, P_ids, n_ent, n_rel_with_ddi, device):
    """S2-style split with removed=P (controlled, not random). Returns
    (G_probe sparse tensor, source_pos (M,2) unique unordered). fact = both
    endpoints NOT in P; source_pos = both endpoints IN P; mixed discarded."""
    h, t = train_ddi_int[:, 0], train_ddi_int[:, 1]
    inP_h = np.isin(h, P_ids); inP_t = np.isin(t, P_ids)
    fact_mask = (~inP_h) & (~inP_t)
    srcpos_mask = inP_h & inP_t
    fact_tri = train_ddi_int[fact_mask]
    # direct leak invariant: NO endpoint of any fact DDI edge is in P -> every
    # p in P has DDI degree 0 in G_probe (fact_tri is G_probe's only DDI source)
    n_leak = int(np.isin(fact_tri[:, :2], P_ids).sum())
    assert n_leak == 0, f"leak: {n_leak} fact-DDI endpoints fall in P"
    epoch_kg = np.concatenate([fact_tri, kg_tri], axis=0) if len(kg_tri) else fact_tri
    G = build_sparse_kg_from_triplets(np.asarray(epoch_kg, np.int64), n_ent,
                                      n_rel_with_ddi, device=device)
    # canonicalize source positives: unordered, unique, drop self-pairs
    sp = np.sort(train_ddi_int[srcpos_mask][:, :2], axis=1)
    sp = sp[sp[:, 0] != sp[:, 1]]
    sp = np.unique(sp, axis=0)
    return G, sp


def _sample_neg_in_P(P_ids, all_pos_pairs, n_neg, rng):
    """Sample up to n_neg unordered non-edge pairs, BOTH drugs in P, excluding
    known positives. Capacity-aware: caps at C(|P|,2) - |pos|."""
    pos_set = {(int(min(a, b)), int(max(a, b))) for a, b in all_pos_pairs}
    P = np.asarray(P_ids)
    cap = len(P) * (len(P) - 1) // 2 - len(pos_set)
    n_neg = min(n_neg, max(cap, 0))
    out, seen, tries = [], set(), 0
    while len(out) < n_neg and tries < n_neg * 200 + 1000:
        a, b = rng.choice(P, 2, replace=False)
        key = (int(min(a, b)), int(max(a, b)))
        if key not in pos_set and key not in seen:
            out.append([a, b]); seen.add(key)
        tries += 1
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="the ep18 S2 run dir")
    ap.add_argument("--ckpt", default="checkpoint-ep018__n00018")
    ap.add_argument("--dataset", default="ddi800")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--k", type=int, default=80, help="# pseudo-emerging drugs P per draw "
                    "(control via logged frac_ddi_removed + cold_head_on_Gprobe, not K itself)")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda")

    leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), args.dataset, "binary",
                       args.split, args.fold)
    tr = leaf.train.reset_index(drop=True)
    cold_test = leaf.test
    merged_kg = Path(leaf.resources.kg.source) / MERGED_EDGES

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT / "Code" / "runs" / f"{ts}__extract_emergnn_s2_sparse_transfer__{args.dataset}_{args.split}_{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- setup core (n_epochs=0 = build model/entity2id/KG/morgan, NO training) --
    core = _S2Core(
        n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=args.batch_size,
        n_epochs=0, device="cuda", backbone_kg_source="merged", merged_kg_path=merged_kg,
        weight_decay=1e-8, shuffle_train_mode="S2", shuffle_ratio=0.8, log_step_every=50,
        eval_strategy="no", save_strategy="no", load_best_model_at_end=False, run_dir=run_dir)
    core._tr_df = tr
    core._cold_val_df = leaf.val.sample(min(len(leaf.val), 2000), random_state=args.seed).reset_index(drop=True)
    _ds = make_dataset(tr, leaf.val, leaf.resources)
    core.fit(_ds, _ds)   # n_epochs=0 -> setup only (build model/entity2id/KG/morgan), NO training
    _load_trainable(core, Path(args.run_dir) / "checkpoints" / args.ckpt)   # frozen ep18
    print(f"[setup] core built + ep18 loaded ({args.ckpt})", flush=True)

    e2i = core._entity2id
    n_ent = int(core._n_ent); ddi_rel = int(core._n_base_rel)
    n_rel_with_ddi = int(core._n_base_rel_with_ddi)
    kg_tri = np.asarray(core._kg_triplets, dtype=np.int64)
    logdeg = np.log1p(np.bincount(np.concatenate([kg_tri[:, 0], kg_tri[:, 1]]),
                                  minlength=n_ent).astype(np.float64))
    # train DDI positives as entity-id triplets
    tp = tr[tr["y_bin"] == 1]
    a = tp["drug_a_id"].astype(str).map(e2i); b = tp["drug_b_id"].astype(str).map(e2i)
    ok = a.notna() & b.notna()
    a = a[ok].to_numpy(np.int64); b = b[ok].to_numpy(np.int64)
    train_ddi_int = np.stack([a, b, np.full(len(a), ddi_rel, np.int64)], axis=1)
    train_drugs = np.unique(np.concatenate([a, b]))
    cold_ia, cold_ib, yc = _pairs(cold_test, e2i)
    dc = np.vstack([logdeg[cold_ia], logdeg[cold_ib]]).T.astype(np.float32)
    print(f"[data] train_drugs={len(train_drugs)} train_pos_edges={len(a)} cold={len(yc)}", flush=True)

    all_tc, all_tcr, all_do, per_draw = [], [], [], []
    for d in range(args.draws):
        rng = np.random.default_rng(args.seed + d)
        P = rng.choice(train_drugs, size=min(args.k, len(train_drugs)), replace=False)
        G, src_pos = _build_g_probe(train_ddi_int, kg_tri, P, n_ent, n_rel_with_ddi, device)
        src_neg = _sample_neg_in_P(P, src_pos, len(src_pos), rng)
        n = min(len(src_pos), len(src_neg))
        if n < 100:
            print(f"[draw {d}] balanced source only n={n} (<100); skip", flush=True); continue
        sp = src_pos[rng.choice(len(src_pos), n, replace=False)] if len(src_pos) > n else src_pos
        sn = src_neg[rng.choice(len(src_neg), n, replace=False)] if len(src_neg) > n else src_neg
        assert len(sp) == len(sn) == n, "source not balanced"
        s_ia = np.concatenate([sp[:, 0], sn[:, 0]])
        s_ib = np.concatenate([sp[:, 1], sn[:, 1]])
        ys = np.concatenate([np.ones(n), np.zeros(n)]).astype(np.float32)
        ds_ = np.vstack([logdeg[s_ia], logdeg[s_ib]]).T.astype(np.float32)
        # extract enc_ht for source + cold on the SAME G_probe
        Zs = _embed_logit_on_kg(core, s_ia, s_ib, G, args.batch_size, device)[0]
        Zc, lc = _embed_logit_on_kg(core, cold_ia, cold_ib, G, args.batch_size, device)
        cold_head = roc_auc_score(yc, 1.0 / (1.0 + np.exp(-lc)))
        tc = _transfer(Zs, ys, Zc, yc, RANKS)
        Rs, Rc = _residualize(Zs, ds_, Zc, dc)
        tcr = _transfer(Rs, ys, Rc, yc, RANKS)
        do = float(roc_auc_score(yc, LogisticRegression(max_iter=2000).fit(ds_, ys.astype(int)).predict_proba(dc)[:, 1]))
        all_tc.append(tc); all_tcr.append(tcr); all_do.append(do)
        frac_removed = (len(a) - int((~np.isin(a, P) & ~np.isin(b, P)).sum())) / len(a)
        per_draw.append({"draw": d, "K": int(len(P)), "src_pos_total": int(len(src_pos)),
                         "n_balanced": int(n), "frac_ddi_removed": round(frac_removed, 3),
                         "cold_head_on_Gprobe": round(cold_head, 4),
                         "transfer_peak": round(max(tc), 4), "transfer_resid_peak": round(max(tcr), 4),
                         "degree_only": round(do, 4)})
        print(f"[draw {d}] P={len(P)} src_pos={len(src_pos)} n_bal={n} frac_ddi_removed={frac_removed:.3f} "
              f"cold_head(Gprobe)={cold_head:.3f} | transfer peak={max(tc):.3f}@rank{RANKS[int(np.argmax(tc))]} "
              f"resid peak={max(tcr):.3f} degree_only={do:.3f}", flush=True)

    if not all_tc:
        print("[abort] no valid draws (increase --k)", flush=True); return
    tc_m, tc_s = np.mean(all_tc, 0), np.std(all_tc, 0)
    tcr_m = np.mean(all_tcr, 0)
    print("\n### EmerGNN S2 SPARSE-SOURCE transfer (mean over %d draws)" % len(all_tc))
    print(" TRANSFER cold      :", [f"{m:.3f}" for m in tc_m], "peak=%.3f@rank%d" % (tc_m.max(), RANKS[int(tc_m.argmax())]))
    print(" TRANSFER cold std  :", [f"{s:.3f}" for s in tc_s])
    print(" TRANSFER-resid cold:", [f"{m:.3f}" for m in tcr_m], "peak=%.3f" % tcr_m.max())
    print(" degree_only(source->cold) mean=%.3f" % np.mean(all_do))
    out = {"ranks": RANKS, "transfer_mean": tc_m.tolist(), "transfer_std": tc_s.tolist(),
           "transfer_resid_mean": tcr_m.tolist(), "degree_only_mean": float(np.mean(all_do)),
           "per_draw": per_draw, "k": args.k, "draws": len(all_tc), "ckpt": args.ckpt}
    (run_dir / "sparse_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"[done] {run_dir}/sparse_transfer.json", flush=True)


if __name__ == "__main__":
    main()

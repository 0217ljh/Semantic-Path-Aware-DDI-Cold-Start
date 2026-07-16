"""R-GCN under the LOCKED S2 protocol (protocol 2, pure cold-style training),
the node-paradigm twin of the EmerGNN S2 rank analysis.

Supersedes the fact-protocol R-GCN for the main line: training is now DRUG-level
S2 (not edge-level fact split), matching EmerGNN so the R-GCN vs EmerGNN low-rank
comparison is protocol-consistent.

PROTOCOL (locked):
- TRAIN: each step, mark a random `ratio`-complement (~20%) of TRAIN drugs as
  emerging; remove ALL their DDI edges; supervised targets = emerging-emerging
  positive pairs (sparse query) + balanced emerging-emerging sampled negatives;
  the message graph = bio KG + kept-kept DDI facts (as one relation). Per-step
  reshuffle. This replicates shuffle_train's S2 branch with an explicit removed
  set (so we can sample emerging-emerging negatives).
- SELECT: cold-val AUROC (no warm).
- EVAL / EXTRACT (leak-free): every scored pair's own DDI edge absent. cold
  (unseen x unseen) is scored on the eval KG = bio + ALL train DDI facts (cold
  drugs are unseen -> own edge absent automatically). enc step = drug pair ->
  frozen R-GCN over the KG -> Z (uses the model's learned parameters).

MEASUREMENTS (all on cold, per the locked spec): head_cold, within-cold(oracle),
degree_only, degree-residual, sparse-source transfer (emerging-emerging source
on G_probe, multi-draw). Writes results.json + pilot_Z.npz.

Run (project root, WSL conda env project_1):
  python Code/scripts/train_rgcn_rank_s2.py --fold fold0 --epochs 1500 --amp
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError("root")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))

from my_code.models.spmn_v1.retrieval import (  # noqa: E402
    MergedKG, N_TYPES, DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
from train_rgcn_rank_pilot import build_relation_edges, RGCNPairModel  # noqa: E402
from train_gcn_rank_pilot import _pairs_to_idx  # noqa: E402
from analyze_rank_within_split import _within_split_curve, _balance  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# --------------------------------------------------------------------------- #
# S2 split + negative sampling
# --------------------------------------------------------------------------- #
def _s2_split(ddi_pairs, drugs, ratio, rng):
    """shuffle_train S2 with an explicit removed set. ddi_pairs (N,2) int entity
    ids of train POSITIVE DDIs. Returns (fact_pairs, tgt_pos_pairs, removed_arr)."""
    n_remove = len(drugs) - int(len(drugs) * ratio)
    removed = rng.choice(drugs, size=n_remove, replace=False)
    rem_set = set(removed.tolist())
    inR_a = np.isin(ddi_pairs[:, 0], removed)
    inR_b = np.isin(ddi_pairs[:, 1], removed)
    fact = ddi_pairs[(~inR_a) & (~inR_b)]           # both kept -> facts
    tgt = ddi_pairs[inR_a & inR_b]                   # both removed -> targets
    return fact, tgt, removed


def _sample_neg(pool, pos_set, n, rng):
    """Sample n unordered non-edge pairs, both endpoints in `pool`, excluding
    pos_set (a set of canonical (min,max) tuples). Capacity-aware: only counts
    positives whose BOTH endpoints are inside `pool`."""
    pool = np.asarray(pool)
    poolset = {int(p) for p in pool}
    inpool_pos = sum(1 for u, v in pos_set if u in poolset and v in poolset)
    cap = len(pool) * (len(pool) - 1) // 2 - inpool_pos
    n = min(n, max(cap, 0))
    out, seen, tries = [], set(), 0
    while len(out) < n and tries < n * 200 + 2000:
        a, b = rng.choice(pool, 2, replace=False)
        key = (int(min(a, b)), int(max(a, b)))
        if key not in pos_set and key not in seen:
            out.append([a, b]); seen.add(key)
        tries += 1
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def _aug_graph(bio_ei, bio_et, ddi_pairs, ddi_rel, device):
    """bio edges + DDI pairs (both directions, one relation slot)."""
    if len(ddi_pairs) == 0:
        return bio_ei, bio_et
    a = ddi_pairs[:, 0]; b = ddi_pairs[:, 1]
    di = torch.from_numpy(np.vstack([np.concatenate([a, b]),
                                     np.concatenate([b, a])]).astype(np.int64)).to(device)
    dt = torch.full((di.shape[1],), ddi_rel, dtype=torch.long, device=device)
    return torch.cat([bio_ei, di], dim=1), torch.cat([bio_et, dt])


# --------------------------------------------------------------------------- #
# Measurements (shared with the EmerGNN analysis)
# --------------------------------------------------------------------------- #
def _transfer(Zsrc, ysrc, Zc, yc, ranks):
    mu, sd = Zsrc.mean(0, keepdims=True), Zsrc.std(0, keepdims=True) + 1e-8
    Ss, Sc = (Zsrc - mu) / sd, (Zc - mu) / sd
    ctr = Ss.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ss - ctr, full_matrices=False)
    ysrc = ysrc.astype(int)
    out = []
    for r in ranks:
        rr = min(r, Vt.shape[0])
        clf = LogisticRegression(max_iter=2000).fit((Ss - ctr) @ Vt[:rr].T, ysrc)
        out.append(float(roc_auc_score(yc, clf.predict_proba((Sc - ctr) @ Vt[:rr].T)[:, 1])))
    return out


def _within(Z, y, ranks, seed=0):
    """Leak-free within-split curve (standardize+PCA fit on TRAIN folds only),
    reusing analyze_rank_within_split._within_split_curve."""
    Zb, yb = _balance(Z, y)
    _, cur = _within_split_curve(Zb, yb, ranks, n_splits=5, seed=seed)
    return [cur["mean"][r] for r in ranks if r in cur["mean"]]


def _deg_within(d, y, seed=0):
    y = y.astype(int); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    rng = np.random.RandomState(seed); m = min(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, m, False), rng.choice(neg, m, False)])
    db, yb = d[idx], y[idx]; aucs = []
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=seed).split(db, yb):
        clf = LogisticRegression(max_iter=2000).fit(db[tri], yb[tri])
        aucs.append(roc_auc_score(yb[tei], clf.predict_proba(db[tei])[:, 1]))
    return float(np.mean(aucs))


def _resid(Z, d):
    X = np.hstack([d, np.ones((len(d), 1))])
    B, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    return Z - X @ B


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=1500, help="optimizer STEPS (full-batch)")
    ap.add_argument("--ratio", type=float, default=0.8, help="S2 keep-ratio (drugs)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--transfer-k", type=int, default=80, help="pseudo-emerging drugs / draw")
    ap.add_argument("--transfer-draws", type=int, default=6)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}", flush=True)

    kg = MergedKG.from_parquet(DEFAULT_NODES_PATH, DEFAULT_EDGES_PATH)
    n = kg.n_nodes; id2i = kg.id_to_idx
    logdeg = np.log1p(kg.degree.astype(np.float64))
    x = np.zeros((n, N_TYPES + 1), dtype=np.float32)
    x[np.arange(n), kg.type_id.astype(np.int64)] = 1.0
    x[:, N_TYPES] = ((logdeg - logdeg.mean()) / (logdeg.std() + 1e-8)).astype(np.float32)
    x = torch.from_numpy(x).to(device)
    bio_ei, bio_et, num_relations, _, _ = build_relation_edges(DEFAULT_EDGES_PATH, id2i)
    bio_ei = bio_ei.to(device); bio_et = bio_et.to(device)
    ddi_rel = num_relations; aug_nr = num_relations + 1

    _s2 = (ROOT / "Code/data/ddi_unified/binary_cls/drugbank_latest_partial"
           / "inductive" / "S2" / args.fold)
    tr = pd.read_parquet(_s2 / "train.parquet")          # FULL official S2 train
    cold_val = pd.read_parquet(_s2 / "val.parquet")      # cold VAL (selection)
    cold_test = pd.read_parquet(_s2 / "test.parquet")    # cold TEST (reporting only)
    ia_v, ib_v, yv = _pairs_to_idx(cold_val, id2i)
    ia_c, ib_c, yc = _pairs_to_idx(cold_test, id2i)
    deg_c = np.vstack([logdeg[ia_c], logdeg[ib_c]]).T.astype(np.float32)

    def _map_pairs(df):
        a = df["drug_a_id"].astype(str).map(id2i); b = df["drug_b_id"].astype(str).map(id2i)
        ok = a.notna() & b.notna()
        return np.stack([a[ok].to_numpy(np.int64), b[ok].to_numpy(np.int64)], axis=1)
    ddi_pos = _map_pairs(tr[tr["y_bin"] == 1])           # DDI facts + emerging targets
    train_neg = _map_pairs(tr[tr["y_bin"] == 0])         # dataset negatives (EmerGNN-matched)
    train_drugs = np.unique(ddi_pos)
    ddi_deg = np.bincount(ddi_pos.reshape(-1), minlength=n).astype(np.float64)
    print(f"[data] train_drugs={len(train_drugs)} train_pos={len(ddi_pos)} "
          f"train_neg={len(train_neg)} cold={len(yc)} aug_relations={aug_nr}", flush=True)

    model = RGCNPairModel(x.shape[1], args.hidden, args.bottleneck, args.dropout,
                          num_relations=aug_nr, num_bases=args.num_bases,
                          num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT / "Code" / "runs" / f"{ts}__train_rgcn_rank_s2__seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # fixed eval KG (all train DDI facts) for cold-val + cold extraction
    eval_ei, eval_et = _aug_graph(bio_ei, bio_et, ddi_pos, ddi_rel, device)

    @torch.no_grad()
    def _cold_val_auc():   # selection on cold VAL, never on cold TEST
        model.eval()
        h = model.encode(x, eval_ei, eval_et)
        z = model.pair_z(h, torch.as_tensor(ia_v, device=device), torch.as_tensor(ib_v, device=device))
        p = torch.sigmoid(model.out(z).squeeze(-1)).cpu().numpy()
        return roc_auc_score(yv, p)

    best_auc, best_state, bad = -1.0, None, 0
    for step in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed * 100000 + step)
        fact, tgt_pos, removed = _s2_split(ddi_pos, train_drugs, args.ratio, rng)
        if len(tgt_pos) < 20:
            continue
        # EmerGNN-matched negatives: subsample dataset train negatives to target count
        nidx = rng.choice(len(train_neg), size=min(len(tgt_pos), len(train_neg)), replace=False)
        tgt_neg = train_neg[nidx]
        m = min(len(tgt_pos), len(tgt_neg))
        pp = tgt_pos[:m]; nn_ = tgt_neg[:m]
        s_ia = np.concatenate([pp[:, 0], nn_[:, 0]]); s_ib = np.concatenate([pp[:, 1], nn_[:, 1]])
        ys = np.concatenate([np.ones(m), np.zeros(m)]).astype(np.float32)
        aug_ei, aug_et = _aug_graph(bio_ei, bio_et, fact, ddi_rel, device)
        model.train(); opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(x, aug_ei, aug_et,
                              torch.as_tensor(s_ia, device=device), torch.as_tensor(s_ib, device=device))
            loss = F.binary_cross_entropy_with_logits(logits, torch.as_tensor(ys, device=device))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == args.epochs:
            ca = _cold_val_auc()
            print(f"[step {step}/{args.epochs}] loss={loss.item():.4f} cold_val_auc={ca:.4f}", flush=True)
            if ca > best_auc:
                best_auc, bad = ca, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += args.eval_every
                if bad >= args.patience:
                    print(f"[early-stop] no cold_val improvement ~{args.patience} steps", flush=True); break

    model.load_state_dict(best_state); model.eval()
    print(f"[select] cold-best cold_val_auc={best_auc:.4f}", flush=True)

    # -- cold extraction on the eval KG (leak-free: cold drugs unseen) --------
    with torch.no_grad():
        h = model.encode(x, eval_ei, eval_et)
        Zc = model.pair_z(h, torch.as_tensor(ia_c, device=device), torch.as_tensor(ib_c, device=device)).cpu().numpy().astype(np.float32)
        lc = model.out(torch.as_tensor(Zc, device=device)).squeeze(-1).cpu().numpy()
    head_cold = roc_auc_score(yc, 1 / (1 + np.exp(-lc)))

    # -- within-cold + degree (oracle, primary) ------------------------------
    wc = _within(Zc, yc, RANKS)
    do = _deg_within(deg_c, yc)
    wcr = _within(_resid(Zc, deg_c), yc, RANKS)

    # -- sparse-source transfer (deployable, supporting; multi-draw) ---------
    tc_draws, tcr_draws, per_draw = [], [], []
    for d in range(args.transfer_draws):
        rng = np.random.default_rng(args.seed + 1000 + d)
        P = rng.choice(train_drugs, size=min(args.transfer_k, len(train_drugs)), replace=False)
        inP_a = np.isin(ddi_pos[:, 0], P); inP_b = np.isin(ddi_pos[:, 1], P)
        fact_pr = ddi_pos[(~inP_a) & (~inP_b)]
        assert int(np.isin(fact_pr.reshape(-1), P).sum()) == 0, "leak: P in fact"
        srcpos = np.unique(np.sort(ddi_pos[inP_a & inP_b], axis=1), axis=0)
        srcpos = srcpos[srcpos[:, 0] != srcpos[:, 1]]
        srcneg = _sample_neg(P, {(int(min(u, v)), int(max(u, v))) for u, v in srcpos}, len(srcpos), rng)
        nb = min(len(srcpos), len(srcneg))
        if nb < 100:
            print(f"[draw {d}] balanced n={nb} (<100); skip", flush=True); continue
        sp = srcpos[:nb]; sn = srcneg[:nb]
        s_ia = np.concatenate([sp[:, 0], sn[:, 0]]); s_ib = np.concatenate([sp[:, 1], sn[:, 1]])
        ys = np.concatenate([np.ones(nb), np.zeros(nb)]).astype(np.float32)
        ds_ = np.vstack([logdeg[s_ia], logdeg[s_ib]]).T.astype(np.float32)
        g_ei, g_et = _aug_graph(bio_ei, bio_et, fact_pr, ddi_rel, device)
        with torch.no_grad():
            hg = model.encode(x, g_ei, g_et)
            Zs = model.pair_z(hg, torch.as_tensor(s_ia, device=device), torch.as_tensor(s_ib, device=device)).cpu().numpy().astype(np.float32)
            Zcg = model.pair_z(hg, torch.as_tensor(ia_c, device=device), torch.as_tensor(ib_c, device=device)).cpu().numpy().astype(np.float32)
            lcg = model.out(torch.as_tensor(Zcg, device=device)).squeeze(-1).cpu().numpy()
        ch = roc_auc_score(yc, 1 / (1 + np.exp(-lcg)))
        tc = _transfer(Zs, ys, Zcg, yc, RANKS)
        tcr = _transfer(_resid(Zs, ds_), ys, _resid(Zcg, deg_c), yc, RANKS)
        tc_draws.append(tc); tcr_draws.append(tcr)
        frac = (len(ddi_pos) - len(fact_pr)) / len(ddi_pos)
        per_draw.append({"draw": d, "n_bal": int(nb), "frac_ddi_removed": round(frac, 3),
                         "cold_head_on_Gprobe": round(ch, 4), "transfer_peak": round(max(tc), 4),
                         "transfer_resid_peak": round(max(tcr), 4)})
        print(f"[draw {d}] n={nb} frac_removed={frac:.3f} cold_head(Gprobe)={ch:.3f} "
              f"transfer peak={max(tc):.3f}@rank{RANKS[int(np.argmax(tc))]} resid={max(tcr):.3f}", flush=True)

    tc_m = np.mean(tc_draws, 0).tolist() if tc_draws else []
    tcr_m = np.mean(tcr_draws, 0).tolist() if tc_draws else []
    np.savez_compressed(run_dir / "pilot_Z.npz", Z_cold=Zc, y_cold=yc.astype(np.float32),
                        logit_cold=lc, deg_cold=deg_c,
                        deg_ddi_cold=np.vstack([np.log1p(ddi_deg[ia_c]), np.log1p(ddi_deg[ib_c])]).T.astype(np.float32))
    results = {"model": "R-GCN-S2", "cold_best_val": round(best_auc, 4), "head_cold": round(head_cold, 4),
               "ranks": RANKS, "within_cold": wc, "degree_only": round(do, 4),
               "within_cold_residual": wcr, "transfer_mean": tc_m, "transfer_resid_mean": tcr_m,
               "transfer_draws": len(tc_draws), "per_draw": per_draw}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("\n### R-GCN S2 (cold n=%d)" % len(yc))
    print(" head_cold=%.3f degree_only=%.3f" % (head_cold, do))
    print(" WITHIN cold        :", [round(v, 3) for v in wc], "max=%.3f" % max(wc))
    print(" WITHIN cold RESID  :", [round(v, 3) for v in wcr], "max=%.3f" % max(wcr))
    if tc_m:
        print(" TRANSFER cold      :", [round(v, 3) for v in tc_m], "peak=%.3f@rank%d" % (max(tc_m), RANKS[int(np.argmax(tc_m))]))
        print(" TRANSFER cold RESID:", [round(v, 3) for v in tcr_m], "peak=%.3f" % max(tcr_m))
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()

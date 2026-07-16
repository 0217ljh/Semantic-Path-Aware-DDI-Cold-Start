"""R-GCN MULTI-CLASS (DDI-type) under the locked S2 protocol — the artifact
control for the low-rank claim.

WHY: the binary low-rank result (transfer peaks at rank ~1) is partly a 1-D
output-head artifact (a binary scorer needs only 1 discriminative direction).
Multi-class has a K-dim head, so the model needs ~K directions; if the deployable
TRANSFER curve still saturates at rank << K (and << dim), the low-rankness is NOT
just head dimensionality -> the low-rank claim is much stronger. Cold-start for
multi-type DDI is still UNSEEN-DRUG (CSMDDI S2: both drugs new, predict the type),
so we reuse the S2 protocol verbatim.

PROTOCOL (locked, = binary R-GCN S2, minus negatives):
- Task: K-way DDI-type classification (y_cls_train), NO negatives (every pair
  interacts). cold eval DROPS y_cls_train==-1 (types unseen in train).
- TRAIN: each step, mark ~20% train drugs emerging, remove ALL their DDI edges;
  targets = emerging-emerging pairs with their type; softmax CE. Message graph =
  bio KG + kept-kept DDI facts as TYPED edges (relation = n_base_rel + class,
  matching EmerGNN multi_cls). Per-step reshuffle.
- SELECT: cold-val macro-F1.
- EVAL/EXTRACT (leak-free): cold scored on eval KG = bio + ALL train typed DDI
  (cold drugs unseen -> own edge absent). Z = pre-scorer pair rep (last layer).
- METRICS: accuracy + macro-F1 + macro-AUROC (head); rank curves = macro-AUROC.
- CRITERION: transfer macro-AUROC saturates at rank << K and << dim -> low-rank
  is not a head-dimensionality artifact.

Run:
  python Code/scripts/train_rgcn_multicls_s2.py --fold fold0 --epochs 1500 --amp
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize


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

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


class RGCNMultiClass(RGCNPairModel):
    """Binary R-GCN with a K-way head (out overridden to Linear(bottleneck, K))."""

    def __init__(self, in_dim, hidden, bottleneck, dropout, num_relations,
                 num_bases, n_classes, num_layers=2):
        super().__init__(in_dim, hidden, bottleneck, dropout, num_relations,
                         num_bases, num_layers)
        self.out = nn.Linear(bottleneck, n_classes)   # override 1 -> K

    def forward(self, x, edge_index, edge_type, ia, ib):
        h = self.encode(x, edge_index, edge_type)
        z = self.pair_z(h, ia, ib)
        return self.out(z), z                          # (n, K), z


# --------------------------------------------------------------------------- #
def _s2_split(pairs, cls, drugs, ratio, rng):
    n_remove = len(drugs) - int(len(drugs) * ratio)
    removed = rng.choice(drugs, size=n_remove, replace=False)
    inR_a = np.isin(pairs[:, 0], removed); inR_b = np.isin(pairs[:, 1], removed)
    fm = (~inR_a) & (~inR_b); tm = inR_a & inR_b
    return pairs[fm], cls[fm], pairs[tm], cls[tm], removed


def _aug_typed(bio_ei, bio_et, pairs, cls, n_base_rel, device):
    """bio edges + TYPED DDI edges (relation = n_base_rel + class, both directions)."""
    if len(pairs) == 0:
        return bio_ei, bio_et
    a, b = pairs[:, 0], pairs[:, 1]
    rel = (n_base_rel + cls).astype(np.int64)
    di = torch.from_numpy(np.vstack([np.concatenate([a, b]),
                                     np.concatenate([b, a])]).astype(np.int64)).to(device)
    dt = torch.from_numpy(np.concatenate([rel, rel])).to(device)
    return torch.cat([bio_ei, di], dim=1), torch.cat([bio_et, dt])


def _macro_auc(y, proba, classes):
    """One-vs-rest macro AUROC over `classes` (proba columns aligned to classes).
    Skips classes absent (or fully present) in y."""
    Y = label_binarize(y, classes=list(classes))
    aucs = []
    for k in range(len(classes)):
        s = Y[:, k].sum()
        if 0 < s < len(y):
            aucs.append(roc_auc_score(Y[:, k], proba[:, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def _topk_acc(y, proba, classes, k=5):
    """top-k accuracy: is the true label among the k highest-proba classes."""
    classes = np.asarray(classes)
    kk = min(k, proba.shape[1])
    topk = classes[np.argsort(-proba, axis=1)[:, :kk]]
    return float(np.mean([yi in row for yi, row in zip(y, topk)]))


def _transfer_mc(Zsrc, ysrc, Zc, yc, ranks):
    """PCA+multinomial-logreg fit on SOURCE, (macro-AUROC, top5-acc) on cold, per
    rank. Returns (auc_list, top5_list); nan lists if degenerate class support."""
    nan = [float("nan")] * len(ranks)
    if len(Zsrc) == 0 or len(Zc) == 0 or np.unique(ysrc).size < 2 or np.unique(yc).size < 2:
        return nan, nan
    mu, sd = Zsrc.mean(0, keepdims=True), Zsrc.std(0, keepdims=True) + 1e-8
    Ss, Sc = (Zsrc - mu) / sd, (Zc - mu) / sd
    ctr = Ss.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ss - ctr, full_matrices=False)
    auc, top5 = [], []
    for r in ranks:
        rr = min(r, Vt.shape[0])
        clf = LogisticRegression(max_iter=2000).fit((Ss - ctr) @ Vt[:rr].T, ysrc)
        proba = clf.predict_proba((Sc - ctr) @ Vt[:rr].T)
        auc.append(_macro_auc(yc, proba, clf.classes_))
        top5.append(_topk_acc(yc, proba, clf.classes_, 5))
    return auc, top5


def _within_mc(Z, y, ranks, seed=0, n_splits=5):
    """Leak-free within-split macro-AUROC: standardize+PCA fit on TRAIN folds only,
    multinomial logreg, macro-AUROC on held-out fold. Restricts to classes with
    >= n_splits samples (StratifiedKFold requirement)."""
    vc = np.bincount(y, minlength=int(y.max()) + 1)
    keep = np.isin(y, np.where(vc >= n_splits)[0])
    Z, y = Z[keep], y[keep]
    if len(y) == 0 or np.unique(y).size < 2:
        return [float("nan")] * len(ranks)
    res = {r: [] for r in ranks}
    for tri, tei in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(Z, y):
        Ztr, Zte, ytr, yte = Z[tri], Z[tei], y[tri], y[tei]
        mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
        Str, Ste = (Ztr - mu) / sd, (Zte - mu) / sd
        ctr = Str.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Str - ctr, full_matrices=False)
        for r in ranks:
            rr = min(r, Vt.shape[0])
            clf = LogisticRegression(max_iter=2000).fit(
                (Str - ctr) @ Vt[:rr].T, ytr)
            proba = clf.predict_proba((Ste - ctr) @ Vt[:rr].T)
            res[r].append(_macro_auc(yte, proba, clf.classes_))
    return [float(np.nanmean(res[r])) for r in ranks]


def _deg_mc(dtr, ytr, dc, yc):
    clf = LogisticRegression(max_iter=2000).fit(dtr, ytr)
    return _macro_auc(yc, clf.predict_proba(dc), clf.classes_)


def _head_metrics(logits, y, classes):
    """accuracy + macro-F1 + macro-AUROC of the model's K-way head on cold."""
    proba = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    pred = proba.argmax(1)
    acc = accuracy_score(y, pred)
    mf1 = f1_score(y, pred, average="macro", labels=classes, zero_division=0)
    mauc = _macro_auc(y, proba, list(range(proba.shape[1])))
    top5 = _topk_acc(y, proba, list(range(proba.shape[1])), 5)
    return float(acc), float(mf1), float(mauc), float(top5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--dataset", default="drugbank_latest_partial",
                    help="multi_cls dataset under ddi_unified/multi_cls/ "
                         "(e.g. drugbank_deng = ~65-type, K=56 in S2 train)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-bases", type=int, default=16)
    ap.add_argument("--bottleneck", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--transfer-k", type=int, default=80)
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
    bio_ei, bio_et, num_base_rel, _, _ = build_relation_edges(DEFAULT_EDGES_PATH, id2i)
    bio_ei = bio_ei.to(device); bio_et = bio_et.to(device)

    _mc = (ROOT / "Code/data/ddi_unified/multi_cls" / args.dataset
           / "inductive" / "S2" / args.fold)
    tr = pd.read_parquet(_mc / "train.parquet")
    va = pd.read_parquet(_mc / "val.parquet")
    te = pd.read_parquet(_mc / "test.parquet")

    def _map(df):
        a = df["drug_a_id"].astype(str).map(id2i); b = df["drug_b_id"].astype(str).map(id2i)
        ok = a.notna() & b.notna() & (df["y_cls_train"].to_numpy() >= 0)   # drop unseen-class (-1)
        return (np.stack([a[ok].to_numpy(np.int64), b[ok].to_numpy(np.int64)], axis=1),
                df["y_cls_train"].to_numpy()[ok.to_numpy()].astype(np.int64))
    tr_pairs, tr_cls = _map(tr)
    va_pairs, va_cls = _map(va)
    te_pairs, te_cls = _map(te)
    K = int(tr_cls.max()) + 1                       # class ids are 0..K-1 in train
    assert np.array_equal(np.unique(tr_cls), np.arange(K)), "train class ids not contiguous 0..K-1"
    assert (va_cls >= 0).all() and (va_cls < K).all(), "val class id out of [0,K)"
    assert (te_cls >= 0).all() and (te_cls < K).all(), "test class id out of [0,K)"
    num_relations = num_base_rel + K
    train_drugs = np.unique(tr_pairs)
    ddi_deg = np.bincount(tr_pairs.reshape(-1), minlength=n).astype(np.float64)
    deg_c = np.vstack([logdeg[te_pairs[:, 0]], logdeg[te_pairs[:, 1]]]).T.astype(np.float32)
    deg_v = np.vstack([logdeg[va_pairs[:, 0]], logdeg[va_pairs[:, 1]]]).T.astype(np.float32)
    print(f"[data] K={K} train={len(tr_pairs)} val={len(va_pairs)} test={len(te_pairs)} "
          f"num_relations={num_relations}", flush=True)

    model = RGCNMultiClass(x.shape[1], args.hidden, args.bottleneck, args.dropout,
                           num_relations=num_relations, num_bases=args.num_bases,
                           n_classes=K, num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT / "Code" / "runs" / f"{ts}__train_rgcn_multicls_s2__{args.dataset}__seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # fixed eval KG (all train typed DDI) for val + cold extraction
    eval_ei, eval_et = _aug_typed(bio_ei, bio_et, tr_pairs, tr_cls, num_base_rel, device)

    @torch.no_grad()
    def _val_macro_f1():
        model.eval()
        h = model.encode(x, eval_ei, eval_et)
        lo = model.out(model.pair_z(h, torch.as_tensor(va_pairs[:, 0], device=device),
                                    torch.as_tensor(va_pairs[:, 1], device=device)))
        pred = lo.argmax(1).cpu().numpy()
        return f1_score(va_cls, pred, average="macro", labels=list(range(K)), zero_division=0)

    best, best_state, bad = -1.0, None, 0
    for step in range(1, args.epochs + 1):
        rng = np.random.default_rng(args.seed * 100000 + step)
        # ONE S2 shuffle per step -> fact (both kept) + targets (both emerging)
        fact_p, fact_c, tgt_p, tgt_c, _ = _s2_split(tr_pairs, tr_cls, train_drugs, args.ratio, rng)
        if len(tgt_p) < 20:
            continue
        aug_ei, aug_et = _aug_typed(bio_ei, bio_et, fact_p, fact_c, num_base_rel, device)
        model.train(); opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(x, aug_ei, aug_et,
                              torch.as_tensor(tgt_p[:, 0], device=device),
                              torch.as_tensor(tgt_p[:, 1], device=device))
            loss = F.cross_entropy(logits, torch.as_tensor(tgt_c, device=device))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % args.eval_every == 0 or step == args.epochs:
            vf1 = _val_macro_f1()
            print(f"[step {step}/{args.epochs}] loss={loss.item():.4f} val_macro_f1={vf1:.4f}", flush=True)
            if vf1 > best:
                best, bad = vf1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += args.eval_every
                if bad >= args.patience:
                    print(f"[early-stop] no val_macro_f1 improvement ~{args.patience} steps", flush=True); break

    if best_state is None:
        raise RuntimeError("no checkpoint selected (all steps skipped or eval never ran)")
    model.load_state_dict(best_state); model.eval()
    print(f"[select] cold-best val_macro_f1={best:.4f}", flush=True)

    with torch.no_grad():
        h = model.encode(x, eval_ei, eval_et)
        Zc = model.pair_z(h, torch.as_tensor(te_pairs[:, 0], device=device),
                          torch.as_tensor(te_pairs[:, 1], device=device)).cpu().numpy().astype(np.float32)
        lc = model.out(torch.as_tensor(Zc, device=device)).cpu().numpy()
    acc, mf1, mauc, top5 = _head_metrics(lc, te_cls, list(range(K)))
    print(f"[head cold] accuracy={acc:.4f} macro_f1={mf1:.4f} macro_auroc={mauc:.4f} top5_acc={top5:.4f}", flush=True)

    # within-cold + degree (macro-AUROC vs rank)
    wc = _within_mc(Zc, te_cls, RANKS)
    do = _deg_mc(deg_v, va_cls, deg_c, te_cls)

    # sparse-source transfer (macro-AUROC vs rank, multi-draw)
    tc_draws, tc5_draws, per_draw = [], [], []
    for d in range(args.transfer_draws):
        rng = np.random.default_rng(args.seed + 1000 + d)
        P = rng.choice(train_drugs, size=min(args.transfer_k, len(train_drugs)), replace=False)
        inP_a = np.isin(tr_pairs[:, 0], P); inP_b = np.isin(tr_pairs[:, 1], P)
        fact_p = tr_pairs[(~inP_a) & (~inP_b)]; fact_c = tr_cls[(~inP_a) & (~inP_b)]
        assert int(np.isin(fact_p.reshape(-1), P).sum()) == 0, "leak: P in fact"
        src_p = tr_pairs[inP_a & inP_b]; src_c = tr_cls[inP_a & inP_b]
        keep = np.isin(te_cls, np.unique(src_c))
        if (len(src_p) < 150 or np.unique(src_c).size < 2
                or keep.sum() == 0 or np.unique(te_cls[keep]).size < 2):
            print(f"[draw {d}] degenerate (src={len(src_p)} src_cls={np.unique(src_c).size} "
                  f"cold_kept={int(keep.sum())}); skip", flush=True); continue
        g_ei, g_et = _aug_typed(bio_ei, bio_et, fact_p, fact_c, num_base_rel, device)
        with torch.no_grad():
            hg = model.encode(x, g_ei, g_et)
            Zs = model.pair_z(hg, torch.as_tensor(src_p[:, 0], device=device),
                              torch.as_tensor(src_p[:, 1], device=device)).cpu().numpy().astype(np.float32)
            Zcg = model.pair_z(hg, torch.as_tensor(te_pairs[:, 0], device=device),
                               torch.as_tensor(te_pairs[:, 1], device=device)).cpu().numpy().astype(np.float32)
        tc, tc5 = _transfer_mc(Zs, src_c, Zcg[keep], te_cls[keep], RANKS)
        tc_draws.append(tc); tc5_draws.append(tc5)
        frac = (len(tr_pairs) - len(fact_p)) / len(tr_pairs)
        peak = float(np.nanmax(tc)) if not np.all(np.isnan(tc)) else float("nan")
        rk = RANKS[int(np.nanargmax(tc))] if not np.all(np.isnan(tc)) else -1
        per_draw.append({"draw": d, "src": int(len(src_p)), "cold_kept": int(keep.sum()),
                         "src_classes": int(np.unique(src_c).size),
                         "frac_ddi_removed": round(frac, 3), "transfer_peak": round(peak, 4)})
        print(f"[draw {d}] src={len(src_p)} cold_kept={int(keep.sum())} src_cls={np.unique(src_c).size} "
              f"frac_removed={frac:.3f} transfer peak={peak:.3f}@rank{rk}", flush=True)

    tc_m = np.nanmean(tc_draws, 0).tolist() if tc_draws else []
    tc5_m = np.nanmean(tc5_draws, 0).tolist() if tc5_draws else []
    results = {"model": "R-GCN-multicls-S2", "K": K, "dim": args.bottleneck,
               "cold_best_val_macro_f1": round(best, 4),
               "head_cold": {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
                             "macro_auroc": round(mauc, 4), "top5_acc": round(top5, 4)},
               "ranks": RANKS, "within_cold_macroauc": wc, "degree_only_macroauc": round(do, 4),
               "transfer_macroauc_mean": tc_m, "transfer_top5_mean": tc5_m,
               "transfer_draws": len(tc_draws), "per_draw": per_draw}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print("\n### R-GCN MULTI-CLASS S2 (K=%d, dim=%d, cold n=%d)" % (K, args.bottleneck, len(te_cls)))
    print(" head: acc=%.3f macro_f1=%.3f macro_auroc=%.3f top5=%.3f | degree_only=%.3f" % (acc, mf1, mauc, top5, do))
    _wmax = np.nanmax(wc) if (wc and not np.all(np.isnan(wc))) else float("nan")
    print(" WITHIN cold (macroAUC):", [round(v, 3) for v in wc], "max=%.3f" % _wmax)
    if tc_m and not np.all(np.isnan(tc_m)):
        print(" TRANSFER macroAUC:", [round(v, 3) for v in tc_m],
              "peak=%.3f@rank%d" % (np.nanmax(tc_m), RANKS[int(np.nanargmax(tc_m))]))
        if tc5_m and not np.all(np.isnan(tc5_m)):
            print(" TRANSFER top5-acc:", [round(v, 3) for v in tc5_m],
                  "peak=%.3f@rank%d" % (np.nanmax(tc5_m), RANKS[int(np.nanargmax(tc5_m))]))
        print(" -> criterion: transfer saturates at rank << K=%d and << dim=%d ?" % (K, args.bottleneck))
    else:
        print(" TRANSFER: no valid (non-NaN) draws", flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    main()

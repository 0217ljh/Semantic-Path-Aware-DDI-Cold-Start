"""EmerGNN FACT-protocol rank analysis (rspmm-preserved), the path-paradigm twin
of train_rgcn_rank_fact_protocol.py.

Design (codex-reviewed 2026-07-05):
- TRAIN via the baseline FIXED fit: shuffle_train_mode="FIXED" keeps a SINGLE
  deterministic 80/20 fact/target split of the seen-seen train DDI positives.
  fact (80%) are DDI message edges (seen drugs stay DDI-enriched, high-degree
  query seeds -> no S2-style warm explosion); targets (20%) are supervised /
  held out of the graph. rspmm kernel untouched.
- EXTRACT on the SAME deterministic FACT KG (bio + fact_pos), NOT the default
  eval_kg (= bio + ALL train DDI), because a predicted pair whose own edge is a
  message edge gives enc_ht a trivial h->t direct path = strong leakage that
  breaks the transfer probe. We rebuild the FIXED split, build a fact-KG sparse
  tensor (build_sparse_kg_from_triplets with n_base_rel_with_ddi, dummy DDI slot
  = n_kg_rel), and extract with a custom batched enc_ht (NOT predict_proba, which
  hardwires _eval_kg).
- Transfer-probe SOURCE (Z_train) = FIXED targets (20% positives, edge-absent) +
  train negatives (y_bin==0, never in graph). warm = warm_test (held-out seen-
  seen), cold = cold_test (S2 unseen). All edge-absent -> clean.
- Select warm-best across per-epoch checkpoints (like the R-GCN pilot's early
  stop on warm_val); the thesis is about the WARM-optimal model.
- Saves pilot_Z.npz (Z_/raw_/y_/logit_/deg_) so analyze_rank_sufficiency.py
  (transfer) and analyze_rank_within_split.py run unchanged.

Run:
  CUDA_HOME=/home/lakestar_ljh/miniconda3 EMERGNN_BACKEND=rspmm EMERGNN_KG_SCOPE=full \
    PYTHONUNBUFFERED=1 python -u Code/scripts/extract_emergnn_fact_protocol_rank.py --epochs 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
os.environ.setdefault("EMERGNN_BACKEND", "rspmm")
os.environ.setdefault("EMERGNN_KG_SCOPE", "full")

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_dataset  # noqa: E402
from baseline.emergnn._per_mode_rspmm import _PerModeEmerGNN_RSPMM  # noqa: E402
from baseline.emergnn._rspmm_utils import build_sparse_kg_from_triplets  # noqa: E402
from baseline.emergnn.shuffle_utils import shuffle_train  # noqa: E402
from extract_emergnn_multickpt_rank import _three_way, _pairs  # noqa: E402
from analyze_rank_within_split import _within_split_curve, _balance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"
RANKS_SURF = [1, 2, 4, 8, 16, 32, 64, 128]


def _transfer_curve(Ztr, ytr, Zw, yw, Zc, yc, ranks):
    """PCA-on-train + logreg-on-train, applied to warm/cold -> AUROC per rank."""
    mu, sd = Ztr.mean(0, keepdims=True), Ztr.std(0, keepdims=True) + 1e-8
    Str, Sw, Sc = (Ztr - mu) / sd, (Zw - mu) / sd, (Zc - mu) / sd
    center = Str.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Str - center, full_matrices=False)
    ytr = ytr.astype(int); warm, cold = [], []
    for r in ranks:
        clf = LogisticRegression(max_iter=2000, C=1.0).fit((Str - center) @ Vt[:r].T, ytr)
        warm.append(float(roc_auc_score(yw, clf.predict_proba((Sw - center) @ Vt[:r].T)[:, 1])))
        cold.append(float(roc_auc_score(yc, clf.predict_proba((Sc - center) @ Vt[:r].T)[:, 1])))
    return warm, cold


def _within_curve(Z, y, ranks):
    Zb, yb = _balance(Z, y)
    _, cur = _within_split_curve(Zb, yb, ranks, n_splits=3, seed=0)
    return [cur["mean"][r] for r in ranks if r in cur["mean"]]


def _rank_summary(aucs, ranks):
    """r90* = rank to reach 90% of BEST attainable AUROC (codex: normalize to best,
    not full-rank, so inverted/declining curves stay well-defined)."""
    ranks = ranks[:len(aucs)]
    a = np.maximum.accumulate(np.asarray(aucs, dtype=float))  # monotone envelope
    best, base = float(a[-1]), float(a[0])
    denom = best - base
    r90 = ranks[0] if denom <= 1e-6 else next(
        (ranks[i] for i, v in enumerate((a - base) / denom) if v >= 0.90), ranks[-1])
    return {"best": round(best, 4), "full": round(float(aucs[-1]), 4),
            "rank_of_best": int(ranks[int(np.argmax(aucs))]), "r90star": int(r90)}


def _build_small_kg(full_edges_path, drug_ids, out_path, hops):
    """Keep only the k-hop bio subgraph around DDI drugs -> a much smaller/faster
    KG. Seen DDI edges are injected separately (fact-protocol) so 'has DDI edges'
    is preserved. Cached to out_path (baseline _data, __mine)."""
    out_path = Path(out_path)
    if out_path.is_file():
        return out_path, len(pd.read_parquet(out_path))
    e = pd.read_parquet(full_edges_path)
    src, dst = e["src"].astype(str), e["dst"].astype(str)
    keep = src.isin(drug_ids) | dst.isin(drug_ids)
    for _ in range(max(hops - 1, 0)):
        frontier = set(src[keep]) | set(dst[keep])
        keep = keep | src.isin(frontier) | dst.isin(frontier)
    small = e[keep].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    small.to_parquet(out_path, index=False)
    return out_path, len(small)


@torch.no_grad()
def _embed_logit_on_kg(core, ia, ib, kg, batch_size, device):
    """Batched pre-scorer embedding + logit on an EXPLICIT sparse KG (rspmm
    enc_ht), bypassing predict_proba's hardwired _eval_kg."""
    m = core._model
    m.eval()
    embs, logits = [], []
    for s in range(0, len(ia), batch_size):
        h = torch.as_tensor(ia[s:s + batch_size], device=device)
        t = torch.as_tensor(ib[s:s + batch_size], device=device)
        emb = m.enc_ht(h, t, kg)
        logit = m.Wr(emb).squeeze(-1)
        embs.append(emb.cpu().numpy().astype(np.float32))
        logits.append(logit.cpu().numpy().astype(np.float32))
    return np.concatenate(embs), np.concatenate(logits)


def _rebuild_fact_kg(core, tr, device):
    """Reconstruct the deterministic FIXED fact split the model trained on and
    build its sparse KG. Returns (fact_kg_sparse, target_pos_int (N,2))."""
    e2i = core._entity2id
    tr_pos = tr[tr["y_bin"] == 1]
    a = tr_pos["drug_a_id"].astype(str).map(e2i)
    b = tr_pos["drug_b_id"].astype(str).map(e2i)
    ok = a.notna() & b.notna()
    a = a[ok].to_numpy(np.int64); b = b[ok].to_numpy(np.int64)
    n_kg_rel = int(core._n_base_rel)                      # dummy DDI relation slot
    train_ddi_int = np.stack([a, b, np.full(len(a), n_kg_rel, dtype=np.int64)], axis=1)
    fact_kg_tri, targets = shuffle_train(
        train_ddi_int, np.asarray(core._kg_triplets, dtype=np.int64),
        "FIXED", ratio=core.shuffle_ratio)
    fact_kg = build_sparse_kg_from_triplets(
        np.asarray(fact_kg_tri, dtype=np.int64), int(core._n_ent),
        int(core._n_base_rel_with_ddi), device=device)
    return fact_kg, np.asarray(targets, dtype=np.int64)[:, :2]


class _LeanCore(_PerModeEmerGNN_RSPMM):
    """Lean checkpoints: save ONLY the ~0.42MB trainable weights, dropping the
    716MB fixed Morgan buffer 'ent_feat' (and the KG). Training protocol UNCHANGED
    (inherits fit / FIXED / fact-KG -- seen DDI edges stay in the graph). The scan
    loads these weights into the LIVE core._model, which already holds ent_feat
    (Morgan) + the fact KG. ~1.6GB -> <1MB per checkpoint."""

    def save(self, path):
        # TrainProgress.step_count RESETS each epoch (train_progress.py:106), so the
        # parent's "checkpoint-step100" name COLLIDES across epochs and later epochs
        # overwrite earlier ones -> the trajectory is lost. Write to a globally
        # monotonic, unique dir so every save survives.
        self._nsave = getattr(self, "_nsave", 0) + 1
        p = Path(path)
        out = p.with_name(f"{p.name}__n{self._nsave:05d}")
        out.mkdir(parents=True, exist_ok=True)
        sd = {k: v.cpu() for k, v in self._model.state_dict().items() if k != "ent_feat"}
        torch.save(sd, out / "trainable.pt")

    def _validate(self, val):
        """Per-epoch: print BOTH warm_val and cold_val AUROC (on the fact KG) to
        the log. Returns cold_val AUROC (what the fit tracks for best/scheduler).
        Requires _tr_df / _warm_val_df / _cold_val_df set on the core before fit."""
        if getattr(self, "_fact_kg_cache", None) is None:
            self._fact_kg_cache, _ = _rebuild_fact_kg(self, self._tr_df, self.device)
        e2i = self._entity2id

        def _auc(df):
            ia, ib, y = _pairs(df, e2i)
            _, logit = _embed_logit_on_kg(self, ia, ib, self._fact_kg_cache, self.batch_size, self.device)
            return float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit))))

        wa, ca = _auc(self._warm_val_df), _auc(self._cold_val_df)
        print(f"    [val warm/cold] warm_val_auc={wa:.4f}  cold_val_auc={ca:.4f}", flush=True)
        return ca


def _load_trainable(core, cp):
    """Load a lean checkpoint's trainable weights INTO core._model in place
    (ent_feat / Morgan kept; strict=False allows the missing ent_feat key).
    Asserts the ONLY missing key is ent_feat and there are no unexpected keys,
    so a mismatched/corrupt checkpoint fails loudly instead of loading partially."""
    sd = torch.load(Path(cp) / "trainable.pt", map_location=core.device)
    missing, unexpected = core._model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys in {cp}: {unexpected}"
    extra_missing = set(missing) - {"ent_feat"}
    assert not extra_missing, f"missing (non-ent_feat) keys in {cp}: {extra_missing}"
    return core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ddi800")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--shuffle-ratio", type=float, default=0.8,
                    help="fraction of seen-seen train positives kept as DDI facts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=0,
                    help="checkpoint every N optimizer steps (0 = per-epoch). Use a small "
                         "--epochs with this to finely resolve cold's early peak (~1.6GB/ckpt).")
    ap.add_argument("--auc-sample", type=int, default=2000)
    ap.add_argument("--train-sample", type=int, default=12000)
    ap.add_argument("--kg-hops", type=int, default=0,
                    help="0 = full merged KG; k>0 = keep only the k-hop bio subgraph around "
                         "DDI drugs (much faster; seen DDI edges are added separately as facts).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda")

    leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), args.dataset, "binary",
                       args.split, args.fold)
    tr, warm_val, warm_test = _three_way(leaf.train, args.seed)
    cold_val, cold_test = leaf.val, leaf.test
    print(f"[run] tr={tr.shape} warm_val={warm_val.shape} warm_test={warm_test.shape} "
          f"cold_val={cold_val.shape} cold_test={cold_test.shape}", flush=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__extract_emergnn_fact_protocol_rank__{args.dataset}_{args.split}_{args.fold}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.kg_hops > 0:
        drug_ids = set(pd.concat([leaf.train["drug_a_id"], leaf.train["drug_b_id"],
                                  leaf.val["drug_a_id"], leaf.val["drug_b_id"],
                                  leaf.test["drug_a_id"], leaf.test["drug_b_id"]]).astype(str))
        small_out = (ROOT / "Code/baseline/emergnn/_data" /
                     f"edges__small_h{args.kg_hops}__{args.dataset}_{args.split}_{args.fold}__mine.parquet")
        merged_kg, n_small = _build_small_kg(
            Path(leaf.resources.kg.source) / MERGED_EDGES, drug_ids, small_out, args.kg_hops)
        print(f"[small-kg] hops={args.kg_hops} -> {n_small} bio edges (subgraph around DDI drugs)",
              flush=True)
    else:
        merged_kg = Path(leaf.resources.kg.source) / MERGED_EDGES
    # --save-steps N > 0: checkpoint every N optimizer steps (FINE, to catch
    # cold's early-epoch peak) instead of per-epoch. WARNING: each EmerGNN
    # checkpoint is ~1.6GB (redundant fixed Morgan matrix + KG triplets), so keep
    # N large-ish and --epochs small (e.g. --epochs 1 --save-steps 50 -> ~7 ckpts
    # ~11GB). save_total_limit set high so the EARLY (peak) checkpoints survive.
    _step_ckpt = args.save_steps > 0
    _save_strategy = "steps" if _step_ckpt else "epoch"
    _save_limit = (args.epochs * 400 // max(args.save_steps, 1) + 5) if _step_ckpt else args.epochs + 2
    core = _LeanCore(   # lean checkpoints (<1MB each); training protocol unchanged
        n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=args.batch_size,
        n_epochs=args.epochs, device="cuda", backbone_kg_source="merged",
        merged_kg_path=merged_kg, weight_decay=1e-8, shuffle_train_mode="FIXED",
        shuffle_ratio=args.shuffle_ratio, log_step_every=50, eval_strategy="epoch",
        save_strategy=_save_strategy, save_steps=(args.save_steps or 500),
        save_total_limit=_save_limit, load_best_model_at_end=True, run_dir=run_dir)

    # data for the per-epoch warm/cold _validate override (subsampled for speed)
    core._tr_df = tr
    core._warm_val_df = warm_val.sample(min(len(warm_val), args.auc_sample),
                                        random_state=args.seed).reset_index(drop=True)
    core._cold_val_df = cold_val.sample(min(len(cold_val), args.auc_sample),
                                        random_state=args.seed).reset_index(drop=True)

    ds = make_dataset(tr, cold_val, leaf.resources)   # fit val -> our _validate prints warm+cold
    t0 = time.time()
    core.fit(ds, ds)
    print(f"[fit] FIXED done in {time.time()-t0:.0f}s (rspmm, dense-query -> no explosion expected)",
          flush=True)

    # -- rebuild the FIXED fact KG once (same split the model trained on) -----
    fact_kg, target_pos = _rebuild_fact_kg(core, tr, device)
    e2i = core._entity2id
    logdeg = np.log1p(np.bincount(
        np.concatenate([np.asarray(core._kg_triplets)[:, 0], np.asarray(core._kg_triplets)[:, 1]]),
        minlength=int(core._n_ent)).astype(np.float64))
    print(f"[fact-kg] rebuilt: fact edges kept, targets(pos)={target_pos.shape} "
          f"(these are the clean Z_train positives)", flush=True)

    # clean labeled Z_train = FIXED target positives + train negatives (y_bin==0)
    tr_neg = tr[tr["y_bin"] == 0]
    na = tr_neg["drug_a_id"].astype(str).map(e2i); nb = tr_neg["drug_b_id"].astype(str).map(e2i)
    nok = na.notna() & nb.notna()
    neg_ia = na[nok].to_numpy(np.int64); neg_ib = nb[nok].to_numpy(np.int64)
    train_ia = np.concatenate([target_pos[:, 0], neg_ia])
    train_ib = np.concatenate([target_pos[:, 1], neg_ib])
    train_y = np.concatenate([np.ones(len(target_pos), np.float32), np.zeros(len(neg_ia), np.float32)])

    def _sub(ia, ib, y, n):
        if n >= len(y):
            return ia, ib, y
        idx = np.random.default_rng(args.seed).choice(len(y), n, replace=False)
        return ia[idx], ib[idx], y[idx]

    warm_pk = _pairs(warm_test, e2i)
    cold_pk = _pairs(cold_test, e2i)
    wv_pk = _pairs(warm_val, e2i)

    def _auc_on_fact(ia, ib, y):
        _, logit = _embed_logit_on_kg(core, ia, ib, fact_kg, args.batch_size, device)
        return float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit))))

    # -- PER-EPOCH rank scan (+ warm-best selection) on the fact KG ----------
    # For EACH epoch checkpoint we compute the transfer + within-split rank
    # curves (subsampled, fast) so the warm/cold rank DYNAMICS over epochs are
    # visible per epoch (user 2026-07-05: watch warm r90* climb -> confirm the
    # low warm rank was undertraining, stop when satisfied).
    ckpts = sorted([p for p in (run_dir / "checkpoints").glob("checkpoint-*")
                    if (p / "trainable.pt").is_file()],
                   key=lambda p: int(p.name.rsplit("__n", 1)[-1]))
    wvs = _sub(*wv_pk, args.auc_sample)
    tr_s = _sub(train_ia, train_ib, train_y, 4000)
    wt_s = _sub(*warm_pk, 2000)
    ct_s = _sub(*cold_pk, 2000)
    scores = []
    surface = {"epochs": [], "ranks": RANKS_SURF,
               "transfer_warm": [], "transfer_cold": [], "within_warm": [], "within_cold": []}
    for cp in ckpts:
        _load_trainable(core, cp)                    # weights into live core._model (Morgan+KG kept)
        wa = _auc_on_fact(*wvs)
        Ztr = _embed_logit_on_kg(core, tr_s[0], tr_s[1], fact_kg, args.batch_size, device)[0]
        Zw = _embed_logit_on_kg(core, wt_s[0], wt_s[1], fact_kg, args.batch_size, device)[0]
        Zc = _embed_logit_on_kg(core, ct_s[0], ct_s[1], fact_kg, args.batch_size, device)[0]
        tw, tc = _transfer_curve(Ztr, tr_s[2], Zw, wt_s[2], Zc, ct_s[2], RANKS_SURF)
        ww, wc = _within_curve(Zw, wt_s[2], RANKS_SURF), _within_curve(Zc, ct_s[2], RANKS_SURF)
        scores.append((cp, wa))
        surface["epochs"].append(cp.name)
        for k, v in [("transfer_warm", tw), ("transfer_cold", tc),
                     ("within_warm", ww), ("within_cold", wc)]:
            surface[k].append(v)
        sW, sC = _rank_summary(tw, RANKS_SURF), _rank_summary(tc, RANKS_SURF)
        wW, wC = _rank_summary(ww, RANKS_SURF), _rank_summary(wc, RANKS_SURF)
        # SAME-MODEL contrast: warm and cold both from THIS checkpoint (one probe
        # fit on THIS Z_train, applied to warm & cold). cold reported as best-rank
        # + r90* so a full-rank inversion is never the reported value (user 2026-
        # 07-05). cold being sub-optimal at the warm-best model is the THESIS,
        # not a bug. Per-epoch trajectory lives in surface.json.
        print(f"[epoch-rank] {cp.name}: warm_val={wa:.4f} | "
              f"TRANSFER warm(best={sW['best']} r90*={sW['r90star']}) "
              f"cold(best={sC['best']} full={sC['full']} r90*={sC['r90star']}) | "
              f"WITHIN warm(best={wW['best']} r90*={wW['r90star']}) "
              f"cold(best={wC['best']} r90*={wC['r90star']})", flush=True)
        torch.cuda.empty_cache()
    (run_dir / "surface.json").write_text(json.dumps(surface, indent=2))

    # single warm-best model (by warm_val); pilot_Z extracts warm AND cold from
    # this ONE model -> a controlled same-encoder contrast.
    if scores:
        best_cp, best_wa = max(scores, key=lambda s: s[1])
        _load_trainable(core, best_cp)               # warm-best weights into live model
        print(f"[select] warm-best (single model for pilot_Z) = {best_cp.name} "
              f"warm_val={best_wa:.4f}", flush=True)

    # -- extract train/warm/cold on the fact KG ------------------------------
    trp = _sub(train_ia, train_ib, train_y, args.train_sample)
    packs = {"train": trp, "warm": warm_pk, "cold": cold_pk}
    arrs = {}
    for name, (ia, ib, y) in packs.items():
        emb, logit = _embed_logit_on_kg(core, ia, ib, fact_kg, args.batch_size, device)
        arrs[f"Z_{name}"] = emb; arrs[f"raw_{name}"] = emb
        arrs[f"y_{name}"] = y.astype(np.float32); arrs[f"logit_{name}"] = logit
        arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
        a = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
        print(f"[extract] {name}: n={len(y)} head_auc={a:.4f} emb={emb.shape}", flush=True)
    np.savez_compressed(run_dir / "pilot_Z.npz", **arrs)

    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "model_label": "EmerGNN-FIXED-fact", "epochs": args.epochs,
        "shuffle_ratio": args.shuffle_ratio, "n_target_pos": int(len(target_pos)),
        "per_epoch_warm_val": [(cp.name, round(w, 4)) for cp, w in scores],
        "note": "FACT-protocol: FIXED training (seen keep DDI facts), extract on rebuilt fact KG "
                "(bio+fact_pos), Z_train = FIXED targets + train negs (edge-absent). rspmm preserved. "
                "SAME-MODEL contrast: warm+cold both from the warm-best model; cold reported best-rank "
                "+ r90* (full-rank inversion never the reported value). Per-epoch trajectory in surface.json."},
        indent=2))
    print(f"[done] {run_dir}\nNEXT:\n  python Code/scripts/analyze_rank_sufficiency.py --z "
          f"{run_dir/'pilot_Z.npz'} --key raw --title EmerGNN-fact-transfer\n"
          f"  python Code/scripts/analyze_rank_within_split.py --z {run_dir/'pilot_Z.npz'} "
          f"--key raw --title EmerGNN-fact-within", flush=True)


if __name__ == "__main__":
    main()

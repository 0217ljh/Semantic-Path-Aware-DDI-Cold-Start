"""Lean rank extraction from ALREADY-SAVED EmerGNN per-epoch checkpoints.

The full pipeline (extract_emergnn_multickpt_rank.py) trained 20 epochs fine but
its extraction phase was too slow + near-OOM at batch 128 (enc_ht hidden tensor
= n_ent x B x n_dim = 174967 x 128 x 64 ~= 5.7 GB each). This reuses the saved
checkpoints (NO retraining) and extracts safely:
  - batch 32 (small hiddens, no OOM)
  - torch.cuda.empty_cache() + del after every loaded checkpoint
  - warm_val / cold_val AUROC scan on a SUBSAMPLE (fast) to pick warm-best/cold-best
  - train SUBSAMPLE for the extraction (plenty for PCA + linear probe)
Splits are reproduced with the SAME seed as the training run (default 42).

Run:
  CUDA_HOME=/home/lakestar_ljh/miniconda3 EMERGNN_BACKEND=rspmm EMERGNN_KG_SCOPE=full \
    PYTHONUNBUFFERED=1 python -u Code/scripts/extract_emergnn_from_ckpts.py \
    --run-dir Code/runs/2026-07-05_04-46-08__extract_emergnn_multickpt_rank__ddi800_cold_s2_fold0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Code" / "scripts"))
os.environ.setdefault("EMERGNN_BACKEND", "rspmm")
os.environ.setdefault("EMERGNN_KG_SCOPE", "full")

from data_utils import unified_loader as U  # noqa: E402
from baseline.emergnn._per_mode_rspmm import _PerModeEmerGNN_RSPMM  # noqa: E402
from extract_emergnn_multickpt_rank import _three_way, _pairs, _embed_logit, _ensure_eval_kg  # noqa: E402


def _sub(pk, n, seed):
    ia, ib, y = pk
    if n >= len(y):
        return ia, ib, y
    idx = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
    return ia[idx], ib[idx], y[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default="ddi800")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--auc-sample", type=int, default=1500)
    ap.add_argument("--train-sample", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda")
    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)

    leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), args.dataset, "binary",
                       args.split, args.fold)
    tr, warm_val, warm_test = _three_way(leaf.train, args.seed)
    cold_val, cold_test = leaf.val, leaf.test

    ckpts = sorted((run_dir / "checkpoints").glob("checkpoint-ep*"))
    print(f"[ckpt] {len(ckpts)} checkpoints in {run_dir.name}", flush=True)

    def load_ckpt(cp):
        c = _ensure_eval_kg(_PerModeEmerGNN_RSPMM.load(cp))
        return c

    def auc(c, pk):
        ia, ib, y = pk
        _, logit = _embed_logit(c, ia, ib, args.batch_size, device)
        return float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit))))

    # -- scan: warm_val / cold_val AUROC per checkpoint (subsampled) ----------
    scores = []
    t0 = time.time()
    for cp in ckpts:
        c = load_ckpt(cp)
        e2i = c._entity2id
        wv = _sub(_pairs(warm_val, e2i), args.auc_sample, args.seed)
        cv = _sub(_pairs(cold_val, e2i), args.auc_sample, args.seed)
        wva, cva = auc(c, wv), auc(c, cv)
        scores.append((cp, wva, cva))
        print(f"[scan] {cp.name}: warm_val={wva:.4f} cold_val={cva:.4f} "
              f"(elapsed {time.time()-t0:.0f}s)", flush=True)
        del c
        torch.cuda.empty_cache()

    warm_best = max(scores, key=lambda s: s[1])
    cold_best = max(scores, key=lambda s: s[2])
    print(f"[select] warm-best={warm_best[0].name} warm_val={warm_best[1]:.4f}; "
          f"cold-best={cold_best[0].name} cold_val={cold_best[2]:.4f}", flush=True)

    trip = None
    saved = {}

    def extract(cp, tag):
        c = load_ckpt(cp)
        e2i = c._entity2id
        nonlocal trip
        if trip is None:
            trip = np.asarray(c._eval_kg_triplets, dtype=np.int64)  # load() restores this, not _kg_triplets
        logdeg = np.log1p(np.bincount(np.concatenate([trip[:, 0], trip[:, 1]]),
                                      minlength=int(c._n_ent)).astype(np.float64))
        trp = _sub(_pairs(tr, e2i), args.train_sample, args.seed)
        packs = {"train": trp, "warm": _pairs(warm_test, e2i), "cold": _pairs(cold_test, e2i)}
        arrs = {}
        for name, (ia, ib, y) in packs.items():
            emb, logit = _embed_logit(c, ia, ib, args.batch_size, device)
            arrs[f"Z_{name}"] = emb; arrs[f"raw_{name}"] = emb
            arrs[f"y_{name}"] = y.astype(np.float32); arrs[f"logit_{name}"] = logit
            arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
            a = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
            print(f"[extract:{tag}] {name}: n={len(y)} head_auc={a:.4f}", flush=True)
        np.savez_compressed(run_dir / f"pilot_Z__{tag}.npz", **arrs)
        del c
        torch.cuda.empty_cache()
        return {"ckpt": cp.name}

    saved["warm_best"] = extract(warm_best[0], "warm_best")
    if cold_best[0] == warm_best[0]:
        # same epoch -> reuse the file
        import shutil
        shutil.copy(run_dir / "pilot_Z__warm_best.npz", run_dir / "pilot_Z__cold_best.npz")
        saved["cold_best"] = {"ckpt": cold_best[0].name, "note": "== warm_best"}
    else:
        saved["cold_best"] = extract(cold_best[0], "cold_best")

    (run_dir / "meta_from_ckpts.json").write_text(json.dumps({
        "model_label": "EmerGNN-Morgan", "run_dir": run_dir.name,
        "per_epoch": [(cp.name, round(w, 4), round(c, 4)) for cp, w, c in scores],
        "warm_best": {"ckpt": warm_best[0].name, "warm_val": round(warm_best[1], 4)},
        "cold_best": {"ckpt": cold_best[0].name, "cold_val": round(cold_best[2], 4)},
        "note": "lean re-extraction from saved checkpoints (batch32, empty_cache, subsampled). "
                "cold-best == deployed (fit selected on leaf.val=cold_val). ONE model per npz."}, indent=2))
    print(f"[done] {run_dir}\nNEXT: analyze_rank_sufficiency.py --z {run_dir}/pilot_Z__warm_best.npz "
          f"--key raw --title EmerGNN-Morgan-warmbest  (+ cold_best)", flush=True)


if __name__ == "__main__":
    main()

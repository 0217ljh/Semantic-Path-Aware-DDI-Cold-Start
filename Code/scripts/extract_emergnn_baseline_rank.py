"""Full-KG EmerGNN rank-analysis via the PROVEN-FAST baseline core.

The custom pilot (train_emergnn_rspmm_rank_pilot.py) trains at ~3.8s/step — its
BACKWARD is ~20x slower than the baseline rspmm core (0.166s/step) for a reason we
could not pin down by code inspection (same model, kernel, KG size). So instead of
fighting it, this rides the baseline's fit (EmerGNNUnifiedBinary.fit, 0.166s/step)
to TRAIN, then extracts the pre-scorer pair representation embed=cat([head_hid,
tail_hid]) with a forward-only pass (no slow backward) for the rank analysis.

- train/warm/cold: warm = a seen-drug hold-out of leaf.train (fit trains on the
  rest, selects on warm); cold = leaf.test (S2 unseen drugs). Matches the R-GCN
  rank protocol.
- Feature: the baseline's NATIVE seed for S2 = Morgan fingerprint. So this is the
  MORGAN-seed EmerGNN (codex's sensitivity run / a path-paradigm data point), NOT
  the knowledge-only structural variant. Stated honestly in the output.
- Saves pilot_Z.npz (Z_/raw_/y_/logit_/deg_ per split) → analyze_rank_sufficiency.py.

Run (from project root):
  CUDA_HOME=/home/lakestar_ljh/miniconda3 EMERGNN_BACKEND=rspmm EMERGNN_KG_SCOPE=full \
    PYTHONUNBUFFERED=1 python -u Code/scripts/extract_emergnn_baseline_rank.py --epochs 5
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
from baseline.emergnn.binary_cls.baseline_unified import EmerGNNUnifiedBinary  # noqa: E402


def _split_seen(df: pd.DataFrame, frac: float, seed: int):
    """Hold out `frac` of leaf.train as the SEEN-drug warm set; train on the rest."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    k = int(round(frac * len(df)))
    warm = df.iloc[perm[:k]].reset_index(drop=True)
    tr = df.iloc[perm[k:]].reset_index(drop=True)
    return tr, warm


def _pairs(df: pd.DataFrame, e2i: dict):
    a = df["drug_a_id"].astype(str).map(e2i)
    b = df["drug_b_id"].astype(str).map(e2i)
    bad = ~(a.notna() & b.notna())
    if int(bad.sum()):
        raise ValueError(f"{int(bad.sum())} pairs reference drugs not in EmerGNN KG")
    return a.to_numpy(np.int64), b.to_numpy(np.int64), df["y_bin"].to_numpy(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ddi800")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warm-frac", type=float, default=0.12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda")

    leaf = U.load_leaf(str(ROOT / "Code/data/ddi_unified"), args.dataset, "binary",
                       args.split, args.fold)
    tr, warm = _split_seen(leaf.train, args.warm_frac, args.seed)
    print(f"[run] leaf train={leaf.train.shape} -> tr={tr.shape} warm(held-out seen)={warm.shape}; "
          f"cold(test)={leaf.test.shape}", flush=True)

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}__extract_emergnn_baseline_rank__{args.dataset}_{args.split}_{args.fold}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- train via the baseline's FAST fit ------------------------------------
    # Selection = the OFFICIAL inductive val (leaf.val), per codex 2026-07-05:
    #   * NEVER select on cold/S2-test (leakage).
    #   * the custom held-out-seen `warm` is EVALUATION-ONLY (rank extraction),
    #     NOT the fit val — passing it as val_df broke the baseline's _validate
    #     (it builds negatives via val.splits.val_s2, giving a bogus AUC 0.185).
    # load_best_model_at_end=True (core default) -> after fit, core._model is the
    # leaf.val-BEST checkpoint = the deployed model. ONE model; warm AND cold are
    # both extracted from it (the "same encoder, different input" divergence).
    # Robustness at an EARLY checkpoint = re-run this script with --epochs 3 (a
    # separate single model), NOT mixing checkpoints within one divergence.
    model = EmerGNNUnifiedBinary(n_epochs=args.epochs, device="cuda", run_dir=str(run_dir))
    t0 = time.time()
    model.fit(tr, leaf.val, resources=leaf.resources)
    print(f"[fit] baseline fit done in {time.time()-t0:.0f}s (selected on leaf.val)", flush=True)

    core = model._core
    m = core._model
    e2i = core._entity2id
    kg = core._eval_kg           # baseline's coalesced eval KG (train_ddi + base KG)
    m.eval()

    # per-entity log-degree from the KG triplets (for the degree-only control)
    trip = np.asarray(core._kg_triplets, dtype=np.int64)
    n_ent = int(core._n_ent)
    if len(trip):
        deg = np.bincount(np.concatenate([trip[:, 0], trip[:, 1]]),
                          minlength=n_ent).astype(np.float64)
    else:
        deg = np.zeros(n_ent, dtype=np.float64)
    logdeg = np.log1p(deg)

    packs = {"train": tr, "warm": warm, "cold": leaf.test}
    arrs = {}
    with torch.no_grad():
        for name, df in packs.items():
            ia, ib, y = _pairs(df, e2i)
            embs, logits = [], []
            for s in range(0, len(y), args.batch_size):
                h = torch.as_tensor(ia[s:s + args.batch_size], device=device)
                t = torch.as_tensor(ib[s:s + args.batch_size], device=device)
                emb = m.enc_ht(h, t, kg)          # pre-scorer rep (feat='M': cat([head_hid,tail_hid]))
                logit = m.Wr(emb).squeeze(-1)
                embs.append(emb.cpu().numpy().astype(np.float32))
                logits.append(logit.cpu().numpy().astype(np.float32))
            emb = np.concatenate(embs); logit = np.concatenate(logits)
            arrs[f"Z_{name}"] = emb          # no bottleneck: Z == raw == embed
            arrs[f"raw_{name}"] = emb
            arrs[f"y_{name}"] = y.astype(np.float32)
            arrs[f"logit_{name}"] = logit
            arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
            auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
            print(f"[extract] {name}: n={len(y)} head_auc={auc:.4f} embed={emb.shape}", flush=True)

    np.savez_compressed(run_dir / "pilot_Z.npz", **arrs)
    meta = {"run_id": run_id, "dataset": args.dataset, "split": args.split, "fold": args.fold,
            "epochs": args.epochs, "n_ent": n_ent, "model_label": "EmerGNN-Morgan",
            "feat": "M(native Morgan seed)", "selection": "leaf.val (official inductive val)",
            "warm": "held-out-seen (eval-only)", "cold": "leaf.test (S2 unseen)",
            "note": "path-paradigm / Morgan-seeded provisional row; knowledge-only structural = follow-up. "
                    "ONE model (leaf.val-best): warm+cold both extracted from it. Robustness = rerun --epochs 3."}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] saved {run_dir/'pilot_Z.npz'}\n"
          f"NEXT: python Code/scripts/analyze_rank_sufficiency.py "
          f"--z {run_dir/'pilot_Z.npz'} --key raw --title EmerGNN-full-DDI", flush=True)


if __name__ == "__main__":
    main()

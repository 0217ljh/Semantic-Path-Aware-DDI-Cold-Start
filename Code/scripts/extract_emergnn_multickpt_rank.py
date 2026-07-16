"""Full-KG EmerGNN (Morgan-seed, paper-faithful baseline core) rank analysis with
PER-EPOCH checkpoints, so warm and cold can each be measured at THEIR OWN best
epoch (user 2026-07-05: cold-generalization peaks early, warm peaks late).

Design (codex + user locked):
- Train ONCE via the PAPER-FAITHFUL baseline core (_PerModeEmerGNN_RSPMM: original
  torchdrug generalized_rspmm kernel, enc_ht bidirectional flow, shuffle_train
  inductive protocol, ReduceLROnPlateau, Morgan seed) — the fast proven path.
  save_strategy='epoch' dumps a checkpoint per epoch.
- Splits (no selection/measurement leakage):
    tr        = leaf.train minus two seen hold-outs   (TRAIN)
    warm_val  = seen hold-out                          (SELECT warm-best)
    warm_test = seen hold-out                          (MEASURE warm rank)
    cold_val  = leaf.val (official inductive val)       (SELECT cold-best; also the fit val)
    cold_test = leaf.test (S2 unseen)                  (MEASURE cold rank)
- After training, load every epoch checkpoint, compute warm_val & cold_val AUROC
  ourselves (bypassing the baseline _validate, which broke on a custom val), pick
  warm-best (argmax warm_val) and cold-best (argmax cold_val) epochs.
- ONE-MODEL rule preserved: for EACH selected checkpoint we extract BOTH warm_test
  and cold_test reps from that SAME model → a self-contained divergence. We never
  mix warm-from-one-ckpt with cold-from-another WITHIN one divergence. We report:
    * warm-best model: its warm_test rank (+ cold_test for reference)
    * cold-best model: its cold_test rank (+ warm_test for reference)
    * deployed (cold_val-best, = baseline's load_best_model_at_end) as a 3rd row
- Saves pilot_Z__<ckpt>.npz per selected checkpoint (train/warm/cold) for
  analyze_rank_sufficiency.py.

Run:
  CUDA_HOME=/home/lakestar_ljh/miniconda3 EMERGNN_BACKEND=rspmm EMERGNN_KG_SCOPE=full \
    PYTHONUNBUFFERED=1 python -u Code/scripts/extract_emergnn_multickpt_rank.py --epochs 10
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

MERGED_EDGES = "edges__drugbank_hetionet_primekg__mask1.parquet"


def _three_way(df: pd.DataFrame, seed: int, val_frac: float = 0.12, test_frac: float = 0.12):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    nv = int(round(val_frac * len(df)))
    nt = int(round(test_frac * len(df)))
    warm_val = df.iloc[perm[:nv]].reset_index(drop=True)
    warm_test = df.iloc[perm[nv:nv + nt]].reset_index(drop=True)
    tr = df.iloc[perm[nv + nt:]].reset_index(drop=True)
    return tr, warm_val, warm_test


def _pairs(df: pd.DataFrame, e2i: dict):
    a = df["drug_a_id"].astype(str).map(e2i)
    b = df["drug_b_id"].astype(str).map(e2i)
    bad = ~(a.notna() & b.notna())
    if int(bad.sum()):
        raise ValueError(f"{int(bad.sum())} pairs reference drugs not in EmerGNN KG")
    return a.to_numpy(np.int64), b.to_numpy(np.int64), df["y_bin"].to_numpy(np.float32)


def _ensure_eval_kg(c):
    """After _PerModeEmerGNN_RSPMM.load(), _eval_kg is None (rebuilt lazily only
    inside predict_proba). Our _embed_logit uses core._eval_kg directly, so
    rebuild it once here (codex 2026-07-05 must-fix)."""
    if getattr(c, "_eval_kg", None) is None:
        c._eval_kg = c._build_kg(c._eval_kg_triplets, c._n_base_rel_with_ddi)
    return c


@torch.no_grad()
def _embed_logit(core, ia, ib, batch_size, device):
    m = core._model; kg = core._eval_kg
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


def _auc(core, ia, ib, y, batch_size, device):
    _, logit = _embed_logit(core, ia, ib, batch_size, device)
    return float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ddi800")
    ap.add_argument("--split", default="cold_s2")
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
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
    run_id = f"{ts}__extract_emergnn_multickpt_rank__{args.dataset}_{args.split}_{args.fold}"
    run_dir = ROOT / "Code" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    merged_kg = Path(leaf.resources.kg.source) / MERGED_EDGES
    core = _PerModeEmerGNN_RSPMM(
        n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=32,
        n_epochs=args.epochs, device="cuda", backbone_kg_source="merged",
        merged_kg_path=merged_kg, weight_decay=1e-8, shuffle_train_mode="FIXED",
        log_step_every=50, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=args.epochs + 2, load_best_model_at_end=True, run_dir=run_dir)

    ds = make_dataset(tr, cold_val, leaf.resources)   # fit val = cold_val (official inductive val)
    t0 = time.time()
    core.fit(ds, ds)
    print(f"[fit] done in {time.time()-t0:.0f}s (paper-faithful baseline core, Morgan seed)", flush=True)

    e2i = core._entity2id
    wv = _pairs(warm_val, e2i); cv = _pairs(cold_val, e2i)
    wt = _pairs(warm_test, e2i); ct = _pairs(cold_test, e2i); trp = _pairs(tr, e2i)
    trip = np.asarray(core._kg_triplets, dtype=np.int64)
    logdeg = np.log1p(np.bincount(np.concatenate([trip[:, 0], trip[:, 1]]),
                                  minlength=int(core._n_ent)).astype(np.float64))

    # -- per-epoch: compute warm_val / cold_val AUROC ourselves --------------
    ckpt_root = run_dir / "checkpoints"
    ckpts = sorted(ckpt_root.glob("checkpoint-ep*")) if ckpt_root.is_dir() else []
    print(f"[ckpt] found {len(ckpts)} epoch checkpoints", flush=True)
    scores = []
    for cp in ckpts:
        try:
            c = _ensure_eval_kg(_PerModeEmerGNN_RSPMM.load(cp))
        except Exception as e:  # noqa: BLE001
            print(f"[ckpt] load FAIL {cp.name}: {e}", flush=True)
            continue
        wva = _auc(c, wv[0], wv[1], wv[2], args.batch_size, device)
        cva = _auc(c, cv[0], cv[1], cv[2], args.batch_size, device)
        scores.append((cp, wva, cva))
        print(f"[ckpt] {cp.name}: warm_val_auc={wva:.4f} cold_val_auc={cva:.4f}", flush=True)

    def _extract_from(core_obj, tag: str, val_note: str):
        arrs = {}
        for name, (ia, ib, y) in {"train": trp, "warm": wt, "cold": ct}.items():
            emb, logit = _embed_logit(core_obj, ia, ib, args.batch_size, device)
            arrs[f"Z_{name}"] = emb; arrs[f"raw_{name}"] = emb
            arrs[f"y_{name}"] = y.astype(np.float32); arrs[f"logit_{name}"] = logit
            arrs[f"deg_{name}"] = np.vstack([logdeg[ia], logdeg[ib]]).T.astype(np.float32)
            auc = roc_auc_score(y, 1.0 / (1.0 + np.exp(-logit)))
            print(f"[extract:{tag}] {name}: n={len(y)} head_auc={auc:.4f}", flush=True)
        np.savez_compressed(run_dir / f"pilot_Z__{tag}.npz", **arrs)
        return {"tag": tag, "selected_by": val_note}

    selected = {}
    if scores:
        warm_best = max(scores, key=lambda s: s[1])
        cold_best = max(scores, key=lambda s: s[2])
        print(f"[select] warm-best={warm_best[0].name} (warm_val={warm_best[1]:.4f}); "
              f"cold-best={cold_best[0].name} (cold_val={cold_best[2]:.4f})", flush=True)
        cwb = _ensure_eval_kg(_PerModeEmerGNN_RSPMM.load(warm_best[0]))
        selected["warm_best"] = _extract_from(cwb, "warm_best", f"warm_val={warm_best[1]:.4f}")
        if cold_best[0] == warm_best[0]:
            print("[select] cold-best == warm-best (same epoch); reusing", flush=True)
            selected["cold_best"] = _extract_from(cwb, "cold_best", f"cold_val={cold_best[2]:.4f}")
        else:
            ccb = _ensure_eval_kg(_PerModeEmerGNN_RSPMM.load(cold_best[0]))
            selected["cold_best"] = _extract_from(ccb, "cold_best", f"cold_val={cold_best[2]:.4f}")
    # deployed = baseline's load_best_model_at_end state (already in `core`)
    selected["deployed"] = _extract_from(core, "deployed", "cold_val-best (baseline load_best)")

    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "model_label": "EmerGNN-Morgan", "epochs": args.epochs,
        "per_epoch": [(cp.name, round(w, 4), round(c, 4)) for cp, w, c in scores],
        "selected": selected,
        "note": "paper-faithful baseline core (Morgan seed); warm/cold each at own-best ckpt; "
                "one-model rule: each pilot_Z__*.npz has warm+cold from ONE checkpoint."}, indent=2))
    print(f"[done] {run_dir}\nNEXT (per row): analyze_rank_sufficiency.py "
          f"--z {run_dir}/pilot_Z__warm_best.npz --key raw --title EmerGNN-Morgan-warmbest  (and cold_best/deployed)",
          flush=True)


if __name__ == "__main__":
    main()

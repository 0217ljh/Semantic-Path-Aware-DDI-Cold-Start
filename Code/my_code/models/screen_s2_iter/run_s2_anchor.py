"""S2-only EmerGNN anchor (single sub-model, paper-faithful Morgan+E flow).

Direct call into _PerModeEmerGNN to train ONLY the S2 sub-model (mode=S2,
batch=32, wd=1e-8, feat='M'), evaluate on test_s2. Fast (~30 min vs 7h
multimode) since we skip s0 and s1 entirely.

This is the anchor for all S2-iterate variants. Target: beat its test_s2 AUC
by ≥1pt with insight-level architectural innovation (not hyperparameter tuning).

Usage:
  python -u Code/my_code/models/screen_s2_iter/run_s2_anchor.py \\
      --backbone-kg-source drugbank --epochs 100 --seed 42 \\
      --tag s2anchor_drugbank_seed42
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
from sklearn.metrics import log_loss, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from baseline.emergnn._per_mode import _PerModeEmerGNN  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MERGED_KG = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone-kg-source", "--kg-source", dest="backbone_kg_source", choices=["drugbank", "merged"], default="drugbank")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-dim", type=int, default=64)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--feat", choices=["M", "E"], default="M",
                   help="M=Morgan FP (paper-faithful s2), E=learned embed")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"s2anchor_{args.backbone_kg_source}_seed{args.seed}_e{args.epochs}"
    with RunLogger(script="run_s2_anchor", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            kg_source=args.backbone_kg_source, epochs=args.epochs, seed=args.seed,
            feat=args.feat, batch_size=args.batch_size,
            shuffle_train_mode="S2",
        )
        _run(args, rl)


def _eval_s2(model, ds) -> dict:
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    try:
        neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    except Exception:
        return {"auc": float("nan"), "nll": float("nan"),
                "n_pos": int(len(pos)), "n_neg": 0}
    y_pos = model.predict_proba(pos)
    y_neg = model.predict_proba(neg)
    y_score = np.concatenate([y_pos, y_neg])
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "nll": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
    }


def _run(args, rl):
    print(f"[s2anchor] config: {vars(args)}")
    print(f"[s2anchor] loading {PKL.name}...")
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    print(f"[s2anchor] drugs={len(ds.drug_set)} train={len(ds.splits.train)} "
          f"test_s2={len(ds.splits.test_s2)}")

    kwargs = dict(
        n_dim=args.n_dim, length=args.length,
        feat=args.feat,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size, n_epochs=args.epochs,
        backbone_kg_source=args.backbone_kg_source,
        merged_kg_path=(str(MERGED_KG) if args.backbone_kg_source == "merged" else None),
        weight_decay=args.weight_decay,
        shuffle_train_mode="S2",
        log_step_every=50,
        eval_strategy="epoch", save_strategy="no",
        load_best_model_at_end=True,
        run_dir=str(rl.run_dir),
    )
    trainer = _PerModeEmerGNN(**kwargs)

    print(f"[s2anchor] fit start at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()
    trainer.fit(ds, val=ds)
    fit_sec = time.time() - t0
    print(f"[s2anchor] fit done in {fit_sec/3600:.2f}h ({fit_sec:.0f}s)")

    m = _eval_s2(trainer, ds)
    print(f"[s2anchor] test_s2: AUC={m['auc']:.4f} NLL={m['nll']:.4f}")

    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "fit_sec": fit_sec,
        "metrics": {"test_s2": m},
    }
    (rl.run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
    print(f"[s2anchor] results saved -> {rl.run_dir / 'results.json'}")


if __name__ == "__main__":
    main()

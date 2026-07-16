"""Run R3 — PMP-style MLP_score replacement readout (round 4 attempt 3 / paper Section 4).

Mirrors run_v3_meet_mask.py CLI + R3-specific flags.

Examples (from project root):

  # Smoke test 1 epoch:
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\
Semantic-Path-Aware-DDI-Cold-Start && python -u \
Code/my_code/models/screen_s2_v3_multimodal/run_v3_pmp.py \
--epochs 1 --tag r3_smoke --seed 42 --deterministic"

  # Canonical R3 main (seed42, 100 epochs):
  python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_pmp.py \
      --epochs 100 --tag r3_main_seed42 --seed 42 --deterministic

  # Controls:
  python -u .../run_v3_pmp.py --epochs 100 --tag r3_shuf_mpool \
      --seed 42 --r3-shuf-mediator-pool --deterministic
  python -u .../run_v3_pmp.py --epochs 100 --tag r3_zero_mpool \
      --seed 42 --r3-zero-m-pool --deterministic
  python -u .../run_v3_pmp.py --epochs 100 --tag r3_additive_init \
      --seed 42 --r3-additive-init --deterministic
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError("project root not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v3_pmp_trainer import (  # noqa: E402
    _PerModeEmerGNN_V3PMP,
    DEFAULT_MEDIATOR_PARQUET,
)
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"


def _set_deterministic(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_s2(model, ds):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    yp = model.predict_proba(pos)
    yn = model.predict_proba(neg)
    s = np.concatenate([yp, yn])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    out = {
        "auc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "nll": float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }
    try:
        ch = model.predict_channels(pd.concat([pos, neg], ignore_index=True))
        out["auc_emergnn"] = float(roc_auc_score(y, ch["emergnn"]))
        # For R3 the "count"/"i4" slots both carry the residual; report once.
        out["auc_residual"] = float(roc_auc_score(y, ch["count"]))
    except Exception as exc:
        print(f"[r3] channel diag failed: {exc}", flush=True)
    return out, (pos, neg, s, y)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--v2i4-hidden", type=int, default=32)
    p.add_argument("--v2i4-beta-init", type=float, default=0.5)
    p.add_argument("--r3-mediator-parquet", type=Path, default=DEFAULT_MEDIATOR_PARQUET)
    p.add_argument("--r3-dropout", type=float, default=0.2)
    p.add_argument("--r3-shuf-mediator-pool", action="store_true",
                   help="K1: shuffle m_pool batch row order before MLP_score.")
    p.add_argument("--r3-zero-m-pool", action="store_true",
                   help="K2: zero m_pool input (still pass m_present_bit).")
    p.add_argument("--r3-additive-init", action="store_true",
                   help="K3: init MLP_score to mimic v2i4 additive at epoch 0.")
    p.add_argument("--r3-shuffle-seed", type=int, default=12345)
    p.add_argument("--deterministic", action="store_true",
                   help="Set torch/np/cuda seeds at process start.")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    if args.deterministic:
        _set_deterministic(args.seed)

    tag = args.tag or f"r3_seed{args.seed}"
    with RunLogger(script="run_v3_pmp", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            epochs=args.epochs, seed=args.seed,
            r3_dropout=args.r3_dropout,
            r3_shuf_mediator_pool=args.r3_shuf_mediator_pool,
            r3_zero_m_pool=args.r3_zero_m_pool,
            r3_additive_init=args.r3_additive_init,
            r3_shuffle_seed=args.r3_shuffle_seed,
            deterministic=args.deterministic,
            length=args.length,
            r3_mediator_parquet=str(args.r3_mediator_parquet),
        )
        print(f"[r3] config: {vars(args)}", flush=True)

        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))

        trainer = _PerModeEmerGNN_V3PMP(
            n_dim=64, length=args.length, feat="M",
            learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50,
            eval_strategy="epoch", save_strategy="no",
            load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            v2i4_hidden=args.v2i4_hidden, v2i4_beta_init=args.v2i4_beta_init,
            r3_mediator_parquet=args.r3_mediator_parquet,
            r3_dropout=args.r3_dropout,
            r3_shuf_mediator_pool=args.r3_shuf_mediator_pool,
            r3_zero_m_pool=args.r3_zero_m_pool,
            r3_additive_init=args.r3_additive_init,
            r3_shuffle_seed=args.r3_shuffle_seed,
        )

        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(
            f"[r3] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"emergnn={m.get('auc_emergnn')} residual={m.get('auc_residual')} "
            f"fit={fit_sec/3600:.2f}h "
            f"(v2i4 anchor 0.7804; hard-stop 0.775)",
            flush=True,
        )

        results = {
            "config": vars(args),
            "fit_sec": fit_sec,
            "metrics": {"test_s2": m},
            "r3_summary": getattr(trainer, "_r3_summary", None),
            "r3_last_m_present_mean": getattr(trainer, "_r3_last_m_present_mean", None),
        }
        results["config"]["r3_mediator_parquet"] = str(args.r3_mediator_parquet)
        (rl.run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

        np.savez(
            rl.run_dir / "test_s2_scores.npz",
            pair_a=np.concatenate(
                [pos["drug_a_id"].astype(str).values, neg["drug_a_id"].astype(str).values]
            ),
            pair_b=np.concatenate(
                [pos["drug_b_id"].astype(str).values, neg["drug_b_id"].astype(str).values]
            ),
            y_true=y, y_score=s,
        )

        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[r3] run_dir: {rl.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

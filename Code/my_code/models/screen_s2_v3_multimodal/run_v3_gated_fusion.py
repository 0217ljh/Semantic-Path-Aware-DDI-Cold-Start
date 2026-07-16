"""Run R1 — gated 3-way replacement fusion (round 4 attempt 2 / fusion reform).

Mirrors run_v3_meet_mask.py CLI + R1-specific flags.

Examples (from project root):

  # Smoke test 1 epoch:
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\
Semantic-Path-Aware-DDI-Cold-Start && python -u \
Code/my_code/models/screen_s2_v3_multimodal/run_v3_gated_fusion.py \
--epochs 1 --tag r1_smoke --seed 42 --deterministic"

  # Canonical R1 main (seed42, 100 epochs):
  python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_gated_fusion.py \
      --epochs 100 --tag r1_main_seed42 --seed 42 --deterministic

  # Controls:
  python -u .../run_v3_gated_fusion.py --epochs 100 --tag r1_shuf_pair_feat \
      --seed 42 --r1-shuf-pair-feat --deterministic
  python -u .../run_v3_gated_fusion.py --epochs 100 --tag r1_freeze_gate \
      --seed 42 --r1-freeze-gate-uniform --deterministic
  python -u .../run_v3_gated_fusion.py --epochs 100 --tag r1_zero_gate \
      --seed 42 --r1-zero-gate-emergnn-only --deterministic
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

from my_code.models.screen_s2_v3_multimodal.v3_gated_fusion_trainer import (  # noqa: E402
    _PerModeEmerGNN_V3GatedFusion,
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
        out["auc_count_only"] = float(roc_auc_score(y, ch["count"]))
        out["auc_i4_only"] = float(roc_auc_score(y, ch["i4"]))
    except Exception as exc:
        print(f"[r1] channel diag failed: {exc}", flush=True)
    return out, (pos, neg, s, y)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--v2i4-hidden", type=int, default=32)
    p.add_argument("--v2i4-beta-init", type=float, default=0.5)
    p.add_argument("--r1-gate-hidden", type=int, default=32)
    p.add_argument("--r1-shuf-pair-feat", action="store_true",
                   help="K1: shuffle pair_feat batch row order before gate_mlp.")
    p.add_argument("--r1-freeze-gate-uniform", action="store_true",
                   help="K2: hardcode gate = [1/3, 1/3, 1/3] frozen.")
    p.add_argument("--r1-zero-gate-emergnn-only", action="store_true",
                   help="K3: hardcode gate = [1, 0, 0] frozen.")
    p.add_argument("--r1-shuffle-seed", type=int, default=12345)
    p.add_argument("--deterministic", action="store_true",
                   help="Set torch/np/cuda seeds at process start.")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    if args.deterministic:
        _set_deterministic(args.seed)

    tag = args.tag or f"r1_seed{args.seed}"
    with RunLogger(script="run_v3_gated_fusion", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            epochs=args.epochs, seed=args.seed,
            r1_gate_hidden=args.r1_gate_hidden,
            r1_shuf_pair_feat=args.r1_shuf_pair_feat,
            r1_freeze_gate_uniform=args.r1_freeze_gate_uniform,
            r1_zero_gate_emergnn_only=args.r1_zero_gate_emergnn_only,
            r1_shuffle_seed=args.r1_shuffle_seed,
            deterministic=args.deterministic,
            length=args.length,
        )
        print(f"[r1] config: {vars(args)}", flush=True)

        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))

        trainer = _PerModeEmerGNN_V3GatedFusion(
            n_dim=64, length=args.length, feat="M",
            learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50,
            eval_strategy="epoch", save_strategy="no",
            load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            v2i4_hidden=args.v2i4_hidden, v2i4_beta_init=args.v2i4_beta_init,
            r1_gate_hidden=args.r1_gate_hidden,
            r1_shuf_pair_feat=args.r1_shuf_pair_feat,
            r1_freeze_gate_uniform=args.r1_freeze_gate_uniform,
            r1_zero_gate_emergnn_only=args.r1_zero_gate_emergnn_only,
            r1_shuffle_seed=args.r1_shuffle_seed,
        )

        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(
            f"[r1] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"emergnn={m.get('auc_emergnn')} count_only={m.get('auc_count_only')} "
            f"i4_only={m.get('auc_i4_only')} fit={fit_sec/3600:.2f}h "
            f"(v2i4 anchor 0.7804; hard-stop 0.775)",
            flush=True,
        )

        results = {
            "config": vars(args),
            "fit_sec": fit_sec,
            "metrics": {"test_s2": m},
            "r1_last_gate_entropy": getattr(trainer, "_r1_last_gate_entropy", None),
            "r1_last_gate_means": getattr(trainer, "_r1_last_gate_means", None),
        }
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
        print(f"[r1] run_dir: {rl.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

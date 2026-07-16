"""Run Phase D: MNAH + I4 + joint binary + 215-class type task.

Primary: binary test_s2 AUROC (vs MNAH 0.772 / v2i4 0.7804).
Secondary: multi-class macro-F1 on test_s2 positives (user's "mutually-reinforcing" criterion).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

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

from my_code.models.screen_s2_v3_multimodal.v2i4d_trainer import _PerModeEmerGNN_V2I4D  # noqa
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"


def _eval_s2(model, ds):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    yp = model.predict_proba(pos); yn = model.predict_proba(neg)
    s = np.concatenate([yp, yn]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    out = {"auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "nll": float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7)))}
    type_metrics = model.eval_type_macro_f1(ds)
    out["type_macro_f1"] = type_metrics["macro_f1"]
    out["type_micro_f1"] = type_metrics["micro_f1"]
    out["n_type_eval"] = type_metrics["n_eval"]
    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lambda-type", type=float, default=0.3)
    p.add_argument("--type-hidden", type=int, default=64)
    p.add_argument("--v2i4-shuffle-control", action="store_true")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"v2i4d_lam{args.lambda_type}_seed{args.seed}"
    with RunLogger(script="run_v2i4d", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed, lambda_type=args.lambda_type)
        print(f"[v2i4d] config: {vars(args)}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))
        trainer = _PerModeEmerGNN_V2I4D(
            n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50, eval_strategy="epoch",
            save_strategy="no", load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            v2i4_hidden=32, v2i4_beta_init=0.5,
            v2i4_shuffle_control=args.v2i4_shuffle_control,
            lambda_type=args.lambda_type, type_hidden=args.type_hidden,
        )
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0
        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(f"[v2i4d] test_s2: binary AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
              f"type_macroF1={m['type_macro_f1']:.4f} type_microF1={m['type_micro_f1']:.4f} "
              f"(n_type_eval={m['n_type_eval']}, K=215) "
              f"fit={fit_sec/3600:.2f}h  (prior: MNAH 0.772, v2i4 0.7804)")
        (rl.run_dir / "results.json").write_text(json.dumps(
            {"config": vars(args), "fit_sec": fit_sec, "metrics": {"test_s2": m}}, indent=2))
        np.savez(rl.run_dir / "test_s2_scores.npz",
                 pair_a=np.concatenate([pos["drug_a_id"].astype(str).values,
                                        neg["drug_a_id"].astype(str).values]),
                 pair_b=np.concatenate([pos["drug_b_id"].astype(str).values,
                                        neg["drug_b_id"].astype(str).values]),
                 y_true=y, y_score=s)
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[v2i4d] run_dir: {rl.run_dir}")


if __name__ == "__main__":
    main()

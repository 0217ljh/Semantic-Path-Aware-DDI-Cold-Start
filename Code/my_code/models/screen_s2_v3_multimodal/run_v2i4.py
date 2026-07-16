"""Run E-i4: MNAH + structured-LLM mechanistic pair features.

Primary comparison: MNAH 0.772 vs MNAH+i4. Hard-stop rule (codex 019e67af):
lift < +1.5pt AND shuffle close to main -> pivot to backbone or mechanism paper.

Examples:
  python -u .../run_v2i4.py --epochs 100 --tag v2i4_main
  python -u .../run_v2i4.py --epochs 100 --v2i4-shuffle-control --tag v2i4_shufctrl
  python -u .../run_v2i4.py --epochs 100 --v2i4-random-control --tag v2i4_randctrl
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

from my_code.models.screen_s2_v3_multimodal.v2i4_trainer import _PerModeEmerGNN_V2I4  # noqa
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"


def _eval_s2(model, ds):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    yp = model.predict_proba(pos); yn = model.predict_proba(neg)
    s = np.concatenate([yp, yn]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    out = {"auc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)),
           "nll": float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7)))}
    try:
        ch = model.predict_channels(pd.concat([pos, neg], ignore_index=True))
        out["auc_emergnn"] = float(roc_auc_score(y, ch["emergnn"]))
        out["auc_count_only"] = float(roc_auc_score(y, ch["count"]))
        out["auc_i4_only"] = float(roc_auc_score(y, ch["i4"]))
    except Exception as exc:
        print(f"[v2i4] channel diag failed: {exc}", flush=True)
    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3,
                   help="EmerGNN path-flow length (Phase C: 4-5 codex 019e7771)")
    p.add_argument("--v2i4-hidden", type=int, default=32)
    p.add_argument("--v2i4-beta-init", type=float, default=0.5)
    p.add_argument("--v2i4-shuffle-control", action="store_true")
    p.add_argument("--v2i4-random-control", action="store_true")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"v2i4_seed{args.seed}"
    with RunLogger(script="run_v2i4", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed,
                    shuffle_control=args.v2i4_shuffle_control,
                    random_control=args.v2i4_random_control)
        print(f"[v2i4] config: {vars(args)}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))
        trainer = _PerModeEmerGNN_V2I4(
            n_dim=64, length=args.length, feat="M", learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50, eval_strategy="epoch",
            save_strategy="no", load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            v2i4_hidden=args.v2i4_hidden, v2i4_beta_init=args.v2i4_beta_init,
            v2i4_shuffle_control=args.v2i4_shuffle_control,
            v2i4_random_control=args.v2i4_random_control,
        )
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0
        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(f"[v2i4] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
              f"emergnn={m.get('auc_emergnn')} count_only={m.get('auc_count_only')} "
              f"i4_only={m.get('auc_i4_only')} fit={fit_sec/3600:.2f}h "
              f"(MNAH baseline 0.772; codex go-bar >+1.5pt)")
        (rl.run_dir / "results.json").write_text(json.dumps(
            {"config": vars(args), "fit_sec": fit_sec, "metrics": {"test_s2": m}}, indent=2))
        np.savez(rl.run_dir / "test_s2_scores.npz",
                 pair_a=np.concatenate([pos["drug_a_id"].astype(str).values,
                                        neg["drug_a_id"].astype(str).values]),
                 pair_b=np.concatenate([pos["drug_b_id"].astype(str).values,
                                        neg["drug_b_id"].astype(str).values]),
                 y_true=y, y_score=s)
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[v2i4] run_dir: {rl.run_dir}")


if __name__ == "__main__":
    main()

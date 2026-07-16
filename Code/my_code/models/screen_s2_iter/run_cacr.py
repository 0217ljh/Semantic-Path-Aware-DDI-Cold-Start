"""Run CACR (Cold-Aware Consistency Regularization) S2-only training."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
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

from my_code.models.screen_s2_iter.cacr_trainer import _PerModeEmerGNN_CACR  # noqa
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MERGED_KG = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def _eval_s2(model, ds):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    y_pos = model.predict_proba(pos); y_neg = model.predict_proba(neg)
    y_score = np.concatenate([y_pos, y_neg])
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "nll": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
    }


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
    p.add_argument("--feat", choices=["M", "E"], default="M")
    p.add_argument("--cacr-lambda", type=float, default=0.5)
    p.add_argument("--cacr-drop-rate", type=float, default=0.75)
    p.add_argument("--cacr-warmup-epochs", type=int, default=5)
    p.add_argument("--cacr-drop-relations", choices=["ddi_only", "all"],
                   default="ddi_only")
    p.add_argument("--cacr-nondrug-drop-rate", type=float, default=0.0)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or (
        f"cacr_lam{args.cacr_lambda}_drop{args.cacr_drop_rate}_warm{args.cacr_warmup_epochs}"
        f"_{args.backbone_kg_source}_seed{args.seed}"
    )
    with RunLogger(script="run_cacr", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            kg_source=args.backbone_kg_source, epochs=args.epochs, seed=args.seed,
            cacr_lambda=args.cacr_lambda, cacr_drop_rate=args.cacr_drop_rate,
            cacr_warmup_epochs=args.cacr_warmup_epochs,
        )
        print(f"[cacr] config: {vars(args)}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))
        kwargs = dict(
            n_dim=args.n_dim, length=args.length, feat=args.feat,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            n_epochs=args.epochs,
            backbone_kg_source=args.backbone_kg_source,
            merged_kg_path=(str(MERGED_KG) if args.backbone_kg_source == "merged" else None),
            weight_decay=args.weight_decay, shuffle_train_mode="S2",
            log_step_every=50, eval_strategy="epoch", save_strategy="no",
            load_best_model_at_end=True, run_dir=str(rl.run_dir),
            cacr_lambda=args.cacr_lambda, cacr_drop_rate=args.cacr_drop_rate,
            cacr_warmup_epochs=args.cacr_warmup_epochs,
            cacr_drop_relations=args.cacr_drop_relations,
            cacr_nondrug_drop_rate=args.cacr_nondrug_drop_rate,
        )
        trainer = _PerModeEmerGNN_CACR(**kwargs)
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0
        m = _eval_s2(trainer, ds)
        print(f"[cacr] test_s2: AUC={m['auc']:.4f} NLL={m['nll']:.4f}, fit={fit_sec/3600:.2f}h")
        payload = {
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "fit_sec": fit_sec, "metrics": {"test_s2": m},
        }
        (rl.run_dir / "results.json").write_text(json.dumps(payload, indent=2))
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])


if __name__ == "__main__":
    main()

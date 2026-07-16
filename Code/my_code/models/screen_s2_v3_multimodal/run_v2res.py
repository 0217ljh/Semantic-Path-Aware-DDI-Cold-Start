"""Run E-frag: shared/residual molecular<->KG alignment on top of KG backbone.

Mirrors run_v2.py but instantiates _PerModeEmerGNN_V2RES. Same legacy seed42 split/KG so
test_s2 AUROC is directly comparable to MNAH 0.772. Success = lift over 0.772 (>0.775 real).
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

from my_code.models.screen_s2_v3_multimodal.v2res_trainer import _PerModeEmerGNN_V2RES  # noqa
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
        ap = pd.concat([pos, neg], ignore_index=True)
        c, e, a = model._predict_branches(ap)
        out["auc_emergnn_only"] = float(roc_auc_score(y, e))
        out["auc_mol_head"] = float(roc_auc_score(y, a))
    except Exception as exc:
        print(f"[v2res] branch diag failed: {exc}", flush=True)
    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lambda-align", type=float, default=0.05)
    p.add_argument("--lambda-orth", type=float, default=0.02)
    p.add_argument("--lambda-dec", type=float, default=0.002)
    p.add_argument("--lambda-std", type=float, default=0.02)
    p.add_argument("--res-d", type=int, default=128)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"v2res_frag_seed{args.seed}"
    with RunLogger(script="run_v2res", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed, lambda_align=args.lambda_align,
                    lambda_orth=args.lambda_orth, mol_src="fragments")
        print(f"[v2res] config: {vars(args)}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))
        trainer = _PerModeEmerGNN_V2RES(
            n_dim=64, length=3, feat="M", learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50, eval_strategy="epoch",
            save_strategy="no", load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            res_d=args.res_d, lambda_align=args.lambda_align, lambda_orth=args.lambda_orth,
            lambda_dec=args.lambda_dec, lambda_std=args.lambda_std, mol_src="fragments",
        )
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0
        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(f"[v2res] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
              f"emergnn_only={m.get('auc_emergnn_only')} mol_head={m.get('auc_mol_head')} "
              f"fit={fit_sec/3600:.2f}h  (KG baseline 0.772; success >0.775)")
        (rl.run_dir / "results.json").write_text(json.dumps(
            {"config": vars(args), "fit_sec": fit_sec, "metrics": {"test_s2": m}}, indent=2))
        np.savez(rl.run_dir / "test_s2_scores.npz",
                 pair_a=np.concatenate([pos["drug_a_id"].astype(str).values,
                                        neg["drug_a_id"].astype(str).values]),
                 pair_b=np.concatenate([pos["drug_b_id"].astype(str).values,
                                        neg["drug_b_id"].astype(str).values]),
                 y_true=y, y_score=s)
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[v2res] run_dir: {rl.run_dir}")


if __name__ == "__main__":
    main()

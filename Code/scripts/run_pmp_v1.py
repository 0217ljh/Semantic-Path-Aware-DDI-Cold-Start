"""Run PMP v1 (Pair-Mediator Pooling) cold-start DDI experiment.

Algorithm 1 (locked). EmerGNN backbone + PMP Layer 2 attention pool aux head.
Minimal change from MNAH: replaces 22-d scalar count head with attention pool
over typed mediator embeddings.

Examples:
  # seed 42, single rep
  python Code/scripts/run_pmp.py --epochs 100 --tag pmp_v1_main_seed42

  # control: relation embeddings detached (sanity check)
  python Code/scripts/run_pmp.py --epochs 100 --tag pmp_v1_seed42 --pmp-hidden 128
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

from my_code.models.pmp_v1.pmp_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP, DEFAULT_PMP_CACHE,
)
from my_code.models.pmp_v1.precompute_pmp_cache import ensure_pmp_cache  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"


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
    # Per-branch diagnostic
    try:
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, e, a = model._predict_branches(all_pairs)
        out["auc_combined_branch"] = float(roc_auc_score(y, c))
        out["auc_emergnn_only"] = float(roc_auc_score(y, e))
        out["auc_pmp_only"] = float(roc_auc_score(y, a))
    except Exception as exc:
        print(f"[pmp] per-branch diag failed: {exc}", flush=True)
    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3,
                   help="EmerGNN path-flow length (default 3, same as MNAH)")
    p.add_argument("--n-dim", type=int, default=64,
                   help="EmerGNN backbone + PMP head embedding dim (default 64)")
    p.add_argument("--pmp-cache", type=str, default=str(DEFAULT_PMP_CACHE),
                   help="path to per-drug mediator dict pickle "
                        "(auto-built on cache MISS)")
    p.add_argument("--pmp-force-rebuild-cache", action="store_true",
                   help="force rebuild PMP cache even if it exists")
    p.add_argument("--pmp-hidden", type=int, default=128)
    p.add_argument("--pmp-dropout", type=float, default=0.2)
    p.add_argument("--pmp-init-beta", type=float, default=1.0)
    p.add_argument("--pmp-max-mediators", type=int, default=None,
                   help="cap mediators per pair (default: no cap)")
    p.add_argument("--log-step-every", type=int, default=50)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    # Auto-detect + auto-build PMP cache (HIT -> skip; MISS -> build from merged KG).
    # Done before RunLogger so cache-build log goes to console (not run dir).
    ensure_pmp_cache(cache_path=args.pmp_cache, force=args.pmp_force_rebuild_cache)

    tag = args.tag or f"pmp_v1_seed{args.seed}"
    with RunLogger(script="run_pmp", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed)
        print(f"[pmp] config: {vars(args)}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))

        trainer = _PerModeEmerGNN_PMP(
            n_dim=args.n_dim,
            length=args.length,
            feat="M",
            learning_rate=1e-3,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            backbone_kg_source="drugbank",
            weight_decay=1e-8,
            shuffle_train_mode="S2",
            log_step_every=args.log_step_every,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=True,
            run_dir=str(rl.run_dir),
            # MNAH parent hyperparams (inherited residual fusion + raw_beta init)
            mnah_hidden=32,
            mnah_dropout=0.2,
            mnah_init_beta=args.pmp_init_beta,
            mnah_text_cache=None,
            # PMP-specific
            pmp_cache_path=args.pmp_cache,
            pmp_hidden=args.pmp_hidden,
            pmp_dropout=args.pmp_dropout,
            pmp_max_mediators=args.pmp_max_mediators,
        )

        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(
            f"[pmp] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"emergnn_only={m.get('auc_emergnn_only')} "
            f"pmp_only={m.get('auc_pmp_only')} "
            f"fit={fit_sec / 3600:.2f}h "
            f"(MNAH baseline 0.7670; v2i4 baseline 0.7804; PMP-v1 target ≥ 0.78)"
        )

        results_path = rl.run_dir / "results.json"
        results_path.write_text(json.dumps({
            "config": vars(args),
            "fit_sec": fit_sec,
            "metrics": {"test_s2": m},
        }, indent=2))

        np.savez(
            rl.run_dir / "test_s2_scores.npz",
            pair_a=np.concatenate([
                pos["drug_a_id"].astype(str).values,
                neg["drug_a_id"].astype(str).values,
            ]),
            pair_b=np.concatenate([
                pos["drug_b_id"].astype(str).values,
                neg["drug_b_id"].astype(str).values,
            ]),
            y_true=y,
            y_score=s,
        )
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[pmp] run_dir: {rl.run_dir}")


if __name__ == "__main__":
    main()

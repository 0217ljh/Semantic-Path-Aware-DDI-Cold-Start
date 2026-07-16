"""Run Meeting-Node Auxiliary Head (MNAH) v2 Stage 1 on S2 cold-start.

Mirrors run_s2_anchor.py but instantiates _PerModeEmerGNN_MNAH.

Usage:
  python -u Code/my_code/models/screen_s2_v2_meetnode/run_mnah.py \\
      --backbone-kg-source drugbank --epochs 100 --seed 42 \\
      --tag mnah_v2s1_drugbank_seed42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

from my_code.models.screen_s2_v2_meetnode.mnah_trainer import _PerModeEmerGNN_MNAH  # noqa
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MERGED_KG = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def _eval_s2(model, ds, dump_logits_path=None):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    y_pos = model.predict_proba(pos); y_neg = model.predict_proba(neg)
    y_score = np.concatenate([y_pos, y_neg])
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if dump_logits_path is not None:
        # logit = log(p/(1-p)); pos-then-neg order to match motif gate dump
        pclip = np.clip(y_score, 1e-7, 1 - 1e-7)
        logit = np.log(pclip / (1 - pclip))
        np.savez(dump_logits_path,
                 y_true=y_true, mnah_prob=y_score, mnah_logit=logit,
                 pair_a=np.concatenate([pos["drug_a_id"].astype(str).values, neg["drug_a_id"].astype(str).values]),
                 pair_b=np.concatenate([pos["drug_b_id"].astype(str).values, neg["drug_b_id"].astype(str).values]))
        print(f"[mnah] dumped test_s2 logits -> {dump_logits_path}", flush=True)
    out = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "nll": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
    }
    # Per-branch diagnostics
    try:
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, e, a = model._predict_branches(all_pairs)
        out["auc_emergnn_only"] = float(roc_auc_score(y_true, e))
        out["auc_aux_only"] = float(roc_auc_score(y_true, a))
        out["auc_combined"] = float(roc_auc_score(y_true, c))
    except Exception as exc:
        print(f"[mnah] branch-AUC diag failed: {exc}", flush=True)
    return out


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
    p.add_argument("--mnah-hidden", type=int, default=32)
    p.add_argument("--mnah-dropout", type=float, default=0.2)
    p.add_argument("--mnah-init-beta", type=float, default=1.0)
    p.add_argument("--mnah-shuffle-control", action="store_true",
                   help="Codex r13 control: scramble pair->feature binding. "
                        "Combined should collapse to emergnn-only if gain is real.")
    p.add_argument("--mnah-feat-cache", type=str, default=None,
                   help="Override feature cache parquet (e.g. degree-only control cache).")
    p.add_argument("--mnah-text-cache", type=str, default=None,
                   help="Stage 2 (i4): PubMedBERT shared-mediator text feature cache to concat.")
    p.add_argument("--release-seed", type=int, default=None,
                   help="If set, load split via from_release_dir(KG/drugbank, seed=N) "
                        "instead of the legacy seed42 PKL — for 2nd-split generalization (codex r18).")
    p.add_argument("--dump-logits", type=str, default=None,
                   help="Path to save test_s2 logits (npz) for the complementarity gate.")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"mnah_v2s1_h{args.mnah_hidden}_b{args.mnah_init_beta}_{args.backbone_kg_source}_seed{args.seed}"
    with RunLogger(script="run_mnah", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            kg_source=args.backbone_kg_source, epochs=args.epochs, seed=args.seed,
            mnah_hidden=args.mnah_hidden, mnah_dropout=args.mnah_dropout,
            mnah_init_beta=args.mnah_init_beta, release_seed=args.release_seed,
        )
        print(f"[mnah] config: {vars(args)}")
        from data_utils import PairDataset
        if args.release_seed is not None:
            ds = PairDataset.from_release_dir(ROOT / "Code/data/KG/drugbank",
                                              seed=args.release_seed)
            print(f"[mnah] loaded RELEASE split seed={args.release_seed} "
                  f"(train={len(ds.splits.train)} test_s2={len(ds.splits.test_s2)})")
        else:
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
            mnah_hidden=args.mnah_hidden,
            mnah_dropout=args.mnah_dropout,
            mnah_init_beta=args.mnah_init_beta,
            mnah_shuffle_control=args.mnah_shuffle_control,
            mnah_feat_cache=args.mnah_feat_cache,
            mnah_text_cache=args.mnah_text_cache,
        )
        trainer = _PerModeEmerGNN_MNAH(**kwargs)
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0
        m = _eval_s2(trainer, ds, dump_logits_path=args.dump_logits)
        print(f"[mnah] test_s2: AUC={m['auc']:.4f} NLL={m['nll']:.4f}, fit={fit_sec/3600:.2f}h")
        if "auc_emergnn_only" in m:
            print(f"[mnah]   emergnn-only AUC: {m['auc_emergnn_only']:.4f}")
            print(f"[mnah]   aux-only     AUC: {m['auc_aux_only']:.4f}")
            print(f"[mnah]   combined     AUC: {m['auc_combined']:.4f}")
        payload = {
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "fit_sec": fit_sec, "metrics": {"test_s2": m},
        }
        (rl.run_dir / "results.json").write_text(json.dumps(payload, indent=2))
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])


if __name__ == "__main__":
    main()

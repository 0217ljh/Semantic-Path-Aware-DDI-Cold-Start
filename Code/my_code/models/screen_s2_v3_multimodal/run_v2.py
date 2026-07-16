"""Run v2 learned-routing (i1 routing + multimodal alignment) on S2 cold-start.

Mirrors run_mnah.py but instantiates _PerModeEmerGNN_V2. Same legacy seed42 split / KG so
test_s2 AUROC is directly comparable to MNAH ~0.77.

Examples:
  # main v2 (cross-attention effect channel)
  python -u Code/my_code/models/screen_s2_v3_multimodal/run_v2.py --epochs 100 --tag v2_main
  # sanity: effect-shuffle control (effect channel should collapse)
  python -u .../run_v2.py --epochs 100 --v2-effect-shuffle-control --tag v2_effshuf
  # ablation: bag-pool effect channel (PD subgroup should drop)
  python -u .../run_v2.py --epochs 100 --v2-effect-mode bagpool --tag v2_bagpool
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

from my_code.models.screen_s2_v3_multimodal.v2_trainer import _PerModeEmerGNN_V2  # noqa
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MERGED_KG = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
PKPD = ROOT / "Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv"


def _load_pkpd_drugtype():
    """Map ddi_type -> {PK,PD,Mixed}. Returns dict or None if unavailable."""
    try:
        df = pd.read_csv(PKPD)
    except Exception as exc:
        print(f"[v2] pk/pd labels unavailable: {exc}", flush=True)
        return None
    return df


def _eval_s2(model, ds):
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    y_pos = model.predict_proba(pos); y_neg = model.predict_proba(neg)
    y_score = np.concatenate([y_pos, y_neg])
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    out = {
        "auc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "nll": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
    }
    try:
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, e, a = model._predict_branches(all_pairs)
        out["auc_emergnn_only"] = float(roc_auc_score(y_true, e))
        out["auc_head_only"] = float(roc_auc_score(y_true, a))
        out["auc_combined"] = float(roc_auc_score(y_true, c))
    except Exception as exc:
        print(f"[v2] branch-AUC diag failed: {exc}", flush=True)
    return out, (pos, neg, y_score, y_true)


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
    p.add_argument("--v2-align-dim", type=int, default=128)
    p.add_argument("--v2-d", type=int, default=64)
    p.add_argument("--v2-topk", type=int, default=32)
    p.add_argument("--v2-effect-mode", choices=["xattn", "bagpool", "count"], default="xattn")
    p.add_argument("--v2-freq-debias", type=float, default=1.0)
    p.add_argument("--v2-effect-shuffle-control", action="store_true")
    p.add_argument("--v2-eff-topk-pair", type=int, default=0,
                   help="codex patch: keep top-k of K*K candidate effect pairs (0=all).")
    p.add_argument("--v2-eff-attn-temp", type=float, default=1.0,
                   help="effect attention temperature (<1 sharpens).")
    p.add_argument("--v2-eff-pair-dropout", type=float, default=0.0,
                   help="train-time dropout on candidate effect pairs.")
    p.add_argument("--v2-eff-aux-bce", type=float, default=0.0,
                   help="codex 019e6272: aux BCE deep-supervision weight on eff_logit (0=off).")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"v2_{args.v2_effect_mode}_seed{args.seed}"
    with RunLogger(script="run_v2", tag=tag, seed=args.seed) as rl:
        rl.set_meta(kg_source=args.backbone_kg_source, epochs=args.epochs, seed=args.seed,
                    v2_effect_mode=args.v2_effect_mode,
                    v2_effect_shuffle_control=args.v2_effect_shuffle_control)
        print(f"[v2] config: {vars(args)}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))
        kwargs = dict(
            n_dim=args.n_dim, length=args.length, feat=args.feat,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source=args.backbone_kg_source,
            merged_kg_path=(str(MERGED_KG) if args.backbone_kg_source == "merged" else None),
            weight_decay=args.weight_decay, shuffle_train_mode="S2",
            log_step_every=50, eval_strategy="epoch", save_strategy="no",
            load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=args.mnah_hidden, mnah_dropout=args.mnah_dropout,
            mnah_init_beta=args.mnah_init_beta,
            v2_align_dim=args.v2_align_dim, v2_d=args.v2_d, v2_topk=args.v2_topk,
            v2_effect_mode=args.v2_effect_mode, v2_freq_debias=args.v2_freq_debias,
            v2_effect_shuffle_control=args.v2_effect_shuffle_control,
            v2_eff_topk_pair=args.v2_eff_topk_pair,
            v2_eff_attn_temp=args.v2_eff_attn_temp,
            v2_eff_pair_dropout=args.v2_eff_pair_dropout,
            v2_eff_aux_bce=args.v2_eff_aux_bce,
        )
        trainer = _PerModeEmerGNN_V2(**kwargs)
        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0
        m, (pos, neg, y_score, y_true) = _eval_s2(trainer, ds)
        print(f"[v2] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} NLL={m['nll']:.4f} "
              f"fit={fit_sec/3600:.2f}h")
        if "auc_emergnn_only" in m:
            print(f"[v2]   emergnn-only AUC: {m['auc_emergnn_only']:.4f}")
            print(f"[v2]   head-only    AUC: {m['auc_head_only']:.4f}")
            print(f"[v2]   combined     AUC: {m['auc_combined']:.4f}")
        payload = {
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "fit_sec": fit_sec, "metrics": {"test_s2": m},
        }
        (rl.run_dir / "results.json").write_text(json.dumps(payload, indent=2))
        # dump per-pair scores + per-channel breakdown for PK/PD + gate analysis
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        ch = trainer.predict_channels(all_pairs)
        np.savez(rl.run_dir / "test_s2_scores.npz",
                 pair_a=np.concatenate([pos["drug_a_id"].astype(str).values,
                                        neg["drug_a_id"].astype(str).values]),
                 pair_b=np.concatenate([pos["drug_b_id"].astype(str).values,
                                        neg["drug_b_id"].astype(str).values]),
                 y_true=y_true, y_score=y_score,
                 ch_combined=ch["combined"], ch_emergnn=ch["emergnn"], ch_head=ch["head"],
                 ch_mol=ch["mol"], ch_eff=ch["eff"], ch_g=ch["g"])
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[v2] run_dir: {rl.run_dir}")


if __name__ == "__main__":
    main()

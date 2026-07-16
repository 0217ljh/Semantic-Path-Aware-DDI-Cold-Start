"""Screen 3 training entry — Meet-in-Middle pooling.

Trains EmerGNN_MIM multimode for a single (init_variant, projection,
junction_type) combination. Mirrors run_screen1.py but with MIM-specific
arguments (junction_type, max_junctions, use_mim).

Example:
  python -u Code/my_code/models/screen3_meet_in_middle/run_screen3.py \\
      --init-variant D --projection P1 --junction-type J1 --max-junctions 32 \\
      --epochs 100 --seed 42 --tag screen3_J1_D_seed42

Variants from first_step_plan.md §4.6:
  R0: --use-mim False         (terminal-only anchor = Screen 1 equivalent)
  R1: --junction-type J1      (primary)
  R2: --junction-type J3-PK
  R3: --junction-type J3-PD
  R4: --junction-type J3-both
  R5: (no terminal — not yet supported; would require disabling head_hid/tail_hid)
  R6: --junction-type Jr      (random control)
  R7: --junction-type J7      (2-hop)
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

from my_code.models.screen1_tag_init import init_features as ift  # noqa: E402
from my_code.models.screen3_meet_in_middle._per_mode_mim import _PerModeEmerGNN_MIM  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
MERGED_EDGES = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def _eval_split(model, ds, split_name: str) -> dict:
    pos = getattr(ds.splits, f"test_{split_name}")[["drug_a_id", "drug_b_id"]]
    try:
        neg = ds.get_negatives(f"test_{split_name}")[["drug_a_id", "drug_b_id"]]
    except Exception as e:
        return {"auc": float("nan"), "nll": float("nan"),
                "n_pos": int(len(pos)), "n_neg": 0}
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": float("nan"), "nll": float("nan"),
                "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    y_pos = model.predict_proba(pos)
    y_neg = model.predict_proba(neg)
    y_score = np.concatenate([y_pos, y_neg])
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "nll": float(log_loss(y_true, np.clip(y_score, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)), "n_neg": int(len(neg)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-variant", choices=["A", "B", "C", "D", "D-name", "E", "F"],
                   default="D")
    p.add_argument("--projection", choices=["P1", "P2", "NONE"], default="P1")
    p.add_argument("--junction-type", choices=["J1", "J3-PK", "J3-PD", "J3-both", "J7", "Jr"],
                   default="J1")
    p.add_argument("--max-junctions", type=int, default=32)
    p.add_argument("--use-mim", type=lambda s: s.lower() in ("true", "1", "yes"),
                   default=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-dim", type=int, default=64)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--shuffle-train-mode", choices=["S0", "S1", "S2"], default="S2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--freeze-init", action="store_true")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.init_variant not in ("A", "B", "C") and args.projection == "NONE":
        raise ValueError(f"variant {args.init_variant} requires --projection P1/P2")

    tag = args.tag or (
        f"screen3_{args.junction_type}_{args.init_variant}_{args.projection}_"
        f"seed{args.seed}_e{args.epochs}"
    )
    if args.smoke:
        tag = f"{tag}__smoke"
        args.epochs = 1

    with RunLogger(script="run_screen3", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            init_variant=args.init_variant,
            projection=args.projection,
            junction_type=args.junction_type,
            use_mim=args.use_mim,
            shuffle_train_mode=args.shuffle_train_mode,
            epochs=args.epochs, seed=args.seed,
        )
        _run(args, rl)


def _run(args, rl):
    print(f"[screen3] config: {vars(args)}")
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    print(f"[screen3] drugs={len(ds.drug_set)} train={len(ds.splits.train)}")

    print(f"[screen3] building init: variant={args.init_variant} projection={args.projection}")
    node_ids, init = ift.build_init(
        args.init_variant, projection=args.projection, n_dim=args.n_dim, seed=args.seed,
    )
    print(f"[screen3] init shape={init.shape}")

    trainer = _PerModeEmerGNN_MIM(
        external_init=init, external_init_node_ids=node_ids,
        junction_type=args.junction_type, max_junctions=args.max_junctions,
        merged_edges_for_junctions=str(MERGED_EDGES),
        use_mim=args.use_mim, freeze_init=args.freeze_init,
        n_dim=args.n_dim, length=args.length, n_epochs=args.epochs,
        shuffle_train_mode=args.shuffle_train_mode,
        batch_size=args.batch_size, weight_decay=args.weight_decay,
        backbone_kg_source="drugbank",  # for cross-baseline comparability
        eval_strategy="epoch", save_strategy="no", load_best_model_at_end=True,
        run_dir=str(rl.run_dir),
    )

    print(f"[screen3] fit starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()
    trainer.fit(ds, val=ds)
    fit_sec = time.time() - t0
    print(f"[screen3] fit done in {fit_sec/3600:.2f}h")

    # Evaluate ON THE shuffle_train_mode's test split only (single sub-model)
    split_for_test = {"S0": "s0", "S1": "s1", "S2": "s2"}[args.shuffle_train_mode]
    m = _eval_split(trainer, ds, split_for_test)
    print(f"[screen3] test_{split_for_test}: AUC={m['auc']:.4f} NLL={m['nll']:.4f}")

    results = {f"test_{split_for_test}": m}
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "fit_sec": fit_sec, "metrics": results,
    }
    (rl.run_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"[screen3] results saved -> {rl.run_dir / 'results.json'}")

    rl.set_metrics(
        fit_time_s=fit_sec,
        **{f"auc_{split_for_test}": m["auc"], f"nll_{split_for_test}": m["nll"]},
    )


if __name__ == "__main__":
    main()

"""Screen 1 training entry script.

Trains EmerGNN multimode + TAG init for a single (variant, projection)
combination. Saves results to a per-run directory under `Code/runs/`.

Example:
  python -u Code/my_code/models/screen1_tag_init/run_screen1.py \\
      --variant D --projection P1 --epochs 100 --seed 42 \\
      --tag screen1_D_P1_seed42

Constraints (per first_step_plan.md §0):
  - seed 42 only
  - 800-drug DrugBank cold-start PKL
  - backbone_kg_source = "drugbank" (matches existing EmerGNN multimode anchor)
  - The merged-KG PubMedBERT init is aligned to the drugbank-KG entity
    vocab at fit time (entities not in merged-KG cache get zero vec; print
    coverage stats).
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
    raise FileNotFoundError("project root not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.screen1_tag_init import init_features as ift  # noqa: E402
from my_code.models.screen1_tag_init.multimode import EmerGNNTAGBaseline  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
ALL_VARIANTS = ["A", "B", "C", "D", "D-name", "E", "F"]


def _eval_split(model, ds, split_name: str) -> dict:
    """Run predict_proba on test_<split> pos + neg, return AUC / NLL / sizes."""
    pos = getattr(ds.splits, f"test_{split_name}")[["drug_a_id", "drug_b_id"]]
    try:
        neg = ds.get_negatives(f"test_{split_name}")[["drug_a_id", "drug_b_id"]]
    except Exception as e:
        print(f"  [_eval] test_{split_name}: no negatives ({e})")
        return {"auc": float("nan"), "nll": float("nan"),
                "n_pos": int(len(pos)), "n_neg": 0}
    if len(pos) == 0 or len(neg) == 0:
        return {"auc": float("nan"), "nll": float("nan"),
                "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    y_pos = model.predict_proba(pos)
    y_neg = model.predict_proba(neg)
    y_score = np.concatenate([y_pos, y_neg])
    y_true = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    auc = float(roc_auc_score(y_true, y_score))
    # Avoid log(0) issues
    eps = 1e-7
    y_clip = np.clip(y_score, eps, 1 - eps)
    nll = float(log_loss(y_true, y_clip))
    return {"auc": auc, "nll": nll, "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=ALL_VARIANTS, required=True)
    p.add_argument("--projection", choices=["P1", "P2", "NONE"], default="P1",
                   help="Projection for D/D-name/E/F variants (P1=random Gaussian, "
                        "P2=PCA, NONE=no projection — only valid for A/B/C).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-dim", type=int, default=64)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--backbone-kg-source", "--kg-source", dest="backbone_kg_source", choices=["drugbank", "merged"], default="drugbank")
    p.add_argument("--freeze-init", action="store_true",
                   help="Freeze ent_kg.weight (default: trainable)")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Quick smoke test: 1 epoch + no negatives eval")
    p.add_argument("--log-step-every", type=int, default=50)
    args = p.parse_args()

    if args.variant not in ("A", "B", "C") and args.projection == "NONE":
        raise ValueError(f"variant {args.variant} requires --projection P1 or P2")

    tag = args.tag or f"screen1_{args.variant}_{args.projection}_seed{args.seed}_e{args.epochs}"
    if args.smoke:
        tag = f"{tag}__smoke"
        args.epochs = 1

    with RunLogger(script="run_screen1", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            variant=args.variant,
            projection=args.projection,
            kg_source=args.backbone_kg_source,
            epochs=args.epochs,
            seed=args.seed,
        )
        _run(args, rl)


def _run(args, rl):
    run_dir = rl.run_dir
    print(f"[screen1] config: {vars(args)}")

    # ----- load data -----
    print(f"[screen1] loading {PKL.name}...")
    t0 = time.time()
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))
    print(f"[screen1] loaded in {time.time()-t0:.1f}s: drugs={len(ds.drug_set)}  "
          f"train={len(ds.splits.train)}  test_s2={len(ds.splits.test_s2)}")

    # ----- build init tensor -----
    print(f"[screen1] building init: variant={args.variant} projection={args.projection}")
    t0 = time.time()
    node_ids, init = ift.build_init(
        args.variant, projection=args.projection, n_dim=args.n_dim, seed=args.seed,
    )
    print(f"[screen1] init shape={init.shape}, "
          f"nonzero rows={(init.abs().sum(dim=1) > 0).sum().item()}/{init.shape[0]:,}, "
          f"built in {time.time()-t0:.1f}s")

    # ----- instantiate Screen 1 multimode trainer -----
    print(f"[screen1] instantiating EmerGNNTAGBaseline...")
    kwargs = dict(
        external_init=init,
        external_init_node_ids=node_ids,
        variant_tag=f"{args.variant}_{args.projection}",
        freeze_init=args.freeze_init,
        n_dim=args.n_dim,
        length=args.length,
        n_epochs=args.epochs,
        backbone_kg_source=args.backbone_kg_source,
        merged_kg_path=(
            str(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
            if args.backbone_kg_source == "merged" else None
        ),
        log_step_every=args.log_step_every,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=True,
        run_dir=str(run_dir),
    )
    model = EmerGNNTAGBaseline(**kwargs)

    # ----- fit -----
    print(f"[screen1] fit starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0 = time.time()
    model.fit(ds, val=ds)
    fit_sec = time.time() - t0
    print(f"[screen1] fit completed in {fit_sec/3600:.2f}h ({fit_sec:.0f}s)")

    # ----- evaluate on test_s0 / test_s1 / test_s2 -----
    print(f"[screen1] evaluating on test_s0 / test_s1 / test_s2...")
    results = {}
    for split in ("s0", "s1", "s2"):
        m = _eval_split(model, ds, split)
        results[f"test_{split}"] = m
        print(f"  test_{split}: AUC={m['auc']:.4f} NLL={m['nll']:.4f} "
              f"n_pos={m['n_pos']} n_neg={m['n_neg']}")

    # ----- save results.json -----
    results_path = run_dir / "results.json"
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "fit_sec": fit_sec,
        "metrics": results,
    }
    results_path.write_text(json.dumps(payload, indent=2))
    print(f"[screen1] results saved -> {results_path}")

    # ----- write metrics row to RunLogger index.csv (codex #2 WARN #4) -----
    rl.set_metrics(
        fit_time_s=fit_sec,
        auc_s0=results["test_s0"]["auc"],
        auc_s1=results["test_s1"]["auc"],
        auc_s2=results["test_s2"]["auc"],
        nll_s0=results["test_s0"]["nll"],
        nll_s1=results["test_s1"]["nll"],
        nll_s2=results["test_s2"]["nll"],
    )


if __name__ == "__main__":
    main()

"""Run PMP v1.2 cold-start DDI experiment (created 2026-06-03).

v1.2 = v1.1 architecture with the cluster path's e_i replaced from the fixed
shared cluster_embed[i] to the pair-conditional
h_a^i = AttnPool(mediator_embed[m] for m in members(i) ∩ N(a)). The
within-cluster pool and the rest of the stack are unchanged from v1.1.

Reuses the v1.1 cluster cache (schema v2) without modification.

Examples:
  # seed 42 single rep, ndim 32 (matches v1.1 main run config)
  python Code/scripts/run_pmp_v1_2.py --epochs 100 --seed 42 --n-dim 32 \
      --tag pmp_v1_2_ndim32_seed42

  # disable per-(drug, cluster) cap (default 64) — large drugs use full neighborhood
  python Code/scripts/run_pmp_v1_2.py --epochs 100 --seed 42 --n-dim 32 \
      --cluster-max-mediators-per-drug 0 --tag pmp_v1_2_uncapped
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

from my_code.models.pmp_v1.pmp_trainer import DEFAULT_PMP_CACHE  # noqa: E402
from my_code.models.pmp_v1.precompute_pmp_cache import ensure_pmp_cache  # noqa: E402
from my_code.models.pmp_v1.v1_1.precompute_cluster_cache import (  # noqa: E402
    ensure_cluster_cache,
)
from my_code.models.pmp_v1.v1_2.pmp_v1_2_trainer import (  # noqa: E402
    _PerModeEmerGNN_PMP_v1_2, DEFAULT_CLUSTER_CACHE,
)
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def _pkl_for_seed(seed: int) -> Path:
    """Resolve seed{N}.pkl matching --seed. Fail loud if missing."""
    p = PKL_DIR / f"seed{int(seed)}.pkl"
    if not p.is_file():
        raise FileNotFoundError(
            f"Dataset pickle for seed={seed} not found at {p}. "
            f"Available pkls in {PKL_DIR}: "
            f"{sorted(x.name for x in PKL_DIR.glob('seed*.pkl'))}"
        )
    return p


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
    # v1.2 reports 3 diagnostic branches via _predict_branches_v1_2:
    #   combined, cluster_only, within_only
    try:
        all_pairs = pd.concat([pos, neg], ignore_index=True)
        c, cluster_only, within_only = model._predict_branches_v1_2(all_pairs)
        out["auc_combined_branch"] = float(roc_auc_score(y, c))
        out["auc_cluster_only"] = float(roc_auc_score(y, cluster_only))
        out["auc_within_only"] = float(roc_auc_score(y, within_only))
    except Exception as exc:
        print(f"[pmp-v1.2] per-branch diag failed: {exc}", flush=True)
    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3,
                   help="EmerGNN path-flow length (legacy; v1.2 does not "
                        "use EmerGNN in scoring but the backbone is still "
                        "built for shuffle_train protocol).")
    p.add_argument("--n-dim", type=int, default=32,
                   help="Embedding dim for mediator/cluster/etc. (default 32)")

    # v1 PMP cache
    p.add_argument("--pmp-cache", type=str, default=str(DEFAULT_PMP_CACHE),
                   help="path to PMP mediator dict pickle "
                        "(auto-built on cache MISS)")
    p.add_argument("--pmp-force-rebuild-cache", action="store_true")

    # v1.1 cluster cache (schema v2) — reused as-is by v1.2
    p.add_argument("--cluster-cache", type=str,
                   default=str(DEFAULT_CLUSTER_CACHE),
                   help="path to v1.1 cluster cache (schema v2) pickle "
                        "(auto-built on cache MISS). v1.2 reuses this schema.")
    p.add_argument("--cluster-force-rebuild-cache", action="store_true")

    # v1.2 module hyperparams
    p.add_argument("--cluster-hidden", type=int, default=128)
    p.add_argument("--cluster-dropout", type=float, default=0.2)
    p.add_argument(
        "--cluster-max-mediators-per-drug", type=int, default=64,
        help="Cap on per-(drug, cluster) mediator list for the pair-conditional "
             "cluster pool. 0 disables the cap (use full neighborhood).",
    )

    p.add_argument("--log-step-every", type=int, default=50)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    # Auto-detect + auto-build both caches.
    ensure_pmp_cache(
        cache_path=args.pmp_cache,
        force=args.pmp_force_rebuild_cache,
    )
    ensure_cluster_cache(
        cache_path=args.cluster_cache,
        force=args.cluster_force_rebuild_cache,
    )

    cap_arg = (
        None if args.cluster_max_mediators_per_drug == 0
        else int(args.cluster_max_mediators_per_drug)
    )

    tag = args.tag or f"pmp_v1_2_seed{args.seed}"
    with RunLogger(script="run_pmp_v1_2", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed)
        print(f"[pmp-v1.2] config: {vars(args)}")
        pkl_path = _pkl_for_seed(args.seed)
        print(f"[pmp-v1.2] using dataset: {pkl_path}")
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(pkl_path))

        trainer = _PerModeEmerGNN_PMP_v1_2(
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
            # MNAH parent hyperparams (kept to satisfy fit() scaffold; v1.2
            # does NOT use the residual fusion path)
            mnah_hidden=32,
            mnah_dropout=0.2,
            mnah_init_beta=0.0,  # v1 PMP residual effectively off
            mnah_text_cache=None,
            # v1 PMP cache (mediator vocab + n1/n2)
            pmp_cache_path=args.pmp_cache,
            pmp_hidden=128,
            pmp_dropout=0.2,
            pmp_max_mediators=None,
            # v1.1 cluster cache (reused)
            cluster_cache_path=args.cluster_cache,
            cluster_hidden=args.cluster_hidden,
            cluster_dropout=args.cluster_dropout,
            # v1.2 only
            cluster_max_mediators_per_drug=cap_arg,
        )

        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_s2(trainer, ds)
        print(
            f"[pmp-v1.2] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"cluster_only={m.get('auc_cluster_only')} "
            f"within_only={m.get('auc_within_only')} "
            f"fit={fit_sec / 3600:.2f}h "
            f"(v1.1 ndim32 ref 0.7751)"
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
        print(f"[pmp-v1.2] run_dir: {rl.run_dir}")


if __name__ == "__main__":
    main()

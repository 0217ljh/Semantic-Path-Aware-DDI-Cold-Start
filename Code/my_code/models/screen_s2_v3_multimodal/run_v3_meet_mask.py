"""Run D2 — meeting-node-aware propagation (Round 4 D2).

Mirrors run_v3_llm_edge.py CLI + adds D2 flags per
Notes/Log/d2_meet_mask_design.md §3.5.

Examples (from project root):

  # K4 parity (deterministic seed; --d2-disable):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\\
Semantic-Path-Aware-DDI-Cold-Start && python -u \\
Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py \\
--epochs 5 --tag d2_k4_parity --seed 42 --d2-disable --deterministic"

  # Canonical D2 main (seed42, 100 epochs):
  python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py \\
      --epochs 100 --tag d2_main_seed42 --seed 42 --deterministic

  # Controls:
  python -u .../run_v3_meet_mask.py --epochs 100 --tag d2_shuf_mediators \\
      --seed 42 --d2-shuf-mediators --deterministic
  python -u .../run_v3_meet_mask.py --epochs 100 --tag d2_rand_uniform \\
      --seed 42 --d2-rand-mediators-uniform --deterministic
  python -u .../run_v3_meet_mask.py --epochs 100 --tag d2_freeze_alpha \\
      --seed 42 --d2-freeze-alpha-meet --deterministic
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
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

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

from my_code.models.screen_s2_v3_multimodal.v3_meet_mask_trainer import (  # noqa: E402
    _PerModeEmerGNN_V3MeetMask,
    DEFAULT_MEDIATOR_PARQUET,
)
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
I4_JSON = ROOT / "Code/data/_cache/llm_pharma/i4_typed_sets.json"


def _set_deterministic(seed: int) -> None:
    """Per D2 design §3.5 deterministic seeding contract (CP-1 round 2 add)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: torch.use_deterministic_algorithms(True) is intentionally NOT set;
    # it requires CUBLAS_WORKSPACE_CONFIG env var and disables several sparse
    # kernel paths in EmerGNN. The seeding above is sufficient for trajectory
    # reproducibility in practice.


def _coverage_by_cardinality(pos: pd.DataFrame, neg: pd.DataFrame, mediator_parquet: Path) -> dict:
    """Per-pair mediator-cardinality quartile breakdown for CP-3 diagnostics."""
    if not mediator_parquet.is_file():
        return {}
    print(f"[d2] loading mediator parquet for coverage stats: {mediator_parquet}", flush=True)
    df = pd.read_parquet(mediator_parquet)
    cmap = {}
    for row in df.itertuples(index=False):
        a, b = str(row.drug_a_id), str(row.drug_b_id)
        ca, cb = (a, b) if a <= b else (b, a)
        cmap[(ca, cb)] = int(row.n_mediators)

    def card_array(d: pd.DataFrame) -> np.ndarray:
        out = []
        for a, b in zip(d["drug_a_id"].astype(str), d["drug_b_id"].astype(str)):
            ca, cb = (a, b) if a <= b else (b, a)
            out.append(cmap.get((ca, cb), 0))
        return np.asarray(out)

    return {"pos_card": card_array(pos), "neg_card": card_array(neg)}


def _eval_s2(model, ds, mediator_parquet: Path):
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
    try:
        ch = model.predict_channels(pd.concat([pos, neg], ignore_index=True))
        out["auc_emergnn"] = float(roc_auc_score(y, ch["emergnn"]))
        out["auc_count_only"] = float(roc_auc_score(y, ch["count"]))
        out["auc_i4_only"] = float(roc_auc_score(y, ch["i4"]))
    except Exception as exc:
        print(f"[d2] channel diag failed: {exc}", flush=True)

    # Mediator-cardinality quartile breakdown.
    card_info = _coverage_by_cardinality(pos, neg, mediator_parquet)
    if card_info:
        all_card = np.concatenate([card_info["pos_card"], card_info["neg_card"]])
        # Quartiles by cardinality.
        try:
            bucket_edges = np.unique(np.percentile(all_card, [25, 50, 75]))
            buckets = np.digitize(all_card, bucket_edges)
            out["mediator_cardinality_breakdown"] = {}
            for b in np.unique(buckets):
                mask = buckets == b
                if mask.sum() == 0:
                    continue
                n_pos_b = int(mask[: len(pos)].sum())
                n_neg_b = int(mask[len(pos) :].sum())
                if n_pos_b == 0 or n_neg_b == 0:
                    continue
                sb = s[mask]; yb = y[mask]
                out["mediator_cardinality_breakdown"][f"q{int(b)}"] = {
                    "auc": float(roc_auc_score(yb, sb)),
                    "n_pos": n_pos_b, "n_neg": n_neg_b,
                    "card_range": [
                        int(all_card[mask].min()), int(all_card[mask].max())
                    ],
                }
        except Exception as exc:
            print(f"[d2] cardinality breakdown failed: {exc}", flush=True)

    return out, (pos, neg, s, y)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3)
    p.add_argument("--v2i4-hidden", type=int, default=32)
    p.add_argument("--v2i4-beta-init", type=float, default=0.5)
    p.add_argument("--d2-mediator-parquet", type=Path, default=DEFAULT_MEDIATOR_PARQUET)
    p.add_argument("--d2-disable", action="store_true",
                   help="K4 parity smoke: skip mask construction.")
    p.add_argument("--d2-freeze-alpha-meet", action="store_true",
                   help="K3: alpha_meet=0 frozen + requires_grad=False.")
    p.add_argument("--d2-shuf-mediators", action="store_true",
                   help="K1: cardinality-bucketed derangement.")
    p.add_argument("--d2-rand-mediators-kind-matched", action="store_true",
                   help="K2a: degree+kind-matched random replacement.")
    p.add_argument("--d2-rand-mediators-uniform", action="store_true",
                   help="K2b: uniform random non-drug entities.")
    p.add_argument("--d2-alpha-init", type=float, default=0.0,
                   help="Initial alpha_meet per layer (default 0.0).")
    p.add_argument("--d2-shuffle-seed", type=int, default=12345)
    p.add_argument("--deterministic", action="store_true",
                   help="Set torch/np/cuda seeds at process start.")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    if args.deterministic:
        _set_deterministic(args.seed)

    tag = args.tag or f"d2_seed{args.seed}"
    with RunLogger(script="run_v3_meet_mask", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            epochs=args.epochs, seed=args.seed,
            d2_disable=args.d2_disable,
            d2_freeze_alpha_meet=args.d2_freeze_alpha_meet,
            d2_shuf_mediators=args.d2_shuf_mediators,
            d2_rand_mediators_kind_matched=args.d2_rand_mediators_kind_matched,
            d2_rand_mediators_uniform=args.d2_rand_mediators_uniform,
            d2_alpha_init=args.d2_alpha_init,
            d2_shuffle_seed=args.d2_shuffle_seed,
            deterministic=args.deterministic,
            length=args.length,
            d2_mediator_parquet=str(args.d2_mediator_parquet),
        )
        print(f"[d2] config: {vars(args)}", flush=True)

        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))

        trainer = _PerModeEmerGNN_V3MeetMask(
            n_dim=64, length=args.length, feat="M",
            learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50,
            eval_strategy="epoch", save_strategy="no",
            load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            v2i4_hidden=args.v2i4_hidden, v2i4_beta_init=args.v2i4_beta_init,
            d2_mediator_parquet=args.d2_mediator_parquet,
            d2_disable=args.d2_disable,
            d2_freeze_alpha_meet=args.d2_freeze_alpha_meet,
            d2_shuf_mediators=args.d2_shuf_mediators,
            d2_rand_mediators_kind_matched=args.d2_rand_mediators_kind_matched,
            d2_rand_mediators_uniform=args.d2_rand_mediators_uniform,
            d2_alpha_init=args.d2_alpha_init,
            d2_shuffle_seed=args.d2_shuffle_seed,
        )

        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_s2(trainer, ds, args.d2_mediator_parquet)
        print(
            f"[d2] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"emergnn={m.get('auc_emergnn')} count_only={m.get('auc_count_only')} "
            f"i4_only={m.get('auc_i4_only')} fit={fit_sec/3600:.2f}h "
            f"(v2i4 anchor 0.7804; hard-stop 0.775)",
            flush=True,
        )

        results = {
            "config": vars(args),
            "fit_sec": fit_sec,
            "metrics": {"test_s2": m},
            "d2_summary": getattr(trainer, "_d2_summary", None),
        }
        results["config"]["d2_mediator_parquet"] = str(args.d2_mediator_parquet)
        (rl.run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

        np.savez(
            rl.run_dir / "test_s2_scores.npz",
            pair_a=np.concatenate(
                [pos["drug_a_id"].astype(str).values, neg["drug_a_id"].astype(str).values]
            ),
            pair_b=np.concatenate(
                [pos["drug_b_id"].astype(str).values, neg["drug_b_id"].astype(str).values]
            ),
            y_true=y, y_score=s,
        )

        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m["auc"], nll_s2=m["nll"])
        print(f"[d2] run_dir: {rl.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

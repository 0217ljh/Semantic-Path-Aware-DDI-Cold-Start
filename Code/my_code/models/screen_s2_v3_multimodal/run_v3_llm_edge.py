"""Run D1 — v3 LLM-edge trainer (round 4 backbone integration).

Mirrors run_v2i4.py CLI. Adds D1 flags (--d1-disable / --d1-shuf-token /
--d1-rand-token / --d1-drop-i4-head / --d1-edges-parquet). Same hard-stop
discipline (combined < 0.775 STOP — see Notes/Log/round4_backbone_diff_plan.md §6
and Notes/Log/d1_llm_edge_design.md §4).

Examples (from project root):
  # K4 parity smoke (5 epochs, must match v2i4 trajectory):
  wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/\
Semantic-Path-Aware-DDI-Cold-Start && python -u \
Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py \
--epochs 5 --tag d1_k4_parity --d1-disable"

  # Canonical D1 main (seed42, 100 epochs):
  python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py \
      --epochs 100 --tag d1_main_seed42 --seed 42

  # Controls:
  python -u .../run_v3_llm_edge.py --epochs 100 --tag d1_shuf_token --d1-shuf-token
  python -u .../run_v3_llm_edge.py --epochs 100 --tag d1_rand_token --d1-rand-token
  python -u .../run_v3_llm_edge.py --epochs 100 --tag d1_drop_i4_head --d1-drop-i4-head
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
    raise FileNotFoundError("project root not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.screen_s2_v3_multimodal.v3_llm_edge_trainer import (  # noqa
    _PerModeEmerGNN_V3LLMEdge,
    DEFAULT_EDGES_PARQUET,
)
from my_code.utils.run_logger import RunLogger  # noqa

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
I4_JSON = ROOT / "Code/data/_cache/llm_pharma/i4_typed_sets.json"


def _covered_buckets(pos: pd.DataFrame, neg: pd.DataFrame) -> dict:
    """Tag each test pair by LLM-cache coverage (both/one/neither_covered).

    See Notes/Log/d1_llm_edge_design.md §2.4 covered-vs-uncovered S2 breakdown.
    """
    if not I4_JSON.is_file():
        return {}
    i4_keys = set(json.loads(I4_JSON.read_text(encoding="utf-8")).keys())

    def tag(df: pd.DataFrame) -> np.ndarray:
        a_in = df["drug_a_id"].astype(str).isin(i4_keys).to_numpy()
        b_in = df["drug_b_id"].astype(str).isin(i4_keys).to_numpy()
        tags = np.full(len(df), "neither_covered", dtype=object)
        tags[(a_in & b_in)] = "both_covered"
        tags[(a_in ^ b_in)] = "one_covered"
        return tags

    return {"pos_tags": tag(pos), "neg_tags": tag(neg)}


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

    try:
        ch = model.predict_channels(pd.concat([pos, neg], ignore_index=True))
        out["auc_emergnn"] = float(roc_auc_score(y, ch["emergnn"]))
        out["auc_count_only"] = float(roc_auc_score(y, ch["count"]))
        out["auc_i4_only"] = float(roc_auc_score(y, ch["i4"]))
    except Exception as exc:
        print(f"[d1] channel diag failed: {exc}", flush=True)

    # Covered-vs-uncovered breakdown (per design §2.4 / CP-1 round 1 add).
    cov = _covered_buckets(pos, neg)
    if cov:
        tags = np.concatenate([cov["pos_tags"], cov["neg_tags"]])
        out["coverage_breakdown"] = {}
        for bucket in ("both_covered", "one_covered", "neither_covered"):
            mask = tags == bucket
            if mask.sum() == 0:
                continue
            n_pos_b = int(mask[: len(pos)].sum())
            n_neg_b = int(mask[len(pos) :].sum())
            if n_pos_b == 0 or n_neg_b == 0:
                out["coverage_breakdown"][bucket] = {
                    "auc": None, "n_pos": n_pos_b, "n_neg": n_neg_b
                }
                continue
            sb = s[mask]
            yb = y[mask]
            out["coverage_breakdown"][bucket] = {
                "auc": float(roc_auc_score(yb, sb)),
                "n_pos": n_pos_b, "n_neg": n_neg_b,
            }

    return out, (pos, neg, s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--length", type=int, default=3,
                   help="EmerGNN path-flow length (D1 first pass = 3 for apples-to-apples vs 0.7804).")
    p.add_argument("--v2i4-hidden", type=int, default=32)
    p.add_argument("--v2i4-beta-init", type=float, default=0.5)
    p.add_argument("--v2i4-shuffle-control", action="store_true",
                   help="(inherited from v2i4) permute the readout-side i4 pair-feature drug->set binding.")
    p.add_argument("--v2i4-random-control", action="store_true",
                   help="(inherited from v2i4) random-token control for the readout-side i4 head.")
    p.add_argument("--d1-edges-parquet", type=Path, default=DEFAULT_EDGES_PARQUET)
    p.add_argument("--d1-disable", action="store_true",
                   help="K4 parity smoke: skip backbone LLM-edge injection.")
    p.add_argument("--d1-shuf-token", action="store_true",
                   help="K1 control: permute drug->LLM-token binding (backbone side).")
    p.add_argument("--d1-rand-token", action="store_true",
                   help="K2 control: replace each LLM-edge token with a globally unique rand_<i>.")
    p.add_argument("--d1-drop-i4-head", action="store_true",
                   help="K3 control: freeze β_i4=0; force backbone to carry LLM signal alone.")
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"d1_seed{args.seed}"
    with RunLogger(script="run_v3_llm_edge", tag=tag, seed=args.seed) as rl:
        rl.set_meta(
            epochs=args.epochs, seed=args.seed,
            v2i4_shuffle_control=args.v2i4_shuffle_control,
            v2i4_random_control=args.v2i4_random_control,
            d1_disable=args.d1_disable,
            d1_shuf_token=args.d1_shuf_token,
            d1_rand_token=args.d1_rand_token,
            d1_drop_i4_head=args.d1_drop_i4_head,
            length=args.length,
            d1_edges_parquet=str(args.d1_edges_parquet),
        )
        print(f"[d1] config: {vars(args)}", flush=True)

        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(PKL))

        trainer = _PerModeEmerGNN_V3LLMEdge(
            n_dim=64, length=args.length, feat="M",
            learning_rate=1e-3, batch_size=args.batch_size,
            n_epochs=args.epochs, backbone_kg_source="drugbank", weight_decay=1e-8,
            shuffle_train_mode="S2", log_step_every=50,
            eval_strategy="epoch", save_strategy="no",
            load_best_model_at_end=True, run_dir=str(rl.run_dir),
            mnah_hidden=32, mnah_dropout=0.2, mnah_init_beta=1.0,
            v2i4_hidden=args.v2i4_hidden, v2i4_beta_init=args.v2i4_beta_init,
            v2i4_shuffle_control=args.v2i4_shuffle_control,
            v2i4_random_control=args.v2i4_random_control,
            d1_edges_parquet=args.d1_edges_parquet,
            d1_disable=args.d1_disable,
            d1_shuf_token=args.d1_shuf_token,
            d1_rand_token=args.d1_rand_token,
            d1_drop_i4_head=args.d1_drop_i4_head,
        )

        t0 = time.time()
        trainer.fit(ds, val=ds)
        fit_sec = time.time() - t0

        m, (pos, neg, s, y) = _eval_s2(trainer, ds)

        print(
            f"[d1] test_s2: AUC={m['auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"emergnn={m.get('auc_emergnn')} count_only={m.get('auc_count_only')} "
            f"i4_only={m.get('auc_i4_only')} fit={fit_sec/3600:.2f}h "
            f"(v2i4 anchor 0.7804; hard-stop 0.775)",
            flush=True,
        )
        if "coverage_breakdown" in m:
            print(f"[d1] coverage breakdown: {m['coverage_breakdown']}", flush=True)

        results = {
            "config": vars(args),
            "fit_sec": fit_sec,
            "metrics": {"test_s2": m},
            "d1_injection_summary": getattr(trainer, "_d1_summary", None),
        }
        # Normalize args path for JSON.
        results["config"]["d1_edges_parquet"] = str(args.d1_edges_parquet)
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
        print(f"[d1] run_dir: {rl.run_dir}", flush=True)


if __name__ == "__main__":
    main()

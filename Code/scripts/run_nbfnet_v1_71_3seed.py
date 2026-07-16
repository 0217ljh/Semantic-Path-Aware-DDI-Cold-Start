"""Run NBFNet v1.71 cold-start S2 on the 800-drug 3-seed parquet (seeds 42/43/44).

Sibling of run_nbfnet_v1_71.py. The original loads the legacy seed42 *pickle*
via PairDataset; this variant loads the regenerated 3-seed *parquet* splits
(Code/data/coldddi_legacy/800drug_3seed/seed{N}/{train,val_s2,test_s2}.parquet,
produced by prepare_800drug_seeds.py) so NBFNet trains on the SAME full-G1xG1
protocol and is evaluated on the SAME static val_s2/test_s2 negatives as the
spmn_v2 adapter. This enables a per-seed paired fusion comparison (NBFNet vs
adapter vs rank-blend) across seeds 42/43/44.

It does NOT modify or reimplement any NBFNet logic: `_build_kg_inputs`,
`_eval_split`, the trainer (NBFNetTrainerV171) and RunLogger are imported and
reused verbatim from run_nbfnet_v1_71. The only difference is the data source,
which is fed through a small duck-typed dataset shim exposing the subset of the
PairDataset interface those functions touch (`.splits.train/val_s2/test_s2`,
`.splits.items()`, `.get_negatives(split)`, `.kg`). Read-only on data; writes a
new run dir under Code/runs/.

Only --kg-source merged is supported here (the shim carries no DrugBank KG
object); the merged KG is rebuilt from the same drug pool so seed42 entity2id
matches the legacy run.

Example (1 seed):
  python Code/scripts/run_nbfnet_v1_71_3seed.py --seed 43 --epochs 100 \
      --n-layers 6 --n-dim 32 --batch-size 32 --tag nbfnet_v1_71_3seed_merged_seed43
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Sibling script in the same dir; importing it sets up sys.path (Code/) and
# re-exports the trainer / logger / helper functions we reuse verbatim.
import run_nbfnet_v1_71 as nbf  # noqa: E402

ROOT = nbf.ROOT
THREE_SEED_DIR = ROOT / "Code/data/coldddi_legacy/800drug_3seed"


class _Splits:
    """Minimal stand-in for PairDataset.splits (positives only)."""

    def __init__(self, train: pd.DataFrame, val_s2: pd.DataFrame,
                 test_s2: pd.DataFrame) -> None:
        self.train = train
        self.val_s2 = val_s2
        self.test_s2 = test_s2

    def items(self):
        return [("train", self.train), ("val_s2", self.val_s2),
                ("test_s2", self.test_s2)]


class _DSShim:
    """Duck-typed PairDataset exposing only what _build_kg_inputs / _eval_split use."""

    def __init__(self, splits: _Splits, negatives: dict[str, pd.DataFrame]) -> None:
        self.splits = splits
        self._neg = negatives
        self.kg = None  # merged KG branch only reads optional kg.drug_ids (hasattr-guarded)

    def get_negatives(self, split: str) -> pd.DataFrame:
        return self._neg[split]


def _load_shim(seed: int) -> _DSShim:
    base = THREE_SEED_DIR / f"seed{seed}"
    if not base.is_dir():
        raise FileNotFoundError(
            f"3-seed split dir for seed={seed} not found at {base}. "
            f"Run Code/scripts/prepare_800drug_seeds.py first. "
            f"Available: {sorted(x.name for x in THREE_SEED_DIR.glob('seed*')) if THREE_SEED_DIR.is_dir() else '[none]'}"
        )
    pos: dict[str, pd.DataFrame] = {}
    neg: dict[str, pd.DataFrame] = {}
    for name in ("train", "val_s2", "test_s2"):
        df = pd.read_parquet(base / f"{name}.parquet")
        for col in ("drug_a_id", "drug_b_id"):
            df[col] = df[col].astype(str)
        lab = df["label"].astype(int)
        pos[name] = df.loc[lab == 1, ["drug_a_id", "drug_b_id"]].reset_index(drop=True)
        neg[name] = df.loc[lab == 0, ["drug_a_id", "drug_b_id"]].reset_index(drop=True)
    splits = _Splits(pos["train"], pos["val_s2"], pos["test_s2"])
    return _DSShim(splits, neg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-dim", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--neg-ratio", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--kg-source", type=str, default="merged", choices=["merged"])
    p.add_argument("--shuffle-train-mode", type=str, default="S2")
    p.add_argument("--shuffle-ratio", type=float, default=0.8)
    p.add_argument("--log-step-every", type=int, default=50)
    p.add_argument("--disable-batched-bf", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="disable AMP autocast")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 matmul/cudnn")
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"nbfnet_v1_71_3seed_merged_seed{args.seed}"
    with nbf.RunLogger(script="run_nbfnet_v1_71_3seed", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed, kg_source=args.kg_source,
                    shuffle_train_mode=args.shuffle_train_mode)
        print(f"[nbfnet-v1.71-3seed] config: {vars(args)}", flush=True)

        ds = _load_shim(args.seed)
        print(f"[nbfnet-v1.71-3seed] using 3-seed parquet: {THREE_SEED_DIR / f'seed{args.seed}'} "
              f"train_pos={len(ds.splits.train)} val_s2_pos={len(ds.splits.val_s2)} "
              f"test_s2_pos={len(ds.splits.test_s2)}", flush=True)

        (base_kg, train_ddi, entity2id, n_ent, n_base_rel, ddi_rel_id) = nbf._build_kg_inputs(
            ds, kg_source=args.kg_source)
        print(f"[nbfnet-v1.71-3seed] kg_source={args.kg_source} KG inputs: n_ent={n_ent} "
              f"n_base_rel={n_base_rel} ddi_rel_id={ddi_rel_id} "
              f"n_base_kg_edges={len(base_kg)} n_train_ddi={len(train_ddi)}", flush=True)

        trainer = nbf.NBFNetTrainerV171(
            d=args.n_dim,
            n_layers=args.n_layers,
            mlp_hidden=args.mlp_hidden,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            neg_ratio=args.neg_ratio,
            early_stop_patience=args.early_stop_patience,
            shuffle_train_mode=args.shuffle_train_mode,
            shuffle_ratio=args.shuffle_ratio,
            use_batched_bf=not args.disable_batched_bf,
            log_step_every=args.log_step_every,
            eval_strategy="epoch",
            device="cuda",
            seed=args.seed,
            use_amp=not args.no_amp,
            amp_dtype=args.amp_dtype,
            use_tf32=not args.no_tf32,
            eval_every_n_epochs=args.eval_every_n_epochs,
        )
        trainer.setup_graph(base_kg, train_ddi, n_nodes=n_ent,
                            n_base_rel=n_base_rel, ddi_rel_id=ddi_rel_id)
        trainer.init_model()

        val_pos = ds.splits.val_s2[["drug_a_id", "drug_b_id"]]
        val_neg = ds.get_negatives("val_s2")[["drug_a_id", "drug_b_id"]]

        t0 = time.time()
        fit_metrics = trainer.fit(val_pos, val_neg, entity2id, rl.run_dir)
        fit_sec = time.time() - t0

        m_s2, (pos, neg, s, y) = nbf._eval_split(trainer, ds, entity2id, "test_s2")
        print(
            f"[nbfnet-v1.71-3seed] test_s2: AUC={m_s2['auc']:.4f} AUPRC={m_s2['auprc']:.4f} "
            f"NLL={m_s2['nll']:.4f} best_ep={fit_metrics.get('best_epoch')} "
            f"fit={fit_sec/3600:.2f}h train={fit_metrics.get('train_sec',0)/3600:.2f}h "
            f"eval={fit_metrics.get('eval_sec',0)/3600:.2f}h",
            flush=True,
        )

        (rl.run_dir / "results.json").write_text(json.dumps({
            "config": vars(args),
            "fit_sec": fit_sec,
            "fit_metrics": fit_metrics,
            "metrics": {"test_s2": m_s2},
        }, indent=2))
        np.savez(
            rl.run_dir / "test_s2_scores.npz",
            pair_a=np.concatenate([pos["drug_a_id"].astype(str).values,
                                   neg["drug_a_id"].astype(str).values]),
            pair_b=np.concatenate([pos["drug_b_id"].astype(str).values,
                                   neg["drug_b_id"].astype(str).values]),
            y_true=y, y_score=s,
        )
        rl.set_metrics(fit_time_s=fit_sec, auc_s2=m_s2["auc"], nll_s2=m_s2["nll"])
        print(f"[nbfnet-v1.71-3seed] run_dir: {rl.run_dir}", flush=True)


if __name__ == "__main__":
    main()

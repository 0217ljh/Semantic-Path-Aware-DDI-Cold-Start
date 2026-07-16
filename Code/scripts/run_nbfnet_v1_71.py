"""Run NBFNet v1.71 cold-start S2 binary DDI experiment (created 2026-06-06).

v1.71 = v1.7 paper-faithful model (unchanged) + first-wave runtime acceleration
in the trainer: TF32, AMP autocast (bf16 default), eval cadence, train/eval
timing split. BF math unchanged (precision-only differences). v1.7 stays the
0.7911 merged-KG reference.

Example (1-epoch timing test vs v1.7's ~858s/epoch on merged):
  python Code/scripts/run_nbfnet_v1_71.py --epochs 1 --seed 42 --kg-source merged \
      --n-layers 3 --n-dim 16 --batch-size 16 --tag nbfnet_v1_71_timing
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
    raise FileNotFoundError("could not locate project root (Code/data/KG)")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from baseline.emergnn.kg_builder import N_BASE_REL, build_kg_from_kb  # noqa: E402
from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet  # noqa: E402
from my_code.models.nbfnet_v1_71.nbfnet_trainer import NBFNetTrainerV171  # noqa: E402
from my_code.utils.run_logger import RunLogger  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
MERGED_EDGES = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"


def _pkl_for_seed(seed: int) -> Path:
    p = PKL_DIR / f"seed{int(seed)}.pkl"
    if not p.is_file():
        raise FileNotFoundError(
            f"Dataset pickle for seed={seed} not found at {p}. "
            f"Available: {sorted(x.name for x in PKL_DIR.glob('seed*.pkl'))}"
        )
    return p


def _kg_to_kb_dict(kg) -> dict:
    required = ("enzymes", "targets", "transporters", "carriers", "pathways")
    missing = [a for a in required if not hasattr(kg, a)]
    if missing:
        raise TypeError(
            f"NBFNet KG build requires attributes {required}; got "
            f"{type(kg).__name__} (missing {missing})."
        )
    return {
        "my_enzyme_list": kg.enzymes,
        "my_target_list": kg.targets,
        "my_transporter_list": kg.transporters,
        "my_carrier_list": kg.carriers,
        "my_pathway_list": kg.pathways,
    }


def _build_kg_inputs(ds, kg_source: str = "merged"):
    """Build NBFNet setup_graph inputs from a PairDataset (same as run_nbfnet)."""
    kg = ds.kg
    drug_ids: set[str] = set()
    for _name, df in ds.splits.items():
        drug_ids.update(df["drug_a_id"].astype(str))
        drug_ids.update(df["drug_b_id"].astype(str))
    if hasattr(kg, "drug_ids"):
        drug_ids.update(kg.drug_ids)
    drug_id_list = sorted(drug_ids)

    if kg_source == "merged":
        if not MERGED_EDGES.is_file():
            raise FileNotFoundError(f"merged KG parquet not found at {MERGED_EDGES}")
        art = build_kg_from_merged_parquet(MERGED_EDGES, drug_id_list, verbose=True)
        n_kg_rel = int(art["n_rel"])
    elif kg_source == "drugbank":
        kb = _kg_to_kb_dict(kg)
        art = build_kg_from_kb(kb, drug_id_list, keep_only_known_drugs=True)
        n_kg_rel = N_BASE_REL
    else:
        raise ValueError(f"unknown kg_source={kg_source!r}")

    entity2id = art["entity2id"]
    n_ent = int(art["n_ent"])
    base_kg_triplets = np.asarray(art["triplets"], dtype=np.int64)
    ddi_rel_id = n_kg_rel
    n_base_rel_with_ddi = n_kg_rel + 1

    pos_df = ds.splits.train
    a = pos_df["drug_a_id"].astype(str).map(entity2id)
    b = pos_df["drug_b_id"].astype(str).map(entity2id)
    valid = a.notna() & b.notna()
    if not valid.all():
        print(f"[nbfnet-v1.71] dropping {(~valid).sum()} train DDI rows with unknown drugs",
              flush=True)
    train_ddi_triplets = np.stack(
        [
            a[valid].astype(np.int64).to_numpy(),
            b[valid].astype(np.int64).to_numpy(),
            np.full(int(valid.sum()), ddi_rel_id, dtype=np.int64),
        ],
        axis=1,
    )
    return (base_kg_triplets, train_ddi_triplets, entity2id, n_ent,
            n_base_rel_with_ddi, ddi_rel_id)


def _eval_split(trainer, ds, drug_id_map, split: str) -> tuple[dict, tuple]:
    pos = getattr(ds.splits, split)[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives(split)[["drug_a_id", "drug_b_id"]]
    yp = trainer.predict_proba(pos, drug_id_map)
    yn = trainer.predict_proba(neg, drug_id_map)
    s = np.concatenate([yp, yn])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    out = {
        "auc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "nll": float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
    }
    return out, (pos, neg, s, y)


def main():
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
    p.add_argument("--kg-source", type=str, default="merged", choices=["merged", "drugbank"])
    p.add_argument("--shuffle-train-mode", type=str, default="S2")
    p.add_argument("--shuffle-ratio", type=float, default=0.8)
    p.add_argument("--log-step-every", type=int, default=50)
    p.add_argument("--disable-batched-bf", action="store_true")
    # --- v1.71 first-wave acceleration flags ---
    p.add_argument("--no-amp", action="store_true", help="disable AMP autocast")
    p.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 matmul/cudnn")
    p.add_argument("--eval-every-n-epochs", type=int, default=1)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    tag = args.tag or f"nbfnet_v1_71_seed{args.seed}"
    with RunLogger(script="run_nbfnet_v1_71", tag=tag, seed=args.seed) as rl:
        rl.set_meta(epochs=args.epochs, seed=args.seed, kg_source=args.kg_source,
                    shuffle_train_mode=args.shuffle_train_mode)
        print(f"[nbfnet-v1.71] config: {vars(args)}", flush=True)

        pkl_path = _pkl_for_seed(args.seed)
        print(f"[nbfnet-v1.71] using dataset: {pkl_path}", flush=True)
        from data_utils import PairDataset
        ds = PairDataset.from_pkl(str(pkl_path))

        (base_kg, train_ddi, entity2id, n_ent, n_base_rel, ddi_rel_id) = _build_kg_inputs(
            ds, kg_source=args.kg_source)
        print(f"[nbfnet-v1.71] kg_source={args.kg_source} KG inputs: n_ent={n_ent} "
              f"n_base_rel={n_base_rel} ddi_rel_id={ddi_rel_id} "
              f"n_base_kg_edges={len(base_kg)} n_train_ddi={len(train_ddi)}", flush=True)

        trainer = NBFNetTrainerV171(
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
            # v1.71 accel
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

        m_s2, (pos, neg, s, y) = _eval_split(trainer, ds, entity2id, "test_s2")
        print(
            f"[nbfnet-v1.71] test_s2: AUC={m_s2['auc']:.4f} AUPRC={m_s2['auprc']:.4f} "
            f"NLL={m_s2['nll']:.4f} best_ep={fit_metrics.get('best_epoch')} "
            f"fit={fit_sec/3600:.2f}h train={fit_metrics.get('train_sec',0)/3600:.2f}h "
            f"eval={fit_metrics.get('eval_sec',0)/3600:.2f}h "
            f"(refs: EmerGNN 0.7458, MNAH 0.7670, v1.5A 0.7764, v2i4 0.7804; v1.7 merged 0.7911)",
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
        print(f"[nbfnet-v1.71] run_dir: {rl.run_dir}", flush=True)


if __name__ == "__main__":
    main()

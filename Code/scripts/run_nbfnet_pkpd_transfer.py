"""NBFNet v1.7 PK/PD transfer experiment (train on one bucket, test on another).

Same KG / model / config as the exp1 reference run (kg_source=merged, L=3,
dim=16, 100 epochs) so results are comparable to the existing ALL->PD number
(test_s2 PD AUC 0.727). The ONLY thing that changes is which positives are
injected into the vKG + supervised (train-pairs) and which are evaluated
(test-pairs). Negatives for eval are the standard split negatives (the
negative-sample question is analyzed separately, post-hoc).

Reuses the exact KG build of run_nbfnet (build_kg_from_merged_parquet over the
full drug vocab), so node ids / base KG are identical across all transfer runs.

Run (from project root, via WSL conda env project_1):
  python Code/scripts/run_nbfnet_pkpd_transfer.py \
    --train-pairs Code/experiments/pkpd_transfer/data/train_PK.csv \
    --val-pairs   Code/experiments/pkpd_transfer/data/val_s2_PK.csv \
    --test-pairs  Code/experiments/pkpd_transfer/data/test_s2_PD.csv \
    --tag exp2_PKtrain_PDtest
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
    raise FileNotFoundError("could not locate project root")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet  # noqa: E402
from my_code.models.nbfnet_v1_7.nbfnet_trainer import NBFNetTrainer  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"
MERGED_EDGES = ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
EXP_DIR = ROOT / "Code/experiments/pkpd_transfer"


def _build_kg(ds):
    """Identical KG build to run_nbfnet (merged, full drug vocab)."""
    kg = ds.kg
    drug_ids: set[str] = set()
    for _n, df in ds.splits.items():
        drug_ids.update(df["drug_a_id"].astype(str))
        drug_ids.update(df["drug_b_id"].astype(str))
    if hasattr(kg, "drug_ids"):
        drug_ids.update(kg.drug_ids)
    drug_id_list = sorted(drug_ids)
    art = build_kg_from_merged_parquet(MERGED_EDGES, drug_id_list, verbose=True)
    entity2id = art["entity2id"]
    n_ent = int(art["n_ent"])
    base_kg = np.asarray(art["triplets"], dtype=np.int64)
    n_kg_rel = int(art["n_rel"])
    ddi_rel_id = n_kg_rel
    n_base_rel = n_kg_rel + 1
    return entity2id, n_ent, base_kg, n_base_rel, ddi_rel_id


def _pairs_to_ddi(pairs_df, entity2id, ddi_rel_id):
    a = pairs_df["drug_a_id"].astype(str).map(entity2id)
    b = pairs_df["drug_b_id"].astype(str).map(entity2id)
    valid = a.notna() & b.notna()
    if not valid.all():
        print(f"[transfer] dropping {(~valid).sum()} pairs with unknown drugs", flush=True)
    return np.stack([
        a[valid].astype(np.int64).to_numpy(),
        b[valid].astype(np.int64).to_numpy(),
        np.full(int(valid.sum()), ddi_rel_id, dtype=np.int64),
    ], axis=1)


def _eval(trainer, pos_df, neg_df, entity2id):
    yp = trainer.predict_proba(pos_df[["drug_a_id", "drug_b_id"]], entity2id)
    yn = trainer.predict_proba(neg_df[["drug_a_id", "drug_b_id"]], entity2id)
    s = np.concatenate([yp, yn])
    y = np.concatenate([np.ones(len(pos_df)), np.zeros(len(neg_df))])
    return {
        "auc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "nll": float(log_loss(y, np.clip(s, 1e-7, 1 - 1e-7))),
        "n_pos": int(len(pos_df)), "n_neg": int(len(neg_df)),
    }, (s, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-pairs", required=True, type=str)
    p.add_argument("--val-pairs", required=True, type=str)
    p.add_argument("--test-pairs", required=True, type=str)
    p.add_argument("--extra-test-pairs", nargs="*", default=[],
                   help="additional test-pair CSVs evaluated with the SAME trained model "
                        "(train once, eval many). Each saved separately by file stem.")
    p.add_argument("--test-neg-split", type=str, default="test_s2")
    p.add_argument("--val-neg-split", type=str, default="val_s2")
    p.add_argument("--tag", required=True, type=str)
    # config defaults match exp1 (merged L3 dim16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-dim", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--neg-ratio", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--shuffle-train-mode", type=str, default="S2")
    p.add_argument("--shuffle-ratio", type=float, default=0.8)
    p.add_argument("--log-step-every", type=int, default=50)
    p.add_argument("--disable-batched-bf", action="store_true")
    args = p.parse_args()

    out_dir = EXP_DIR / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[transfer] config: {vars(args)}", flush=True)
    print(f"[transfer] out_dir: {out_dir}", flush=True)

    from data_utils import PairDataset
    pkl = PKL_DIR / f"seed{args.seed}.pkl"
    ds = PairDataset.from_pkl(str(pkl))

    entity2id, n_ent, base_kg, n_base_rel, ddi_rel_id = _build_kg(ds)

    train_pairs = pd.read_csv(args.train_pairs)
    val_pairs = pd.read_csv(args.val_pairs)
    test_pairs = pd.read_csv(args.test_pairs)
    train_ddi = _pairs_to_ddi(train_pairs, entity2id, ddi_rel_id)
    print(f"[transfer] n_ent={n_ent} n_base_rel={n_base_rel} ddi_rel_id={ddi_rel_id} "
          f"base_kg={len(base_kg)} train_ddi={len(train_ddi)} "
          f"val_pos={len(val_pairs)} test_pos={len(test_pairs)}", flush=True)

    val_neg = ds.get_negatives(args.val_neg_split)[["drug_a_id", "drug_b_id"]]
    test_neg = ds.get_negatives(args.test_neg_split)[["drug_a_id", "drug_b_id"]]

    trainer = NBFNetTrainer(
        d=args.n_dim, n_layers=args.n_layers, mlp_hidden=args.mlp_hidden,
        n_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        neg_ratio=args.neg_ratio, early_stop_patience=args.early_stop_patience,
        shuffle_train_mode=args.shuffle_train_mode, shuffle_ratio=args.shuffle_ratio,
        use_batched_bf=not args.disable_batched_bf, log_step_every=args.log_step_every,
        eval_strategy="epoch", device="cuda", seed=args.seed,
    )
    trainer.setup_graph(base_kg, train_ddi, n_nodes=n_ent,
                        n_base_rel=n_base_rel, ddi_rel_id=ddi_rel_id)
    trainer.init_model()

    t0 = time.time()
    fit_metrics = trainer.fit(val_pairs[["drug_a_id", "drug_b_id"]], val_neg,
                              entity2id, out_dir)
    fit_sec = time.time() - t0

    # Evaluate the primary test set + every extra test set with the SAME model.
    test_metrics = {}
    for tp_path in [args.test_pairs, *args.extra_test_pairs]:
        stem = Path(tp_path).stem
        tp = pd.read_csv(tp_path)
        m, (s, y) = _eval(trainer, tp, test_neg, entity2id)
        test_metrics[stem] = m
        print(f"[transfer] {args.tag} | {stem}: AUC={m['auc']:.4f} "
              f"AUPRC={m['auprc']:.4f} NLL={m['nll']:.4f} "
              f"n_pos={m['n_pos']} n_neg={m['n_neg']}", flush=True)
        np.savez(
            out_dir / f"test_scores__{stem}.npz",
            pair_a=np.concatenate([tp["drug_a_id"].astype(str).values,
                                   test_neg["drug_a_id"].astype(str).values]),
            pair_b=np.concatenate([tp["drug_b_id"].astype(str).values,
                                   test_neg["drug_b_id"].astype(str).values]),
            y_true=y, y_score=s,
        )

    (out_dir / "results.json").write_text(json.dumps({
        "tag": args.tag, "config": vars(args),
        "train_pairs": str(args.train_pairs),
        "test_pairs": [args.test_pairs, *args.extra_test_pairs],
        "fit_sec": fit_sec, "fit_metrics": fit_metrics, "test_metrics": test_metrics,
    }, indent=2))
    print(f"[transfer] fit={fit_sec/3600:.2f}h  saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()

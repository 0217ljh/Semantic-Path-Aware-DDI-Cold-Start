"""Smoke test: EmerGNN multilabel (TWOSIDES) unified port.

Subsamples twoside multilabel cold_s2 fold0 to a few hundred pairs, trains 1
epoch, and checks: fit runs; predict shape (n, 200); non-degenerate; 200-multihot
correctly built (spot-check y_label_ids -> bits); metrics compute.

Run:
  wsl bash -ic "conda activate project_1 && cd <root> && \
    python Code/scripts/smoke_emergnn_multilabel.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from data_utils import unified_loader as U  # noqa: E402
from data_utils.leaf_adapter import make_multilabel_bundle, _dense_multihot  # noqa: E402
from baseline.emergnn.multi_label_cls._core_twoside import (  # noqa: E402
    BaseModelTwoside, _fold_to_repro_dirname,
)

DATA_ROOT = ROOT / "Code" / "data" / "ddi_unified"
N_SUB = 300          # subsample train pairs
N_TEST_SUB = 60      # subsample test pairs
FOLD = "fold0"


def main() -> None:
    print("[smoke] loading leaf twosides/multilabel/cold_s2/fold0", flush=True)
    leaf = U.load_leaf(str(DATA_ROOT), "twosides", "multilabel", "cold_s2", FOLD)
    res = leaf.resources
    n_labels = int(res.meta["labels"]["n_labels"])
    print(f"[smoke] n_labels={n_labels} train={leaf.train.shape} test={leaf.test.shape}")
    assert n_labels == 200, n_labels

    parent = U.unified.layout_dir(str(DATA_ROOT), "twosides", "multilabel", "cold_s2")
    fdir = parent / FOLD
    tr_links = pd.read_parquet(fdir / "train_pair_links.parquet")
    te_links = pd.read_parquet(fdir / "test_pair_links.parquet")

    # ---- subsample train pairs (keep paired pos/neg) ----
    tr_links_sub = tr_links.iloc[:N_SUB].reset_index(drop=True)
    keep_ids = set(tr_links_sub["pos_pair_id"]) | set(tr_links_sub["neg_pair_id"])
    train_sub = leaf.train[leaf.train["pair_id"].isin(keep_ids)].reset_index(drop=True)

    # small val = reuse a slice of train links so eval branch runs
    val_links_sub = tr_links.iloc[N_SUB:N_SUB + 40].reset_index(drop=True)
    val_keep = set(val_links_sub["pos_pair_id"]) | set(val_links_sub["neg_pair_id"])
    val_sub = leaf.train[leaf.train["pair_id"].isin(val_keep)].reset_index(drop=True)

    bundle = make_multilabel_bundle(train_sub, val_sub, tr_links_sub, val_links_sub,
                                    n_labels=n_labels)
    print(f"[smoke] bundle train_pos_ht={bundle['train_pos_ht'].shape} "
          f"train_pos_y={bundle['train_pos_y'].shape} "
          f"val_pos_ht={bundle['val_pos_ht'].shape}")

    # ---- multihot spot-check: pick a pos pair, compare bits vs its y_label_ids ----
    first_pos_id = int(tr_links_sub.iloc[0]["pos_pair_id"])
    pos_row = train_sub[train_sub["pair_id"] == first_pos_id].iloc[0]
    gold_ids = sorted(int(i) for i in pos_row["y_label_ids"])
    mh = _dense_multihot([pos_row["y_label_ids"]], n_labels)[0]
    got_bits = sorted(np.nonzero(mh)[0].tolist())
    assert got_bits == gold_ids, (got_bits, gold_ids)
    assert mh.sum() == len(gold_ids)
    # verify the paired neg carries the SAME multihot (endpoint corruption)
    assert np.array_equal(bundle["train_pos_y"][0], bundle["train_neg_y"][0])
    print(f"[smoke] multihot OK: pair {first_pos_id} -> {len(gold_ids)} bits "
          f"{got_bits[:6]}{'...' if len(got_bits) > 6 else ''}")

    # ---- fit 1 epoch ----
    core = BaseModelTwoside(
        kg_source=res.kg.source, split_code=res.meta["split_code"], fold=FOLD,
        n_labels=n_labels, n_dim=64, length=3, feat="M",
        learning_rate=3e-3, weight_decay=1e-6, batch_size=32,
        test_batch_size=16, n_epochs=1, device="auto", log_step_every=5,
    )
    print("[smoke] fitting 1 epoch (real KG)...", flush=True)
    core.fit(bundle)

    # ---- confirm cumulative KG graphs: train < valid < test ----
    ec = core._edge_counts
    print(f"[smoke] edge counts: train={ec['train_graph_edges']} "
          f"valid={ec['valid_graph_edges']} test={ec['test_graph_edges']}")
    assert ec["train_graph_edges"] < ec["valid_graph_edges"] < ec["test_graph_edges"], ec

    # ---- predict on subsampled test ----
    test_sub = leaf.test.iloc[:N_TEST_SUB].reset_index(drop=True)
    ht = test_sub[["drug_a_id", "drug_b_id"]].to_numpy().astype(np.int64)
    scores = core.predict_proba(ht)
    print(f"[smoke] predict scores shape={scores.shape}")
    assert scores.shape == (len(test_sub), n_labels), scores.shape
    assert np.isfinite(scores).all(), "non-finite scores"
    assert scores.min() >= 0.0 and scores.max() <= 1.0, (scores.min(), scores.max())
    non_degenerate = float(scores.std()) > 1e-6
    print(f"[smoke] scores min={scores.min():.4f} max={scores.max():.4f} "
          f"mean={scores.mean():.4f} std={scores.std():.6f} "
          f"non_degenerate={non_degenerate}")
    assert non_degenerate, "degenerate (constant) predictions"

    # ---- metrics compute (reuse runner's multilabel metric on full-test predict) ----
    from run_baseline_unified import _multilabel_metrics
    full_ht = leaf.test[["drug_a_id", "drug_b_id"]].to_numpy().astype(np.int64)
    full_scores = core.predict_proba(full_ht)
    metrics = _multilabel_metrics(leaf.test, full_scores, te_links, n_labels, train_sub)
    print(f"[smoke] metrics={metrics}")
    assert np.isfinite(metrics["macro_auroc"]), "macro_auroc not finite"
    assert np.isfinite(metrics["macro_auprc"]), "macro_auprc not finite"
    assert "n_zero_pos_labels_train" in metrics
    assert "test_positive_on_unseen_label_rate" in metrics

    print("[smoke] PASS", flush=True)


if __name__ == "__main__":
    main()

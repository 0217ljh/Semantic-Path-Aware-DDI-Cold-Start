"""Run a UNIFIED baseline on one leaf (task/dataset/regime/split/fold) of ddi_unified.

Loads the leaf via data_utils.unified_loader, trains a `UnifiedBaseline` on its
train/val, predicts on its test pairs, computes task metrics, and writes a run dir
(results.json + train.log) per CLAUDE.md run-logging.

Example (EmerGNN binary, drugbank_latest_full, cold-S2, fold0):
  python Code/scripts/run_baseline_unified.py --baseline emergnn --task binary \
      --dataset ddi_full --split cold_s2 --fold fold0 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from data_utils import unified_loader as U  # noqa: E402
from baseline.unified_base import get_unified  # noqa: E402

#: (baseline, task) -> module that defines + registers the UnifiedBaseline subclass.
_MODULES = {
    ("emergnn", "binary"): "baseline.emergnn.binary_cls.baseline_unified",
    ("emergnn", "multiclass"): "baseline.emergnn.multi_cls.baseline_unified",
    ("emergnn", "multilabel"): "baseline.emergnn.multi_label_cls.baseline_unified",
    ("ssi_ddi", "binary"): "baseline.ssi_ddi.binary_cls.baseline_unified",
    ("ssi_ddi", "multiclass"): "baseline.ssi_ddi.multi_cls.baseline_unified",
    ("ssi_ddi", "multilabel"): "baseline.ssi_ddi.multi_label_cls.baseline_unified",
    ("hdn_ddi", "binary"): "baseline.hdn_ddi.binary_cls.baseline_unified",
    ("hdn_ddi", "multiclass"): "baseline.hdn_ddi.multi_cls.baseline_unified",
    ("hdn_ddi", "multilabel"): "baseline.hdn_ddi.multi_label_cls.baseline_unified",
    ("sumgnn", "binary"): "baseline.sumgnn.binary_cls.baseline_unified",
    ("sumgnn", "multiclass"): "baseline.sumgnn.multi_cls.baseline_unified",
    ("sumgnn", "multilabel"): "baseline.sumgnn.multi_label_cls.baseline_unified",
    ("mrcgnn", "binary"): "baseline.mrcgnn.binary_cls.baseline_unified",
    ("mrcgnn", "multiclass"): "baseline.mrcgnn.multi_cls.baseline_unified",
    ("mrcgnn", "multilabel"): "baseline.mrcgnn.multi_label_cls.baseline_unified",
    ("tiger", "binary"): "baseline.tiger.binary_cls.baseline_unified",
    ("tiger", "multiclass"): "baseline.tiger.multi_cls.baseline_unified",
    ("tiger", "multilabel"): "baseline.tiger.multi_label_cls.baseline_unified",
    ("mkg_fenn", "binary"): "baseline.mkg_fenn.binary_cls.baseline_unified",
    ("mkg_fenn", "multiclass"): "baseline.mkg_fenn.multi_cls.baseline_unified",
    ("mkg_fenn", "multilabel"): "baseline.mkg_fenn.multi_label_cls.baseline_unified",
    ("knowddi", "multiclass"): "baseline.knowddi.multi_cls.baseline_unified",
    ("knowddi", "multilabel"): "baseline.knowddi.multi_label_cls.baseline_unified",
    ("knowddi", "binary"): "baseline.knowddi.binary_cls.baseline_unified",
}


def _binary_metrics(y_true, y_score) -> dict:
    """Binary DDI (codex 5-metric set; PRIMARY = auprc). + score-dispersion diagnostics."""
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 matthews_corrcoef, roc_auc_score)
    pred = (y_score >= 0.5).astype(int)
    disp = {"score_min": float(y_score.min()), "score_max": float(y_score.max()),
            "score_mean": float(y_score.mean()), "score_std": float(y_score.std())}
    return {"auprc": float(average_precision_score(y_true, y_score)),   # PRIMARY
            "auroc": float(roc_auc_score(y_true, y_score)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, pred)),
            "mcc": float(matthews_corrcoef(y_true, pred)),
            "n_test": int(len(y_true)), "n_pos": int(y_true.sum()), **disp}


def _multiclass_metrics(y_true, scores, oov_mask) -> dict:
    """Multiclass DDI-event (codex 5-metric set; PRIMARY = macro_f1). argmax over the
    GLOBAL class axis -> OOV-in-train gold rows (oov_mask) can't be predicted, so they
    are automatically WRONG in all 5 metrics (codex); oov_target_rate is a diagnostic."""
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 precision_score, recall_score)
    pred = scores.argmax(axis=1)
    return {"macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),  # PRIMARY
            "accuracy": float(accuracy_score(y_true, pred)),
            "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(y_true, pred)),
            "oov_target_rate": float(np.mean(oov_mask)),   # diagnostic (not a headline metric)
            "n_test": int(len(y_true)), "n_classes_gold": int(len(set(y_true)))}


def _multilabel_metrics(test_df, scores, pair_links, n_labels, train_df) -> dict:
    """Macro AUROC + AUPRC over the n_labels labels, averaged ONLY over labels with
    BOTH classes present in the (paired) test set. Faithful to upstream TWOSIDES
    evaluate (base_model.py:86-106): for each label r, positives = pos-pair scores
    at column r for rows whose multihot[r]>0; negatives = the endpoint-corrupting
    neg-pair scores at column r for the SAME rows.

    Plus support stats:
      n_zero_pos_labels_train: # labels with no positive train pair.
      test_positive_on_unseen_label_rate: fraction of test positive (row,label)
        activations whose label never appeared as a positive in train.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    def _multihot(df):
        m = np.zeros((len(df), n_labels), dtype=np.float32)
        for i, ids in enumerate(df["y_label_ids"].to_numpy()):
            idx = np.asarray(list(ids), dtype=np.int64) if ids is not None else np.zeros(0, np.int64)
            if idx.size:
                m[i, idx] = 1.0
        return m

    by_id = test_df.reset_index(drop=True)
    id2row = {int(pid): i for i, pid in enumerate(by_id["pair_id"].to_numpy())}
    pos_rows = np.array([id2row[int(p)] for p in pair_links["pos_pair_id"]], dtype=np.int64)
    neg_rows = np.array([id2row[int(p)] for p in pair_links["neg_pair_id"]], dtype=np.int64)
    pos_scores = scores[pos_rows]                    # (P, n_labels)
    neg_scores = scores[neg_rows]                    # (P, n_labels)
    pos_multihot = _multihot(by_id.iloc[pos_rows].reset_index(drop=True))

    rocs, prs = [], []
    micro_score, micro_label = [], []          # pooled decisions across all labels (micro)
    n_labels_scored = 0
    for r in range(n_labels):
        idx = pos_multihot[:, r] > 0
        k = int(idx.sum())
        if k == 0:
            continue
        score = np.concatenate([pos_scores[idx, r], neg_scores[idx, r]])
        label = np.concatenate([np.ones(k), np.zeros(k)])
        if label.min() == label.max():
            continue
        rocs.append(roc_auc_score(label, score))
        prs.append(average_precision_score(label, score))
        micro_score.append(score); micro_label.append(label)
        n_labels_scored += 1

    # train support stats
    train_pos = train_df[train_df["is_positive"] == 1]
    train_label_pos = np.zeros(n_labels, dtype=np.int64)
    for ids in train_pos["y_label_ids"].to_numpy():
        idx = np.asarray(list(ids), dtype=np.int64) if ids is not None else np.zeros(0, np.int64)
        if idx.size:
            train_label_pos[idx] += 1
    seen_labels = set(np.nonzero(train_label_pos)[0].tolist())
    n_zero_pos_labels_train = int((train_label_pos == 0).sum())

    total_act, unseen_act = 0, 0
    for ids in test_df[test_df["is_positive"] == 1]["y_label_ids"].to_numpy():
        idx = np.asarray(list(ids), dtype=np.int64) if ids is not None else np.zeros(0, np.int64)
        for lb in idx.tolist():
            total_act += 1
            if lb not in seen_labels:
                unseen_act += 1

    # micro (pooled over all label decisions); F1/accuracy at threshold 0.5 (sigmoid probs)
    from sklearn.metrics import accuracy_score as _acc, f1_score as _f1
    if micro_score:
        ms = np.concatenate(micro_score); ml = np.concatenate(micro_label)
        mp = (ms >= 0.5).astype(int)
        micro_auprc = float(average_precision_score(ml, ms))
        micro_f1 = float(_f1(ml, mp, zero_division=0))
        label_micro_accuracy = float(_acc(ml, mp))
    else:
        micro_auprc = micro_f1 = label_micro_accuracy = float("nan")

    disp = {"score_min": float(scores.min()), "score_max": float(scores.max()),
            "score_mean": float(scores.mean()), "score_std": float(scores.std())}
    return {
        # codex 5-metric set (PRIMARY = macro_auprc); F1/acc at threshold 0.5
        "macro_auprc": float(np.mean(prs)) if prs else float("nan"),   # PRIMARY
        "macro_auroc": float(np.mean(rocs)) if rocs else float("nan"),
        "micro_auprc": micro_auprc,
        "micro_f1": micro_f1,
        "label_micro_accuracy": label_micro_accuracy,
        # diagnostics
        "n_labels_scored": n_labels_scored,
        "n_zero_pos_labels_train": n_zero_pos_labels_train,
        "test_positive_on_unseen_label_rate": float(unseen_act / total_act) if total_act else 0.0,
        "n_test": int(len(test_df)), **disp,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--task", required=True, choices=["binary", "multiclass", "multilabel"])
    ap.add_argument("--dataset", required=True, help="group key: ddi_full/ddi800/deng/ryu/twosides")
    ap.add_argument("--split", required=True, choices=["transductive", "cold_s1", "cold_s2"])
    ap.add_argument("--fold", default="fold0")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--root", default=str(ROOT / "Code/data/ddi_unified"))
    args = ap.parse_args()

    key = (args.baseline, args.task)
    if key not in _MODULES:
        raise SystemExit(f"no unified baseline registered for {key}; have {list(_MODULES)}")
    __import__(_MODULES[key])  # registers the subclass
    cls = get_unified(args.baseline, args.task)

    run_id = f"{args.baseline}__{args.dataset}__{args.split}__{args.fold}"
    run_dir = ROOT / "Code/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] loading leaf {args.dataset}/{args.task}/{args.split}/{args.fold}", flush=True)
    leaf = U.load_leaf(args.root, args.dataset, args.task, args.split, args.fold)
    print(f"[run] train={leaf.train.shape} val={leaf.val.shape} test={leaf.test.shape} "
          f"regime={leaf.regime}/{leaf.split_code}", flush=True)

    model = cls(n_epochs=args.epochs, device=args.device, run_dir=run_dir)
    t0 = time.time()
    model.fit_leaf(leaf)
    fit_s = time.time() - t0

    scores = np.asarray(model.predict(leaf.test))
    if args.task == "binary":
        metrics = _binary_metrics(leaf.test["y_bin"].to_numpy(), scores)
    elif args.task == "multiclass":
        oov = (leaf.test["y_cls_train"].to_numpy() == -1)
        metrics = _multiclass_metrics(leaf.test["y_cls"].to_numpy(), scores, oov)
    elif args.task == "multilabel":
        n_labels = int(leaf.resources.meta["labels"]["n_labels"])
        parent = U.unified.layout_dir(args.root, args.dataset, args.task, args.split)
        test_links = pd.read_parquet(parent / args.fold / "test_pair_links.parquet")
        metrics = _multilabel_metrics(leaf.test, scores, test_links, n_labels, leaf.train)
    else:
        raise SystemExit(f"metrics for task {args.task} not wired yet")

    results = {"run_id": run_id, "baseline": args.baseline, "task": args.task,
               "dataset": args.dataset, "split": args.split, "fold": args.fold,
               "epochs": args.epochs, "fit_time_s": round(fit_s, 1), "metrics": metrics}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[run] DONE {run_id}  metrics={metrics}  fit={fit_s:.1f}s -> {run_dir}", flush=True)
    # Torch/torchdrug CUDA-extension teardown can return a nonzero process exit
    # code in some launch contexts (detached / no-tty) even after a fully
    # successful run. Results are already persisted + flushed above, so force a
    # clean exit 0 to avoid false "failed" status in orchestration/monitoring.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

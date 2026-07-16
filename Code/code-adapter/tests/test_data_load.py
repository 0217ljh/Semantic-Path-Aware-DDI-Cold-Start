"""Stage-1 acceptance test for the code-adapter dataset loader.

Success criteria (user + codex):
  1. loads BOTH datasets (drugbank_ryu, drugbank_latest_partial)
  2. loads different folds (fold0/2/4)
  3. returns a RankData class the pipeline can consume, with correct semantics.

Run (WSL conda env, from project root):
  wsl bash -ic "conda activate project_1 && cd <root> && \
      python Code/code-adapter/tests/test_data_load.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# put Code/code-adapter (hyphen dir, not a package) on sys.path, then import its modules
_ADAPTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ADAPTER))

from specs import TaskSpec          # noqa: E402
from data.loader import (RankData, SUPPORTED_DATASETS, load_rank_data)  # noqa: E402

# expected multiclass class counts + one exact post-filter train count (verified earlier)
N_CLASSES = {"drugbank_ryu": 86, "drugbank_latest_partial": 165}
EXACT_MULTI_TRAIN = {("drugbank_ryu", "fold0"): 71560, ("drugbank_latest_partial", "fold0"): 35351}

_fails: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        _fails.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


def drug_set(pairs: np.ndarray) -> set:
    return set(pairs.reshape(-1).tolist()) if len(pairs) else set()


def check_leaf(ds: str, fold: str, task: TaskSpec, kind: str) -> None:
    print(f"\n== {ds} / {kind} / {fold} ==")
    d = load_rank_data(ds, task, fold)
    check(isinstance(d, RankData), "returns RankData")
    check(len(d.train_pairs) == len(d.train_labels) > 0, "train pairs/labels aligned + nonempty")
    check(len(d.cold_val_pairs) == len(d.cold_val_labels) > 0, "val pairs/labels aligned + nonempty")
    check(len(d.cold_test_pairs) == len(d.cold_test_labels) > 0, "test pairs/labels aligned + nonempty")
    check(d.ref_logdeg is None, "ref_logdeg is None at stage 1 (no KG)")
    check(np.array_equal(d.train_drugs, np.unique(d.train_pairs.reshape(-1))),
          "train_drugs == unique(train_pairs)")

    # drug-pool disjointness (train vs val, train vs test); val-vs-test NOT assumed
    tr, va, te = drug_set(d.train_pairs), drug_set(d.cold_val_pairs), drug_set(d.cold_test_pairs)
    check(len(tr & va) == 0, "train/val drug pools disjoint")
    check(len(tr & te) == 0, "train/test drug pools disjoint")

    if task.is_binary:
        check(d.seen_ddi.shape[1] == 3 and np.all(d.seen_ddi[:, 2].astype(np.int64) == 0),
              "binary seen_ddi type col all 0")
        # train_neg is exactly the y_bin==0 train rows -> count == (#train - #pos)
        n_pos = int((d.train_labels == 1).sum())
        check(len(d.train_neg) == len(d.train_labels) - n_pos, "train_neg count == #train negatives")
        check(set(np.unique(d.train_labels).tolist()) <= {0, 1}, "binary labels in {0,1}")
    else:
        check(len(d.train_neg) == 0, "multiclass train_neg empty")
        for nm, y in (("train", d.train_labels), ("val", d.cold_val_labels), ("test", d.cold_test_labels)):
            check(int(y.min()) >= 0, f"{nm} labels >= 0 (closed-set filtered)")
            check(int(y.max()) < task.n_classes, f"{nm} labels < n_classes")
        check(d.seen_ddi.shape[1] == 3, "multiclass seen_ddi typed (3 cols)")
        if (ds, fold) in EXACT_MULTI_TRAIN:
            exp = EXACT_MULTI_TRAIN[(ds, fold)]
            check(len(d.train_pairs) == exp, f"exact multi train count == {exp}")


def main() -> int:
    for ds in SUPPORTED_DATASETS:
        for fold in ("fold0", "fold2", "fold4"):
            check_leaf(ds, fold, TaskSpec.binary(), "binary")
            check_leaf(ds, fold, TaskSpec.multiclass(N_CLASSES[ds]), "multi")

    # invalid inputs must fail cleanly
    print("\n== invalid-input guards ==")
    for bad_ds in ("nope", "drugbank_deng"):
        try:
            load_rank_data(bad_ds, TaskSpec.binary(), "fold0"); check(False, f"reject dataset {bad_ds}")
        except ValueError:
            check(True, f"reject dataset {bad_ds}")
    try:
        load_rank_data("drugbank_ryu", TaskSpec.binary(), "fold9"); check(False, "reject fold9")
    except ValueError:
        check(True, "reject fold9")
    try:
        load_rank_data("drugbank_ryu", TaskSpec.multiclass(5), "fold0"); check(False, "reject wrong K=5")
    except ValueError:
        check(True, "reject wrong K=5 (label-range guard)")

    print(f"\n{'='*50}\n{'ALL PASS' if not _fails else f'{len(_fails)} FAILURES'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

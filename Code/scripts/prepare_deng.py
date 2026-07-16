"""Materialize drugbank_deng (Deng-65, MRCGNN format) into the unified all-regime layout.

Deng et al. 2020 ("DDIMDL"): ~570 drugs, 65 DDI-event types. Source ships 5 warm CV
folds of `d1,type,d2`. Produces transductive/S0 (from source folds) + inductive/S1,S2
(drug-disjoint 5-fold, generated) for binary + multiclass, via data_utils.mrcgnn_cv.
SMILES from drug_listxiao.csv (covers all used drugs). Read-only on source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))
from data_utils import mrcgnn_cv  # noqa: E402

SRC = ROOT / "Paper/Reference/Original-Code/MRCGNN/Deng's dataset"
OUT = ROOT / "Code/data/ddi_unified"
N_CLASSES = 65
FOLD_FILES = {"train": "ddi_training1xiao.csv", "val": "ddi_validation1xiao.csv",
              "test": "ddi_test1xiao.csv"}


def main() -> None:
    smi = pd.read_csv(SRC / "drug_listxiao.csv", header=None, names=["drug_id", "smiles"])
    label_vocab = pd.DataFrame({"label_idx_global": np.arange(N_CLASSES, dtype=np.int64),
                                "label_name": [f"deng_event_{i}" for i in range(N_CLASSES)]})
    st = mrcgnn_cv.build(group="deng", src_dir=SRC, fold_files=FOLD_FILES,
                         n_source_folds=5, smiles=smi, n_classes=N_CLASSES,
                         label_vocab=label_vocab, out_root=OUT)
    print(f"[deng] done drugbank_deng  drugs={st['drugs']} types={st['n_classes']} "
          f"dropped(cross-unseen)={st['dropped']}", flush=True)


if __name__ == "__main__":
    main()

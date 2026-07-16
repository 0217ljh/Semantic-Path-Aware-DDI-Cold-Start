"""Materialize drugbank_ryu (Ryu DrugBank-86, MRCGNN format) into the unified layout.

Ryu et al. ("DeepDDI"): 1700 drugs, 86 DDI-event types. Source ships 5 warm CV folds of
`d1,type,d2`. Produces transductive/S0 (from source folds) + inductive/S1,S2 (drug-disjoint
5-fold, generated) for binary + multiclass, via data_utils.mrcgnn_cv. SMILES from HDN-DDI's
drug_smiles.csv (covers all 1700); label names from DDI_event.csv. Read-only on source.

Note: ryu has 406 canonical pairs carrying >1 event type and some directional pairs; the
drug-disjoint cold split assigns each pair's fold by its drugs, so all of a pair's type
rows stay in one fold (no cross-fold label leak). Binary canonicalizes+dedups.
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

SRC = ROOT / "Paper/Reference/Original-Code/MRCGNN/Ryu's dataset"
HDN_SMI = ROOT / "Code/reproductions/HDN-DDI/_Original-Dataset/drug_smiles.csv"
OUT = ROOT / "Code/data/ddi_unified"
N_CLASSES = 86
FOLD_FILES = {"train": "ddi_training1.csv", "val": "ddi_validation1.csv",
              "test": "ddi_test1.csv"}


def main() -> None:
    h = pd.read_csv(HDN_SMI)  # drug_id, smiles
    ev = pd.read_csv(SRC / "DDI_event.csv", header=None,
                     names=["label_idx_global", "label_name"])
    ev["label_idx_global"] = ev.label_idx_global.astype(np.int64)
    assert len(ev) == N_CLASSES, f"DDI_event has {len(ev)} rows != {N_CLASSES}"
    st = mrcgnn_cv.build(group="ryu", src_dir=SRC, fold_files=FOLD_FILES,
                         n_source_folds=5, smiles=h[["drug_id", "smiles"]],
                         n_classes=N_CLASSES,
                         label_vocab=ev[["label_idx_global", "label_name"]], out_root=OUT)
    print(f"[ryu] done drugbank_ryu  drugs={st['drugs']} types={st['n_classes']} "
          f"dropped(cross-unseen)={st['dropped']}", flush=True)


if __name__ == "__main__":
    main()

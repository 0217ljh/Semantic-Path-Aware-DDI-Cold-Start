"""Materialize the Deng-65 multi-class DDI-event benchmark into the unified layout.

Deng's dataset (Deng et al. 2020, "DDIMDL"): ~570 drugs, 65 DDI-event types, 5-fold
CV (transductive/warm — the canonical protocol for this benchmark). Source ships as
`d1,type,d2` csvs per fold + a `drug_id,smiles` table. Molecular only: NO KG.

Writes `Code/data/ddi_unified/deng65_5fold/` per the locked schema (codex
ddi_unified_v2): meta.json + drugs.parquet + label_vocab.parquet + splits/fold{0..4}/
{train,val,test}.parquet. Multi-class split columns: pair_id, drug_a_id, drug_b_id,
y_cls (global 0..64), y_cls_train (per-fold train-seen local index; -1 if a val/test
type is unseen in that fold's train — kept for uniformity with the closed-set protocol).

Read-only on the source; only writes new parquet/json under ddi_unified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
SRC = ROOT / "Paper/Reference/Original-Code/MRCGNN/Deng's dataset"
OUT = ROOT / "Code/data/ddi_unified/deng65_5fold"
N_FOLDS = 5
N_CLASSES = 65
FOLD_FILES = {"train": "ddi_training1xiao.csv", "val": "ddi_validation1xiao.csv",
              "test": "ddi_test1xiao.csv"}


def _read_fold(fold: int, split: str) -> pd.DataFrame:
    df = pd.read_csv(SRC / str(fold) / FOLD_FILES[split])
    # columns: d1, type, d2
    if list(df.columns) != ["d1", "type", "d2"]:
        raise ValueError(f"unexpected cols {list(df.columns)} in fold{fold}/{split}")
    out = pd.DataFrame({
        "drug_a_id": df["d1"].astype(str),
        "drug_b_id": df["d2"].astype(str),
        "y_cls": df["type"].astype(np.int64),
    })
    return out


def _y_cls_train(df: pd.DataFrame, train_types: dict[int, int]) -> np.ndarray:
    """Per-fold train-seen local index; -1 if the global type is unseen in train."""
    return np.array([train_types.get(int(t), -1) for t in df["y_cls"]], dtype=np.int64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # drug universe ACTUALLY USED across all 5 folds (codex: filter drugs.parquet to
    # the used union, not the raw 572-row roster).
    used: set[str] = set()
    for fold in range(N_FOLDS):
        for split in ("train", "val", "test"):
            df = _read_fold(fold, split)
            used |= set(df["drug_a_id"]) | set(df["drug_b_id"])

    # drugs.parquet — drug_listxiao.csv (drug_id,smiles, no header) is AUTHORITATIVE
    # and covers all used drugs; drug_smiles.csv (1700-row table) misses 22, so do NOT
    # use it. Filter to the used union.
    smi = pd.read_csv(SRC / "drug_listxiao.csv", header=None, names=["drug_id", "smiles"])
    smi["drug_id"] = smi["drug_id"].astype(str)
    smi = smi.drop_duplicates("drug_id")
    miss = used - set(smi["drug_id"])
    if miss:
        raise ValueError(f"[deng65] {len(miss)} used drugs lack SMILES: {sorted(miss)[:5]}")
    smi = smi[smi["drug_id"].isin(used)].reset_index(drop=True)
    drugs = pd.DataFrame({
        "drug_id": smi["drug_id"],
        "source_drug_id": smi["drug_id"],
        "drugbank_id": smi["drug_id"],
        "drug_name": pd.NA,
        "smiles": smi["smiles"].astype(str),
        "morgan_fp_1024": pd.NA,
        "description": pd.NA,
        "smiles_source": "drug_listxiao.csv",
    })
    drugs.to_parquet(OUT / "drugs.parquet", index=False)
    drug_set = set(drugs["drug_id"])

    # global label vocab (0..64). Deng folds carry no event names -> synthesize.
    label_vocab = pd.DataFrame({
        "label_idx_global": np.arange(N_CLASSES, dtype=np.int64),
        "label_name": [f"deng_event_{i}" for i in range(N_CLASSES)],
    })
    label_vocab.to_parquet(OUT / "label_vocab.parquet", index=False)

    seen_global = set()
    for fold in range(N_FOLDS):
        fdir = OUT / "splits" / f"fold{fold}"
        fdir.mkdir(parents=True, exist_ok=True)
        tr = _read_fold(fold, "train")
        train_type_list = sorted(tr["y_cls"].unique().tolist())
        train_types = {int(t): i for i, t in enumerate(train_type_list)}
        for split in ("train", "val", "test"):
            df = tr if split == "train" else _read_fold(fold, split)
            # OOV drug guard (should be none for a closed molecular benchmark)
            oov = (~df["drug_a_id"].isin(drug_set)) | (~df["drug_b_id"].isin(drug_set))
            if oov.any():
                raise ValueError(f"fold{fold}/{split}: {int(oov.sum())} pairs with OOV drugs")
            df = df.reset_index(drop=True)
            df.insert(0, "pair_id", df.index.astype(np.int64))
            df["y_cls_train"] = _y_cls_train(df, train_types)
            # warm/transductive multiclass invariants (codex): no zero-shot class in
            # val/test, and the train-local index equals the (dense) global label.
            zs = set(df["y_cls"]) - set(train_types)
            assert not zs, f"fold{fold}/{split}: zero-shot classes {sorted(zs)}"
            assert (df["y_cls_train"] == df["y_cls"]).all(), \
                f"fold{fold}/{split}: y_cls_train != y_cls (train labels not dense 0..N-1)"
            df = df[["pair_id", "drug_a_id", "drug_b_id", "y_cls", "y_cls_train"]]
            df.to_parquet(fdir / f"{split}.parquet", index=False)
            seen_global |= set(df["y_cls"].tolist())

    meta = {
        "schema_version": "ddi_unified_v2",
        "dataset_id": "deng65_5fold",
        "dataset_group": "deng65",
        "task": "multiclass",
        "split_type": "transductive",
        "pair_format": "ordered_source",
        "split_ids": [f"fold{i}" for i in range(N_FOLDS)],
        "split_layout": "cv_5fold",
        "modalities": {
            "kg": {"present": False, "scope": None, "source": None, "drug_node_key": None},
            "smiles": {"present": True, "source": "drug_listxiao.csv", "coverage_required": True},
            "morgan_fp": {"present": False, "dim": None, "source": None},
            "text": {"present": False, "fields": [], "source": None},
        },
        "labels": {
            "kind": "multiclass", "n_labels": N_CLASSES,
            "label_vocab_file": "label_vocab.parquet",
            "closed_set_projection": {"field": "y_cls_train", "unknown_value": -1},
        },
        "files": {"drugs": "drugs.parquet", "label_vocab": "label_vocab.parquet",
                  "splits_root": "splits/"},
        "n_drugs": int(len(drugs)),
        "labels_observed_global": int(len(seen_global)),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[deng65] done -> {OUT}  drugs={len(drugs)} labels_observed={len(seen_global)}",
          flush=True)


if __name__ == "__main__":
    main()

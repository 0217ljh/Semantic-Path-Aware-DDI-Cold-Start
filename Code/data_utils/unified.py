"""Unified DDI dataset layout (ddi_unified_v2) — shared writer + split validators.

One canonical on-disk shape for every (corpus, split-protocol, task) so the 10
baselines + our method plug into a single loader. Each prepare_* script parses one
source, builds the frames, and calls :func:`write_dataset`; the validators here
enforce the split semantics (warm closed-set / cold-start drug-disjointness) so a
mis-parsed source cannot silently corrupt a benchmark.

On-disk per dataset_id (under Code/data/ddi_unified/<dataset_id>/):
    meta.json                       # schema/task/split_type/modalities/labels
    drugs.parquet                   # clean USED-drug list (+ smiles/morgan/etc.)
    label_vocab.parquet             # multiclass/multilabel only
    splits/<split_id>/train.parquet
    splits/<split_id>/val.parquet
    splits/<split_id>/test.parquet
    splits/<split_id>/drug_split.parquet   # cold-start only: per-drug role

Split-row columns by task:
    binary      : pair_id, drug_a_id, drug_b_id, y_bin
    multiclass  : pair_id, drug_a_id, drug_b_id, y_cls, y_cls_train
    multilabel  : pair_id, drug_a_id, drug_b_id, is_positive, y_label_ids (list[int])
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = "ddi_unified_v2"
DRUG_COLS = ["drug_id", "source_drug_id", "drugbank_id", "drug_name", "smiles",
             "morgan_fp_1024", "description", "smiles_source"]
# On-disk layout: Code/data/ddi_unified/<TASK_DIRS[task]>/<DATASET_DIRS[group]>/
#   <regime>/<split_code>/<fold>/ . task folder names match the baseline task-subfolder
# convention in CLAUDE.md (binary_cls / multi_cls / multi_label_cls).
TASK_DIRS = {"binary": "binary_cls", "multiclass": "multi_cls",
             "multilabel": "multi_label_cls"}
# dataset_group -> unified dataset folder name. drugbank_latest has a small fast-test
# subset (partial, 800 drugs) and the full DrugBank (full, 1900 drugs).
DATASET_DIRS = {"ddi800": "drugbank_latest_partial", "ddi_full": "drugbank_latest_full",
                "deng": "drugbank_deng", "ryu": "drugbank_ryu", "twosides": "twoside"}
# split_type -> (regime, split_code): transductive/S0, inductive/S1, inductive/S2.
SPLIT_LAYOUT = {"transductive": ("transductive", "S0"),
                "cold_s1": ("inductive", "S1"), "cold_s2": ("inductive", "S2")}


def layout_dir(root, dataset_group: str, task: str, split_type: str) -> Path:
    """Resolve the leaf dataset dir <root>/<task>/<dataset>/<regime>/<split_code>/
    that holds meta.json + drugs.parquet (+ label_vocab) and the per-fold subdirs."""
    regime, split_code = SPLIT_LAYOUT[split_type]
    return (Path(root) / TASK_DIRS[task] / DATASET_DIRS[dataset_group]
            / regime / split_code)


def drug_universe(*frames: pd.DataFrame) -> set[str]:
    """Set of every drug id appearing as an endpoint in the given split frames."""
    s: set[str] = set()
    for df in frames:
        s |= set(df["drug_a_id"].astype(str)) | set(df["drug_b_id"].astype(str))
    return s


# -- split-semantics validators (assert or raise) --------------------------------

def check_multiclass_closed_set(train: pd.DataFrame, val: pd.DataFrame,
                                test: pd.DataFrame) -> dict:
    """Closed-set multiclass invariant (warm AND cold, uniform): y_cls_train maps
    train classes to dense 0..K-1 and is -1 EXACTLY for a val/test class unseen in
    train. Rare classes that land only in eval (real in Ryu-86 / cold-start) are NOT
    an error to materialize — they are scored as unknown (-1) per the closed-set
    protocol. Returns the per-split unseen-class row counts for reporting (no fail)."""
    tt = set(train["y_cls"])
    assert (train["y_cls_train"] >= 0).all(), "mc train: some train rows mapped to -1"
    counts = {}
    for nm, df in (("val", val), ("test", test)):
        in_tr = df["y_cls"].isin(tt)
        assert ((df["y_cls_train"] >= 0) == in_tr).all(), \
            f"mc {nm}: y_cls_train -1 mask != unseen-in-train mask"
        counts[nm] = int((~in_tr).sum())
    return counts


def check_cold_s2(train: pd.DataFrame, evals: list[tuple[str, pd.DataFrame]]) -> None:
    """Cold-start S2 (both-unseen): every eval-split drug is disjoint from train."""
    td = drug_universe(train)
    for nm, df in evals:
        ed = drug_universe(df)
        leak = ed & td
        assert not leak, f"cold-S2 {nm}: {len(leak)} eval drugs leak into train"


def check_cold_s1(train: pd.DataFrame, evals: list[tuple[str, pd.DataFrame]]) -> None:
    """Cold-start S1 (one-unseen): every eval pair has EXACTLY one endpoint seen
    in train (forbid seen-seen and unseen-unseen)."""
    td = drug_universe(train)
    for nm, df in evals:
        a = df["drug_a_id"].astype(str).isin(td)
        b = df["drug_b_id"].astype(str).isin(td)
        assert (a ^ b).all(), (
            f"cold-S1 {nm}: {int((~(a ^ b)).sum())} pairs are not exactly-one-seen "
            f"(seen-seen {int((a & b).sum())}, unseen-unseen {int((~a & ~b).sum())})")


def build_drug_split(train: pd.DataFrame,
                     evals: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Per-drug role table for a cold-start split (codex drug_roles): which drugs
    are train-seen vs eval-only-unseen — makes the drug partition auditable."""
    td = drug_universe(train)
    ed: set[str] = set()
    for _, df in evals:
        ed |= drug_universe(df)
    rows = [(d, "train_seen") for d in sorted(td)]
    rows += [(d, "eval_unseen") for d in sorted(ed - td)]
    return pd.DataFrame(rows, columns=["drug_id", "role"])


# -- writer ----------------------------------------------------------------------

def make_drugs_frame(drug_id, *, smiles=None, source_drug_id=None, drugbank_id=None,
                     drug_name=None, morgan_fp_1024=None, description=None,
                     smiles_source=None) -> pd.DataFrame:
    """Build the canonical drugs.parquet frame; pass aligned sequences or None."""
    drug_id = [str(x) for x in drug_id]
    n = len(drug_id)
    def col(v, default=pd.NA):
        return v if v is not None else [default] * n
    src = [str(x) for x in (source_drug_id if source_drug_id is not None else drug_id)]
    dbk = [str(x) for x in (drugbank_id if drugbank_id is not None else drug_id)]
    return pd.DataFrame({
        "drug_id": drug_id, "source_drug_id": src, "drugbank_id": dbk,
        "drug_name": col(drug_name), "smiles": col(smiles),
        "morgan_fp_1024": col(morgan_fp_1024), "description": col(description),
        "smiles_source": col(smiles_source),
    })[DRUG_COLS]


def write_dataset(out_dir, meta: dict, drugs: pd.DataFrame,
                  label_vocab: pd.DataFrame | None,
                  split_frames: dict[str, dict[str, pd.DataFrame]],
                  drug_splits: dict[str, pd.DataFrame] | None = None) -> None:
    """Materialize one dataset_id to disk in the unified layout.

    split_frames: {split_id: {"train"|"val"|"test": df}}; drug_splits: {split_id: df}.
    Validates: drugs deduped + covers all split drugs; meta has the required keys.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for req in ("dataset_id", "dataset_group", "task", "split_type", "split_ids",
                "modalities", "labels"):
        if req not in meta:
            raise ValueError(f"meta missing required key {req!r}")
    if str(meta["split_type"]).startswith("cold"):
        if not drug_splits or set(drug_splits) != set(split_frames):
            raise ValueError(
                f"cold dataset {meta['dataset_id']} must supply a drug_split for "
                f"every split id (have {sorted(drug_splits or [])}, "
                f"need {sorted(split_frames)})")
    if drugs["drug_id"].duplicated().any():
        raise ValueError("drugs.parquet has duplicate drug_id")
    duni = set(drugs["drug_id"].astype(str))
    for sid, fr in split_frames.items():
        used = drug_universe(*fr.values())
        miss = used - duni
        if miss:
            raise ValueError(f"split {sid}: {len(miss)} drugs not in drugs.parquet "
                             f"(e.g. {sorted(miss)[:5]})")

    drugs.to_parquet(out / "drugs.parquet", index=False)
    if label_vocab is not None:
        label_vocab.to_parquet(out / "label_vocab.parquet", index=False)
    for sid, fr in split_frames.items():
        sd = out / sid                       # folds are direct children of the leaf
        sd.mkdir(parents=True, exist_ok=True)
        for split, df in fr.items():
            df.reset_index(drop=True).to_parquet(sd / f"{split}.parquet", index=False)
        if drug_splits and sid in drug_splits:
            drug_splits[sid].to_parquet(sd / "drug_split.parquet", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def base_meta(dataset_id: str, dataset_group: str, task: str, split_type: str,
              split_ids: list[str], *, kg=None, smiles_source=None,
              n_labels=None, label_vocab_file="label_vocab.parquet",
              morgan_dim=None, pair_format="ordered_source") -> dict:
    """Assemble a meta.json dict with the canonical modality/label blocks.

    pair_format: "canonical" (a<=b, unordered — use for the symmetric binary task so
    pair orientation carries NO label signal) or "ordered_source" (preserve source
    order — required for DIRECTIONAL multiclass DDI-event types).
    """
    is_cold = str(split_type).startswith("cold")
    regime, split_code = SPLIT_LAYOUT[split_type]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_group": dataset_group,
        "task": task,
        "split_type": split_type,
        "regime": regime,
        "split_code": split_code,
        "pair_format": pair_format,
        "split_ids": list(split_ids),
        "modalities": {
            "kg": ({"present": True, "scope": kg.get("scope"), "source": kg.get("source"),
                    "drug_node_key": kg.get("drug_node_key")} if kg else
                   {"present": False, "scope": None, "source": None, "drug_node_key": None}),
            "smiles": {"present": smiles_source is not None, "source": smiles_source,
                       "coverage_required": smiles_source is not None},
            "morgan_fp": {"present": morgan_dim is not None, "dim": morgan_dim,
                          "source": None},
            "text": {"present": False, "fields": [], "source": None},
        },
        "labels": ({"kind": task, "n_labels": int(n_labels),
                    "label_vocab_file": label_vocab_file,
                    "closed_set_projection": ({"field": "y_cls_train", "unknown_value": -1}
                                              if task == "multiclass" else None)}
                   if n_labels is not None else {"kind": task}),
        "files": {"drugs": "drugs.parquet",
                  "label_vocab": label_vocab_file if n_labels is not None else None,
                  "folds_root": ".",   # fold subdirs are direct children of the leaf
                  "drug_split": "drug_split.parquet" if is_cold else None},
    }


__all__ = ["SCHEMA_VERSION", "DRUG_COLS", "TASK_DIRS", "DATASET_DIRS", "SPLIT_LAYOUT",
           "layout_dir", "drug_universe", "check_multiclass_closed_set",
           "check_cold_s2", "check_cold_s1", "build_drug_split", "make_drugs_frame",
           "write_dataset", "base_meta"]

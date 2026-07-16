"""DDI dataset loaders for binary classification and multi-class classification
under strict cold-start G1/G2 splits (Code-Released-Formal format).

Split structure (per seed):
    splits/seed{42,43,44}/
        train.parquet               positives only (G1 × G1)
        val_s0/1/2.parquet          S0/S1/S2 val positives
        test_s0/1/2.parquet         S0/S1/S2 test positives
        negatives/test_s0/1/2.parquet      paired negatives for eval
        negatives/val_s0/1/2.parquet
        train_negatives/epoch_*.parquet    dynamic negatives per epoch
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

# Resolve project root
def _find_root() -> Path:
    cur = Path.cwd().resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError(f"Project root not found from {cur}")

ROOT = _find_root()
SPLITS_DIR = ROOT / "Code" / "data" / "KG" / "drugbank" / "splits"


# -----------------------------------------------------------------------------
# Inventory
# -----------------------------------------------------------------------------
def list_split_files(seed: int = 42) -> dict:
    """Return file inventory for a given seed."""
    d = SPLITS_DIR / f"seed{seed}"
    if not d.is_dir():
        raise FileNotFoundError(f"Split dir not found: {d}")
    return {
        "train":            d / "train.parquet",
        "val_s0":           d / "val_s0.parquet",
        "val_s1":           d / "val_s1.parquet",
        "val_s2":           d / "val_s2.parquet",
        "test_s0":          d / "test_s0.parquet",
        "test_s1":          d / "test_s1.parquet",
        "test_s2":          d / "test_s2.parquet",
        "neg_val_s0":       d / "negatives" / "val_s0.parquet",
        "neg_val_s1":       d / "negatives" / "val_s1.parquet",
        "neg_val_s2":       d / "negatives" / "val_s2.parquet",
        "neg_test_s0":      d / "negatives" / "test_s0.parquet",
        "neg_test_s1":      d / "negatives" / "test_s1.parquet",
        "neg_test_s2":      d / "negatives" / "test_s2.parquet",
        "train_neg_dir":    d / "train_negatives",
        "manifest":         d / "manifest.json",
    }


# -----------------------------------------------------------------------------
# Binary classification: positives + paired negatives, with `label` column
# -----------------------------------------------------------------------------
def load_binary_split(seed: int = 42, split: str = "test_s2",
                      include_negatives: bool = True) -> pd.DataFrame:
    """Return one binary split (train / val_s0|1|2 / test_s0|1|2) with positives
    and (optionally) paired negatives merged. Adds:
        - label: 1 (positive) / 0 (negative)

    Train negatives are NOT included here (use `load_train_negatives_epoch` for that
    since they're resampled per epoch).
    """
    files = list_split_files(seed)
    if split not in files:
        raise ValueError(f"Unknown split '{split}'. Valid: train, val_s0/1/2, test_s0/1/2")

    pos = pq.read_table(files[split]).to_pandas()
    pos["label"] = 1

    if not include_negatives:
        return pos

    if split == "train":
        # train negatives live in train_negatives/epoch_*.parquet (not paired here)
        return pos

    neg_key = "neg_" + split
    if neg_key not in files or not files[neg_key].exists():
        return pos
    neg = pq.read_table(files[neg_key]).to_pandas()
    neg["label"] = 0
    # Negatives may lack description / ddi_type columns; harmonize
    for col in ["description", "ddi_type"]:
        if col not in neg.columns:
            neg[col] = ""
    common = ["drug_a_id", "drug_b_id", "label"]
    extras = [c for c in pos.columns if c not in common]
    for c in extras:
        if c not in neg.columns:
            neg[c] = ""
    return pd.concat([pos, neg], ignore_index=True)[common + extras].reset_index(drop=True)


def load_train_negatives_epoch(seed: int = 42, epoch: int = 0) -> pd.DataFrame:
    """Load a single epoch's training negatives."""
    files = list_split_files(seed)
    p = files["train_neg_dir"] / f"epoch_{epoch}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pq.read_table(p).to_pandas()
    df["label"] = 0
    return df


# -----------------------------------------------------------------------------
# Multi-class classification: positives only, label = ddi_type (top-K)
# -----------------------------------------------------------------------------
def load_multiclass_split(seed: int = 42, split: str = "test_s2",
                          top_k: int | None = 86) -> pd.DataFrame:
    """Return positives only, with `ddi_type` column as the multi-class label.

    If top_k is not None, restrict to the K most frequent ddi_types in the
    seed's TRAIN split (so val/test only contains seen types).
    Returns DataFrame plus extra columns: `ddi_type_idx` (0..K-1 if top-K mode).
    """
    files = list_split_files(seed)
    if split not in {"train", "val_s0", "val_s1", "val_s2",
                     "test_s0", "test_s1", "test_s2"}:
        raise ValueError(split)
    df = pq.read_table(files[split]).to_pandas()
    if "ddi_type" not in df.columns:
        raise RuntimeError(f"ddi_type missing from {files[split]}; multiclass needs it")

    if top_k is None:
        types = sorted(df["ddi_type"].unique())
    else:
        # Derive top-K from TRAIN set (consistent across train/val/test)
        train = pq.read_table(files["train"]).to_pandas()
        type_counts = train["ddi_type"].value_counts()
        types = list(type_counts.head(top_k).index)

    type_to_idx = {t: i for i, t in enumerate(types)}
    df = df[df["ddi_type"].isin(types)].copy()
    df["ddi_type_idx"] = df["ddi_type"].map(type_to_idx).astype("int64")
    df.attrs["type_to_idx"] = type_to_idx  # accessible via df.attrs
    df.attrs["idx_to_type"] = {i: t for t, i in type_to_idx.items()}
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Convenience for the exp1-0512 experiments
# -----------------------------------------------------------------------------
def load_cold_start_S2(seed: int = 42, task: str = "binary",
                       top_k: int | None = 86) -> dict:
    """One-stop loader for the cold-start S2 task.

    Args:
        seed: 42 / 43 / 44
        task: 'binary' or 'multiclass'
        top_k: for multiclass — keep top-K ddi_types (None = keep all 215)

    Returns:
        dict with train / val / test DataFrames. For binary, includes paired negatives
        in val/test. For multiclass, only positives.
    """
    if task == "binary":
        train = load_binary_split(seed, "train", include_negatives=False)
        val = load_binary_split(seed, "val_s2")
        test = load_binary_split(seed, "test_s2")
    elif task == "multiclass":
        train = load_multiclass_split(seed, "train", top_k=top_k)
        val = load_multiclass_split(seed, "val_s2", top_k=top_k)
        test = load_multiclass_split(seed, "test_s2", top_k=top_k)
    else:
        raise ValueError(f"task must be 'binary' or 'multiclass', got {task}")
    return {"train": train, "val": val, "test": test, "seed": seed, "task": task, "top_k": top_k}


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("== Inventory for seed=42 ==")
    for k, v in list_split_files(42).items():
        if isinstance(v, Path):
            exists = "OK " if v.exists() else "MISS"
            print(f"  [{exists}]  {k:<18}  {v.relative_to(ROOT)}")

    print("\n== Binary task, S2 test ==")
    df = load_binary_split(42, "test_s2")
    print(f"  rows: {len(df):,}  cols: {list(df.columns)}")
    print(f"  label dist: {df['label'].value_counts().to_dict()}")

    print("\n== Multiclass task, S2 test, top-86 ==")
    df = load_multiclass_split(42, "test_s2", top_k=86)
    print(f"  rows after top-86 filter: {len(df):,}")
    print(f"  number of unique classes kept: {df['ddi_type_idx'].nunique()}")

    print("\n== Cold-start S2 packaged ==")
    data = load_cold_start_S2(42, task="binary")
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(data[k]):,} rows")

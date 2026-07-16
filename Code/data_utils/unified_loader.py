"""Reader for the unified DDI benchmark (ddi_unified_v2) — the consumer side.

`data_utils.unified` WRITES leaves; this module READS one leaf
(task/dataset/regime/split/fold) into an in-memory :class:`Leaf` that the per-task
baseline interface (see `baseline.unified_base`) consumes. Decision B (codex thread
019f19fb): a baseline trains on ONE leaf's own train/val/test — there is no shared
train across regimes and no cross-regime routing. The leaf is single-regime; cold-aware
baselines read `resources.drug_split` (always present, even for S0) for role info.

Layout per leaf: <root>/<TASK_DIRS[task]>/<dataset>/<regime>/<split>/<fold>/
  {train,val,test}.parquet (+ drug_split.parquet for cold, + *_pair_links.parquet for
  multilabel); leaf-level meta.json + drugs.parquet (+ label_vocab.parquet).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_utils import unified


@dataclass(frozen=True)
class KGHandle:
    """Dataset-scoped KG pointer resolved from meta. Baselines that need a live graph
    build it from `source` (an absolute path) with their own KG builder; KG-free
    baselines ignore it. `present=False` means the leaf declares no KG."""
    present: bool
    source: str | None          # absolute path to the KG dir/files (or None)
    scope: str | None
    drug_node_key: str | None


@dataclass
class LeafResources:
    """Everything a baseline needs about a leaf besides the train/val/test rows."""
    drugs: pd.DataFrame                 # drug_id, smiles, morgan_fp_1024, drugbank_id, ...
    drug_split: pd.DataFrame            # drug_id, role ∈ {train,val,test,unused} (always set)
    kg: KGHandle
    meta: dict
    label_vocab: pd.DataFrame | None    # multiclass/multilabel only
    # Runtime-only leaf identity (added 2026-07-01 for per-epoch train negatives,
    # design B). Default None keeps existing constructions non-breaking.
    fold: str = ""                      # fold0..4
    fold_dir: "Path | None" = None      # <root>/<task>/<dataset>/<regime>/<split>/<fold>


@dataclass
class Leaf:
    """One materialized (task, dataset, regime, split, fold) training/eval unit."""
    task: str                           # binary | multiclass | multilabel
    dataset: str                        # drugbank_latest_full | ... | twoside
    regime: str                         # transductive | inductive
    split_code: str                     # S0 | S1 | S2
    fold: str                           # fold0..4
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    resources: LeafResources


def leaf_dir(root, dataset_group: str, task: str, split_type: str) -> Path:
    """Path to the leaf parent (holds meta + folds). Mirrors unified.layout_dir."""
    return unified.layout_dir(root, dataset_group, task, split_type)


def _project_root(start) -> Path:
    """Walk up from `start` to the repo root (the dir holding Code/data/KG). KG pointers
    in meta are stored relative to this root, not to the ddi_unified data root."""
    p = Path(start).resolve()
    for c in [p, *p.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    return p


def list_folds(leaf_parent: Path) -> list[str]:
    return sorted(p.name for p in Path(leaf_parent).glob("fold*") if p.is_dir())


def _synth_drug_split(drugs: pd.DataFrame) -> pd.DataFrame:
    """Transductive S0 ships no drug_split; synthesize one over the FULL drugs.parquet
    (every drug = role 'train', since all drugs are seen in a warm split) so cold-aware
    baselines get a complete, uniform seen-drug universe (codex: cover all drugs, not
    just those appearing in this fold's pair rows)."""
    return pd.DataFrame({"drug_id": sorted(drugs["drug_id"].astype(str)), "role": "train"})


def load_leaf(root, dataset_group: str, task: str, split_type: str, fold: str) -> Leaf:
    """Load one leaf. `dataset_group` is the unified group key (ddi_full/ddi800/deng/
    ryu/twosides); `task` ∈ {binary, multiclass, multilabel}; `split_type` ∈
    {transductive, cold_s1, cold_s2}; `fold` e.g. 'fold0'."""
    parent = leaf_dir(root, dataset_group, task, split_type)
    meta = json.loads((parent / "meta.json").read_text())
    fdir = parent / fold
    if not fdir.is_dir():
        raise FileNotFoundError(f"missing fold {fold} under {parent}")
    train = pd.read_parquet(fdir / "train.parquet")
    val = pd.read_parquet(fdir / "val.parquet")
    test = pd.read_parquet(fdir / "test.parquet")

    drugs = pd.read_parquet(parent / "drugs.parquet")
    lv_path = parent / "label_vocab.parquet"
    label_vocab = pd.read_parquet(lv_path) if lv_path.is_file() else None
    ds_path = fdir / "drug_split.parquet"
    drug_split = (pd.read_parquet(ds_path) if ds_path.is_file()
                  else _synth_drug_split(drugs))

    kgm = meta.get("modalities", {}).get("kg", {})
    src = kgm.get("source")
    if src and not Path(src).is_absolute():
        src = str((_project_root(root) / src).resolve())
    kg = KGHandle(present=bool(kgm.get("present")), source=src,
                  scope=kgm.get("scope"), drug_node_key=kgm.get("drug_node_key"))

    res = LeafResources(drugs=drugs, drug_split=drug_split, kg=kg, meta=meta,
                        label_vocab=label_vocab, fold=fold, fold_dir=fdir)
    return Leaf(task=meta["task"], dataset=meta["dataset_id"], regime=meta["regime"],
                split_code=meta["split_code"], fold=fold,
                train=train, val=val, test=test, resources=res)


def iter_leaves(root, dataset_group: str, task: str, split_type: str):
    """Yield all folds of one (dataset, task, split_type) as Leaf objects."""
    parent = leaf_dir(root, dataset_group, task, split_type)
    for fold in list_folds(parent):
        yield load_leaf(root, dataset_group, task, split_type, fold)


__all__ = ["KGHandle", "LeafResources", "Leaf", "leaf_dir", "list_folds",
           "load_leaf", "iter_leaves"]

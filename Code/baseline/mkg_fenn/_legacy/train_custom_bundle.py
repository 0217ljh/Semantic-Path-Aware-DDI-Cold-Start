"""
MKG-FENN  fair-negative-sample data-migration script (train_custom_bundle.py)

Core method (unchanged from upstream):
  - GNN1-4 : 4-channel knowledge-graph GNN (from Code and Datasets/code/modeltask1.py)
  - FusionLayer : 4-way feature concatenation + MLP
  - nn.CrossEntropyLoss + Adam
  - Symmetric augmentation: at train time swap (A,B) -> (B,A) to mint
    an equal-sized mirror batch

Migration (adapter for our bundle layout):
  - event_num=2  (binary: DDI / no-DDI)
  - 4 KGs are built dynamically from the bundle data:
      KG1 (drug-protein/entity): bundle.extra['kb'] (targets/enzymes/transporters/carriers)
      KG2 (drug-substructure)  : SMILES -> Morgan fingerprint bit positions (radius=2, nBits=512)
      KG3 (drug-DDI)           : training positive pairs (bidirectional);
                                 cold-start drugs get a self-reference edge
      KG4 (drug-property)      : SMILES -> RDKit molecular descriptors (5 props, binned)
  - Fair negatives: from bundle.extra['train_neg_epochs']; rotated each epoch
  - Early-stopping metric: AUC-ROC (mean of val_s1 + val_s2)
  - Outputs: <output_dir>/s0/metric.json, s0/val/metric.json, ..., Baseline.csv/.xlsx

Notes:
  - FusionLayer calls idx.numpy(), which requires the index tensor to live on CPU
  - This script forces CPU execution (matches the upstream paper code;
    modeltask1.py also builds Tensors on CPU internally)
  - For GNN2/GNN4 we add a ghost placeholder entry for drugs with no SMILES
    (a dedicated trailing slot), mirroring the ghost mechanism GNN1 already
    has; for GNN3, cold-start drugs use a self-reference (drug_idx, 0) so
    GNN3.arrge cannot trigger an out-of-range surplus-ghost lookup.

Run example:
  cd /mnt/d/My-Research/03-Projects/ColdDDI/Code-Released/baseline/MKG-FENN
  python train_custom_bundle.py \
      --data_path /path/to/bundle.pkl \
      --output_dir ./outputs \
      --epochs 50 --batch_size 256 --lr 1e-2 --dropout 0.3 \
      --embedding_num 128 --neighbor_sample_size 6
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np

# numpy 1.x <-> 2.x pickle compatibility shim
# Bundles were pickled under numpy 2.x (which uses `numpy._core.*`).
# project_1 env runs numpy 1.x (which uses `numpy.core.*`), so alias the paths.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    for _sub in ("numeric", "multiarray", "umath", "_multiarray_umath",
                  "fromnumeric", "_methods", "arrayprint"):
        try:
            _mod = __import__("numpy.core." + _sub, fromlist=[_sub])
            sys.modules.setdefault("numpy._core." + _sub, _mod)
        except ImportError:
            pass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. Paths & model imports
# ──────────────────────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_baseline_root = os.path.dirname(_script_dir)  # Baseline/
_code_dir = os.path.join(_script_dir, "Code and Datasets", "code")
if _code_dir not in sys.path:
    # Insert at FRONT so MKG-FENN-NEW's modeltask1 wins over the original
    sys.path.insert(0, _code_dir)

# NEW version of modeltask1 — same architecture/parameters as original,
# but neighbor sampling is moved out of forward (precompute_adj per epoch)
# and all inputs are GPU buffers.
from modeltask1 import FusionLayer, GNN1, GNN2, GNN3, GNN4  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
# 1. Arguments
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="MKG-FENN bundle migration (fair negatives)")
parser.add_argument("--data_path", type=str, default=None,
                    help="bundle .pkl path; if omitted, derived from --data_dir + --seed.")
parser.add_argument("--data_dir", type=str,
                    default="/mnt/d/My-Research/03-Projects/ColdDDI/Code-Released/dataset/1900drug/P0",
                    help="directory containing the bundle; used with --seed when "
                         "--data_path is not provided.")
parser.add_argument("--output_dir", type=str,
                    default=os.path.join(_baseline_root, "Output", "MKG-FENN", "my"),
                    help="output root (a timestamp subdir is appended automatically)")
# Hyper-parameters (aligned with paper best values)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=1e-8)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--embedding_num", type=int, default=128)
parser.add_argument("--neighbor_sample_size", type=int, default=6)
parser.add_argument("--patience", type=int, default=15, help="early-stop patience (epochs)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=int, default=0,
                    help="GPU device id (-1 forces CPU). NEW version actually uses GPU.")
parser.add_argument("--resample_every_n_batches", type=int, default=0,
                    help=("Resample neighbor adjacency every N training batches. "
                          "0 = once per epoch (fast, default; matches what most KG-GNN "
                          "implementations do). 1 = original per-batch behaviour (slow). "
                          "Recommended for closer-to-original variance: 50-100."))
# Fast-test knob
parser.add_argument("--max_steps", type=int, default=None,
                    help="cap on steps per epoch (for quick smoke tests)")
# MKG-FENN internal parameters (do not edit)
parser.add_argument("--event_num", type=int, default=2, help="fixed to 2 (binary classification)")
# KG2 Morgan-fingerprint hyper-parameters
parser.add_argument("--fp_radius", type=int, default=2)
parser.add_argument("--fp_nbits", type=int, default=512)
# KG4 molecular-property bin count
parser.add_argument("--n_bins", type=int, default=10)

args = parser.parse_args()
if args.data_path is None:
    args.data_path = os.path.join(
        args.data_dir,
        f"latest_drugbank_ddi-Binary_cls-{args.seed}"
        f"+cold_start_split_fair_step-and-fair_negatives_step.pkl",
    )

# ──────────────────────────────────────────────────────────────────────────────
# 2. Output directory & logging
# ──────────────────────────────────────────────────────────────────────────────
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
run_output_dir = os.path.join(args.output_dir, _ts)
os.makedirs(run_output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(run_output_dir, "train.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
log.info("MKG-FENN fair-negative training  dataset=%s  output=%s", _dataset_name, run_output_dir)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Random seed
# ──────────────────────────────────────────────────────────────────────────────
def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(args.seed)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Load bundle
# ──────────────────────────────────────────────────────────────────────────────
def load_bundle(path):
    # Inject paths that may carry the original Preprocessor module so the
    # pickle can be deserialised successfully.
    _extra_paths = [
        "/mnt/d/My-Research/03-Projects/ColdDDI/Code-Released/baseline",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
        os.path.dirname(path),
    ]
    for p in _extra_paths:
        if p and os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    last_exc = None
    for enc in ("latin1", "utf-8", None):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f, encoding=enc) if enc else pickle.load(f)
            return data
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"failed to load bundle: {path}") from last_exc


bundle = load_bundle(args.data_path)

# FoldBundle uses attribute access; also support a plain-dict layout.
def _get(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)

df_all: pd.DataFrame = _get(bundle, "df").copy()
split = _get(bundle, "split")
extra = _get(bundle, "extra")

# Normalise column names: the bundle uses drug_a_id/drug_b_id; this script
# uses d1/d2 internally.
if "drug_a_id" in df_all.columns and "d1" not in df_all.columns:
    df_all = df_all.rename(columns={"drug_a_id": "d1", "drug_b_id": "d2"})

kb: dict = extra.get("kb", {})          # targets/enzymes/transporters/carriers/pathways

# drug_id2smiles lives inside kb, not at the extra top level.
drug_id2smiles: dict = (
    kb.get("drug_id2smiles", {})
    if isinstance(kb, dict)
    else getattr(kb, "drug_id2smiles", {})
)
# DataFrame compatibility: if it is a DataFrame, build a dict from the
# drug_id / smiles columns.
if hasattr(drug_id2smiles, "iterrows"):
    drug_id2smiles = {
        str(r["drug_id"]): str(r["smiles"])
        for _, r in drug_id2smiles.iterrows()
        if pd.notna(r.get("smiles", None))
    }

_raw_neg = extra.get("train_neg_epochs", [])
# train_neg_epochs may be stored as a dict {0: [...], 1: [...]} or a list.
if isinstance(_raw_neg, dict):
    train_neg_epochs = [_raw_neg[k] for k in sorted(_raw_neg.keys())]
else:
    train_neg_epochs = list(_raw_neg)


# split is a SimpleNamespace; this helper also supports plain-dict access.
def _split_get(sp, key, default=None):
    if isinstance(sp, dict):
        return sp.get(key, default)
    return getattr(sp, key, default)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Train / val / test splits  (S0/S1/S2)
# ──────────────────────────────────────────────────────────────────────────────
train_idx    = _split_get(split, "train_idx", [])
val_idx_s0   = _split_get(split, "val_idx_s0", [])
val_idx_s1   = _split_get(split, "val_idx_s1", [])
val_idx_s2   = _split_get(split, "val_idx_s2", [])
test_idx_s0  = _split_get(split, "test_idx_s0", [])
test_idx_s1  = _split_get(split, "test_idx_s1", [])
test_idx_s2  = _split_get(split, "test_idx_s2", [])
val_idx_s0_neg  = _split_get(split, "val_idx_s0_neg", [])
val_idx_s1_neg  = _split_get(split, "val_idx_s1_neg", [])
val_idx_s2_neg  = _split_get(split, "val_idx_s2_neg", [])
test_idx_s0_neg = _split_get(split, "test_idx_s0_neg", [])
test_idx_s1_neg = _split_get(split, "test_idx_s1_neg", [])
test_idx_s2_neg = _split_get(split, "test_idx_s2_neg", [])


def _merge_and_fetch(pos_idx, neg_idx):
    """Concatenate positive + negative index lists (both integer row
    numbers into ``df_all``) into a single DataFrame.  Val / test
    splits must include both classes for AUC to be defined."""
    pos_df = df_all.iloc[pos_idx].reset_index(drop=True)
    if not neg_idx:
        return pos_df
    neg_df = df_all.iloc[neg_idx].reset_index(drop=True)
    return pd.concat([pos_df, neg_df], ignore_index=True)


train_df = df_all.iloc[train_idx].reset_index(drop=True)
val_dfs = {
    "s0": _merge_and_fetch(val_idx_s0, val_idx_s0_neg),
    "s1": _merge_and_fetch(val_idx_s1, val_idx_s1_neg),
    "s2": _merge_and_fetch(val_idx_s2, val_idx_s2_neg),
}
test_dfs = {
    "s0": _merge_and_fetch(test_idx_s0, test_idx_s0_neg),
    "s1": _merge_and_fetch(test_idx_s1, test_idx_s1_neg),
    "s2": _merge_and_fetch(test_idx_s2, test_idx_s2_neg),
}

# Training positives
train_pos_df = train_df[train_df["label"] == 1].reset_index(drop=True)
n_neg_epochs = len(train_neg_epochs)

log.info("Training set: positives=%d  negative-epochs=%d", len(train_pos_df), n_neg_epochs)
from load_custom_data import summarize_train_negatives
summarize_train_negatives(train_neg_epochs, len(train_pos_df), logger=log)
for sn in ["s0", "s1", "s2"]:
    v, t = val_dfs[sn], test_dfs[sn]
    log.info(
        "  val_%s: %d (pos=%d, neg=%d)  test_%s: %d (pos=%d, neg=%d)",
        sn, len(v), (v["label"] == 1).sum(), (v["label"] == 0).sum(),
        sn, len(t), (t["label"] == 1).sum(), (t["label"] == 0).sum(),
    )

# ──────────────────────────────────────────────────────────────────────────────
# 6. Build dict1 (drug_id -> integer index)
# ──────────────────────────────────────────────────────────────────────────────
all_drug_ids: set = set()
for col in ("d1", "d2"):
    if col in df_all.columns:
        all_drug_ids.update(df_all[col].astype(str).tolist())

dict1: dict = {}  # drug_id(str) → int idx
for did in sorted(all_drug_ids):
    dict1[did] = len(dict1)

n_drugs = len(dict1)
drug_name_list = list(range(n_drugs))   # [0, 1, 2, ..., n_drugs-1]
log.info("Drug count (dict1): %d", n_drugs)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Build the 4 knowledge graphs
# ──────────────────────────────────────────────────────────────────────────────
GHOST = "GHOST"   # ghost-node identifier


def _add_ghost_if_empty(kg, all_drug_idxs, ghost_tail, ghost_rel):
    """For every drug in ``kg`` that has zero neighbours, append a ghost entry."""
    for idx in all_drug_idxs:
        if len(kg[idx]) == 0:
            kg[idx].append((ghost_tail, ghost_rel))


# ───── KG1: drug → chemical entity (targets/enzymes/transporters/carriers/pathways) ─────
# Supports two kb layouts: (1) dict keys targets/enzymes/...;
# (2) DataFrame keys my_target_list/my_enzyme_list/...
# Using (2), unseen drugs that have entity records also enter KG1, helping
# cold-start similarity; KG3 still uses training positives only and does not
# pull unseen history.
def _kg1_drug_col(df):
    for c in ["drug_id", "drug_a_id", "DrugBank ID", "drugbank_id", "d1", "id"]:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) > 0 else None


def _kg1_entity_col(df, rel_name):
    # Entity column: prefer the column named after the relation, then
    # common entity-name columns, then the second column.
    cand = [rel_name, "target", "enzyme", "transporter", "carrier", "pathway", "name", "gene", "Gene Name", "Uniprot ID", "entity"]
    for c in cand:
        if c in df.columns:
            return c
    if len(df.columns) >= 2:
        return df.columns[1]
    return None


def build_kg1(kb_data, dict1):
    """
    KG1: drug → chemical entity
    Relation types: targets / enzymes / transporters / carriers / pathways.
    Prefer kb's dict keys targets/enzymes/...; otherwise fall back to the
    DataFrame keys my_target_list/my_enzyme_list/..., so unseen drugs that
    have entity records still receive a KG1 vector.  GNN1 already includes
    a ghost mechanism.
    """
    kg1 = defaultdict(list)
    entity_to_idx: dict = {}
    rel_to_idx: dict = {}
    if not isinstance(kb_data, dict):
        kb_data = getattr(kb_data, "__dict__", {}) or {}

    # 1) Try the dict layout: targets, enzymes, ...
    categories_std = ["targets", "enzymes", "transporters", "carriers", "pathways"]
    categories_my = ["my_target_list", "my_enzyme_list", "my_transporter_list", "my_carrier_list", "my_pathway_list"]
    use_dataframe = False
    for cat in categories_std:
        cat_data = kb_data.get(cat)
        if isinstance(cat_data, dict) and len(cat_data) > 0:
            break
    else:
        use_dataframe = any(kb_data.get(k) is not None for k in categories_my)

    if not use_dataframe:
        for cat in categories_std:
            cat_data = kb_data.get(cat, {})
            if not isinstance(cat_data, dict):
                continue
            if cat not in rel_to_idx:
                rel_to_idx[cat] = len(rel_to_idx)
            rel_idx = rel_to_idx[cat]
            for drug_id, entities in cat_data.items():
                drug_id = str(drug_id)
                if drug_id not in dict1:
                    continue
                drug_idx = dict1[drug_id]
                if isinstance(entities, str):
                    entities = [entities]
                for ent in entities:
                    ent = str(ent).strip() if ent is not None else ""
                    if not ent:
                        continue
                    if ent not in entity_to_idx:
                        entity_to_idx[ent] = len(entity_to_idx)
                    kg1[drug_idx].append((entity_to_idx[ent], rel_idx))
    else:
        # 2) DataFrame layout: my_target_list, my_enzyme_list, ...
        rel_name_map = {
            "my_target_list": "targets",
            "my_enzyme_list": "enzymes",
            "my_transporter_list": "transporters",
            "my_carrier_list": "carriers",
            "my_pathway_list": "pathways",
        }
        for kb_key in categories_my:
            cat_data = kb_data.get(kb_key)
            if cat_data is None or not isinstance(cat_data, pd.DataFrame) or cat_data.empty:
                continue
            df = cat_data
            drug_col = _kg1_drug_col(df)
            rel_name = rel_name_map.get(kb_key, kb_key)
            entity_col = _kg1_entity_col(df, rel_name)
            if drug_col is None:
                continue
            if rel_name not in rel_to_idx:
                rel_to_idx[rel_name] = len(rel_to_idx)
            rel_idx = rel_to_idx[rel_name]
            for _, row in df.iterrows():
                try:
                    drug_id = str(row[drug_col])
                except Exception:
                    continue
                if drug_id not in dict1:
                    continue
                drug_idx = dict1[drug_id]
                if entity_col is not None and entity_col in df.columns:
                    ent = row[entity_col]
                else:
                    ent = None
                    for c in df.columns:
                        if c == drug_col:
                            continue
                        ent = row[c]
                        break
                if ent is None or (isinstance(ent, float) and pd.isna(ent)):
                    continue
                ent = str(ent).strip()
                if not ent:
                    continue
                if ent not in entity_to_idx:
                    entity_to_idx[ent] = len(entity_to_idx)
                kg1[drug_idx].append((entity_to_idx[ent], rel_idx))

    tail_len = len(entity_to_idx)   # GNN1 internally adds +1 for the ghost slot
    rel_len = max(len(rel_to_idx), 1)
    log.info("KG1: entities=%d, relations=%d (from %s)", tail_len, rel_len, "DataFrame my_*" if use_dataframe else "dict targets/...")
    return kg1, tail_len, rel_len


# ───── KG2: drug → SMILES substructure (Morgan fingerprint bits) ─────
def build_kg2(drug_id2smiles, dict1, radius=2, nbits=512):
    """
    KG2: drug → substructure bit
    tail     = Morgan-fingerprint bit position (0..nbits-1); ghost bit = nbits
    relation = 0 (include); ghost rel = 1
    GNN2 has no built-in ghost mechanism, so for drugs with zero
    neighbours we append a (ghost_bit, ghost_rel) entry here.
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        # Silence the RDKit console logger to avoid a flood of
        # DEPRECATION WARNING lines.
        RDLogger.DisableLog('rdApp.*')
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False
        log.warning("RDKit unavailable; KG2 will use ghost entries for every drug")

    kg2 = defaultdict(list)
    rel_include = 0
    ghost_bit = nbits
    ghost_rel = 1

    if rdkit_ok:
        for drug_id, smiles in drug_id2smiles.items():
            drug_id = str(drug_id)
            if drug_id not in dict1 or not smiles:
                continue
            drug_idx = dict1[drug_id]
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
            for bit in fp.GetOnBits():
                kg2[drug_idx].append((bit, rel_include))

    _add_ghost_if_empty(kg2, dict1.values(), ghost_bit, ghost_rel)

    tail_len = nbits + 1    # 0..nbits-1 (real) + ghost slot
    rel_len = 2             # 0=include, 1=ghost
    log.info("KG2: tail_len=%d (fp_nbits=%d+ghost), rel_len=%d", tail_len, nbits, rel_len)
    return kg2, tail_len, rel_len


# ───── KG3: drug → drug (DDI training positive pairs, bidirectional) ─────
def build_kg3(train_pos_df, dict1):
    """
    KG3: drug-drug DDI (training positive pairs only, bidirectional)
    tail     = drug_idx (0..n_drugs-1); GNN3.ent_embed has size len(dict1)
    relation = 0 (has_ddi)
    For drugs with no DDI neighbours (cold-start), append a self-reference
    (drug_idx, 0) so GNN3.arrge does not trigger an out-of-range
    surplus-ghost lookup.
    """
    kg3 = defaultdict(list)
    rel_ddi = 0

    for _, row in train_pos_df.iterrows():
        d1 = str(row["d1"])
        d2 = str(row["d2"])
        if d1 not in dict1 or d2 not in dict1:
            continue
        i1, i2 = dict1[d1], dict1[d2]
        kg3[i1].append((i2, rel_ddi))
        kg3[i2].append((i1, rel_ddi))

    # Cold-start self-reference (avoids GNN3 surplus-ghost overflow)
    for did, didx in dict1.items():
        if len(kg3[didx]) == 0:
            kg3[didx].append((didx, rel_ddi))

    tail_len = len(dict1)   # GNN3 ent_embed actually uses len(dict1)
    rel_len = 1             # only one relation type
    log.info("KG3: drugs with DDI training positives=%d, tail_len=%d", len(kg3), tail_len)
    return kg3, tail_len, rel_len


# ───── KG4: drug → molecular property (RDKit descriptors, binned) ─────
_KG4_PROPS = ["MolMR", "MolLogP", "MolWt", "NumRotatableBonds", "NumAliphaticRings"]
_PROP2IDX = {p: i for i, p in enumerate(_KG4_PROPS)}


def build_kg4(drug_id2smiles, dict1, n_bins=10):
    """
    KG4: drug → molecular property
    tail     = property-type index (0..len(_KG4_PROPS)-1); ghost = n_props
    relation = binned-value index (0..n_bins-1);          ghost = n_bins
    """
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors as RDDesc
        # Silence the RDKit console logger to avoid a flood of
        # DEPRECATION WARNING lines.
        RDLogger.DisableLog('rdApp.*')
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False
        log.warning("RDKit unavailable; KG4 will use ghost entries for every drug")

    n_props = len(_KG4_PROPS)
    ghost_prop = n_props
    ghost_rel = n_bins

    # First collect property values for every drug.
    raw_vals = {p: [] for p in _KG4_PROPS}   # prop -> list of (drug_idx, value)

    if rdkit_ok:
        _desc_fn = {p: getattr(RDDesc, p, None) for p in _KG4_PROPS}
        for drug_id, smiles in drug_id2smiles.items():
            drug_id = str(drug_id)
            if drug_id not in dict1 or not smiles:
                continue
            drug_idx = dict1[drug_id]
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            for p, fn in _desc_fn.items():
                if fn is None:
                    continue
                try:
                    v = float(fn(mol))
                    raw_vals[p].append((drug_idx, v))
                except Exception:
                    pass

    # Global binning (use np.percentile to compute bin edges per property).
    bin_edges = {}
    for p, entries in raw_vals.items():
        if len(entries) < 2:
            bin_edges[p] = None
        else:
            vals = [e[1] for e in entries]
            pcts = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(vals, pcts)
            edges[-1] += 1e-9
            bin_edges[p] = edges

    kg4 = defaultdict(list)
    for p, entries in raw_vals.items():
        prop_idx = _PROP2IDX[p]
        edges = bin_edges[p]
        for drug_idx, val in entries:
            if edges is None:
                bin_id = 0
            else:
                bin_id = int(np.searchsorted(edges[1:], val, side="left"))
                bin_id = min(bin_id, n_bins - 1)
            kg4[drug_idx].append((prop_idx, bin_id))

    _add_ghost_if_empty(kg4, dict1.values(), ghost_prop, ghost_rel)

    tail_len = n_props + 1   # 0..n_props-1 (real) + ghost slot
    rel_len = n_bins + 1     # 0..n_bins-1 (real) + ghost slot
    log.info("KG4: tail_len=%d (props=%d+ghost), rel_len=%d", tail_len, n_props, rel_len)
    return kg4, tail_len, rel_len


# ──── Build the 4 KGs ────
log.info("Building KG1 (drug-entity)...")
kg1, kg1_tail, kg1_rel = build_kg1(kb, dict1)
log.info("Building KG2 (drug-substructure)...")
kg2, kg2_tail, kg2_rel = build_kg2(drug_id2smiles, dict1, radius=args.fp_radius, nbits=args.fp_nbits)
log.info("Building KG3 (drug-DDI)...")
kg3, kg3_tail, kg3_rel = build_kg3(train_pos_df, dict1)
log.info("Building KG4 (drug-property)...")
kg4, kg4_tail, kg4_rel = build_kg4(drug_id2smiles, dict1, n_bins=args.n_bins)

dataset = {
    "dataset1": kg1,
    "dataset2": kg2,
    "dataset3": kg3,
    "dataset4": kg4,
}
tail_len = {
    "dataset1": kg1_tail,
    "dataset2": kg2_tail,
    "dataset3": kg3_tail,
    "dataset4": kg4_tail,
}
relation_len = {
    "dataset1": kg1_rel,
    "dataset2": kg2_rel,
    "dataset3": kg3_rel,
    "dataset4": kg4_rel,
}

# ──────────────────────────────────────────────────────────────────────────────
# 8a. Cold start (S1/S2): similarity + val_adj / test_adj
#     (only when the bundle contains drug_groups)
# ──────────────────────────────────────────────────────────────────────────────
def _jaccard_sim(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    numerator = matrix @ matrix.T
    row_sum = matrix.sum(axis=1, keepdims=True)
    denominator = row_sum + row_sum.T - numerator
    out = np.zeros_like(numerator)
    np.divide(numerator, np.maximum(denominator, 1e-9), out=out)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _find_dif_sim(raw_matrix):
    X = np.asarray(raw_matrix)
    sim = (X[:, None, :] == X[None, :, :]).sum(axis=2).astype(np.float64)
    np.fill_diagonal(sim, 0)
    return sim


def _build_feature_matrices(kg1, kg2, kg3, kg4, kg1_tail, kg2_tail, n_drugs):
    n_props = len(_KG4_PROPS)
    fm1 = np.zeros((n_drugs, kg1_tail), dtype=np.float64)
    fm2 = np.zeros((n_drugs, kg2_tail), dtype=np.float64)
    fm3 = np.zeros((n_drugs, n_drugs), dtype=np.float64)
    fm4 = np.zeros((n_drugs, n_props), dtype=np.float64)
    for i in range(n_drugs):
        for (tail, _) in kg1.get(i, []):
            if tail < kg1_tail:
                fm1[i, tail] = 1
        for (tail, _) in kg2.get(i, []):
            if tail < kg2_tail:
                fm2[i, tail] = 1
        for (tail, _) in kg3.get(i, []):
            if tail < n_drugs:
                fm3[i, tail] = 1
        for (prop_idx, bin_id) in kg4.get(i, []):
            if prop_idx < n_props:
                fm4[i, prop_idx] = float(bin_id)
    return fm1, fm2, fm3, fm4


def _build_adj_for_g2(g2_idx, g1_idx, drug_sim_list, n_drugs):
    g2_idx = set(g2_idx)
    g1_idx = set(g1_idx)
    adj = [defaultdict(list) for _ in range(4)]
    for k in range(4):
        sim = drug_sim_list[k]
        for j in g2_idx:
            if j >= sim.shape[0]:
                continue
            target_list = sim[j] if hasattr(sim[j], "__iter__") and not isinstance(sim[j], (int, float)) else sim[j, :]
            if hasattr(target_list, "tolist"):
                target_list = target_list.tolist()
            else:
                target_list = list(target_list)
            max_v = 0.0
            current_p = []
            for p in range(len(target_list)):
                if p in g2_idx or p == j:
                    continue
                if p not in g1_idx:
                    continue
                v = target_list[p]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if v > max_v:
                    max_v = v
                    current_p = [p]
                elif max_v > 0 and np.isclose(v, max_v):
                    current_p.append(p)
            if not current_p:
                current_p = [j]
            adj[k][j] = current_p
    return adj


val_adj_s1 = None
val_adj_s2 = None
test_adj_s1 = None
test_adj_s2 = None

# Prefer extra.drug_groups; fall back to the legacy extra['g1_drugs'] / extra['g2_drugs'].
drug_groups = extra.get("drug_groups") if isinstance(extra, dict) else getattr(extra, "drug_groups", None)
if drug_groups is None:
    if isinstance(extra, dict):
        g1_fallback = extra.get("g1_drugs")
        g2_fallback = extra.get("g2_drugs")
    else:
        g1_fallback = getattr(extra, "g1_drugs", None)
        g2_fallback = getattr(extra, "g2_drugs", None)
    if g1_fallback is not None or g2_fallback is not None:
        drug_groups = {"g1": g1_fallback or [], "g2": g2_fallback or []}

if drug_groups is not None:
    g1_list = drug_groups.get("g1", []) if isinstance(drug_groups, dict) else getattr(drug_groups, "g1", [])
    g2_list = drug_groups.get("g2", []) if isinstance(drug_groups, dict) else getattr(drug_groups, "g2", [])
    g1_idx = set()
    g2_idx = set()
    for did in (g1_list or []):
        did = str(did)
        if did in dict1:
            g1_idx.add(dict1[did])
    for did in (g2_list or []):
        did = str(did)
        if did in dict1:
            g2_idx.add(dict1[did])
    if g1_idx and g2_idx:
        fm1, fm2, fm3, fm4 = _build_feature_matrices(kg1, kg2, kg3, kg4, kg1_tail, kg2_tail, n_drugs)
        drug_sim1 = _jaccard_sim(fm1)
        drug_sim2 = _jaccard_sim(fm2)
        drug_sim3 = _jaccard_sim(fm3)
        drug_sim4 = _find_dif_sim(fm4)
        drug_sim_list = [drug_sim1, drug_sim2, drug_sim3, drug_sim4]
        val_adj_s1 = _build_adj_for_g2(g2_idx, g1_idx, drug_sim_list, n_drugs)
        val_adj_s2 = _build_adj_for_g2(g2_idx, g1_idx, drug_sim_list, n_drugs)
        test_adj_s1 = _build_adj_for_g2(g2_idx, g1_idx, drug_sim_list, n_drugs)
        test_adj_s2 = _build_adj_for_g2(g2_idx, g1_idx, drug_sim_list, n_drugs)
        log.info("Cold start: g1=%d, g2=%d, val_adj/test_adj built (S1/S2)", len(g1_idx), len(g2_idx))
    else:
        log.info("Cold start: drug_groups present but g1 or g2 is empty or disjoint with dict1; skipping adj construction")
else:
    log.info("Cold start: no drug_groups; similar-known replacement disabled for S1/S2")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Convert DataFrames into the model's numpy input arrays
# ──────────────────────────────────────────────────────────────────────────────
def df_to_xy(df: pd.DataFrame) -> tuple:
    """Return ``(x: ndarray[n,2], y: ndarray[n])`` — only includes drug pairs whose ids are in dict1."""
    rows_x, rows_y = [], []
    for _, row in df.iterrows():
        d1, d2 = str(row["d1"]), str(row["d2"])
        if d1 not in dict1 or d2 not in dict1:
            continue
        rows_x.append([dict1[d1], dict1[d2]])
        rows_y.append(int(row["label"]))
    if not rows_x:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.array(rows_x, dtype=np.int64), np.array(rows_y, dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Instantiate the model (Cold Start wrapper: at S1/S2 val/test time,
#    replace unseen-drug embeddings with the mean of similar known drugs).
# ──────────────────────────────────────────────────────────────────────────────
class MKG_FENN_Wrapper(nn.Module):
    """Wrap GNN1-GNN4 + FusionLayer.  When ``train_or_test=1`` and a
    ``test_adj`` is supplied, the unseen-drug positions are overwritten
    with the mean embedding of their similar-known neighbours."""

    def __init__(self, gnn1, gnn2, gnn3, gnn4, fusion):
        super().__init__()
        self.gnn1 = gnn1
        self.gnn2 = gnn2
        self.gnn3 = gnn3
        self.gnn4 = gnn4
        self.fusion = fusion

    def forward(self, idx, train_or_test=0, test_adj=None):
        g1, idx = self.gnn1(idx)
        g2, g1, idx = self.gnn2((g1, idx))
        g3, g2, g1, idx = self.gnn3((g2, g1, idx))
        g4, g3, g2, g1, idx = self.gnn4((g3, g2, g1, idx))
        if train_or_test == 1 and test_adj is not None:
            embs = [g1, g2, g3, g4]
            for k in range(4):
                for i, pos in test_adj[k].items():
                    if not pos:
                        continue
                    embs[k][i] = embs[k][pos].mean(dim=0)
        return self.fusion((g4, g3, g2, g1, idx))


args.n_drug = n_drugs   # consumed inside the model (GNN1 embedding size)

_gnn1 = GNN1(dataset, tail_len, relation_len, args, dict1, drug_name_list)
_gnn2 = GNN2(dataset, tail_len, relation_len, args, dict1, drug_name_list)
_gnn3 = GNN3(dataset, tail_len, relation_len, args, dict1, drug_name_list)
_gnn4 = GNN4(dataset, tail_len, relation_len, args, dict1, drug_name_list)
_fusion = FusionLayer(args)
net = MKG_FENN_Wrapper(_gnn1, _gnn2, _gnn3, _gnn4, _fusion)
log.info("Model parameter count: %d", sum(p.numel() for p in net.parameters()))

# ── NEW: place model on GPU (vs original which ran on CPU) ──
if args.device >= 0 and torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{args.device}")
else:
    DEVICE = torch.device("cpu")
net = net.to(DEVICE)
log.info("Model placed on device: %s", DEVICE)

optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)


# ── NEW: helper to call precompute_adj on all 4 GNNs (per epoch) ──
def _resample_all_gnn_adj(net, rng=None):
    """Call precompute_adj on all 4 GNN sub-modules.
    Replicates the 'fresh neighbor sample per forward' from the original,
    but at epoch granularity (much faster, semantically near-equivalent)."""
    net.gnn1.precompute_adj(rng)
    net.gnn2.precompute_adj(rng)
    net.gnn3.precompute_adj(rng)
    net.gnn4.precompute_adj(rng)

# ──────────────────────────────────────────────────────────────────────────────
# 10. Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def do_compute_metrics(probs_np, labels_np):
    """
    probs_np : ndarray [N, 2] (softmax output)
    labels_np: ndarray [N]    (0/1)
    """
    preds = np.argmax(probs_np, axis=1)
    pos_prob = probs_np[:, 1]
    acc = float(accuracy_score(labels_np, preds))
    try:
        auc_roc = float(roc_auc_score(labels_np, pos_prob))
        if np.isnan(auc_roc):
            auc_roc = 0.5
    except Exception:
        auc_roc = 0.5
    try:
        prec_c, rec_c, _ = precision_recall_curve(labels_np, pos_prob)
        auc_prc = float(auc(rec_c, prec_c))
        if np.isnan(auc_prc):
            auc_prc = 0.0
    except Exception:
        auc_prc = 0.0
    f1 = float(f1_score(labels_np, preds, average="binary", zero_division=0))
    prec = float(precision_score(labels_np, preds, average="binary", zero_division=0))
    sens = float(recall_score(labels_np, preds, average="binary", zero_division=0))
    return acc, auc_roc, auc_prc, f1, prec, sens


def evaluate_split_xy(x_np, y_np, train_or_test=0, test_adj=None):
    """Evaluate on numpy inputs; returns (loss, probs[N,2], labels[N]).
    When ``train_or_test=1`` and ``test_adj`` is not None, Cold Start is
    enabled: unseen-drug positions are replaced with the mean of similar
    known drugs."""
    if len(x_np) == 0:
        return 0.0, np.zeros((0, 2)), np.zeros(0)
    net.eval()
    all_probs, total_loss = [], 0.0
    bs = args.batch_size
    n = len(x_np)
    with torch.no_grad():
        for start in range(0, n, bs):
            xb = torch.LongTensor(x_np[start:start + bs]).to(DEVICE)
            yb = torch.LongTensor(y_np[start:start + bs]).to(DEVICE)
            logits = net(xb, train_or_test=train_or_test, test_adj=test_adj)
            total_loss += float(loss_fn(logits, yb).item()) * len(xb)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    probs_all = np.concatenate(all_probs, axis=0)
    loss_avg = total_loss / max(n, 1)
    return loss_avg, probs_all, y_np


# ──────────────────────────────────────────────────────────────────────────────
# 11. Save helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_metric_json(metrics_dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def save_inference_csv(probas_pred, ground_truth, subset_df, path):
    """Save per-sample predictions to inference.csv (optionally attach d1/d2 if provided).
    probas_pred: (N,) or (N,2); if 2D, use column 1 as positive-class probability."""
    probas_pred = np.asarray(probas_pred)
    if probas_pred.ndim == 2:
        probas_pred = probas_pred[:, 1]
    pred_label = (probas_pred >= 0.5).astype(int)
    true_label = np.asarray(ground_truth, dtype=int)
    out = pd.DataFrame(
        {
            "predicted_prob": probas_pred,
            "predicted_label": pred_label,
            "true_label": true_label,
        }
    )
    if subset_df is not None and len(subset_df) == len(out):
        if "d1" in subset_df.columns:
            out.insert(0, "d1", subset_df["d1"].values)
        if "d2" in subset_df.columns:
            out.insert(1, "d2", subset_df["d2"].values)
    out.to_csv(path, index=False)


def evaluate_and_save_split(split_name, df_split, output_dir, subdir=None, train_or_test=0, test_adj=None):
    """
    Evaluate one split and save both metric.json and inference.csv to:
      subdir=None  → <output_dir>/<split_name>/
      subdir="val" → <output_dir>/<split_name>/val/
    """
    x_np, y_np = df_to_xy(df_split)
    if len(x_np) == 0:
        log.info("  %s: no valid samples; skipping", split_name)
        return
    loss_avg, probs, labels = evaluate_split_xy(x_np, y_np, train_or_test=train_or_test, test_adj=test_adj)
    acc, auc_roc, auc_prc, f1, prec, sens = do_compute_metrics(probs, labels)
    n_total = int(len(labels))
    n_pos = int((labels == 1).sum())
    n_neg = n_total - n_pos
    m = {
        "loss": loss_avg, "accuracy": acc,
        "auc_roc": auc_roc, "auc_prc": auc_prc,
        "f1_score": f1, "precision": prec, "sensitivity": sens,
        "n_total": n_total, "n_positive": n_pos, "n_negative": n_neg,
    }
    split_dir = (
        os.path.join(output_dir, split_name)
        if subdir is None
        else os.path.join(output_dir, split_name, subdir)
    )
    os.makedirs(split_dir, exist_ok=True)
    save_metric_json(m, os.path.join(split_dir, "metric.json"))
    save_inference_csv(probs, labels, df_split, os.path.join(split_dir, "inference.csv"))
    log.info(
        "  [%s%s] Loss=%.4f Acc=%.4f AUC-ROC=%.4f AUC-PRC=%.4f "
        "F1=%.4f P=%.4f R=%.4f (N=%d, Pos=%d, Neg=%d)",
        split_name, f"/{subdir}" if subdir else "",
        loss_avg, acc, auc_roc, auc_prc, f1, prec, sens, n_total, n_pos, n_neg,
    )


BASELINE_CSV_COLUMNS = [
    "model", "dataset", "split",
    "val-loss", "val-accuracy", "val-auc_roc", "val-auc_prc",
    "val-f1_score", "val-precision", "val-sensitivity",
    "val-n_total", "val-n_positive", "val-n_negative",
    "test-loss", "test-accuracy", "test-auc_roc", "test-auc_prc",
    "test-f1_score", "test-precision", "test-sensitivity",
    "test-n_total", "test-n_positive", "test-n_negative",
]


def save_baseline_csv(output_dir, model_name, dataset_name):
    rows = []
    for sn in ("s0", "s1", "s2"):
        test_path = os.path.join(output_dir, sn, "metric.json")
        val_path = os.path.join(output_dir, sn, "val", "metric.json")
        test_m, val_m = {}, {}
        for p, d in [(test_path, test_m), (val_path, val_m)]:
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        d.update(json.load(f))
                except Exception:
                    pass
        if not test_m and not val_m:
            continue
        row = {
            "model": model_name, "dataset": dataset_name, "split": sn,
            "val-loss": val_m.get("loss"), "val-accuracy": val_m.get("accuracy"),
            "val-auc_roc": val_m.get("auc_roc"), "val-auc_prc": val_m.get("auc_prc"),
            "val-f1_score": val_m.get("f1_score"), "val-precision": val_m.get("precision"),
            "val-sensitivity": val_m.get("sensitivity"),
            "val-n_total": val_m.get("n_total"), "val-n_positive": val_m.get("n_positive"),
            "val-n_negative": val_m.get("n_negative"),
            "test-loss": test_m.get("loss"), "test-accuracy": test_m.get("accuracy"),
            "test-auc_roc": test_m.get("auc_roc"), "test-auc_prc": test_m.get("auc_prc"),
            "test-f1_score": test_m.get("f1_score"), "test-precision": test_m.get("precision"),
            "test-sensitivity": test_m.get("sensitivity"),
            "test-n_total": test_m.get("n_total"), "test-n_positive": test_m.get("n_positive"),
            "test-n_negative": test_m.get("n_negative"),
        }
        rows.append(row)
    if not rows:
        log.warning("save_baseline_csv: no valid val/test metrics; skipping")
        return
    df_out = pd.DataFrame(rows, columns=BASELINE_CSV_COLUMNS)
    csv_path = os.path.join(output_dir, "Baseline.csv")
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    log.info("Baseline.csv saved: %s", csv_path)
    try:
        xlsx_path = os.path.join(output_dir, "Baseline.xlsx")
        df_out.to_excel(xlsx_path, index=False)
        log.info("Baseline.xlsx saved: %s", xlsx_path)
    except Exception as e:
        log.warning("Baseline.xlsx save failed: %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Training loop
# ──────────────────────────────────────────────────────────────────────────────
def train_loop():
    """
    Main training loop:
    - Each epoch uses the corresponding rotated negative set
      (train_neg_epochs[epoch % n_neg_epochs]).
    - Symmetric augmentation: each epoch processes both (A,B) and
      (B,A) directions (preserves the original MKG-FENN design).
    - Early-stop signal: training continues as long as ANY split's val
      AUC-ROC improves.
    - The best checkpoint for each of S0/S1/S2 is saved separately.
    """
    best_val_auc = {"s0": -1.0, "s1": -1.0, "s2": -1.0}
    best_val_f1  = {"s0":  0.0, "s1":  0.0, "s2":  0.0}
    no_improve = 0

    # Pre-convert val splits to numpy data once.
    val_xy = {sn: df_to_xy(df) for sn, df in val_dfs.items()}

    # Check whether each val split contains negatives.  When a split is
    # single-class, AUC is undefined and we switch its early-stop signal
    # to F1.
    _val_single_class = {}
    for sn, df in val_dfs.items():
        if "label" in df.columns and df["label"].nunique() < 2:
            _val_single_class[sn] = True
            log.warning(
                "val_%s contains a single label class only (all positives); "
                "AUC is undefined — falling back to F1 as the early-stop "
                "metric for this split", sn
            )
        else:
            _val_single_class[sn] = False

    t_start_total = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()

        # ── NEW: precompute neighbor adjacency for this epoch (was per-batch in original) ──
        _epoch_rng = np.random.RandomState(args.seed + epoch)   # deterministic per epoch
        _resample_all_gnn_adj(net, rng=_epoch_rng)

        # ── Fetch this epoch's negative sample ──
        if n_neg_epochs > 0:
            neg_df = train_neg_epochs[(epoch - 1) % n_neg_epochs]
            # extra['train_neg_epochs'] may be a list of dicts or a DataFrame already.
            if isinstance(neg_df, list):
                neg_df = pd.DataFrame(neg_df)
            neg_df = neg_df.copy().reset_index(drop=True)
            # Normalise column names: drug_a_id/drug_b_id -> d1/d2.
            if "drug_a_id" in neg_df.columns and "d1" not in neg_df.columns:
                neg_df = neg_df.rename(columns={"drug_a_id": "d1", "drug_b_id": "d2"})
            if "label" not in neg_df.columns:
                neg_df["label"] = 0
        else:
            neg_df = pd.DataFrame(columns=["d1", "d2", "label"])

        combined_df = pd.concat([train_pos_df, neg_df], ignore_index=True)
        train_x, train_y = df_to_xy(combined_df)

        if len(train_x) == 0:
            log.warning("Epoch %d: empty training data; skipping", epoch)
            continue

        # ── Symmetric augmentation (core: swap A,B to mint a mirror sample with the same label) ──
        train_x_flip = train_x.copy()
        train_x_flip[:, [0, 1]] = train_x_flip[:, [1, 0]]
        train_x_aug = np.concatenate([train_x, train_x_flip], axis=0)
        train_y_aug = np.concatenate([train_y, train_y], axis=0)

        x_t = torch.LongTensor(train_x_aug)
        y_t = torch.LongTensor(train_y_aug)
        train_data = TensorDataset(x_t, y_t)
        train_iter = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

        net.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        step = 0
        for xb, yb in train_iter:
            # Optional sub-epoch resampling (matches original per-batch variance more closely
            # at the cost of speed). step starts at 0, so first batch uses the per-epoch sample.
            if args.resample_every_n_batches > 0 and step > 0 \
                    and step % args.resample_every_n_batches == 0:
                _resample_all_gnn_adj(net, rng=np.random)

            optimizer.zero_grad()
            xb = torch.LongTensor(xb).to(DEVICE)
            yb = torch.LongTensor(yb).to(DEVICE)
            out = net(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item())
            train_acc_sum += float(accuracy_score(
                torch.argmax(out.detach(), dim=1).cpu().numpy(),
                yb.cpu().numpy(),
            ))
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break

        train_loss_avg = train_loss_sum / max(step, 1)
        train_acc_avg = train_acc_sum / max(step, 1)

        # ── Validation evaluation ──
        net.eval()
        val_aucs = {}
        val_f1s = {}
        with torch.no_grad():
            for sn in ["s0", "s1", "s2"]:
                vx, vy = val_xy[sn]
                if len(vx) == 0:
                    continue
                # To match the upstream repository behaviour exactly, the
                # similar-drug cold-start replacement is NOT enabled here;
                # we always use the plain GNN forward path
                # (train_or_test=0, test_adj=None).
                _, probs, labels = evaluate_split_xy(vx, vy, train_or_test=0, test_adj=None)
                _, auc_roc, _, f1, _, _ = do_compute_metrics(probs, labels)
                val_aucs[sn] = auc_roc
                val_f1s[sn] = f1

        # Each split tracks its own best metric and saves its own best
        # checkpoint independently.
        # - When AUC > 0.5, AUC is the primary metric (s0/s1 require
        #   >0.5; s2 cold-start allows any improvement).
        # - When AUC <= 0.5 (single-class val set or degenerate),
        #   F1 is used as the early-stop signal instead.
        any_split_improved = False
        for sn in list(val_aucs.keys()):
            auc_roc = val_aucs[sn]
            f1      = val_f1s.get(sn, 0.0)
            if np.isnan(auc_roc):
                auc_roc = 0.0
            use_f1_fallback = _val_single_class.get(sn, False) or auc_roc <= 0.5
            if use_f1_fallback:
                # F1 fallback: save whenever F1 improves (no floor required).
                if f1 > best_val_f1[sn] + 1e-4:
                    best_val_f1[sn] = f1
                    best_val_auc[sn] = auc_roc  # record the corresponding AUC (may be 0.5)
                    torch.save({"epoch": epoch, "model_state_dict": net.state_dict()},
                               os.path.join(run_output_dir, f"best_model_{sn}.pt"))
                    any_split_improved = True
            else:
                floor = 0.5 if sn in ("s0", "s1") else -1.0
                threshold = max(best_val_auc[sn] + 1e-5, floor)
                if auc_roc > threshold:
                    best_val_auc[sn] = auc_roc
                    best_val_f1[sn]  = f1
                    torch.save({"epoch": epoch, "model_state_dict": net.state_dict()},
                               os.path.join(run_output_dir, f"best_model_{sn}.pt"))
                    any_split_improved = True

        if any_split_improved:
            no_improve = 0
        else:
            no_improve += 1

        t_epoch = time.time() - t_epoch_start
        best_str = " | ".join(
            f"{k}:AUC={best_val_auc[k]:.4f}/F1={best_val_f1[k]:.4f}" for k in ("s0","s1","s2")
        )
        log.info(
            "Epoch %3d/%d | train_loss=%.4f train_acc=%.4f | "
            "val_s0 AUC=%.4f F1=%.4f | val_s1 AUC=%.4f F1=%.4f | "
            "val_s2 AUC=%.4f F1=%.4f | Best=[%s] | %.1fs",
            epoch, args.epochs,
            train_loss_avg, train_acc_avg,
            val_aucs.get("s0", 0.0), val_f1s.get("s0", 0.0),
            val_aucs.get("s1", 0.0), val_f1s.get("s1", 0.0),
            val_aucs.get("s2", 0.0), val_f1s.get("s2", 0.0),
            best_str, t_epoch,
        )

        if no_improve >= args.patience:
            log.info("Early stop: no improvement on any split for %d consecutive epochs",
                     args.patience)
            break

    total_time = time.time() - t_start_total
    log.info("Training complete: %d epochs, total %.1fs, Best=[%s]",
             epoch, total_time,
             " | ".join(f"{k}:AUC={best_val_auc[k]:.4f}/F1={best_val_f1[k]:.4f}" for k in ("s0","s1","s2")))
    return best_val_auc


# ──────────────────────────────────────────────────────────────────────────────
# 13. Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Hyper-parameters: epochs=%d, batch_size=%d, lr=%g, dropout=%g, "
             "embedding_num=%d, neighbor_sample_size=%d, patience=%d",
             args.epochs, args.batch_size, args.lr, args.dropout,
             args.embedding_num, args.neighbor_sample_size, args.patience)

    best_val_auc = train_loop()

    # ── For each split, load its own best checkpoint and evaluate val + test ──
    log.info("===== Final evaluation (val + test for S0/S1/S2; each split uses its own best model) =====")
    for sn in ["s0", "s1", "s2"]:
        best_model_path = os.path.join(run_output_dir, f"best_model_{sn}.pt")
        if os.path.isfile(best_model_path):
            ckpt = torch.load(best_model_path, map_location="cpu")
            net.load_state_dict(ckpt["model_state_dict"])
            log.info("Loaded best model (%s) epoch=%d, best_val_auc=%.4f",
                     sn, ckpt.get("epoch", 0), best_val_auc[sn])
        else:
            log.warning("No best-model file found for %s; skipping", sn)
            continue
        # Final evaluation also keeps similar-drug replacement disabled,
        # matching the upstream repository's cold-start semantics.
        evaluate_and_save_split(sn, val_dfs[sn], run_output_dir, subdir="val", train_or_test=0, test_adj=None)
        evaluate_and_save_split(sn, test_dfs[sn], run_output_dir, subdir=None, train_or_test=0, test_adj=None)

    save_baseline_csv(run_output_dir, "MKG-FENN", _dataset_name)
    log.info("All done. Output directory: %s", run_output_dir)

"""EmerGNN paper data loader.

Reads `Original-Dataset/DrugBank/data/` (downloaded from Zenodo
10.5281/zenodo.10016715, identical to the OneDrive link in the official
GitHub README) and produces:

- vocab dicts (drug → id, entity → id, relation → id)
- Morgan FP matrix indexed by global entity id (1710 drugs + 32414
  non-drug entities = up to 34124 entities)
- per-setting train/valid/test DDI triplets + train/valid/test KG
  triplets (the latter two are *deltas* per paper convention; the
  builder below cumulates them per paper `load_data.py` lines 117-119).

All file formats and ID conventions match paper's `load_data.py`
verbatim.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_DATA_DIR = Path(__file__).resolve().parent / "_Original-Dataset" / "DrugBank" / "data"
NECESSARY_DIR = REPO_DATA_DIR / "necessary"

# Paper hardcodes 86 DDI classes in load_data.py:67  `self.eval_rel = 86`.
N_DDI_CLASSES: int = 86


def _resolve_necessary(name: str, ext: str) -> Path:
    """Resolve a `necessary/` artefact path per CLAUDE.md §"复现代码规范"
    §2 priority: ``__official`` > ``__mine`` > legacy flat name.

    Logs the resolved source to stderr so pipeline output makes which
    version was used auditable.
    """
    import sys as _sys
    official = NECESSARY_DIR / f"{name}__official.{ext}"
    mine = NECESSARY_DIR / f"{name}__mine.{ext}"
    legacy = REPO_DATA_DIR / f"{name}.{ext}"  # pre-migration flat path
    for candidate, tag in (
        (official, "OFFICIAL"),
        (mine, "MINE (official missing)"),
        (legacy, "LEGACY (flat, pre-2026-05-18 layout)"),
    ):
        if candidate.is_file():
            print(f"[{name}] USING {tag}: {candidate}", file=_sys.stderr)
            return candidate
    raise FileNotFoundError(
        f"necessary artefact '{name}.{ext}' not found at any of: "
        f"{official}, {mine}, {legacy}. Per CLAUDE.md §复现代码规范 §3 "
        f"step 6, the reproduction does NOT auto-build — download via "
        f"upstream Zenodo (10.5281/zenodo.10016715) and place at the "
        f"__official path."
    )


# -----------------------------------------------------------------------------
# Vocab + Morgan
# -----------------------------------------------------------------------------


def load_vocab() -> dict:
    """Load global vocab files (shared across all S0/S1/S2 settings).

    Returns dict with:
        drug2id     : {drugbank_id_str: int}      from node2id.json (1710 drugs)
        entity2id   : {entity_str:      int}      from entity_drug.json (32414 entities)
        relation2id : {relation_str:    int}      from relation2id.json (109 rels)
        n_drug      : 1710
        n_entity    : len(drug2id) + len(entity2id)
        n_relation  : len(relation2id)
    """
    drug2id_raw = json.load(open(_resolve_necessary("node2id", "json")))
    entity2id_raw = json.load(open(_resolve_necessary("entity_drug", "json")))
    relation_raw = json.load(open(_resolve_necessary("relation2id", "json")))
    # `node2id.json` and `entity_drug.json` map (name str → int id) — direct.
    # `relation2id.json` is REVERSED: (id_as_str → description); we only need
    # the int ids, so we use the keys.
    drug2id = {str(k): int(v) for k, v in drug2id_raw.items()}
    entity2id = {str(k): int(v) for k, v in entity2id_raw.items()}
    relation_id_to_text = {int(k): str(v) for k, v in relation_raw.items()}
    n_drug = len(drug2id)
    n_entity = n_drug + len(entity2id)
    n_relation = len(relation_id_to_text)
    return {
        "drug2id": drug2id,
        "entity2id": entity2id,
        "relation_id_to_text": relation_id_to_text,
        "n_drug": n_drug,
        "n_entity": n_entity,
        "n_relation": n_relation,
    }


def load_morgan_features() -> np.ndarray:
    """Load paper's pre-computed Morgan FP (1710, 1024) float32 matrix,
    indexed by `Node ID` (= drug2id value).

    Source: `DB_molecular_feats.pkl` is a pandas DataFrame with columns
    `DrugBank ID`, `Node ID`, `SMILES`, `Morgan_Features`, `RDKit2D_Features`.
    We index by `Node ID` to align with the drug entity ID space.
    """
    pkl_path = _resolve_necessary("DB_molecular_feats", "pkl")
    obj = pickle.load(open(pkl_path, "rb"), encoding="utf-8")
    # obj is a DataFrame (or dict of pandas Series) per Zenodo build
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict):
        df = pd.DataFrame(obj)
    else:
        raise TypeError(f"unexpected DB_molecular_feats.pkl payload: {type(obj)}")
    n_drug = len(df)
    # Each row's Morgan_Features is a numpy array of length 1024
    morgan_dim = len(df["Morgan_Features"].iloc[0])
    # Paper's models.py iterates ``for y in x['Morgan_Features']`` directly,
    # implicitly trusting row order == Node ID order. We verify that
    # invariant so our (Node ID indexed) load is bit-equivalent.
    expected_node_ids = np.arange(n_drug, dtype=int)
    actual_node_ids = df["Node ID"].to_numpy().astype(int)
    if not np.array_equal(actual_node_ids, expected_node_ids):
        raise ValueError(
            f"DB_molecular_feats.pkl Node ID column does not equal arange({n_drug}); "
            f"paper's models.py assumes it does (iterates Morgan_Features in row order)."
        )
    mat = np.zeros((n_drug, morgan_dim), dtype=np.float32)
    for _, row in df.iterrows():
        nid = int(row["Node ID"])
        mat[nid] = np.asarray(row["Morgan_Features"], dtype=np.float32)
    return mat


def build_global_morgan_matrix(vocab: dict) -> np.ndarray:
    """Build (n_entity, 1024) Morgan FP matrix where rows 0..n_drug-1 are
    actual drug fingerprints and rows n_drug..n_entity-1 are zeros
    (non-drug entities have no molecular features in paper's setup).

    Mirrors paper's `models.py:EmerGNN.__init__` where
    `self.ent_kg = nn.Parameter(torch.FloatTensor(mfeat))` holds drug
    features only — but our port's `EmerGNN_MC` expects a (n_ent, 1024)
    matrix covering all entities. Padding non-drug rows with zeros
    matches the implicit behavior of paper's code: `head/tail` in
    `enc_ht` are always drug IDs (0..1709), so non-drug rows are never
    queried during forward.
    """
    morgan_drug = load_morgan_features()
    if morgan_drug.shape[0] != vocab["n_drug"]:
        raise ValueError(
            f"morgan_drug rows {morgan_drug.shape[0]} != n_drug {vocab['n_drug']}"
        )
    n_ent = vocab["n_entity"]
    mat = np.zeros((n_ent, morgan_drug.shape[1]), dtype=np.float32)
    mat[: vocab["n_drug"]] = morgan_drug
    return mat


# -----------------------------------------------------------------------------
# Per-setting txt loaders
# -----------------------------------------------------------------------------


def _read_triplet_txt(path: Path) -> np.ndarray:
    """Read paper's `h t r` per-line int triplet file.

    Empty lines are skipped. Paper uses raw int IDs — same vocab as the
    JSON files; we don't re-map.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append([int(parts[0]), int(parts[1]), int(parts[2])])
    if not rows:
        return np.zeros((0, 3), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def load_setting(setting: str) -> dict:
    """Load one S0 / S1_<seed> / S2_<seed> folder.

    Returns dict with:
        train_ddi, valid_ddi, test_ddi    : (n, 3) int arrays  (h, t, r), r ∈ 0..85
        train_kg                          : (n, 3) base KG used for training
        valid_kg                          : cumulative KG = train_kg + valid_KG.txt delta
        test_kg                           : cumulative KG = train_kg + valid_KG.txt + test_KG.txt deltas
        setting                           : str (passed-through)
    """
    base = REPO_DATA_DIR / setting
    if not base.is_dir():
        raise FileNotFoundError(f"setting folder not found: {base}")

    train_ddi = _read_triplet_txt(base / "train_ddi.txt")
    valid_ddi = _read_triplet_txt(base / "valid_ddi.txt")
    test_ddi = _read_triplet_txt(base / "test_ddi.txt")

    train_kg_raw = _read_triplet_txt(base / "train_KG.txt")
    valid_kg_delta = _read_triplet_txt(base / "valid_KG.txt")
    test_kg_delta = _read_triplet_txt(base / "test_KG.txt")

    # paper load_data.py:117-119 cumulates
    valid_kg = (
        np.concatenate([train_kg_raw, valid_kg_delta], axis=0)
        if len(valid_kg_delta)
        else train_kg_raw
    )
    test_kg = (
        np.concatenate([train_kg_raw, valid_kg_delta, test_kg_delta], axis=0)
        if len(test_kg_delta) or len(valid_kg_delta)
        else train_kg_raw
    )

    return {
        "setting": setting,
        "train_ddi": train_ddi,
        "valid_ddi": valid_ddi,
        "test_ddi": test_ddi,
        "train_kg": train_kg_raw,
        "valid_kg": valid_kg,
        "test_kg": test_kg,
    }


# -----------------------------------------------------------------------------
# Shuffle train (per-epoch fact/target resampling) — paper load_data.py:138-178
# -----------------------------------------------------------------------------


def shuffle_train(
    train_ddi: np.ndarray,
    train_kg: np.ndarray,
    setting: str,
    *,
    ratio: float = 0.8,
    rng: np.random.Generator | None = None,
    extra_kg_ent: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch shuffle — paper `load_data.py:shuffle_train`.

    Args:
        train_ddi: full train DDI triplet array (n, 3).
        train_kg:  base KG triplet array (m, 3).
        setting:   'S0', 'S1_*', or 'S2_*'.
        ratio:     fraction of `ddi_in_kg` to KEEP as "old" entities.
                   Paper default 0.8.
        rng:       optional numpy Generator for reproducibility.
        extra_kg_ent: optional set of entity ids that also appear in the
                   KG (e.g. drugs that show up only in valid_kg / test_kg
                   deltas). Paper's `process_files_kg` 92-109 unions all
                   3 KG splits when populating `ddi_in_kg`, so this kwarg
                   exists to pass that union from the caller. If None,
                   only `train_kg` is used (looser approximation).

    Returns:
        (epoch_kg, train_targets):
          epoch_kg      : (M, 3) int triplets to build the epoch's KG
                          (= fact_triplet ⊕ train_kg in paper)
          train_targets : (N, 3) int triplets — the prediction targets for
                          this epoch

    Semantics per setting prefix (paper code lines 144-178):
      * S0: random 80/20 split of all train_ddi; first 80% go into KG,
            last 20% are targets.
      * S1: for each triplet (h, t, r):
                - if both h, t in kept_ent → fact
                - elif exactly one in kept_ent → target  (one-old + one-new)
      * S2: for each triplet (h, t, r):
                - if both h, t in kept_ent → fact
                - elif neither in kept_ent → target  (two-new)
    """
    if rng is None:
        rng = np.random.default_rng()

    # entities present in train_ddi
    train_ent = set(np.unique(train_ddi[:, :2]).tolist())
    # entities present in train_kg (+ optional extra KG ent from val/test KG)
    kg_ent = set(np.unique(train_kg[:, :2]).tolist()) if len(train_kg) else set()
    if extra_kg_ent:
        kg_ent = kg_ent | extra_kg_ent
    # entities present in BOTH train_ddi and (full) KG
    ddi_in_kg = train_ent & kg_ent

    if setting.startswith("S0"):
        n_all = len(train_ddi)
        if n_all == 0:
            return train_kg.copy(), np.zeros((0, 3), dtype=np.int64)
        perm = rng.permutation(n_all)
        shuffled = train_ddi[perm]
        n_fact = int(n_all * ratio)
        fact = shuffled[:n_fact]
        targets = shuffled[n_fact:]
        epoch_kg = np.concatenate([fact, train_kg], axis=0) if len(train_kg) else fact
        return epoch_kg, targets

    n_in_kg = len(ddi_in_kg)
    if n_in_kg == 0 or ratio >= 1.0:
        # degenerate: no shuffling possible — fall back to using all train_ddi
        # as targets and only train_kg as KG
        return train_kg.copy(), train_ddi.copy()

    n_remove = n_in_kg - int(n_in_kg * ratio)
    removed = set(
        rng.choice(np.array(sorted(ddi_in_kg)), size=n_remove, replace=False).tolist()
    )
    kept = train_ent - removed

    fact_list: list[list[int]] = []
    targets_list: list[list[int]] = []
    if setting.startswith("S1"):
        for h, t, r in train_ddi:
            h_kept = h in kept
            t_kept = t in kept
            if h_kept and t_kept:
                fact_list.append([h, t, r])
            elif h_kept ^ t_kept:  # exactly one in kept
                targets_list.append([h, t, r])
            # else: both removed → discarded this epoch
    elif setting.startswith("S2"):
        for h, t, r in train_ddi:
            h_kept = h in kept
            t_kept = t in kept
            if h_kept and t_kept:
                fact_list.append([h, t, r])
            elif (not h_kept) and (not t_kept):
                targets_list.append([h, t, r])
            # else: one kept + one removed → discarded
    else:
        raise ValueError(f"unknown setting prefix: {setting!r}")

    fact_arr = (
        np.array(fact_list, dtype=np.int64) if fact_list else np.zeros((0, 3), dtype=np.int64)
    )
    targets_arr = (
        np.array(targets_list, dtype=np.int64)
        if targets_list
        else np.zeros((0, 3), dtype=np.int64)
    )
    epoch_kg = (
        np.concatenate([fact_arr, train_kg], axis=0) if len(train_kg) else fact_arr
    )
    return epoch_kg, targets_arr


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------


def _self_check() -> None:
    print("[data_loader] vocab + Morgan + S1_1 sanity check")
    vocab = load_vocab()
    print(
        f"  n_drug={vocab['n_drug']}  n_entity={vocab['n_entity']}  "
        f"n_relation={vocab['n_relation']}"
    )
    morgan_mat = build_global_morgan_matrix(vocab)
    print(
        f"  morgan_matrix: shape={morgan_mat.shape}  "
        f"non-zero rows={(morgan_mat.sum(1) > 0).sum()}"
    )
    s = load_setting("S1_1")
    print(
        f"  S1_1: train_ddi={len(s['train_ddi'])}  valid_ddi={len(s['valid_ddi'])}  "
        f"test_ddi={len(s['test_ddi'])}"
    )
    print(
        f"  S1_1: train_kg={len(s['train_kg'])}  valid_kg(cum)={len(s['valid_kg'])}  "
        f"test_kg(cum)={len(s['test_kg'])}"
    )
    # paper Supp Table 1: S1 seed=1 → train_ddi=137864, valid_ddi=17591, test_ddi=32322
    assert len(s["train_ddi"]) == 137864, "train_ddi rows mismatch paper"
    assert len(s["valid_ddi"]) == 17591, "valid_ddi rows mismatch paper"
    assert len(s["test_ddi"]) == 32322, "test_ddi rows mismatch paper"
    # paper Supp Table 1: |N_B-train| = 1656037
    assert len(s["train_kg"]) == 1656037, "train_kg rows mismatch paper"
    # DDI relation range check
    max_r_ddi = int(max(s["train_ddi"][:, 2].max(), s["valid_ddi"][:, 2].max(), s["test_ddi"][:, 2].max()))
    assert max_r_ddi <= 85, f"max DDI relation id = {max_r_ddi} > 85"
    print("  ✅ all paper-stat assertions passed")
    # shuffle_train check
    rng = np.random.default_rng(42)
    ekg, tgt = shuffle_train(s["train_ddi"], s["train_kg"], "S1_1", rng=rng)
    print(
        f"  shuffle_train(S1_1, ratio=0.8): epoch_kg={len(ekg)}  "
        f"train_targets={len(tgt)}"
    )


if __name__ == "__main__":
    _self_check()

"""EmerGNN per-epoch shuffle_train utility.

Ports paper's :func:`LARS-research/EmerGNN/DrugBank/load_data.py:shuffle_train`
to project use (operates on integer-id triplets). This is paper §Methods'
**core inductive design** — without per-epoch fact/target resampling the
training degenerates to transductive (model always sees all train DDI as
KG facts), losing the inductive-flow simulation that paper relies on.

Each call returns:
  - ``epoch_kg``      : (M, 3) int triplets to build this epoch's KG
                        (= fact_triplet ⊕ base_kg in paper convention)
  - ``train_targets`` : (N, 3) int triplets — prediction targets

Semantics per ``setting`` prefix (matches paper code lines 144-178):
  * S0: random 80/20 split of all train_ddi
  * S1: target = exactly one endpoint in kept (one-old + one-new)
  * S2: target = both endpoints removed (two-new)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def shuffle_train(
    train_ddi: np.ndarray,
    train_kg: np.ndarray,
    setting: str,
    *,
    ratio: float = 0.8,
    rng: Optional[np.random.Generator] = None,
    extra_kg_ent: Optional[set] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch fact/target resampling.

    Args:
        train_ddi: full train DDI triplet array (n, 3), int dtype. Columns =
            (head_entity_id, tail_entity_id, relation_id).
        train_kg:  base KG triplet array (m, 3), int dtype.
        setting:   one of 'S0', 'S1', 'S2' (case-sensitive prefix match).
        ratio:     fraction of ``ddi_in_kg`` to KEEP as "old" entities.
                   Paper default 0.8.
        rng:       optional numpy Generator for reproducibility.
        extra_kg_ent: optional set of KG-only entities (e.g. drugs appearing
                   only in valid/test KG deltas). Paper's
                   ``process_files_kg`` unions all 3 KG splits when
                   populating ``ddi_in_kg``; pass that union here.

    Returns:
        (epoch_kg, train_targets), both (·, 3) int arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    train_ddi = np.asarray(train_ddi, dtype=np.int64)
    train_kg = np.asarray(train_kg, dtype=np.int64) if len(train_kg) else np.zeros((0, 3), dtype=np.int64)

    train_ent = set(np.unique(train_ddi[:, :2]).tolist())
    kg_ent = set(np.unique(train_kg[:, :2]).tolist()) if len(train_kg) else set()
    if extra_kg_ent:
        kg_ent = kg_ent | set(extra_kg_ent)
    ddi_in_kg = train_ent & kg_ent

    # ── S0: random 80/20 split, no inductive simulation ──────────────────
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

    # ── S1 / S2: inductive simulation, mark 20% drugs as "emerging" ──────
    n_in_kg = len(ddi_in_kg)
    if n_in_kg == 0 or ratio >= 1.0:
        # degenerate: no shuffling possible; fall back to "all train_ddi as
        # targets, only train_kg as KG" — equivalent to transductive setup
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
            elif h_kept ^ t_kept:  # exactly one endpoint in kept
                targets_list.append([h, t, r])
            # else: both removed → silently discarded this epoch
    elif setting.startswith("S2"):
        for h, t, r in train_ddi:
            h_kept = h in kept
            t_kept = t in kept
            if h_kept and t_kept:
                fact_list.append([h, t, r])
            elif (not h_kept) and (not t_kept):
                targets_list.append([h, t, r])
            # else: one kept + one removed → silently discarded
    else:
        raise ValueError(f"unknown setting prefix: {setting!r}; expect S0/S1/S2")

    fact_arr = (
        np.array(fact_list, dtype=np.int64)
        if fact_list
        else np.zeros((0, 3), dtype=np.int64)
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


def build_edge_lists_from_triplets(
    triplets: np.ndarray,
    n_ent: int,
    n_base_rel: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (src, dst, rel) edge lists with paper convention:
    forward edges (h, t, r+n_base_rel), reverse (t, h, r), self-loops
    (e, e, 2*n_base_rel) for each entity. Returns numpy int64 arrays.

    Matches ``baseline.emergnn.kg_builder.build_sparse_adj`` /
    ``edges_as_dense_lists`` but without the sparse_coo intermediate that
    triggers huge memory spikes on some torch versions.
    """
    triplets = np.asarray(triplets, dtype=np.int64)
    if len(triplets) == 0:
        idd = np.arange(n_ent, dtype=np.int64)
        return idd, idd, np.full(n_ent, 2 * n_base_rel, dtype=np.int64)
    heads = triplets[:, 0]
    tails = triplets[:, 1]
    rels = triplets[:, 2]
    fwd_r = rels + n_base_rel
    rev_r = rels
    idd = np.arange(n_ent, dtype=np.int64)
    idd_r = np.full(n_ent, 2 * n_base_rel, dtype=np.int64)
    src = np.concatenate([heads, tails, idd])
    dst = np.concatenate([tails, heads, idd])
    rel = np.concatenate([fwd_r, rev_r, idd_r])
    return src, dst, rel

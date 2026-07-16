"""Builder: unified BINARY `Leaf` -> SumGNN's on-disk int-indexed layout (Case-B).

Binary counterpart of :mod:`build_sumgnn_data` (multiclass). It writes the SAME shared
artifacts EXACTLY as the mc builder by REUSING its helpers (``build_entity_index_full`` /
``build_entity_index`` / ``_morgan_bits``): ``entity.txt``, ``relations_2hop.txt``,
``DB_molecular_feats.pkl``, ``entity2id.pkl`` (full-KG default). It does NOT edit the mc
builder.

Binary-specific faithfulness (codex-approved):

  * The train DDI relation (relation slot 0) adjacency is built from TRAIN POSITIVES ONLY.
    ``train.txt`` therefore contains ONLY ``y_bin==1`` rows (``a b 0``). NEGATIVES MUST
    NOT enter ``train.txt`` / the adjacency, or ``process_files_ddi`` would treat them as
    real DDI edges.
  * SEPARATE pos/neg QUERY files per split are written for enclosing-subgraph extraction:
    ``train_pos.txt`` / ``train_neg.txt`` / ``dev_pos.txt`` / ``dev_neg.txt`` /
    ``test_pos.txt`` / ``test_neg.txt``, each row ``drug_a_idx drug_b_idx 0`` (r_label
    slot 0). ``g_label`` (1 pos / 0 neg) is assigned at extraction time, not stored here.
  * NEGATIVES are the leaf's FIXED materialized negatives (``y_bin==0`` rows) used as
    extraction queries ONLY. We do NOT regenerate negatives and do NOT add extra ones.
  * ``dev.txt`` / ``test.txt`` (positives only, ``a b 0``) exist so ``process_files_ddi``
    -- which reads all three file paths to assemble ``entity2id`` -- does not choke; the
    adjacency is still built ONLY from the ``train`` relation-0 rows (see
    ``process_files_ddi``: adjacency loops over ``triplets['train']`` for rel in range).
  * A ``test_manifest.json`` sidecar preserves the ORIGINAL test_df row order + the
    keep-mask (pairs whose both drugs are known), so the wrapper can align the extracted
    test-query predictions (concatenated as [pos, neg]) back to the full test_df rows.

Drugs occupy contiguous ids ``[0, n_drugs)`` (sorted); non-drug KG entities follow. The
single DDI relation occupies slot 0 in ``relation2id`` (built by ``process_files_ddi`` at
extraction time from the ``label`` column of ``train.txt``, which is always 0 here).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# REUSE the mc builder's helpers UNCHANGED (does not edit build_sumgnn_data).
from baseline.sumgnn._data.necessary.build_sumgnn_data import (
    _morgan_bits, build_entity_index, build_entity_index_full)

_PAIR = ["drug_a_id", "drug_b_id"]


def _pos_neg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a binary leaf frame into (positives, negatives) by ``y_bin``.

    Positives = ``y_bin==1`` rows; negatives = ``y_bin==0`` rows (the FIXED materialized
    benchmark negatives). If ``y_bin`` is absent, every row is treated as a positive
    (defensive; unified binary leaves always carry ``y_bin``)."""
    if "y_bin" in df.columns:
        pos = df[df["y_bin"] == 1][_PAIR].reset_index(drop=True)
        neg = df[df["y_bin"] == 0][_PAIR].reset_index(drop=True)
    else:
        pos = df[_PAIR].reset_index(drop=True)
        neg = df.iloc[0:0][_PAIR].reset_index(drop=True)
    return pos, neg


def build_sumgnn_binary_data(
    out_dir: str | Path,
    drugs: pd.DataFrame,
    edges: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    morgan_from: str = "smiles",
    kg_scope: str = "full",
) -> dict:
    """Write SumGNN's on-disk BINARY layout for one leaf. Returns build stats.

    ``train/val/test_df`` carry ``drug_a_id``, ``drug_b_id`` (strings) + ``y_bin`` (1/0).
    Unknown drugs (not in ``drugs``) are skipped (logged in stats). ``kg_scope``: "full"
    (default; whole merged KG) or "drug_incident" (1-hop, legacy).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drug_ids = drugs["drug_id"].astype(str).tolist()
    if kg_scope == "full":
        entity2id, entity_types, tdf = build_entity_index_full(drug_ids, edges)
    elif kg_scope == "drug_incident":
        entity2id, entity_types, tdf = build_entity_index(drug_ids, edges)
    else:
        raise ValueError(f"kg_scope must be 'full' or 'drug_incident'; got {kg_scope!r}")
    n_drugs = int((entity_types == 1).sum())
    n_ent = len(entity_types)

    # entity.txt: one type per line (index order) -- EXACTLY the mc builder.
    np.savetxt(out / "entity.txt", entity_types, fmt="%d")

    # relations_2hop.txt: head tail rel_idx (rel enumerated, sorted) -- EXACTLY the mc.
    rel_list = sorted(tdf["rel"].unique())
    rel2id = {r_: i for i, r_ in enumerate(rel_list)}
    kg = np.column_stack([tdf["h_idx"].to_numpy(),
                          tdf["t_idx"].to_numpy(),
                          tdf["rel"].map(rel2id).to_numpy(dtype=np.int64)])
    np.savetxt(out / "relations_2hop.txt", kg, fmt="%d")

    def _map_pairs(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Map a pair frame to int (a,b) idx; return (rows kept as (n,3) with r=0, keep
        mask over frame's original row order)."""
        a = frame["drug_a_id"].astype(str).map(entity2id)
        b = frame["drug_b_id"].astype(str).map(entity2id)
        keep = (a.notna() & b.notna()).to_numpy()
        rows = np.column_stack([a[keep].to_numpy(dtype=np.int64),
                                b[keep].to_numpy(dtype=np.int64),
                                np.zeros(int(keep.sum()), dtype=np.int64)])  # r_label slot 0
        return rows, keep

    dropped = {}

    # ---- train.txt = POSITIVES ONLY (drives the DDI relation-0 adjacency) --------
    train_pos, train_neg = _pos_neg(train_df)
    tr_pos_rows, tr_pos_keep = _map_pairs(train_pos)
    tr_neg_rows, tr_neg_keep = _map_pairs(train_neg)
    np.savetxt(out / "train.txt", tr_pos_rows, fmt="%d")          # adjacency source (POS only)
    np.savetxt(out / "train_pos.txt", tr_pos_rows, fmt="%d")      # pos query set (g_label 1)
    np.savetxt(out / "train_neg.txt", tr_neg_rows, fmt="%d")      # neg query set (g_label 0)
    dropped["train_pos"] = int((~tr_pos_keep).sum())
    dropped["train_neg"] = int((~tr_neg_keep).sum())

    # ---- dev / test: pos-only "<name>.txt" (so process_files_ddi can read all paths),
    #      plus explicit pos/neg query files for extraction ------------------------
    for frame, name in ((val_df, "dev"), (test_df, "test")):
        pos, neg = _pos_neg(frame)
        pos_rows, pos_keep = _map_pairs(pos)
        neg_rows, neg_keep = _map_pairs(neg)
        np.savetxt(out / f"{name}.txt", pos_rows, fmt="%d")          # positives (read by ddi)
        np.savetxt(out / f"{name}_pos.txt", pos_rows, fmt="%d")      # pos query set
        np.savetxt(out / f"{name}_neg.txt", neg_rows, fmt="%d")      # neg query set
        dropped[f"{name}_pos"] = int((~pos_keep).sum())
        dropped[f"{name}_neg"] = int((~neg_keep).sum())

    # ---- test alignment manifest: original test_df row order + per-row keep mask +
    #      whether each kept row is pos or neg. The extracted test queries are ordered
    #      [all test_pos, then all test_neg]; the wrapper uses this to scatter predicted
    #      probs back onto the full test_df row order (dropped rows -> 0.5 default). ----
    test_a = test_df["drug_a_id"].astype(str).map(entity2id)
    test_b = test_df["drug_b_id"].astype(str).map(entity2id)
    test_known = (test_a.notna() & test_b.notna()).to_numpy()
    test_is_pos = (test_df["y_bin"].to_numpy() == 1) if "y_bin" in test_df.columns \
        else np.ones(len(test_df), dtype=bool)
    manifest = {
        "n_test_rows": int(len(test_df)),
        "known_mask": test_known.astype(int).tolist(),   # 1 if both drugs known
        "is_pos": test_is_pos.astype(int).tolist(),      # 1 if y_bin==1
    }
    (out / "test_manifest.json").write_text(json.dumps(manifest))

    # Morgan feats indexed by drug idx (0..n_drugs-1) -- EXACTLY the mc builder.
    id2smiles = dict(zip(drugs["drug_id"].astype(str), drugs[morgan_from].astype(str)))
    id2drug = {v: k for k, v in entity2id.items() if v < n_drugs}
    mfeat = [_morgan_bits(id2smiles.get(id2drug[i], "")) for i in range(n_drugs)]
    with open(out / "DB_molecular_feats.pkl", "wb") as f:
        pickle.dump({"Morgan_Features": mfeat}, f)

    stats = {"n_ent": n_ent, "n_drugs": n_drugs, "n_kg_edges": int(len(kg)),
             "n_kg_rels": len(rel_list),
             "n_train_pos": int(len(tr_pos_rows)), "n_train_neg": int(len(tr_neg_rows)),
             "dropped_pairs": dropped}
    (out / "_build_stats.json").write_text(json.dumps(stats, indent=2))
    with open(out / "entity2id.pkl", "wb") as f:
        pickle.dump(entity2id, f)
    return stats


__all__ = ["build_sumgnn_binary_data"]

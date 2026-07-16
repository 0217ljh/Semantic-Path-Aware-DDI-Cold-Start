"""Builder: unified BINARY ``Leaf`` -> KnowDDI's on-disk int layout (Case-B GraIL-style).

Binary counterpart of :mod:`build_knowddi_data` (multiclass). It writes the SAME shared KG
artifacts as the mc builder by REUSING its ``build_entity_index_full`` helper (dense
entity indexing over the FULL merged KG), plus the binary-specific query/adjacency layout.
It does NOT edit the mc builder.

Binary-specific faithfulness (mirrors ``build_sumgnn_binary_data``; codex 019f2930):

  * The train DDI relation (relation slot 0) adjacency is built from TRAIN POSITIVES ONLY.
    ``train.txt`` therefore contains ONLY ``y_bin==1`` rows (``a b 0``). NEGATIVES MUST NOT
    enter ``train.txt`` / the adjacency, or ``process_files_ddi`` would treat them as real
    DDI edges.
  * SEPARATE pos/neg QUERY files per split are written for enclosing-subgraph extraction:
    ``train_pos.txt`` / ``train_neg.txt`` / ``valid_pos.txt`` / ``valid_neg.txt`` /
    ``test_pos.txt`` / ``test_neg.txt``, each row ``drug_a_idx drug_b_idx 0`` (r_label slot
    0). ``g_label`` (1 pos / 0 neg) is assigned at extraction time, not stored here.
  * NEGATIVES are the leaf's FIXED materialized negatives (``y_bin==0`` rows) used as
    extraction queries ONLY. We do NOT regenerate negatives and do NOT add extra ones.
  * ``valid.txt`` / ``test.txt`` (positives only, ``a b 0``) exist so ``process_files_ddi``
    -- which reads all three file paths to assemble ``entity2id`` -- does not choke; the
    adjacency is still built ONLY from the ``train`` relation-0 rows (process_files_ddi
    loops ``adj_list`` over ``triplets['train']`` for rel in range(rel)).
  * ``BKG_file.txt`` (whitespace ``h t r``, np.loadtxt-readable) carries the FULL merged KG.
  * A ``test_manifest.json`` sidecar preserves the ORIGINAL test_df row order + the keep-mask
    (pairs whose both drugs are known) + per-row pos/neg flag, so the wrapper can align the
    extracted test-query predictions (concatenated as [pos, neg]) back to the full test_df
    rows (dropped/unknown -> 0.5 neutral).

Dense entity ids (codex 019f2930): the merged-KG builder ``build_entity_index_full`` already
produces a dense contiguous entity index (drugs ``[0, n_drugs)`` first, KG entities follow),
so KnowDDI's adjacency-by-raw-id + ``shape=(len(entity2id), len(entity2id))`` is safe.

NO Morgan molecular features: KnowDDI's GraphSAGE uses a learned ``pre_embed`` table indexed
by entity id (GraphSAGE.py:43), unlike SumGNN; so this builder writes no molecular-feat pkl.

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): reuses only the
sibling ``build_knowddi_data`` helper, NOT ``build_sumgnn_binary_data`` / emergnn.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# REUSE the mc builder's dense entity index helper UNCHANGED (does not edit it).
from baseline.knowddi._data.necessary.build_knowddi_data import build_entity_index_full

_PAIR = ["drug_a_id", "drug_b_id"]


def _pos_neg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a binary leaf frame into (positives, negatives) by ``y_bin``.

    Positives = ``y_bin==1`` rows; negatives = ``y_bin==0`` rows (the FIXED materialized
    benchmark negatives). If ``y_bin`` is absent, every row is a positive (defensive)."""
    if "y_bin" in df.columns:
        pos = df[df["y_bin"] == 1][_PAIR].reset_index(drop=True)
        neg = df[df["y_bin"] == 0][_PAIR].reset_index(drop=True)
    else:
        pos = df[_PAIR].reset_index(drop=True)
        neg = df.iloc[0:0][_PAIR].reset_index(drop=True)
    return pos, neg


def build_knowddi_binary_data(
    out_dir: str | Path,
    drugs: pd.DataFrame,
    edges: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    kg_scope: str = "full",
) -> dict:
    """Write KnowDDI's on-disk BINARY layout for one leaf. Returns build stats.

    ``train/val/test_df`` carry ``drug_a_id``, ``drug_b_id`` (strings) + ``y_bin`` (1/0).
    Unknown drugs (not in ``drugs``) are skipped (logged in stats). ``kg_scope`` = "full"
    (default; whole merged KG).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drug_ids = drugs["drug_id"].astype(str).tolist()
    if kg_scope != "full":
        raise ValueError(f"KnowDDI binary supports kg_scope='full' only; got {kg_scope!r}")
    entity2id, entity_types, tdf = build_entity_index_full(drug_ids, edges)
    n_drugs = int((entity_types == 1).sum())
    n_ent = len(entity_types)

    # BKG_file.txt: head tail rel_idx (whitespace; np.loadtxt-readable). rel enumerated
    # 0-based + sorted for determinism (process_files_ddi offsets them by #DDI-rels==1).
    rel_list = sorted(tdf["rel"].unique())
    rel2id = {r_: i for i, r_ in enumerate(rel_list)}
    bkg = np.column_stack([tdf["h_idx"].to_numpy(),
                           tdf["t_idx"].to_numpy(),
                           tdf["rel"].map(rel2id).to_numpy(dtype=np.int64)])
    np.savetxt(out / "BKG_file.txt", bkg, fmt="%d")

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

    dropped: dict = {}

    # ---- train.txt = POSITIVES ONLY (drives the DDI relation-0 adjacency) --------
    train_pos, train_neg = _pos_neg(train_df)
    tr_pos_rows, tr_pos_keep = _map_pairs(train_pos)
    tr_neg_rows, tr_neg_keep = _map_pairs(train_neg)
    np.savetxt(out / "train.txt", tr_pos_rows, fmt="%d")          # adjacency source (POS only)
    np.savetxt(out / "train_pos.txt", tr_pos_rows, fmt="%d")      # pos query set (g_label 1)
    np.savetxt(out / "train_neg.txt", tr_neg_rows, fmt="%d")      # neg query set (g_label 0)
    dropped["train_pos"] = int((~tr_pos_keep).sum())
    dropped["train_neg"] = int((~tr_neg_keep).sum())

    # ---- valid / test: pos-only "<name>.txt" (so process_files_ddi can read all paths),
    #      plus explicit pos/neg query files for extraction ------------------------
    for frame, name in ((val_df, "valid"), (test_df, "test")):
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

    stats = {"n_ent": n_ent, "n_drugs": n_drugs, "n_bkg_edges": int(len(bkg)),
             "n_bkg_rels": len(rel_list),
             "n_train_pos": int(len(tr_pos_rows)), "n_train_neg": int(len(tr_neg_rows)),
             "dropped_pairs": dropped}
    (out / "_build_stats.json").write_text(json.dumps(stats, indent=2))
    with open(out / "entity2id.pkl", "wb") as f:
        pickle.dump(entity2id, f)
    return stats


__all__ = ["build_knowddi_binary_data"]

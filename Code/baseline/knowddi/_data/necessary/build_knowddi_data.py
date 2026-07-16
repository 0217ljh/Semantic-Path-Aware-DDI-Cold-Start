"""Builder: unified ``Leaf`` -> KnowDDI's on-disk file layout (DrugBank multiclass).

KnowDDI's pipeline (``utils/data_utils.process_files_ddi``) reads a fixed on-disk
schema (verified against the official source at
``Paper/Reference/Original-Code/KnowDDI/pytorch/utils/data_utils.py:4`` and
``train.py:76-80``):

  <out>/train.txt, valid.txt, test.txt   int triples ``h t r`` (whitespace;
                                         loaded via np.loadtxt). r = DDI-event id,
                                         DENSE in [0, K_train). h/t = drug entity idx.
  <out>/BKG_file.txt                     int triples ``h t r`` of the background KG
                                         (r = BKG relation id, 0-based; the pipeline
                                         offsets these by the #DDI-relations at load
                                         time via ``rel + r`` in process_files_ddi).
  <out>/entity2id.pkl                    string->int entity map (drugs first, sorted).
  <out>/train_class_to_global.json       dense-DDI-id -> GLOBAL y_cls list (wrapper
                                         scatters fc-head logits back to global axis).
  <out>/_build_stats.json                build stats (cache stamp).

This ADAPTS the SumGNN full-merged-KG builder (``Code/baseline/sumgnn/_data/
necessary/build_sumgnn_data.build_entity_index_full``) to KnowDDI's DIFFERENT file
format: SumGNN writes ``entity.txt`` + ``relations_2hop.txt`` + Morgan feats;
KnowDDI writes ``train/valid/test.txt`` + ``BKG_file.txt`` and does NOT use Morgan
molecular features (its GraphSAGE uses a learned ``pre_embed`` table indexed by
entity id, GraphSAGE.py:19/29). The ENTITY-INDEXING logic (drugs first sorted, then
KG entities; single-drug edges normalized to (drug_head, tail, rel); full merged KG
by default) is the same standing decision (2026-07-01, kg_scope="full").

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): this is
its own builder, NOT importing ``build_sumgnn_data`` or emergnn. The overlapping
entity-index logic is intentionally re-implemented here (adapted, not shared).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_PAIR = ["drug_a_id", "drug_b_id"]


def build_entity_index_full(drug_ids: list[str], edges: pd.DataFrame):
    """Assign contiguous int ids: drugs first (sorted), then all other KG entities.

    FULL merged-KG variant (keep EVERY edge, not just drug-incident). Single-drug
    edges are normalized to (drug_head, tail, rel); drug-drug and non-drug edges keep
    src->dst. Standing decision 2026-07-01: KG baselines use the FULL merged KG by
    default. Returns (entity2id, entity_types[node->1 drug/0 non-drug], triplet_df
    with int h_idx/t_idx and string rel).
    """
    drug_list = sorted({str(d) for d in drug_ids})
    drug_set = set(drug_list)
    entity2id = {d: i for i, d in enumerate(drug_list)}
    n_drugs = len(drug_list)

    edges = edges.copy()
    edges["relation"] = edges["relation"].astype(str).str.strip()
    src_in = edges["src"].isin(drug_set)
    dst_in = edges["dst"].isin(drug_set)
    flip = (~src_in) & dst_in
    h = edges["src"].where(~flip, edges["dst"])
    t = edges["dst"].where(~flip, edges["src"])
    r = edges["relation"]
    tdf = pd.DataFrame({"head": h.values, "tail": t.values, "rel": r.values})
    tdf = tdf.drop_duplicates(subset=["head", "tail", "rel"]).reset_index(drop=True)

    all_ents = set(tdf["head"]) | set(tdf["tail"])
    non_drug = sorted(all_ents - drug_set)
    for e in non_drug:
        entity2id[e] = len(entity2id)
    n_ent = len(entity2id)

    entity_types = np.zeros(n_ent, dtype=np.int64)
    entity_types[:n_drugs] = 1

    tdf["h_idx"] = tdf["head"].map(entity2id).astype(np.int64)
    tdf["t_idx"] = tdf["tail"].map(entity2id).astype(np.int64)
    return entity2id, entity_types, tdf


def build_knowddi_data(
    out_dir: str | Path,
    drugs: pd.DataFrame,
    edges: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_col: str = "y_cls",
    kg_scope: str = "full",
) -> dict:
    """Write KnowDDI's on-disk layout for one multiclass leaf. Returns build stats.

    ``train/val/test_df`` carry ``drug_a_id``,``drug_b_id`` (strings) + ``label_col``
    (int GLOBAL y_cls). Unknown drugs (not in ``drugs``) are skipped from splits
    (logged). ``kg_scope`` = "full" (default; whole merged KG). DDI-event labels are
    remapped to a DENSE ``[0, K_train)`` block (fc head size = K_train), keeping the
    dense->global list for the wrapper's scatter. Valid/test gold classes unseen in
    train (OOV-in-train) get a SENTINEL dense id == K_train (one beyond the head's
    range) so the head can never predict them and they always count wrong. BKG
    relations are enumerated 0-based (the pipeline offsets them by #DDI-relations via
    ``rel + r`` at load time).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drug_ids = drugs["drug_id"].astype(str).tolist()
    if kg_scope != "full":
        raise ValueError(f"KnowDDI Phase-1 supports kg_scope='full' only; got {kg_scope!r}")
    entity2id, entity_types, tdf = build_entity_index_full(drug_ids, edges)
    n_drugs = int((entity_types == 1).sum())
    n_ent = len(entity_types)

    # BKG_file.txt: head tail rel_idx (rel enumerated 0-based, sorted for determinism).
    rel_list = sorted(tdf["rel"].unique())
    rel2id = {r_: i for i, r_ in enumerate(rel_list)}
    bkg = np.column_stack([tdf["h_idx"].to_numpy(),
                           tdf["t_idx"].to_numpy(),
                           tdf["rel"].map(rel2id).to_numpy(dtype=np.int64)])
    np.savetxt(out / "BKG_file.txt", bkg, fmt="%d")

    # Contiguous train DDI-event vocab. process_files_ddi builds relation2id from the
    # r values it sees in train.txt and the fc head outputs one logit per DDI relation
    # (id 0..K_train-1), so labels MUST be a contiguous [0, K_train) block. Leaf y_cls
    # are GLOBAL (non-contiguous on a subsample) -> remap train-observed classes to
    # dense ids; keep dense->global for the wrapper's scatter.
    #
    # OOV-in-train gold handling (EmerGNN/SumGNN convention): a valid/test gold class
    # that never appears in train has NO dense head slot. Mapping it to dense 0 would
    # give false credit whenever the head predicts class 0 and pollutes the internal
    # VAL macro-F1 that drives best-ckpt selection. Instead map OOV gold to a SENTINEL
    # dense id == K_train (== len(train_globals), i.e. ONE BEYOND the head's output
    # range 0..K_train-1): the fc head (width K_train) can NEVER emit it, so such rows
    # are always counted wrong. Applied to VALID and TEST only; train.txt is unchanged
    # because train only ever carries in-vocab classes (they define the vocab).
    train_globals = sorted(pd.unique(train_df[label_col].to_numpy()).tolist())
    global_to_dense = {int(g): i for i, g in enumerate(train_globals)}
    train_class_to_global = [int(g) for g in train_globals]
    oov_sentinel = len(train_globals)  # == K_train, one beyond head range 0..K_train-1

    def _write_split(df: pd.DataFrame, name: str) -> int:
        a = df["drug_a_id"].astype(str).map(entity2id)
        b = df["drug_b_id"].astype(str).map(entity2id)
        keep = a.notna() & b.notna()
        # train is guaranteed in-vocab; valid/test OOV gold -> sentinel (always wrong).
        default_dense = 0 if name == "train" else oov_sentinel
        dense = df.loc[keep, label_col].map(
            lambda g: global_to_dense.get(int(g), default_dense))
        rows = np.column_stack([a[keep].to_numpy(dtype=np.int64),
                                b[keep].to_numpy(dtype=np.int64),
                                dense.to_numpy(dtype=np.int64)])
        np.savetxt(out / f"{name}.txt", rows, fmt="%d")
        return int((~keep).sum())

    dropped = {
        "train": _write_split(train_df, "train"),
        "valid": _write_split(val_df, "valid"),
        "test": _write_split(test_df, "test"),
    }
    (out / "train_class_to_global.json").write_text(json.dumps(train_class_to_global))

    stats = {"n_ent": n_ent, "n_drugs": n_drugs, "n_bkg_edges": int(len(bkg)),
             "n_bkg_rels": len(rel_list), "n_train_ddi_rels": len(train_globals),
             "oov_sentinel_dense_id": oov_sentinel, "dropped_pairs": dropped}
    (out / "_build_stats.json").write_text(json.dumps(stats, indent=2))
    with open(out / "entity2id.pkl", "wb") as f:
        pickle.dump(entity2id, f)
    return stats


__all__ = ["build_knowddi_data", "build_entity_index_full"]

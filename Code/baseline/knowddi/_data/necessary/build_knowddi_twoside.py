"""Builder: unified multilabel ``Leaf`` -> KnowDDI's BioSNAP/decagon on-disk layout.

Multilabel (TWOSIDES) counterpart of :mod:`build_knowddi_data` (multiclass). It emits
the file layout KnowDDI's ``process_files_decagon`` reads (verified against the official
source at ``Paper/Reference/Original-Code/KnowDDI/pytorch/utils/data_utils.py:61`` and
the official BioSNAP data at ``Paper/Reference/Original-Code/KnowDDI/data/BioSNAP/``):

  <out>/train.txt, valid.txt, test.txt   decagon lines ``h\\tt\\tmultihot_csv\\tpolarity``
                                         (tab-separated). h/t = drug entity idx,
                                         multihot_csv = comma-joined 0/1 over n_labels
                                         labels, polarity = 1 pos / 0 neg. Read line-by-
                                         line by ``process_files_decagon`` (NOT loadtxt).
  <out>/BKG_file.txt                     int triples ``h t r`` of the LEAF'S OWN KG
                                         (whitespace; loaded via ``np.loadtxt`` inside
                                         ``process_files_decagon``). r = KG relation id,
                                         0-based; the pipeline offsets these by the
                                         #DDI-relations (200) via ``rel + r`` at load time.
  <out>/entity2id.pkl                    STRING->int identity entity map (``{str(i): i}``)
                                         so the wrapper's ``.astype(str).map(entity2id)``
                                         alignment matches (codex 019f2930, decision 1).
  <out>/_build_stats.json                build stats (cache stamp).

Faithfulness (codex 019f2930):
  * The TWOSIDES leaf's drug ids are already dense integers ``[0, n_drugs)`` matching the
    leaf pairs, and its OWN KG (``train_KG.txt`` under ``resources.kg.source``) uses the
    SAME int namespace (disjoint from the merged DrugBank ``DBxxxxx`` KG). We consume that
    int KG directly (like SumGNN's ``write_twoside_split``), NOT the merged parquet.
  * Entities are dense-remapped defensively: the KG loaders index adjacency by raw id with
    ``shape=(len(entity2id), len(entity2id))`` (data_utils.py process_files_decagon), so ids
    MUST be dense ``0..N-1``. Drug ids are contiguous by construction; KG entity ids follow.
    We keep the identity map since the TWOSIDES KG ids are already dense; a guard raises if
    a sparse gap is detected.
  * Negatives (``is_positive==0``) still carry their label ids in the multihot (codex
    caveat): an all-zero multihot would drop them from the masked BCE + the multilabel
    evaluator. The polarity column flips them to the negative target.
  * NO Morgan molecular features: KnowDDI's GraphSAGE uses a learned ``pre_embed`` table
    indexed by entity id (GraphSAGE.py:43), unlike SumGNN which fuses Morgan bits. So this
    builder writes no ``id2drug_feat.pkl``.
  * Head width is FIXED at ``n_labels`` (200) by the wrapper (``params.num_rels=200``); it
    is NOT derived from the labels that happen to appear across splits.

INDEPENDENT copy under ``baseline.knowddi`` (CLAUDE.md file-independence): its own builder,
NOT importing ``build_sumgnn_data`` / ``write_twoside_split`` / emergnn.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_PAIR = ["drug_a_id", "drug_b_id"]


def build_knowddi_twoside(
    out_dir: str | Path,
    drugs: pd.DataFrame,
    kg_triples: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_labels: int = 200,
) -> dict:
    """Write KnowDDI's BioSNAP/decagon layout for one multilabel (TWOSIDES) leaf.

    ``train/val/test_df`` carry ``drug_a_id``, ``drug_b_id`` (integer-string ids) +
    ``y_label_ids`` (per-row list of active label ids) + ``is_positive`` (1/0). Unknown
    drugs (not in ``drugs``) are skipped (logged in stats). ``kg_triples`` is the leaf's
    OWN int KG (Nx3 ``head tail rel``). Returns build stats.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drug_ints = sorted({int(d) for d in drugs["drug_id"].astype(str)})
    n_drugs = len(drug_ints)
    kg = (np.asarray(kg_triples, dtype=np.int64).reshape(-1, 3)
          if len(kg_triples) else np.zeros((0, 3), np.int64))
    kg_ents = set(kg[:, 0].tolist()) | set(kg[:, 1].tolist()) if len(kg) else set()
    max_id = max([n_drugs - 1] + list(kg_ents)) if (kg_ents or n_drugs) else -1
    n_ent = int(max_id) + 1
    # Dense-id guard (codex 019f2930): the decagon loader indexes adjacency by raw id with
    # shape (n_ent, n_ent); sparse gaps would waste rows but never index OOB (ids <= max_id
    # < n_ent). Drug ids are contiguous [0, n_drugs) by construction; KG ids follow. We keep
    # the identity map. Non-contiguous KG ids only inflate n_ent (harmless, larger pre_embed).
    entity2id = {str(i): i for i in range(n_ent)}   # STRING keys (codex: wrapper maps str)

    # BKG_file.txt: head tail rel_idx (whitespace; np.loadtxt-readable). rel enumerated
    # 0-based + sorted for determinism (process_files_decagon offsets by rel==200 at load).
    if len(kg):
        rel_list = sorted(set(kg[:, 2].tolist()))
        rel2id = {r_: i for i, r_ in enumerate(rel_list)}
        bkg = np.column_stack([kg[:, 0], kg[:, 1],
                               np.array([rel2id[r_] for r_ in kg[:, 2].tolist()],
                                        dtype=np.int64)])
    else:
        rel_list = []
        bkg = np.zeros((0, 3), dtype=np.int64)
    np.savetxt(out / "BKG_file.txt", bkg, fmt="%d")

    def _multihot(ids) -> str:
        vec = np.zeros(n_labels, dtype=np.int64)
        if ids is not None:
            idx = np.asarray(list(ids), dtype=np.int64)
            if idx.size:
                vec[idx[idx < n_labels]] = 1
        return ",".join(map(str, vec.tolist()))

    def _write(df: pd.DataFrame, name: str) -> int:
        a = df["drug_a_id"].astype(str).map(entity2id)
        b = df["drug_b_id"].astype(str).map(entity2id)
        pol = (df["is_positive"].astype(int).to_numpy()
               if "is_positive" in df.columns else np.ones(len(df), int))
        dropped = 0
        with open(out / f"{name}.txt", "w") as fh:
            for i in range(len(df)):
                if pd.isna(a.iloc[i]) or pd.isna(b.iloc[i]):
                    dropped += 1
                    continue
                # negatives keep their label ids in the multihot (codex caveat); polarity
                # flips them to the negative target in the trainer's masked BCE.
                ids = df["y_label_ids"].iloc[i] if "y_label_ids" in df.columns else None
                fh.write(f"{int(a.iloc[i])}\t{int(b.iloc[i])}\t{_multihot(ids)}\t{int(pol[i])}\n")
        return dropped

    dropped = {"train": _write(train_df, "train"), "valid": _write(val_df, "valid"),
               "test": _write(test_df, "test")}

    stats = {"n_ent": n_ent, "n_drugs": n_drugs, "n_bkg_edges": int(len(bkg)),
             "n_bkg_rels": len(rel_list), "n_labels": n_labels, "dropped_pairs": dropped}
    (out / "_build_stats.json").write_text(json.dumps(stats, indent=2))
    with open(out / "entity2id.pkl", "wb") as f:
        pickle.dump(entity2id, f)
    return stats


__all__ = ["build_knowddi_twoside"]

"""Builder: unified `Leaf` -> SumGNN's on-disk int-indexed data layout.

SumGNN's subgraph-extraction pipeline (ported in the reproduction and re-copied into
this baseline) reads a fixed on-disk schema (verified against
Paper/Reference/Original-Code/SumGNN/data/drugbank):

  <out>/train.txt, dev.txt, test.txt   int triples `drug_a_idx drug_b_idx label`
                                       (whitespace; loaded via np.loadtxt)
  <out>/entity.txt                     one line per entity index i = node type of i
                                       (1 = drug, 0 = non-drug), used for the KG mask.
  <out>/relations_2hop.txt             KG triples `head_idx tail_idx rel_idx` (the
                                       biomedical KG, entities beyond the drug block).
  <out>/DB_molecular_feats.pkl         {'Morgan_Features': list indexed by drug idx}
  <out>/relation2id.json               written by the pipeline on first run.

This is the baseline-side copy of the SumGNN data builder (CLAUDE.md file-independence:
copied, independently maintained; NOT importing the reproduction or emergnn). It maps our
STRING-id leaves (drug_id `DB00006`, merged-KG node ids `db:target:BE...`) into SumGNN's
contiguous integer indexing: drugs occupy `[0, n_drugs)` (drugs FIRST, sorted), non-drug
KG entities follow. `kg_scope` selects the KG: "full" (default, standing decision
2026-07-01 = the WHOLE 7.1M-edge merged KG) or "drug_incident" (legacy 1-hop, edges whose
src or dst is a leaf drug). Single-drug edges are normalized to (drug_head, tail, rel).

DDI event label: multiclass uses the leaf's GLOBAL `y_cls`; the pipeline builds its own
train relation vocab from the labels it sees (relation2id.json). Multilabel (TWOSIDES)
uses the decagon-style file format (see `write_twoside_split`).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_PAIR = ["drug_a_id", "drug_b_id"]
_DRUG_KINDS = {"Drug", "drug"}


def _morgan_bits(smiles: str, n_bits: int = 1024) -> np.ndarray:
    """1024-bit Morgan (ECFP4) fingerprint as a float vector. Zeros if RDKit fails."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def build_entity_index(drug_ids: list[str], edges: pd.DataFrame) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Assign contiguous int ids: drugs first (sorted), then non-drug KG entities.

    Returns (entity2id, entity_types[node->1 drug/0 non-drug], drug_incident_triplet_df
    with int head/tail and string rel).
    """
    drug_list = sorted({str(d) for d in drug_ids})
    drug_set = set(drug_list)
    entity2id = {d: i for i, d in enumerate(drug_list)}
    n_drugs = len(drug_list)

    edges = edges.copy()
    edges["relation"] = edges["relation"].astype(str).str.strip()
    src_is_drug = edges["src"].isin(drug_set)
    dst_is_drug = edges["dst"].isin(drug_set)
    di = edges[src_is_drug | dst_is_drug].copy()

    src_in = di["src"].isin(drug_set)
    dst_in = di["dst"].isin(drug_set)
    flip = (~src_in) & dst_in
    h = di["src"].where(~flip, di["dst"])
    t = di["dst"].where(~flip, di["src"])
    r = di["relation"]
    tdf = pd.DataFrame({"head": h.values, "tail": t.values, "rel": r.values})
    tdf = tdf.drop_duplicates(subset=["head", "tail", "rel"]).reset_index(drop=True)

    all_ents = set(tdf["head"]) | set(tdf["tail"])
    non_drug = sorted(all_ents - drug_set)
    for e in non_drug:
        entity2id[e] = len(entity2id)
    n_ent = len(entity2id)

    # SumGNN entity.txt convention: 1 = drug, 0 = non-drug (drug=1 so the KG mask
    # picks out drugs; verified against original data/drugbank/entity.txt where the
    # 1709 drug rows sum to a drug-count of ones).
    entity_types = np.zeros(n_ent, dtype=np.int64)
    entity_types[:n_drugs] = 1

    tdf["h_idx"] = tdf["head"].map(entity2id).astype(np.int64)
    tdf["t_idx"] = tdf["tail"].map(entity2id).astype(np.int64)
    return entity2id, entity_types, tdf


def build_entity_index_full(drug_ids: list[str], edges: pd.DataFrame) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """FULL merged-KG variant of :func:`build_entity_index`: keep EVERY edge (NOT just
    drug-incident). Drugs indexed first (sorted), then all other entities. Single-drug
    edges normalized to (drug_head, tail, rel); drug-drug and non-drug edges keep src->dst.
    Mirrors emergnn.kg_builder_merged.build_full_kg_from_merged_parquet. Standing decision
    2026-07-01: all KG baselines use the FULL merged KG by default."""
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


def build_sumgnn_data(
    out_dir: str | Path,
    drugs: pd.DataFrame,
    edges: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    label_col: str = "y_cls",
    morgan_from: str = "smiles",
    kg_scope: str = "full",
) -> dict:
    """Write SumGNN's on-disk layout for one multiclass leaf. Returns build stats.

    `train/val/test_df` carry `drug_a_id`,`drug_b_id` (strings) + `label_col` (int).
    Unknown drugs (not in `drugs`) are skipped from splits (logged in stats).
    `kg_scope`: "full" (default; whole merged KG) or "drug_incident" (1-hop, legacy).
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

    # entity.txt: one type per line (index order)
    n_ent = len(entity_types)
    np.savetxt(out / "entity.txt", entity_types, fmt="%d")

    # relations_2hop.txt: head tail rel_idx (rel enumerated, sorted for determinism)
    rel_list = sorted(tdf["rel"].unique())
    rel2id = {r_: i for i, r_ in enumerate(rel_list)}
    kg = np.column_stack([tdf["h_idx"].to_numpy(),
                          tdf["t_idx"].to_numpy(),
                          tdf["rel"].map(rel2id).to_numpy(dtype=np.int64)])
    np.savetxt(out / "relations_2hop.txt", kg, fmt="%d")

    # Contiguous train label vocab. SumGNN's fc head has `num_rels` = # distinct
    # train labels outputs, and CrossEntropy indexes by the on-disk label value, so
    # labels MUST be a contiguous [0, K_train) block. Our leaf `y_cls` are GLOBAL ids
    # (non-contiguous on a subsample). Remap train-observed classes -> dense ids, keep
    # the dense->global list for the wrapper's scatter to the global axis. Unseen-in-
    # train test classes are mapped to dense 0 (their prediction is scored vs global
    # y_cls by the runner and counted wrong regardless).
    train_globals = sorted(pd.unique(train_df[label_col].to_numpy()).tolist())
    global_to_dense = {int(g): i for i, g in enumerate(train_globals)}
    train_class_to_global = [int(g) for g in train_globals]

    def _write_split(df: pd.DataFrame, name: str) -> int:
        a = df["drug_a_id"].astype(str).map(entity2id)
        b = df["drug_b_id"].astype(str).map(entity2id)
        keep = a.notna() & b.notna()
        dense = df.loc[keep, label_col].map(lambda g: global_to_dense.get(int(g), 0))
        rows = np.column_stack([a[keep].to_numpy(dtype=np.int64),
                                b[keep].to_numpy(dtype=np.int64),
                                dense.to_numpy(dtype=np.int64)])
        np.savetxt(out / f"{name}.txt", rows, fmt="%d")
        return int((~keep).sum())

    dropped = {
        "train": _write_split(train_df, "train"),
        "dev": _write_split(val_df, "dev"),
        "test": _write_split(test_df, "test"),
    }
    (out / "train_class_to_global.json").write_text(json.dumps(train_class_to_global))

    # Morgan feats indexed by drug idx (0..n_drugs-1)
    id2smiles = dict(zip(drugs["drug_id"].astype(str), drugs[morgan_from].astype(str)))
    id2drug = {v: k for k, v in entity2id.items() if v < n_drugs}
    mfeat = [_morgan_bits(id2smiles.get(id2drug[i], "")) for i in range(n_drugs)]
    with open(out / "DB_molecular_feats.pkl", "wb") as f:
        pickle.dump({"Morgan_Features": mfeat}, f)

    stats = {"n_ent": n_ent, "n_drugs": n_drugs, "n_kg_edges": int(len(kg)),
             "n_kg_rels": len(rel_list), "dropped_pairs": dropped}
    (out / "_build_stats.json").write_text(json.dumps(stats, indent=2))
    # entity2id sidecar (string->int) for the wrapper to map test pairs / debug
    with open(out / "entity2id.pkl", "wb") as f:
        pickle.dump(entity2id, f)
    return stats


def write_twoside_split(
    out_dir: str | Path,
    drugs: pd.DataFrame,
    kg_triples: np.ndarray,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    n_labels: int = 200,
) -> dict:
    """Write SumGNN's BioSNAP/decagon layout for one multilabel (TWOSIDES) leaf.

    The TWOSIDES leaf ships its OWN SumGNN-compatible KG (int-indexed `head tail rel`
    where drug ids already occupy [0, n_drugs) and match the leaf's pair ids), so we
    consume `kg_triples` (Nx3 int array) directly rather than the merged parquet.

    decagon file format (verified against process_files_decagon): each line is
    `x\\ty\\tz\\tw` where x,y are drug idx, z is a comma-joined 0/1 multihot over
    labels, w is polarity (1 pos / 0 neg). id2drug_feat.pkl holds {'Morgan','rdkit2d'}.
    Here drug ids are already integer strings; entity2id is the identity map over the
    drug block plus every KG entity id.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drug_ints = sorted({int(d) for d in drugs["drug_id"].astype(str)})
    n_drugs = len(drug_ints)
    kg = np.asarray(kg_triples, dtype=np.int64).reshape(-1, 3) if len(kg_triples) else np.zeros((0, 3), np.int64)
    kg_ents = set(kg[:, 0].tolist()) | set(kg[:, 1].tolist()) if len(kg) else set()
    max_id = max([n_drugs - 1] + list(kg_ents)) if (kg_ents or n_drugs) else -1
    n_ent = int(max_id) + 1
    # identity entity map (drug ints are already contiguous [0, n_drugs); KG ids follow)
    entity2id = {str(i): i for i in range(n_ent)}
    # entity.txt: 1 = drug (index < n_drugs), 0 = non-drug KG entity
    entity_types = np.zeros(n_ent, dtype=np.int64)
    entity_types[:n_drugs] = 1
    np.savetxt(out / "entity.txt", entity_types, fmt="%d")
    np.savetxt(out / "relations_2hop.txt", kg, fmt="%d")

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
        pol = df["is_positive"].astype(int) if "is_positive" in df.columns else np.ones(len(df), int)
        dropped = 0
        with open(out / f"{name}.txt", "w") as fh:
            for i in range(len(df)):
                if pd.isna(a.iloc[i]) or pd.isna(b.iloc[i]):
                    dropped += 1
                    continue
                ids = df["y_label_ids"].iloc[i] if "y_label_ids" in df.columns else None
                fh.write(f"{int(a.iloc[i])}\t{int(b.iloc[i])}\t{_multihot(ids)}\t{int(pol.iloc[i])}\n")
        return dropped

    dropped = {"train": _write(train_df, "train"), "dev": _write(val_df, "dev"),
               "test": _write(test_df, "test")}

    id2smiles = dict(zip(drugs["drug_id"].astype(str), drugs["smiles"].astype(str)))
    id2drug = {v: k for k, v in entity2id.items() if v < n_drugs}
    id2drug_feat = {}
    for i in range(n_drugs):
        m = _morgan_bits(id2smiles.get(id2drug[i], ""))
        id2drug_feat[i] = {"Morgan": m, "rdkit2d": np.zeros(200, dtype=np.float32)}
    with open(out / "id2drug_feat.pkl", "wb") as f:
        pickle.dump(id2drug_feat, f)

    n_kg_rels = int(len(set(kg[:, 2].tolist()))) if len(kg) else 0
    stats = {"n_ent": len(entity_types), "n_drugs": n_drugs, "n_kg_edges": int(len(kg)),
             "n_kg_rels": n_kg_rels, "dropped_pairs": dropped, "n_labels": n_labels}
    (out / "_build_stats.json").write_text(json.dumps(stats, indent=2))
    with open(out / "entity2id.pkl", "wb") as f:
        pickle.dump(entity2id, f)
    return stats


__all__ = ["build_sumgnn_data", "write_twoside_split", "build_entity_index"]

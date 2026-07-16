"""Molecular modality m_u — frozen Morgan fingerprint per drug (InfoNCE alignment source).

Pairs with precompute_kg_neighbor_target.py (which builds the alignment TARGET k_u).
m_u is the drug's molecular-structure embedding; the InfoNCE alignment learns proj_m(m_u)
to match proj_k(k_u) on train-graph drugs only (edge-independent, no DDI labels).

Frozen Morgan first (radius 2, 2048 bits) per the v3 plan; ChemBERTa is an optional later
swap. Deterministic — no training, no GPU needed. Drug IDs are DrugBank IDs, shared with
the k_u cache (kg_neighbor_target_pubmedbert.npz) and the 800-drug SMILES csv.

Output: Code/data/_cache/molecular_mu_morgan.npz
  drug_ids (list[str]), m (N_drug x 2048 float32), valid (bool, SMILES parsed ok)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]

SMILES_CSV = PROJECT_ROOT / "Code/data/coldddi_legacy/800drug/drug_smiles__seed42.csv"
OUT = PROJECT_ROOT / "Code/data/_cache/molecular_mu_morgan.npz"

FP_RADIUS = 2
FP_BITS = 2048


def _resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    smiles_col = next((c for c in cols if str(c).lower() == "smiles"), None)
    if smiles_col is None:
        raise ValueError(f"no 'smiles' column found in {SMILES_CSV} (cols={cols})")
    id_candidates = [c for c in cols if c != smiles_col]
    id_col = next(
        (c for c in id_candidates if str(c).lower() in ("drug_id", "id", "drugbank_id", "drug")),
        id_candidates[0] if id_candidates else None,
    )
    if id_col is None:
        raise ValueError(f"no drug id column found in {SMILES_CSV} (cols={cols})")
    return id_col, smiles_col


def main() -> None:
    print(f"[m_u] loading SMILES from {SMILES_CSV} ...", flush=True)
    df = pd.read_csv(SMILES_CSV)
    id_col, smiles_col = _resolve_columns(df)
    print(f"[m_u] id_col={id_col!r} smiles_col={smiles_col!r} rows={len(df)}", flush=True)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

    drug_ids: list[str] = []
    rows: list[np.ndarray] = []
    valid: list[bool] = []
    n_fail = 0
    for did, smi in zip(df[id_col].astype(str), df[smiles_col].astype(str)):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if mol is None:
            n_fail += 1
            drug_ids.append(did)
            rows.append(np.zeros(FP_BITS, dtype=np.float32))
            valid.append(False)
            continue
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        drug_ids.append(did)
        rows.append(fp)
        valid.append(True)

    m = np.vstack(rows).astype(np.float32)
    valid_arr = np.array(valid, dtype=bool)
    print(
        f"[m_u] drugs={len(drug_ids)}, parsed_ok={int(valid_arr.sum())}, "
        f"failed={n_fail}, m shape={m.shape}",
        flush=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, drug_ids=np.array(drug_ids), m=m, valid=valid_arr)
    print(f"[m_u] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

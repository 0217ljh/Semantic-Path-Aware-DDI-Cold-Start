# MKG-FENN Migration Notes

## Source
`D:\My-Research\03-Projects\ColdDDI\Code-Released-Formal\coldddi\baselines\mkg_fenn`

## Migrated
- `2026-06-08` — files copied verbatim from source. No code adaptation yet.

## Files

| File | Source bytes | Purpose |
|---|---|---|
| `baseline.py` | 14284 | `MKGFENNBaseline` class, `fit() / predict_proba() / save() / load()` |
| `model.py` | 13720 | PyTorch model definitions (4 parallel KG-GNN channels) |
| `kg_builder.py` | 7828 | Build 4 parallel KGs (drug-entity, Morgan-FP substructure, drug-DDI, drug-property) |
| `_legacy/` | — | older variants (modeltask1.py, train_custom_bundle.py) |

## Required adaptation when activating

1. **Import paths**:
   Source uses `coldddi.baselines.mkg_fenn.*` and `coldddi.baselines.base` and `coldddi.data.dataset.PairDataset`.
   Destination needs (likely): `baseline.mkg_fenn.*` and `baseline.base` and project's own PairDataset.

2. **Data loader contract**:
   Source expects `PairDataset.drugs` with `smiles` column + `PairDataset.kg` with attributes `enzymes / targets / transporters / carriers / pathways` (DataFrames mapping drugs to entities).
   Destination's KG data is in `Code/data/KG/` and `Code/data/_cache/`. Need adapter layer.

3. **Training data format**:
   Source expects `splits.train` DataFrame with `drug_a_id, drug_b_id` columns.
   Destination splits are in `Code/data/KG/drugbank/splits/seed42/` etc. as parquet. Need adapter.

4. **Baseline contract**:
   Destination has `Code/baseline/base.py` (already there). Verify it matches source `coldddi.baselines.base` ABC. Likely yes (they're sister projects).

## Why kept here (paper context)

This baseline is the **target** of the multi-modal misalignment argument in the v1.6.x AAAI 2027 paper. We will rerun MKG-FENN on the project's cold-start S2 split and compare to our hyperedge + fragment alignment method.

- Expected: MKG-FENN AUC ≈ 0.5 on cold-start S2 (per prior ColdDDI experiments)
- Paper argument: this failure is because MKG-FENN aligns at the **drug-identity level**, which collapses under cold-start. Our framework aligns at the **mechanism level** (fragment ↔ hyperedge), which transfers.

See `Notes/Log/v1_6x_paper_frame.md` Part 2 for the full argument.

## Status

| Item | Status |
|---|---|
| Files copied | ✅ |
| Import adaptation | ⏸ pending (do when activating for experiment) |
| Data adapter | ⏸ pending |
| Training contract verify | ⏸ pending |
| Re-run on cold-start S2 (project KG) | ⏸ pending |

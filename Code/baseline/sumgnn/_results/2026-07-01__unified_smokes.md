# SumGNN unified wrappers — smoke results

**Date**: 2026-07-01
**Scope**: 1-epoch SMOKE of each unified wrapper on a SUBSAMPLED leaf (NOT a metric gate).
Confirms fit runs, predict shape is correct, output is non-degenerate.

## Multiclass (`SumGNNUnifiedMulticlass`, task=multiclass)
- Leaf: `drugbank_latest_partial` (group `ddi800`) multiclass transductive/S0 fold0,
  subsampled to train=300 / val=75 / test=120, merged KG (drug-incident restriction).
- Build stats: 800 drugs -> entity block [0,800); merged KG 7.1M edges -> **128,042
  drug-incident** edges, 18 relations, 15,412 entities; 0 dropped pairs.
- **PASS**: `predict shape=(120, 165)` = (n, K_global) as expected; 120/120 rows
  non-degenerate (mass>0); scores in [0, 0.039].
- Cmd: `python Code/baseline/sumgnn/_smoke_unified.py --task multiclass --n 300 --test-n 120 --epochs 1`

## Multilabel (`SumGNNUnifiedMultilabel`, task=multilabel)
- Leaf: `twoside` multilabel transductive/S0 fold0, subsampled train=300 / val=75 / test=120.
- KG: the TWOSIDES leaf ships its OWN SumGNN-format int-indexed KG
  (`<kg.source>/S0/train_KG.txt`, drug ids already [0, n_drugs) matching leaf pairs) — this
  is used, NOT the merged DrugBank KG (disjoint drug namespace). `number of relations:200`.
- **PASS**: `predict shape=(120, 200)` = (n, n_labels) as expected; 120/120 rows
  non-degenerate; sigmoid scores in [0.437, 0.559] (near 0.5 after 1 untrained epoch).
- Cmd: `python Code/baseline/sumgnn/_smoke_unified.py --task multilabel --n 300 --test-n 120 --epochs 1`

## Data-mapping design (the riskiest part — resolved, no Hetionet-specific preprocessing needed)
- **Multiclass**: our merged KG (`edges__drugbank_hetionet_primekg__mask1.parquet`, string
  node ids like `DB00006` / `db:target:BE...`) is mapped to SumGNN's int layout by
  `build_sumgnn_data`: drugs indexed first `[0, n_drugs)` (marked type 1 in `entity.txt`),
  KG restricted to DRUG-INCIDENT edges normalized to (drug_head, tail, rel) — the same
  restriction EmerGNN's merged builder uses — then re-indexed contiguously. Labels (`y_cls`,
  global) are remapped to a contiguous `[0, K_train)` train vocab (SumGNN's CE head indexes
  by on-disk label value), with a dense->global map for scattering `predict` to the global
  axis. No Hetionet-specific artifacts are required; the merged KG runs directly through
  SumGNN's subgraph BFS.
- **Multilabel (TWOSIDES)**: uses the leaf's own int-indexed KG (already SumGNN-format), so
  no id remapping is needed. `write_twoside_split` emits the decagon `a\tb\tmultihot\tpolarity`
  format + `id2drug_feat.pkl`.

## Caveat flagged for codex review
- **`process_files_decagon` `keeptrainone` default** was flipped `True`->`False` in the
  baseline `_core` copy so the 200-multihot flows to the BCE loss (with `keeptrainone=True`
  the per-link label collapses to scalar 0 and BCE gets a (batch,batch) target — the original
  file's default is self-inconsistent for the BCE branch; the working path is
  `keeptrainone=False`). Confirmed the multilabel smoke passes with this. Worth a second look
  that this matches the paper's TWOSIDES training config exactly.
- **`process_files_ddi/decagon` adjacency dim** was sized by `max_id+1` instead of
  `len(entity2id)` (our leaf-derived entity ids can be non-dense; original drugbank data was
  dense so `len==max+1`). Semantically a no-op on dense data.

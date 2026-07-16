# EmerGNN binary — unified-benchmark wrapper review

- Date: 2026-06-30
- Primary reviewer: Claude (opus-4.8)
- Independent reviewer: codex (gpt-5-codex), thread 019f19fb
- Trigger: migrate EmerGNN to the new ddi_unified benchmark as the first UnifiedBaseline.

## Scope
New files (legacy baseline + reviewed core `_per_mode.py` UNCHANGED):
- `_unified_adapter.py` — adapts a unified `Leaf` to the minimal PairDataset surface
  `_PerModeEmerGNN` reads.
- `binary_cls/baseline_unified.py` — `EmerGNNUnifiedBinary(UnifiedBaseline)`, registered
  "emergnn".
- `Code/scripts/run_baseline_unified.py` — leaf runner.
- Shared infra: `Code/data_utils/unified_loader.py`, `Code/baseline/unified_base.py`.

## Design (decision B, codex 019f19fb)
- One leaf = one regime → ONE `_PerModeEmerGNN`, `shuffle_train_mode = leaf.split_code`
  (S0/S1/S2). Cross-regime G1/G2 routing DROPPED (obsolete — regime fixed per leaf).
- Adapter exposes: splits.train (POS, y_bin==1), splits.val_s2 (val POS), splits.items()
  (train+val pair frames + an all-drugs frame from drugs.parquet so the entity pool covers
  unseen test drugs, since fit() never sees test_df), drugs[[drugbank_id,smiles]] keyed on
  drug_id, get_train_negatives()/get_negatives("val_s2") → leaf's MATERIALIZED y_bin==0 rows.
- Negatives: benchmark-fixed (materialized). `_per_mode` sub-samples them per epoch; NO
  fresh re-sampling. **Caveat (codex): this is NOT the literal paper per-epoch negative
  generation — the benchmark fixes negatives upstream. Faithful in model/core behavior,
  not in negative-gen protocol. Documented here per codex.**
- Per-regime feat bundles kept (paper dispatch): S0→feat=E/batch128/wd1e-6; S1,S2→feat=M/
  batch32/wd1e-8.
- KG: backbone_kg_source="merged", merged_kg_path = <leaf kg.source>/edges__drugbank_
  hetionet_primekg__mask1.parquet. KG-free leaves (deng/ryu) raise → out of scope.

## codex verdict (round: wiring review)
"No blocking issues found." Confirmed: (1) same `ds` as train+val is correct (disjoint
fields, no aliasing); (2) fixed negatives = acceptable benchmark adaptation, document it;
(3) all_drugs injection safe (vocab union only, not treated as DDI edges); (4) keep
per-regime feat as-is for faithfulness.

## Verification
Non-training wiring smoke (PASS): registry resolves; merged-KG edges path exists; adapter
train_pos/val_pos/train_neg/val_neg = 35351/4090/35351/4090 (1:1); g1(seen)=480 g2(held)=320;
entity pool covers ALL test drugs; drugs cols=[drugbank_id,smiles], smiles non-null=1.0.
End-to-end training run: PENDING (user runs the single-fold command).

## Reproduce
python Code/scripts/run_baseline_unified.py --baseline emergnn --task binary \
    --dataset ddi800 --split cold_s2 --fold fold0 --epochs 3   # quick verify (partial)

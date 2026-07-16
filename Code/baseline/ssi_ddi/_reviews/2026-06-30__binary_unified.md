# SSI-DDI binary — unified-benchmark wrapper review

- Date: 2026-06-30
- Primary reviewer: Claude (opus-4.8)
- Independent reviewer: codex (gpt-5-codex), thread 019f1a4f
- Trigger: migrate SSI-DDI to the new ddi_unified benchmark (2nd baseline, molecular template).

## Scope
- `baseline/ssi_ddi/binary_cls/baseline_unified.py` — `SSIDDIUnifiedBinary(UnifiedBaseline)`,
  registered "ssi_ddi", reuses paper-faithful core `SSIDDIBaseline` (baseline/ssi_ddi/baseline.py)
  UNCHANGED.
- `Code/data_utils/leaf_adapter.py` — SHARED leaf->PairDataset adapter (generalized from
  EmerGNN's; EmerGNN's `_unified_adapter.py` now re-exports it; EmerGNN verified still works).

## Design
- SSI-DDI is molecular (substructure GAT + co-attention RESCAL), KG-FREE → applies to all
  datasets. Reads the same legacy PairDataset surface as EmerGNN minus the KG, so the shared
  adapter serves it directly (kg=None unused).
- Negatives = leaf's materialized y_bin==0 rows (benchmark-fixed).

## codex verdict (thread 019f1a4f)
- Shared adapter sound IF treated as a minimal compatibility shim; keep per-baseline contract
  (fail-hard on missing surface reads, conformance test per baseline/regime); "extend as new
  reads appear" is fine with that discipline.
- (multiclass guidance, for later) feed y_cls global, core re-vocabs from train; OOV-in-train
  eval rows = WRONG not dropped; report oov_target_rate.
- SSI 0.558@1ep tiny smoke = sane wiring signal (trains, non-degenerate, both classes present).

## Verification
CPU smoke (ddi800 cold_s2 fold0, subsample train 400 [205 neg/195 pos] / test 200 [87 neg/
113 pos], 1 epoch): molecular graphs built; predict shape (200,) non-degenerate (min 0.407
max 0.534 std 0.030); AUROC 0.558; asserted binary negatives present + non-constant. PASS.
Full single-fold training run: PENDING (user runs).

## Reproduce (full)
python Code/scripts/run_baseline_unified.py --baseline ssi_ddi --task binary \
    --dataset ddi_full --split cold_s2 --fold fold0 --epochs 5

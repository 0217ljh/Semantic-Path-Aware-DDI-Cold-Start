# Per-epoch deterministic train negatives (design B) — unified BINARY benchmark

- **Date**: 2026-07-01
- **Primary**: Claude (opus-4.8) · **Independent**: codex (gpt-5.2-codex, threads 019f1f62 spec, 019f1f75 round1 CHANGES_REQUIRED, 019f1f7b round2 APPROVED)
- **Trigger**: user — restore per-epoch negative resampling that the unified leaf
  adapter had frozen (every epoch identical), while keeping cross-baseline
  comparability + "reconstructable at any time".

## Problem
Unified leaf adapter `_LeafDataset.get_train_negatives(epoch, regenerate=True)`
ignored epoch and returned one FIXED materialized 1:1 negative set → every epoch of
every migrated baseline (EmerGNN/HDN-DDI/SSI-DDI) reused identical negatives. The
native `PairDataset` (dataset.py:470-517) already resamples per epoch via
`build_train_negatives` (seed `base_seed + TRAIN_NEGATIVES_SEED_BASE(1000) + epoch`);
the unified adapter was a regression that dropped it.

## Design B (deterministic generator, no storage) — implemented
- NEW `Code/data_utils/leaf_negatives.py`: `LeafTrainNegatives` thin wrapper over the
  existing `build_train_negatives`. `leaf_base_seed = blake2b(dataset|regime|split|fold)`
  (stable across processes, NOT python hash) → reconstructable any epoch, any time.
  `for_epoch(splits, e)` → build_train_negatives(base_seed, epoch=e, exclude_extra=...).
- `LeafResources` gained runtime fields `fold` + `fold_dir` (defaults → non-breaking);
  `load_leaf` populates them.
- Adapter: `_LeafDataset._epoch_neg`; `regenerate=True` → per-epoch draw; `regenerate=False`
  → fixed materialized negatives (audit anchor). `make_dataset` builds the provider for
  task==binary only (multiclass positives-only / multilabel endpoint-corruption unaffected).
- **exclude_extra = dataset-global positives (union y_bin==1 across ALL regime/split/fold
  leaves, mirroring analyze_unified_audit.py:53-66) ∪ this leaf's fixed val negs ∪ fixed
  test negs.** (Round-1 codex catch: S0/fold0 alone is INCOMPLETE for MRCGNN datasets
  drugbank_deng/ryu → would collide with real positives in other folds. Fixed.)
- Pool = g1 (roles train/train_seen) → S1/S2 cold-start safe, no unseen leakage.

## Verification (ran)
- ddi800 S2 fold0: 1:1 (35351); same epoch twice identical (reconstructable); e0≠e1;
  regenerate=False = fixed materialized; overlap with train-pos / fixed val-neg / fixed
  test-neg all 0; fresh provider reproduces e1 exactly.
- deng (|global_pos|=37264) + ryu (|global_pos|=191164): epochs 0/1/2/50 have 0 overlap
  with the FULL global-positive union; e0≠e1; 1:1 preserved.
- codex round2 019f1f7b: **APPROVED**, no findings.

## Effect on already-accepted baselines
EmerGNN / HDN-DDI / SSI-DDI binary call `get_train_negatives(regenerate=True)` → they now
get per-epoch negatives (paper-faithful), resolving the HDN-DDI "frozen negatives" nit.
All baselines + our method draw from the SAME deterministic sequence → binary comparable.
No paper-gate reruns (simplified acceptance).

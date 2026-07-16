# MRCGNN multiclass — from-scratch port (step-gated) + review

- **Date**: 2026-07-01
- **Primary reviewer**: Claude (opus-4.8, coordinator) + a general-purpose subagent (porter)
- **Independent reviewer**: codex (gpt-5.2-codex) — gated EVERY step
- **Trigger**: user — port MRCGNN (AAAI-2023) as a non-KG baseline; step-gated agent, each
  step reviewed by coordinator + codex before proceeding.

## Paper core idea (MRCGNN, AAAI-2023)
"Multi-relational Contrastive Learning GNN for DDI Event Prediction." Multiclass DDI-event
(Deng 65 / Ryu 86). TrimNet molecular encoder (offline supervised pretrain → per-drug 128-d
features) + 2-layer RGCN over the DDI-event graph + layer-attention fusion + molecular skip
+ DGI-style contrastive on 2 corrupted views + 3-loss training
(1.0·CE + 0.05·BCE_viewA + 0.1·BCE_viewB).

## Port (all NEW under Code/baseline/mrcgnn/, file-independent; upstream at
Paper/Reference/Original-Code/MRCGNN/)
- layers.py (Discriminator/AvgReadout/MLPHead), models.py (MRCGNN nn.Module),
  trimnet_model.py (encoder), mol_features.py (atom/bond featurizer, 55/10 dims),
  _data/necessary/build_trimnet_features.py (offline supervised TrimNet builder), _shared.py
  (detect-and-build + leakage-safe cache key), multi_cls/{baseline.py, baseline_unified.py}.
- registered ("mrcgnn","multiclass") in run_baseline_unified._MODULES.

## Step-gated codex verdicts
- Step 1 (plan): APPROVED_WITH_CHANGES (019f1fe2) — 3 factual corrections (mol features drive
  BOTH branches not skip-only; view B = same adjacency + shuffled relation labels; TrimNet =
  supervised pretrain) + confirmed GLOBAL-K + TrimNet-builder-is-the-crux.
- Step 2 (model core): APPROVED_WITH_NITS (019f1feb) — forward faithful to upstream; nits =
  Step-3/4 contracts (raw attt, pairwise view-B shuffle, mol-features caller contract).
- Step 3 (TrimNet featurization): CHANGES_REQUIRED (019f1ff4) → fixed → OK. 3 real fixes:
  cache key must encode epochs + ordered drug list (else 5-ep smoke reused as 300-ep, or
  row-misaligned features); drop_last small-fold zero-step random-init guard. Re-verified
  (different keys for epochs/order; small fold trains >0 steps).
- Step 4 (training core): APPROVED_WITH_NITS (019f2001) — NO blocking findings; 3-loss, graph,
  both views, hparams, cold-start non-leakage, global-id contract all faithful. Nit (RNG) =
  non-issue (Step 5 confirmed views built-once already).
- Step 5 (wrapper + register): approved (verbatim ssi_ddi-multi pattern; view-once confirmed
  via object-id spy = 1 distinct corrupted-view tensor across epochs).

## Cold-start mechanism (paper-faithful)
DDI graph from leaf TRAIN positives only; node/feature universe = ALL leaf drugs. Unseen S2
drugs have valid TrimNet molecular features (structure-only) but no incident train edge → the
RGCN branch is near-inert for them and the molecular path carries the signal. TrimNet trained
on TRAIN pairs only (leakage-safe cache key = drug set + train triples + K + fold + seed +
epochs + ordered SMILES). Expected MODEST S2 macro-F1 — this is FAITHFUL, not a bug.

## Smoke (coordinator-run) — see _results/2026-07-01__multi_cls__deng_s2_smoke.md
Full unified path end-to-end on deng S2 fold0, 1 epoch: TrimNet cache MISSING→BUILT (570,128),
DONE + 5 metrics, exit 0. accuracy 0.113 (42 gold classes, ~4.7× above random 0.024) —
learning; macro_f1 0.013 (1-epoch + cold-start, expected low).

## Status
MRCGNN multiclass BUILT + step-gated codex-approved + smoke PASS. Non-KG baseline complete
(the only MRCGNN task; paper is multiclass-only). Full runs pending (user).

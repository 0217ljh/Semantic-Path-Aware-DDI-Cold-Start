# v2 Stage 3 design DRAFT (i1 PK/PD dual-channel) — pre-codex

**Status**: DRAFT for codex review (round 17). Do NOT code until reviewed.
**Codex r16 ranked this #1 next step**: PK/PD channels are conceptually orthogonal
to EmerGNN flow, so more likely to move COMBINED than i4 text (which was redundant).

## Motivation (i1)
PK interactions act through the MOLECULAR layer (enzymes/transporters/proteins/
pathways); PD through the EFFECT-SYSTEM layer (side effects/phenotypes/anatomy/
disease). A FLAT meeting-node counts aux (Stage 1) blurs these. Stage 3 splits the
22 counts into two mechanism channels with independent gates — the i1 architectural
prior, NO PK/PD label supervision in training.

## Channel assignment (of the 11 kind groups)
- **PK / molecular channel**: protein_gene, pathway, molecular_function,
  biological_process, cellular_component  (×{1hop,2hop} = 10 dims)
- **PD / effect channel**: side_effect, disease, anatomy  (×{1hop,2hop} = 6 dims)
- **Shared/neutral**: compound, pharmacologic_class, exposure (×2 = 6 dims) — feed BOTH
  or a third neutral channel (decide w/ codex).

## Architecture (minimal increment on Stage 1 counts aux)
```
pk_logit  = MLP_pk(counts[PK dims])
pd_logit  = MLP_pd(counts[PD dims])
combined  = emergnn_logit + softplus(b_pk)*pk_logit + softplus(b_pd)*pd_logit
```
- Two small heads, two independent gates b_pk, b_pd (init so each ≈ 0.5 to sum ~1).
- BCE on combined. Same training/fusion as Stage 1.

## The i1 EVIDENCE (the real payoff, beyond a metric bump)
Use existing `Code/data/KG/drugbank/enriched/ddi_pk_pd_labels.csv` to label test
pairs PK vs PD. Then:
- Compute per-class contribution: for PK-labeled test pairs, is pk_logit's
  contribution to the correct ranking > pd_logit's? And vice versa for PD pairs?
- Expected (i1): PK channel carries PK pairs, PD channel carries PD pairs →
  interpretable specialization. THIS is the publishable i1 result, not just AUC.

## Metrics
- Primary: combined within-run Δ vs Stage 1 flat-counts aux (does splitting help?).
- Primary i1: per-class channel-contribution asymmetry (PK pairs ← PK channel etc.).
- Honest expectation (per codex r16 logic): combined may move little (same info,
  just split); the WIN is interpretability + per-class analysis. Report honestly.

## Controls
- Swapped-channel control: route molecular counts to "PD" head and vice versa.
  If channel specialization is real, swapping should HURT per-class contribution.
- Flat-counts Stage 1 is the baseline (already have 3 runs).

## Codex r17 decisions (PASS-as-bounded-i1-evidence, NOT perf bet)
- Success = PK/PD specialization WITHOUT perf loss (expect combined ≈ Stage 1). Not a number bump.
- Global scalar gates b_pk,b_pd FIRST. Pair-conditional only later if clean.
- PRIMARY metric = per-class channel-ablation difference-in-differences (DiD):
  DiD = (PK_drop_on_PK − PD_drop_on_PK) − (PK_drop_on_PD − PD_drop_on_PD). Positive = specialization.
  Use AUC drop if subgroup large; else log-loss / margin (AUC noisy on small subsets).
- Secondary: pk_logit vs PK-pairness correlation (descriptive); swapped-channel control.
- MANDATORY: label-quality caveat; manually audit 50-100 PK + 50-100 PD; report high-conf subset.
- Stop rule: 3 seeds; keep only if specialization directionally stable. Else consolidate i2+i4.

## DATA MAPPING (confirmed feasible 07:30)
- test_s2.parquet HAS columns [drug_a_id, drug_b_id, description, ddi_type].
- ddi_pk_pd_labels.csv maps ddi_type(215) → pk_pd_label(PK/PD) + matched keywords.
- Path: test positive pair → ddi_type → join labels → PK/PD class. Negatives have no
  type (use only positives for per-class analysis, or rank within class vs shared negs).
- Label audit: sample 50-100 each, hand-check keyword match → report agreement %.

## IMPLEMENTATION STATUS: designed + reviewed + feasibility-confirmed. CODE when GPU frees
  (repeats running ~3h) so trainer can be tested immediately, not sit unrun.

## Open questions for codex (r17) [ANSWERED above]
1. Where to put the 6 neutral dims (compound/pharm-class/exposure)? Both channels,
   own channel, or drop?
2. Is per-class channel-contribution the right i1 metric, or use gradient/attribution?
3. Should gates be per-pair (conditioned on f_u,f_v) instead of global scalars? That
   would be a stronger i1 claim (pair-conditional routing) but more capacity.
4. Given i4 was redundant with flow, is PK/PD splitting of COUNTS also at risk of
   redundancy with flow? How to argue it's different (mechanism categories vs names)?
5. Need PK/PD repeats / how many seeds for the specialization claim?

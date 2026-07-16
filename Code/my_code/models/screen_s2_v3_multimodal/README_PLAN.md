# v3 PLAN — Edge-Independent Multimodal Alignment for Cold-Start DDI

**Status**: DRAFT for codex review loop (do NOT code until PASS).
**Working split**: seed42 800-drug PKL (fast iteration, per user directive).
**Target**: best-method S2 cold-start AUC > 0.80 (MNAH currently ~0.77).

## Problem (from user + ColdDDI benchmark evidence)
Multimodal molecular+KG models COLLAPSE on double-drug cold-start because they fuse the
two modalities ONLY through the supervised DDI edge — absent when both drugs unseen:
- TIGER S2 AUC 0.630 (< molecular-only DSN 0.711, < KG-only EmerGNN 0.707) → fusion HURTS
- MKG-FENN S2 AUC 0.563 (near chance, degenerate)
We need EDGE-INDEPENDENT multimodal alignment so the 2nd modality GAINS, not collapses.

## Assets
- KG modality: MNAH (EmerGNN flow + shared-mediator meeting-node head), ~0.77 S2. KEEP.
- Molecular modality (NEW): hierarchical BRICS-motif graph per drug — cold-start-stable
  (any SMILES → motifs). DSN ~0.81 / HLN ~0.75 both-unseen from chemistry alone (their splits).
- Existing: EmerGNN already ingests Morgan FP (crude molecular). v3 adds STRUCTURED hierarchy + alignment.

## Core thesis
Hierarchical molecular branch + EDGE-INDEPENDENT contrastive alignment (molecule ↔ the
drug's KG neighborhood, never the DDI label) + MNAH fusion = multimodal GAIN under cold-start.

## STAGED design (each stage isolates one question; codex-style minimal increments)

### v3.1 — Molecular branch, NAIVE fusion (the contrast / collapse test)
- Encode each drug's BRICS-motif graph (atom→motif→molecule) → mol embedding m_u, m_v.
  Start simple: a small GNN (e.g. GIN) over motif graph, or even motif-count features
  as a first cut (cheap, like the meeting-node counts that worked).
- Pair molecular logit = MLP([m_u; m_v; m_u*m_v]).
- combined = MNAH_logit + softplus(γ)*mol_logit. NO alignment.
- QUESTION: does naive molecular fusion help or collapse (TIGER-style) on our S2?
- This is the baseline the alignment must beat.

### v3.2 — EDGE-INDEPENDENT contrastive alignment (the key innovation, Aspect 2)
- Auxiliary self-supervised loss (NO DDI label): align each drug's mol embedding m_u to
  its KG-neighborhood embedding k_u (pooled targets/enzymes/pathways from the KG).
  Contrastive (InfoNCE): positive = (m_u, k_u) same drug; negatives = other drugs' k.
  L_align = InfoNCE(m, k). Total = BCE(combined, y) + λ_align * L_align.
- Rationale (KANO/KCL-style): forces the molecular encoder into the KG space using only
  intrinsic structure → transfers to UNSEEN drugs by construction (no DDI edge needed).
- QUESTION: does alignment turn v3.1's result into a real cold-start GAIN?

### v3.3 (only if v3.2 partial) — Gromov-Wasserstein OT alignment (Aspect 2 backup)
- Replace/augment InfoNCE with GW-OT aligning molecular-graph structure to KG-neighborhood
  structure (label-free, no negative sampling). Use if InfoNCE underperforms for unseen drugs.

## Controls (anti-collapse / attribution, per past codex discipline)
- v3.1 vs MNAH: does molecular add or hurt without alignment?
- v3.2 vs v3.1: does alignment specifically recover the gain? (the thesis)
- shuffled-molecule control: scramble molecule↔drug assignment → mol branch should go to chance.
- alignment-only ablation: λ_align=0 vs >0.
- Report within-run Δ and combined AUC, seed42, 3 repeats for any candidate gain.

## Open questions for codex (review round 19)
1. v3.1 molecular encoder: start with motif-COUNT features (cheap, mirrors the meeting-node
   counts that worked) or go straight to a motif-graph GIN? Risk of over-engineering.
2. Is the InfoNCE molecule↔KG-neighborhood alignment the right edge-independent objective,
   or is GW-OT safer given the anisotropy/collapse risks we saw with pooled embeddings?
3. KG-neighborhood embedding k_u: reuse the existing PubMedBERT node embeddings of the
   drug's neighbors? or learned KG embeddings? or the 22-dim meeting-node counts?
4. Does adding a molecular branch risk just re-learning what EmerGNN's Morgan-FP input
   already captures? How to isolate the NEW molecular signal?
5. Staging: is v3.1→v3.2 the right order, or should alignment be built in from the start?
6. Biggest risk this whole multimodal direction doesn't break 0.80 — what's the kill criterion?

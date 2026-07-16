# v2 Stage 2 + Stage 3 design DRAFT (pre-codex, pending Stage 1 confirmation)

**Status**: DRAFT — do NOT implement until Stage 1 (MNAH) is confirmed
(shuffle-control collapses + repeats consistent) and this passes codex review.

## Where Stage 1 leaves us (if confirmed)

MNAH proves the **meeting-node count signal (i2) is additive to EmerGNN flow**
(+2.57pt). But Stage 1 uses only crude 22-dim *counts* of shared mediators by
kind. It does NOT yet use:
- **i4**: which specific mediators (their node-name text semantics)
- **i1**: PK vs PD paradigm structure (molecular- vs effect-layer mediators)
- **i3**: pair-conditional per-instance selection (counts treat all mediators equally)

Stages 2-3 add these, each as an isolated, attributable increment.

## Stage 2 — PubMedBERT mediator semantics (i4)

**Hypothesis**: replacing/augmenting the 22 kind-counts with semantic summaries
of the actual shared-mediator NODES (encoded by PubMedBERT on node.name) gives
further gain, because *which* CYP enzyme / pathway is shared matters, not just
how many.

**Minimal design** (keep attributable):
- Pre-compute PubMedBERT([CLS]) for every KG node name → 768d, project to 64d
  (frozen random projection, per E3 protocol). Cache.
- For pair (u,v): shared-mediator set M (already computed). Aux representation =
  mean (or count-weighted mean) of projected PubMedBERT vectors of nodes in M,
  per kind-group → concat with the 22 counts.
- Feed to a slightly wider aux MLP. Same logit-fusion as Stage 1.
- **Control**: shuffled-name PubMedBERT (E3-style) — semantic vs capacity.

**Expected**: real-name > shuffled-name by ≥1pt (mirrors E3's +5.65pt finding
on init, but here at the readout).

## Stage 3 — PK/PD dual-channel + pair-conditional selection (i1 + i3)

**Hypothesis**: routing molecular-kind mediators (Gene/Protein/Pathway/enzyme/
transporter) to a PK channel and effect-kind (SideEffect/Phenotype/Anatomy/
Disease) to a PD channel, with pair-conditional attention over individual
mediators, captures the i1 asymmetry that a flat readout blurs.

**Minimal design**:
- Two aux sub-heads: mol_repr (PK), eff_repr (PD), each a pair-conditional
  attention-weighted sum over its mediator subset:
  α_m = softmax_m( MLP([f_u; f_v; pubmedbert(m); kind(m)]) )   # i3 per-instance
- combined_logit = emergnn_logit + β_pk·g_pk(mol_repr) + β_pd·g_pd(eff_repr)
- No PK/PD LABEL supervision needed (architectural prior, per i1 doc).
- **Analysis** (not just metric): does β_pk dominate for PK-labeled test pairs and
  β_pd for PD-labeled pairs? Use the existing ddi_pk_pd_labels.csv to check.
  This would be the i1 *evidence*, not just a number.

**Expected**: small additional metric gain + interpretable PK/PD channel
specialization (the publishable i1 story).

## Risks / why staged
- Each stage adds confounds; isolating them lets us attribute gain to i4 vs i1 vs i3.
- If Stage 2 (PubMedBERT) gives ~0 over Stage 1 counts, that's still informative:
  it means kind-counts already capture the mediator signal and node identity adds
  little under S2 — report honestly (would weaken i4's readout-side claim).

## Compute prep — EMBEDDINGS ALREADY EXIST (no re-encoding needed!)
Found cached at `Code/data/KG/_merged_kg/_cache/screen1_tag_init/`:
- `d_name_only__pubmedbert.pt` — dict{node_ids:list[178029], embeddings:[178029,768]}
  = node-NAME PubMedBERT [CLS]. EXACTLY the i4 signal for Stage 2.
- `e_shuffled_text__pubmedbert.pt` — same shape, shuffled-name control (semantic-vs-capacity).
- also d_full_text / f_typename_only available.
Reuse plan: build node_id→768d lookup; for each pair's shared-mediator set M,
mean-pool the 768d vectors per kind-group, project (frozen random or PCA to ~64d),
concat with the 22 counts → aux MLP. Shuffled-name pt gives the E3-style control free.
Saves the ~1 GPU-hour encode. (E3 pipeline: 05_init_comparison.py.)

## Open questions for codex (Stage 2 review, later)
1. Augment 22 counts WITH pubmedbert means, or REPLACE? (augment = safer increment)
2. Frozen random projection vs PCA (E3 found PCA better for init — does it hold at readout?)
3. Pooling over M: mean, count-weighted mean, or attention? (attention = Stage 3, keep Stage 2 simple)
4. Is mean-pooling PubMedBERT vectors too lossy? Alternative: top-k by degree.

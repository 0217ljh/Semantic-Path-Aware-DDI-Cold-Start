# LLM Distillation — How to Induce KG-Complementary Information

**Status**: prompt v1 implemented; rich design space untested
**Logged**: 2026-05-30
**Cache**: `Code/data/_cache/llm_pharma/llm_pharma.jsonl` (1897 drugs, Claude Haiku 4.5)
**Downstream caches**: `llm_text_pubmedbert.npz` (1530 × 768) and `i4_typed_sets.json` (1530 × 10 fields)
**Used by**: E-llm (free-text → INERT) and E-i4 (structured pair features → +0.84pt over MNAH)

This is **the most experimentally informative cache** in our pipeline, because it's the only one
where the same source(LLM output)was used in two ways and produced **completely opposite outcomes**.
That asymmetry tells us exactly what kind of inducement matters.

---

## The core asymmetry — and what it means for prompt design

Same JSONL, two consumption modes, two outcomes:

| Experiment | LLM output used as | Result on MNAH+channel | Shuffle control |
|---|---|---|---|
| **E-llm** | `sanitized_text` → PubMedBERT CLS (768d) → per-drug embedding | 0.7668 (≈ MNAH 0.772) | **0.7658 = main → INERT** |
| **E-i4** | `structured` fields → **pairwise** mechanistic features (13-d) | **0.7804 (+0.84pt over MNAH)** | 0.7643 (clean collapse) |

**Implication**: LLM-distilled signal **is** useful, but **not as free-text-CLS-per-drug**; it's
useful when it becomes **structured + pairwise-composable**. So the question "how to induce useful
info" reduces to "**how to make the LLM emit richer structured / categorical / numeric fields that
compose meaningfully across pairs**" — NOT "how to write longer free-text prose".

This finding is consistent with codex's strategic verdict (`019e67af`): cold-start DDI gain comes
from **explicit pair-level mechanistic features** that the KG topology doesn't traverse directly.
The LLM's job is to extract the per-drug atoms (CYP role, transporter role, class, etc.) that the
downstream model can compose into pair features.

---

## What KG already has vs. what LLM might add (honest inventory)

**KG (DrugBank + Hetionet + PrimeKG, DDI-masked)**:
- Edge-form facts: drug → target / enzyme / transporter / pathway / disease / phenotype / side-effect / anatomy
- Binary or coarse-typed only; e.g. "drug X is enzyme Y's substrate" but not strength or kinetics

**LLM training data plausibly encodes (KG doesn't)**:

| Category | Example | KG limitation |
|---|---|---|
| Quantitative PK | Ki, IC50, half-life, F%, V_d, clearance fractions | KG only has binary edges |
| Inhibition/induction strength | strong vs moderate vs weak CYP3A4 inhibitor | KG flattens to a single edge |
| Mechanistic narrative | "partial agonist with biased GPCR signaling" | KG has drug→target only |
| Clinical context | narrow therapeutic index, dose adjust in CKD | not in our KG |
| Class-level comparison | "atypical within statins because of dual-mode action" | KG nodes don't express intra-class heterogeneity |
| Negative knowledge | "no clinically relevant CYP inhibition" | KG can't express absence |
| Cross-domain semantic ties | PubMed co-occurrence frequency, ADR report bias | KG is a structural graph |
| Risk class associations | "should be cautious with serotonergic agents (class, not drug)" | KG doesn't categorize risk classes |

The promising-but-untested theme is **categorical/numeric features that compose pairwise** —
exactly the I4 axis that gave +0.84pt, just much richer.

---

## Current prompt — what we did

Single-shot, T=0, sanitized for "intrinsic only / no partners / no DDI phrases" (see
`distill_llm_pharmacology.py`). Structured fields requested:
`cyp_substrate, cyp_inhibitor, cyp_inducer, transporter_substrate, transporter_inhibitor,
therapeutic_class, primary_targets, pd_effects, toxicity_mechanisms, clearance` + free-text.

**Coverage measured**: CYP-substrate 50%, CYP-inhibitor 27%, CYP-inducer 2%, transporter-substrate
18%, transporter-inhibitor 3%, class/targets/pd/tox/clearance ~80%. **27.7% leakage-flagged** (had
partner drug name or DDI phrase in raw → sanitizer redacted).

What this prompt **didn't** elicit (gaps a re-distillation could fill):
- No strength categories (just binary substrate/inhibitor)
- No numeric values (Ki, IC50, half-life)
- No clearance pathway fractions
- No therapeutic index / narrow-vs-wide
- No class-level risk descriptions
- No negative knowledge ("does NOT inhibit X")
- No multi-sample consistency

---

## Inducement strategies — ranked by (info value / leakage risk)

### A. Safe + high-EV (recommended for next distillation pass)

**A1 — Quantitative categorical fields for PK enzymes/transporters**

Replace binary `cyp_substrate: ["cyp3a4"]` with strength categories:
```
cyp_3a4: {role: substrate|inhibitor|inducer|none, strength: strong|moderate|weak|na}
cyp_2d6: {...}
... (one entry per CYP / UGT / transporter of clinical interest)
half_life_hours: number or range
clearance_pathway: {hepatic_pct, renal_pct, biliary_pct, other_pct}  # sum=100
therapeutic_index: narrow | moderate | wide
protein_binding_pct: number
oral_bioavailability_pct: number
```

→ Direct upgrade to I4's pairwise features: `strong-inhibitor(a) × primary-substrate(b)` is
**clinically much more predictive** than the current `is-inhibitor × is-substrate` binary product.
**Fully leakage-safe** (per-drug intrinsic). This is the single highest-EV change.

**A2 — Class-level risk profiles (no specific drugs)**

```
Without naming ANY specific drug, list:
  pk_risk_classes: list of compound CLASSES that would interact PK-wise with this drug
  pd_risk_classes: list of compound CLASSES sharing/opposing pharmacodynamic effects
```

→ Yields class-overlap pair features (`drug_a.pk_risk_classes ∩ drug_b.therapeutic_class`).
**Leakage-safe** (class is a generic descriptor, not a partner).

**A3 — Negative knowledge probe**

```
Explicitly answer (yes/no/uncertain):
  has_clinically_relevant_cyp_inhibition: ...
  has_qt_prolongation_risk: ...
  has_serotonergic_activity: ...
  ... (10-15 clinically relevant DDI mechanisms)
```

→ Forces LLM to make absence explicit. **Especially useful for negative-control feature
construction** (e.g. "neither drug has QT risk" should reduce DDI probability).

### B. Medium value / medium risk

**B1 — Multi-sample self-consistency**: same prompt, T=0.5, n=3 samples per drug → vote on each
structured field. Costs 3× API but reduces single-shot noise. Especially useful for the rare/sparse
fields (CYP-inducer 2%, transporter-inhibitor 3% in current run — these may benefit most from
voting).

**B2 — Chain-of-thought structured extraction**:
```
Step 1: Reason about this drug's metabolic fate (3 sentences).
Step 2: Based on that reasoning, fill in the structured fields.
```
Typically produces more accurate structured fields, but reasoning text increases leakage risk
(partner drug names slip into the reasoning). Needs aggressive sanitizer.

**B3 — Few-shot exemplars**: include 2-3 hand-curated examples of "ideal output" (using generic
drugs like aspirin or metformin) to constrain format. May rescue some of the 367 lost records
(records that failed JSON parse or had empty sanitized_text).

**B4 — Model comparison**: Claude Haiku 4.5 (current) vs Claude Sonnet 4.5 vs GPT-4 on 50-drug
subset. Higher-tier models may provide more accurate quantitative fields (Ki, IC50, fractions).
If Sonnet meaningfully better, use it for the high-stakes test drugs only (cost-aware).

### C. High value, high risk (deploy only with strong sanitizer)

**C1 — Mechanistic DDI narrative at CLASS level**

```
Describe MECHANISMS by which this drug could participate in DDIs.
Forbidden: naming any specific drug.
Allowed: phrases like "this drug, as a strong CYP3A4 inhibitor, can reduce
clearance of any CYP3A4 substrate" or "concomitant administration with
serotonergic agents poses theoretical risk".
```

→ Rich free-text **explaining mechanism rationale**. Downstream: train an NER+RE model to extract
(condition, class) pairs from the narrative, use as features. **Risk**: LLM may slip and name a
specific drug ("e.g. with simvastatin..."). Need much stronger sanitizer than current.

**C2 — Quantitative DDI propensity score** (per-drug, 1-10):

```
On a 1-10 scale, estimate:
  pk_inhibition_score: probability this drug inhibits others' metabolism
  pk_substrate_vulnerability: vulnerability to others affecting its metabolism
  pd_overlap_breadth: how many therapeutic areas its PD effects span
  narrow_ti_concern_score: clinical concern due to narrow therapeutic index
```

→ Compact single-drug risk scores. Usable as gating features. **Caveat**: LLM number calibration is
notoriously bad; need a calibration ablation (do 1-10 scores correlate with held-out ground truth?).

### D. Avoid (oracle leakage / not publishable)

- **D1**: directly asking `"Does drug A interact with drug B?"` → straightforward oracle leakage
- **D2**: RAG with literature documents → may inject information that was supposed to be DDI-masked
- **D3**: asking LLM to predict `ddi_type` for unseen pairs → same as D1

---

## Orthogonal levers (not about prompt, but about how we *use* the output)

### E — Replace PubMedBERT CLS with a stronger encoder

E-llm's failure may not be the LLM's fault; it may be PubMedBERT's. CLS on a 2020-era domain
model is much weaker than 2024+ retrieval-tuned encoders (MedCPT, BGE-large, OpenAI's
`text-embedding-3-large`). See `language_encoding.md` for the full discussion. **Re-running E-llm
with a stronger text encoder might rescue the free-text channel**.

### F — Attention-pool instead of CLS

Even with PubMedBERT, attention-pooling over all tokens (instead of taking CLS) typically yields
richer per-document representations. Cheap to test.

### G — Distill to a SMILES → LLM-feature student (backlog I12)

Train a small MLP that takes SMILES Morgan + ChemBERTa as input and predicts the LLM structured
features. Then at test time, unseen drugs get LLM-quality features just from structure. This breaks
the "we need to query LLM for every new drug" deployment problem. From the backlog, never built.

### H — Multi-LLM ensemble

Distill the same drug with 2-3 LLMs (Claude / GPT-4 / a domain-specialized like BioMistral or
Med-PaLM if accessible), combine structured fields with majority voting or weighted average.
**Cost**: 2-3× the current pipeline. Useful if rare-field accuracy is the bottleneck.

---

## Connection to other follow-up logs

| Log | Relation |
|---|---|
| `effect_neighbor_aggregation.md` | Same theme (`adaptive weighting vs fixed rules`), different cache |
| `kg_typed_aggregation.md` | Same theme (`pair-conditioned attention over typed buckets`) |
| `language_encoding.md` | Orthogonal lever E above — replacing PubMedBERT may save E-llm |
| `kg_molecular_redundancy.md` | Deepest question — even with rich LLM-distilled features, can we break out of KG saturation? |
| `molecular_alignment_design.md` | Cross-attention architecture would consume richer LLM features as one modality |
| `fragment_importance_weighting.md` | EDT-Former entropy weighting can compose with LLM features |

**(CORRECTION 2026-05-30, user challenge)**: I previously framed "failures come from
non-pairwise-composable representations" as the unifying principle. This is **overclaim** — it's
**ONE** plausible explanation among several, and the v2 effect-channel failure (which WAS
pair-level via cross-attention, yet still hit chance) directly contradicts the strong reading.

Multiple competing hypotheses are consistent with the observed pattern (I4 +0.84pt, E-llm inert,
v2 effect-channel chance, E-frag inconclusive, MNAH counts +3pt):

| H | Statement | Supports | Tensions with |
|---|---|---|---|
| H1 | Pair-level form is required | I4, MNAH | **v2 effect-channel was pair-level and failed** |
| H2 | KG-complementarity is required (features that KG doesn't already encode via topology) | All cases; CYP inhib→substrate cross is the only KG-edge-missing relation among the failures | Needs R1 ablation to confirm |
| H3 | Representation matched to data scale (low-dim explicit features tractable; high-dim opaque embeddings overfit at 53k training pairs) | I4 (13-d) > E-llm (2304-d) > v2 effect (high-dim opaque) | Doesn't isolate from H2 |
| H4 | Inductive bias matches task structure (DDI = mechanism overlap) | I4's explicit features match; opaque embeddings don't | Philosophical, hard to test |

All four are likely **simultaneously active with different weights**. The cleanest disambiguation is
the **R1 KG-neighborhood ablation** in `kg_molecular_redundancy.md` (isolates H2). EDT-Former
inducement, the new LLM distillation plan, and the cross-attention architecture from
`molecular_alignment_design.md` are all attacking some subset of {H2, H3, H4} simultaneously
without isolated tests of any.

The honest reframe: **on this cold-start + strong-KG-backbone setup, added features need to satisfy
(approximately) (a) information not already KG-traversable AND (b) representation tractable with
limited training pairs AND (c) tooling that exposes pair-level signal to the head — but we have not
isolated which condition is binding.**

---

## Recommended next distillation pass

If we ever re-run the LLM distillation, the highest-EV minimal change is **A1 + A2 + A3**:
**categorical strength fields + class-level risk lists + explicit negative knowledge**.

Concrete prompt skeleton (combining all three):

```
You are a clinical pharmacologist. Without naming any specific drug or DDI,
return STRICT JSON with the following intrinsic-only profile for {drug_name}
(SMILES: {smiles}).

# Quantitative pharmacology
cyp_3a4: {role: substrate|inhibitor|inducer|none, strength: strong|moderate|weak|na}
cyp_2d6: {...}
cyp_2c9: {...}
cyp_2c19: {...}
cyp_1a2: {...}
cyp_2b6: {...}
ugt_1a1: {...}
transporter_pgp: {...}
transporter_bcrp: {...}
transporter_oatp1b1: {...}
transporter_oatp1b3: {...}
transporter_oct2: {...}
half_life_hours: number or [min, max]
clearance_pathway: {hepatic_pct, renal_pct, biliary_pct, other_pct}  # sum=100
therapeutic_index: narrow | moderate | wide
protein_binding_pct: number
oral_bioavailability_pct: number

# Class-level risk profile (no specific drugs)
pk_risk_classes: list of compound CLASSES (e.g., "strong CYP3A4 inhibitors", "P-gp substrates")
pd_overlap_classes: list of compound CLASSES sharing PD effects (e.g., "serotonergic agents")

# Negative knowledge
has_clinically_relevant_cyp_inhibition: yes|no|uncertain
has_qt_prolongation_risk: yes|no|uncertain
has_serotonergic_activity: yes|no|partial|no
has_anticholinergic_activity: yes|no|partial|no
has_cns_depressant_activity: yes|no|partial|no
... (8-12 more clinically relevant negative-knowledge fields)

Hard constraints:
- NO specific drug names anywhere in output (use class descriptions only)
- NO DDI phrases (interact, coadminister, contraindicated with [drug], etc.)
- Express interaction-relevant biology as INTRINSIC properties only
```

**Cost estimate**: 1897 drugs × (1.5× tokens for richer schema) ≈ ~7 min on Haiku, ~$0.5-1.
**Downstream**: new `i4_typed_sets_v2.json` with all these fields; new pair features that include
**strength-weighted overlaps** (e.g. `count(strong_inhib(a) ∩ primary_subst(b))`).

Expected impact: if I4's binary features gave +0.84pt, strength-categorical features could plausibly
add another +0.5 to +1pt — still below the +1.5pt threshold but cleanly testable. Combined with
the encoder upgrade from `language_encoding.md` it might exceed.

---

## The honest unknown

We don't actually know whether ANY of these inducements will move the needle, because the
redundancy thesis (`kg_molecular_redundancy.md`) predicts that LLM-extracted features about a
drug's properties may be redundant with what the KG already encodes via biomedical neighborhoods.

The only way to test is to run the experiment chain in R1 (KG-neighborhood ablation) from
`kg_molecular_redundancy.md`. Under partial-KG conditions, LLM features should become more useful
exactly to the extent that they encode information the KG was masking.

If we ever invest in richer LLM distillation, **pair it with the R1 ablation** to get both:
- empirical lift on full-KG (or not, providing redundancy evidence)
- empirical lift on masked-KG (which would prove the LLM features carry genuinely complementary
  information, even if the full-KG benchmark doesn't show it)

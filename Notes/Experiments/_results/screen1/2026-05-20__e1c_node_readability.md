# E1c Node Readability — Motivation experiment for i4 (TAG init feasibility)

**Date**: 2026-05-20
**Script**: `Code/my_code/models/screen1_tag_init/node_text_builder.py`
**Output stats**: `Code/data/KG/_merged_kg/_cache/screen1_tag_init/node_text_stats.parquet`

## Claim being tested

i4 (node-name text semantics under cold-start): the merged biomedical KG provides human-readable names for ≥80% of biomedical-relevant entities, so a text encoder (PubMedBERT) can produce non-trivial node init even for newly-introduced drugs with no KG neighbors.

**Gate**: ≥80% of biomedical-relevant nodes (Drug, Gene/Protein, SE, Disease, Anatomy, Pathway, Phenotype, biological_process, molecular_function, cellular_component, side_effect, symptom, exposure, pharmacologic_class) have human-readable names that pass a vowel-word + non-ID-shape regex.

## Result

| Bucket | Count | Share |
|---|---|---|
| Total merged-KG nodes | 178,029 | 100% |
| With non-empty text | 174,882 | 98.2% |
| Readable (regex pass) | 128,336 | 72.1% |
| Biomedical-relevant kinds | 128,564 | 72.2% |
| **Readable AND relevant** | **105,123** | **59.0% of all / 81.8% of relevant** |

**Gate met** (≥80% of biomedical-relevant nodes are readable): **PASS** (81.8% ≥ 80%).

## Sanity check on PubMedBERT [CLS] of node text (variant D, full text)

10 representative drug pairs:

| Pair | cosine sim | comment |
|---|---|---|
| Carbamazepine / Diazepam (both CNS-active) | 0.992 | highest among tested |
| Aspirin / Capecitabine | 0.974 | inflated by shared profile template |
| Carbamazepine / Chlordiazepoxide | 0.952 | mixed CNS / anxiolytic |
| Aspirin / Ibuprofen (both NSAID) | 0.934 | high but lower than templated cases |
| **Bivalirudin / Leuprolide (unrelated)** | **0.817** | lowest, as expected |

Avg cosine over 100 random intra-Drug pairs:
- variant D (full profile): 0.964
- variant D-name (drug name only): 0.892
- variant E (within-kind shuffled names): 0.895
- variant F (kind label only): 1.000 ← all-same-class

**Interpretation**: variant D's higher absolute similarity than D-name is partly an artefact of shared profile-text boilerplate (e.g. "Type: small molecule"). Whether this carries useful relative ordering for downstream DDI prediction will be settled by the actual Screen 1 training matrix.

## Top KG node kinds (relevant + readable)

```
biological_process   28,642
gene/protein         27,610
Gene                 20,945
disease              17,080
effect/phenotype     15,311
anatomy              14,033
Biological Process   11,381
molecular_function   11,169
drug                  5,854
Side Effect           5,734
cellular_component    4,176
Molecular Function    2,884
pathway               2,516
Pathway               2,487
Protein               2,482
```

Note PrimeKG (`drug`/`gene/protein`/lowercased) and Hetionet (`Drug`/`Gene`/TitleCase) name kinds differently. The merged KG keeps both vocabularies; TAG encoders see them as different `kind` strings.

## What this enables

- TAG variant D (PubMedBERT [CLS] of full profile + KG name) has non-trivial text for 174,882 / 178,029 = 98.2% of nodes (3,147 ID-only fallback to zero vector)
- Cold-start drug case: a held-out (G2) drug has no DDI edges, but its DrugBank profile is always available, so init is never degraded by missing KG context.

## Decision

i4 motivation is established. Proceed to Screen 1 training matrix.

## Caveats

- Readability regex is heuristic (vowel-word + non-ID-shape). Spot-checked but not labeled by hand.
- Profile-text similarity is high in absolute magnitude (~0.96) because PubMedBERT [CLS] is generally clustered. Relative ordering is the load-bearing signal — Screen 1 training will arbitrate whether real text beats shuffled text on downstream AUC.

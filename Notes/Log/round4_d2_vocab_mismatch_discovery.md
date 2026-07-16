# Round 4 D2 — Critical vocab-mismatch discovery from smoke test

**Date**. 2026-06-01
**Trigger**. D2 CP-2 implementation 1-epoch smoke test (`d2_smoke_1ep` run_id `2026-06-01_15-27-21__run_v3_meet_mask__d2_smoke_1ep__seed42`). The chain wires end-to-end, but the mask is essentially empty after vocab filtering.

## The discovery

Trainer log (verified from `Code/runs/2026-06-01_15-27-21__run_v3_meet_mask__d2_smoke_1ep__seed42/train.log` line containing `mediator-cache summary`):

```
n_pairs_loaded: 319600
n_dropped_pairs_out_of_vocab: 0
n_dropped_mediators_out_of_vocab: 5179531
cardinality_mean_post_mutation: 0.277
cardinality_max_post_mutation: 12
```

Reading the numbers:
- 319,600 pairs × pre-filter mean of 16.5 = ~5.27M total mediators in cache.
- 5.18M out of 5.27M (**98%**) are dropped during the `m in self._entity2id` translation step in `v3_meet_mask_trainer.py:_load_mediator_cache`.
- Surviving mediators: ~88,500 across 319,600 pairs, giving **mean ~0.28 mediators per pair**, max 12.

## Root cause

The D2 mediator builder computes `M_ab` over the **merged KG** (178,029 entities; `Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet`) per design §3.2 step 1, which copies `Code/my_code/models/screen_s2_v2_meetnode/precompute_meet_features.py:81-117`'s neighbor-set logic.

The D2 trainer inherits `_PerModeEmerGNN_V2I4 → _PerModeEmerGNN_MNAH → _PerModeEmerGNN` with `kg_source="drugbank"` (same as v2i4 anchor, per round4_d1_kg_source_deviation.md). The drugbank 5-bucket KG only has ~5,633 entity IDs (drug + enzyme + target + transporter + carrier + pathway). Mediator IDs derived from Hetionet / PrimeKG nodes in the merged KG (the bulk of mediators) are not in the trainer's entity vocab and get silently dropped.

In contrast, MNAH's 22-d count cache (`Code/data/_cache/meet_feat_drugbank_seed42_kgonly_v1.parquet`) **aggregates** mediators by kind-group BEFORE feeding to the trainer, so vocab-mismatch is irrelevant: 22 floats per pair, no per-entity lookup. MNAH's success at 0.7670 combined despite the vocab gap relies on this aggregation.

D2's mask requires per-entity-vocab lookup. The vocab mismatch makes the mask essentially empty.

## Was this caught at CP-1?

No. CP-1 round 1 asked about cardinality (R1) but not about vocab coverage. CP-1 round 2 explicitly flagged C2 leakage difference between MNAH and D2 (union+cross-hop expanded), but did not note that the mediator universe is in the wrong NAMESPACE for the drugbank-vocab trainer. This is an implementation-time-discovered design gap.

CP-1 process improvement (for D3/D4 future designs): **CP-1 must require an explicit vocab-coverage check** — what fraction of mediator IDs (or any per-entity signal IDs) will survive translation to the trainer's entity vocab?

## Three forward paths

**Path A (small build change; keep design intent of broad mediators)**. Build `n1` and `n2` over the merged KG as before, but at the end of the per-pair mediator computation, intersect with `drugbank_5_bucket_entity_set` (loadable via `build_kg_from_kb(kb, drug_id_list)`'s output) so the cached mediator IDs are pre-filtered to drugbank-vocab. Re-run with R1 cardinality probe; likely much smaller now (mediator universe shrinks from ~170k to ~5k non-drug drugbank-vocab entities).

**Path B (build over drugbank 5-bucket only)**. Compute `n1` over drugbank 5-bucket edges directly (drug → enzyme/target/transporter/carrier/pathway). `n2` is empty for the 5-bucket because the schema is bipartite drug↔entity. Mediator = `n1[a] ∩ n1[b]` over 5-bucket only. Aligns with Stage 1 MNAH count cache's exact mediator universe.

**Path C (switch trainer to merged KG)**. Use `kg_source="merged"`. The 178k-entity trainer would naturally accommodate all mediator IDs. But this changes the substrate vs the 0.7804 anchor — apples-to-apples comparison breaks; a new merged-KG v2i4 anchor would need to be run first.

## Decision

**Path A failed empirically**. After implementing Path A and excluding Hetionet's `kind="Compound"` from the mediator candidate pool (Compound IDs are DrugBank drugs by another label, including them as mediators leaks the drug-as-mediator path), the cache reports **mean=0, max=0, zero_fraction=1.0** — verified `Code/data/_cache/meet_mediators/_summary__seed42_drugbank.json` after the Compound-fix re-run with sha16=a1ae9686d0c908aa. The drugbank 5-bucket vocab and merged-KG entity-ID spaces simply do not overlap except for drugs themselves.

**Pivoting to Path B**: compute `n1` directly over the drugbank 5-bucket KG (drug ↔ enzyme/target/transporter/carrier/pathway bipartite). `n2` is empty by schema; mediator(a, b) = `n1[a] ∩ n1[b]` over the 5-bucket only. This aligns exactly with the trainer's entity vocab and matches Stage 1 MNAH's mediator universe (just at per-entity level instead of 22-d aggregated).

Build pivot impact:
- Code: new builder function that uses `build_kg_from_kb(...)` triplets directly (no merged-KG dependency).
- Design: this is no longer the design §3.2 step 3 "union+cross-hop" formula. It's the simpler 1-hop intersection over the bipartite 5-bucket KG. CP-2 must be told.
- Round4 plan §6 deviation: yes, but the design intent (pair-conditional propagation bonus on shared mediators) is preserved; only the mediator universe definition changes.

Rebuild + re-smoke before triggering CP-2. CP-2 disclosure: this entire vocab-mismatch + Path A→B pivot story.

## What this means for D2 expectations

The D2 design intent is still pair-conditional propagation. Path A preserves that. But cardinality of ~3-5 mediators per pair is at the small end — the bonus path has limited "budget" to influence the backbone. The lift may be smaller than the round4 plan §4 D2 expected `combined ≥ 0.785`. CP-3 should weigh this when judging.

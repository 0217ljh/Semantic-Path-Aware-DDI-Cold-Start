# D2 Design Sketch — Meeting-Node-Aware Propagation (pre-staged for potential D1 pivot)

**Status**. Pre-staged sketch. Written 2026-05-31 while K2/K3 controls are running. If D1 CP-3 verdict is NOT_PASS (likely given K1 falsification + capacity confound), this is the pivot target per round4 plan §4 D2. If D1 is rescued by K3, this stays archived for later sequence.

**Not yet authoritative**. Has not gone through CP-1. Numbers / file:line references are spot-checked but the full design needs codex review before any code.

## Mechanism (one paragraph)

EmerGNN's propagation at `Code/baseline/emergnn/model.py:137-180` updates `hiddens[v, b, :]` per layer via per-(batch, relation) gated relation embedding × per-entity hidden state. Currently no batch-specific bonus on meeting-mediator nodes. D2 adds:

    new_hiddens[v, b, :] += α_meet[l] × is_meet(v, a_b, b_b) × hiddens[v, b, :]

at each layer `l` after the message-passing aggregation (after `model.py:178`), where `is_meet` is a (n_ent, B) {0,1} mask saying whether node v is in `(n1[a_b] ∪ n2[a_b]) ∩ (n1[b_b] ∪ n2[b_b])` (the 1-hop-or-2-hop meeting mediator set for the b-th pair). `α_meet[l]` is a learnable scalar per layer initialized at 0 (so D2 starts byte-identical to v2i4 / D1 with d1_disable).

## Data flow

### Precompute cache

New offline builder `Code/my_code/models/screen_s2_v3_multimodal/precompute_meet_mediators.py`:
- Re-uses the meeting-set logic from `Code/my_code/models/screen_s2_v2_meetnode/precompute_meet_features.py:81-117` (1-hop / 2-hop neighbor sets built from the merged KG edges parquet, restricted to non-drug mediators).
- For each canonical (drug_a, drug_b) pair in train/val/test universe + epoch-regenerated negative space, write to `Code/data/_cache/meet_mediators__seed42_drugbank.parquet`:
  - `drug_a_id` (canonical)
  - `drug_b_id` (canonical)
  - `mediator_ent_ids: list[int]` — the EmerGNN entity-vocab indices of nodes in 1-hop OR 2-hop intersection
  - `n_mediators: int` (cardinality, for log diagnostics)

The entity-vocab depends on the trainer's `_setup_graph` output. The builder must be invoked AFTER the trainer's entity vocab is decided (since the cache uses entity indices), OR build keyed by drug-id string and let the trainer translate at lookup time (preferred; matches how `meet_feat` cache works at `mnah_trainer.py:159`).

### Runtime: per-batch mask

In `_combined_logit` of the new D2 trainer:
1. For each row in the batch DataFrame, lookup canonical (drug_a, drug_b) in the cache, retrieve the list of mediator IDs.
2. Build a sparse `meet_mask` of shape (n_ent, B), where `meet_mask[mid, b]=1` for each mid in the b-th pair's mediator list.
3. Pass `meet_mask` into the new model's forward (signature change vs base EmerGNN; that's why we need a new EmerGNN subclass).

## Model subclass (does NOT modify baseline/emergnn/model.py)

New file `Code/my_code/models/screen_s2_v3_multimodal/emergnn_meet_mask.py`:

    class EmerGNNWithMeetMask(EmerGNN):
        def __init__(self, *args, alpha_meet_init=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.alpha_meet = nn.Parameter(
                torch.full((self.L,), float(alpha_meet_init))
            )

        def _propagate(self, source_idx, source_embed, ht_embed,
                       edge_src, edge_dst, edge_rel, meet_mask=None):
            # Mirror parent _propagate up to the per-layer hidden update
            # (model.py:137-178), then ADD the meet-mediator bonus before
            # passing into self.act(self.linear[l](new_hiddens)).
            ...

        def forward(self, head, tail, edge_src, edge_dst, edge_rel,
                    meet_mask=None):
            # Same as parent forward but threads meet_mask into both
            # _propagate calls.
            ...

`α_meet[l]` is the only new trainable parameter (3 floats for L=3 layers). The mask is a non-trainable input.

## D2 trainer

New file `Code/my_code/models/screen_s2_v3_multimodal/v3_meet_mask_trainer.py`:

    class _PerModeEmerGNN_V3MeetMask(_PerModeEmerGNN_V2I4):
        def __init__(self, *, d2_meet_cache=None, d2_alpha_init=0.0,
                     d2_disable=False, d2_shuffle_mask=False, **kwargs):
            super().__init__(**kwargs)
            ...

        def _build_model(self, n_ent, n_base_rel, morgan_features):
            # Use EmerGNNWithMeetMask instead of EmerGNN
            return EmerGNNWithMeetMask(
                n_ent=n_ent, n_base_rel=n_base_rel,
                n_dim=self.n_dim, length=self.length, feat=self.feat,
                morgan_features=morgan_features,
                alpha_meet_init=self.d2_alpha_init,
            )

        def _combined_logit(self, head, tail, edge_src, edge_dst, edge_rel,
                            batch_df):
            meet_mask = (None if self.d2_disable
                         else self._build_meet_mask(batch_df))
            emergnn_logit = self._model(
                head, tail, edge_src, edge_dst, edge_rel,
                meet_mask=meet_mask,
            )
            # rest same as v2i4: count + i4 readout heads.
            ...

The `_build_model` hook does not exist in the parent today. We'd need to identify the right override point. Looking at `mnah_trainer.py:331-336`: the EmerGNN allocation is inline inside `fit()`. We'd need to override `fit()` OR refactor — but the project rule says no refactoring of forbidden files. So override `fit()` in the D2 trainer, duplicating the body but swapping the model class. (Verbose; acceptable.)

## Controls (D2-specific)

| Control | Mechanism | Expected effect |
|---|---|---|
| D2-disable | meet_mask=None at forward time | byte-identical to v2i4 baseline |
| α_meet=0 frozen | freeze all 3 α_meet params at 0 | structurally identical to D2-disable |
| Mask-shuffle | permute meet-mediator lists across pairs (preserve per-pair cardinality, destroy pair-specificity) | If D2 lift survives → mediator IDENTITY doesn't matter, the cardinality alone is the signal |
| Random-mediator | replace each pair's mediator list with a uniformly random set of same size | If D2 lift survives → it's just per-pair "have some mask=1 somewhere" effect |

The mask-shuffle and random-mediator are the analogues of K1 / K2 in D1, but operating on the mediator-set instead of token-edges. Critical: D2 must beat these or it's the same capacity story.

## Risks

- **Per-batch mask construction cost**. For B=32 pairs × n_ent=5633 → 5633×32×4 = 720 KB sparse fill per batch. Likely negligible but should benchmark.
- **α_meet inits at 0 → no gradient signal at start**. Standard fix: init α_meet at a small positive (0.1) so gradient flow starts. CP-1 will decide.
- **Cardinality of meeting set ≥ ~10 typical**. Bonus then applied to ~10 nodes per pair per layer. If too many nodes are flagged, the bonus loses pair-specificity. Sanity-check at builder time: mean / max mediator count per pair.
- **Cross-product with D1**. D2 and D1 are orthogonal mechanisms — could in principle be combined. CP-1 will decide whether to test D1+D2 if D2 alone works.

## Expected lift if it works (per round4 plan §4 D2)

- combined ≥ 0.785
- emergnn-branch ≥ 0.75

These are mid-tier expectations. The D2 mechanism is more targeted than D1 (pair-conditional bonus on the propagation), so should have a smaller capacity-confound risk than D1.

## Pre-CP-1 followups (only if we decide to pivot to D2)

1. Verify the meet-mediator cache cardinality per pair (use existing `meet_feat` cache as a sanity proxy: sum of 22-d counts ≈ |1-hop ∪ 2-hop meeting set|).
2. Confirm `mnah_trainer.fit()` can be cleanly overridden (we already did this for D1's `_setup_graph`).
3. Decide α_meet init (0 vs 0.1 vs softplus(learnable)).
4. Decide whether to keep D1's LLM-edge KG extension simultaneously, or run D2 on the original drugbank 5-bucket KG (apples-to-apples vs v2i4 0.7804).

# D2 Design — Meeting-Node-Aware Propagation (Pair-Specific Backbone Injection)

**Round**. Round 4, direction D2 (per `Notes/Log/round4_backbone_diff_plan.md` §4).

**Trigger**. D1 (LLM-typed KG edge injection) was falsified at CP-3 (see `_reviews/2026-06-01__d1_results__round1.md`). All three D1 controls failed: the +1.28 pp emergnn-branch lift survived both K1 (drug-token binding shuffled) and K2 (cross-drug bridges destroyed). Codex Q7 attribution: "D1 failed because the mechanism is mostly shuffle-invariant. That points away from more global KG-density augmentation and toward explicitly pair-conditional mechanisms."

**Goal (same as D1)**. Push at least one readout-side signal (here: the i2 meeting-node mediator set) into the EmerGNN backbone such that the architecture differs from EmerGNN in a non-readout-only way, while preserving combined AUROC ≥ 0.78 vs the v2i4 anchor 0.7804. Hard-stop combined < 0.775.

**Difference from D1**. D2 makes the per-pair structure explicit in the propagation kernel itself (per-batch attention bonus on meeting-mediator nodes), not implicit in shared KG edges. This is the pair-conditional mechanism that D1 lacked.

**Gate**. CP-1 codex review must PASS (critical = 0, major = 0) before any code is written. Archive to `_reviews/<date>__d2_design__round1.md`.

---

## 1. Three claims (for CP-1 verification)

**Claim C1 — Backbone change is pair-conditional and non-readout-only.** D2 adds a per-layer learnable scalar `α_meet[l]` and a per-(batch, entity) {0, 1} meeting-mediator mask `is_meet[v, b]`. The mask is computed offline (set membership `v ∈ (n1[a_b] ∪ n2[a_b]) ∩ (n1[b_b] ∪ n2[b_b])`) and consumed by the propagation kernel as a bonus `α_meet[l] * is_meet[v, b] * hiddens[v, b, :]` added at every layer t. Reviewer must verify: (a) the bonus is genuinely pair-conditional (mask depends on a_b and b_b together, not on a_b xor b_b alone); (b) the bonus path into combined_logit only goes through the propagation kernel (not through any readout head bypass); (c) D2 is structurally distinct from D1 (D1 modified KG; D2 modifies propagation while leaving KG unchanged).

**Claim C2 — No cold-start leakage; safety comes from builder filtering, not from mask1 alone (CP-1 round 1 fix).**

The mask1 merged KG still contains 4,855 drug→drug edges (verified `refine-logs/LEAKAGE_AUDIT.txt:22` "drug->drug edges (any rel): 4855 (expect 0 for DDI-masked)"), all `het:CrC` from the Hetionet Compound–resembles–Compound relation. These are NOT DDI labels but they are drug-drug structural similarity edges. The MNAH 22-d count cache builder at `precompute_meet_features.py:81-117` filters them out by the construction `if src_is_drug and not dst_is_drug` — only drug→non-drug edges are emitted into n1, similarly for n2. The D2 mediator-set builder must apply the same filter explicitly (and assert no mediator in any pair's set is itself a drug-id).

Additionally, D2's mediator definition differs from the 22-d count cache's:
- 22-d count cache: `n1(a) & n1(b)` and `n2(a) & n2(b)` (per-hop intersection, then aggregated by kind).
- D2 mask: `(n1∪n2)(a) ∩ (n1∪n2)(b)` (union-then-intersection, raw entity IDs preserved).

The D2 union+cross-hop expands the mediator set by adding `n1(a)&n2(b)` and `n2(a)&n1(b)` matches. So the 22-d cache leakage audit does NOT transitively cover D2. **The D2 builder must run a fresh leakage audit at build time**, asserting: (a) no mediator is a drug-id; (b) mediator total count per test_s2 pair matches the union+cross-hop formula; (c) the median, p95, p99, max of `n_mediators` are reported.

Reviewer must verify the builder applies the filter explicitly, runs the per-pair audit, and does NOT cite the 22-d count cache audit as a transitive guarantee.

**Claim C3 — Two structural-parity guards + one runtime guard (CP-1 round 1 fix; `load_best_model_at_end` removed from claim).**

The original C3 incorrectly cited `load_best_model_at_end` as a safety net; codex CP-1 round 1 flagged that `load_best_model_at_end` restores the best VALIDATION checkpoint, not a pre-training v2i4-equivalent state — so it cannot serve as a worst-case bound.

Replaced with three explicit guards:

1. **`--d2-disable` (structural)**. Skips mask construction in `_combined_logit`. Mathematically equivalent to v2i4. CP-2 verifies via deterministic-seed K4 parity smoke (5-epoch loss within 1e-4 mean abs diff vs v2i4 `--deterministic` reference).

2. **`--d2-freeze-alpha-meet` (mathematical)**. `α_meet[l] = 0` for all layers, `requires_grad=False`. The bonus term `α_meet[l] * mask * hidden` is identically zero at every step. Forward is mathematically (not just statistically) equivalent to parent `EmerGNN._propagate`. CP-2 verifies via 5-epoch deterministic comparison: per-epoch loss must equal `--d2-disable` exactly (the only difference is two extra dead-zero adds per layer).

3. **Round4 §6 hard-stop (runtime)**. `combined < 0.775` at end of seed42 main HALTS the experiment before any control/sweep/multi-seed. So D2 cannot stealthily ship a regressed model.

Reviewer must verify the K4 deterministic-parity numerical bound is achievable (we have `torch.manual_seed` etc. wired in `run_v3_meet_mask.py --deterministic`, unlike D1).

---

## 2. Lessons-learned table: D1 → D2

| Dimension | D1 (failed) | D2 |
|---|---|---|
| Where the LLM/meeting signal enters | KG edge list (offline, baked in) | propagation kernel (online per batch) |
| Pair-conditional? | No (shuffle-invariant; K1/K2 confirmed) | Yes (mask depends on both a_b AND b_b) |
| What survives K1-equivalent shuffle | The lift (capacity-driven) | Predicted: lift collapses (signal is pair-binding) |
| Cost of being wrong | Capacity confound; codex Q1 said "honestly attribute to capacity, not semantics" | If K1-equivalent also doesn't drop, mediator IDENTITY doesn't matter — meaningful negative |
| Disable mechanism | `--d1-disable` skips KG injection (verified at K4 structural parity) | `--d2-disable` skips mask construction (CP-2 must verify byte parity with deterministic seed) |

---

## 3. Architecture and data flow

### 3.1 Components and files

| Component | Type | New file path | Modifies existing? |
|---|---|---|---|
| Meeting-mediator builder | CLI script | `Code/my_code/models/screen_s2_v3_multimodal/precompute_meet_mediators.py` | No |
| Mediator cache (offline) | Parquet | `Code/data/_cache/meet_mediators/meet_mediators__seed42_drugbank.parquet` | No |
| EmerGNN subclass with mask | Library | `Code/my_code/models/screen_s2_v3_multimodal/emergnn_meet_mask.py` | No |
| D2 trainer | Library | `Code/my_code/models/screen_s2_v3_multimodal/v3_meet_mask_trainer.py` | No |
| D2 run entrypoint | CLI | `Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py` | No |

**Files that stay untouched** (per round4 plan §6 禁止改 list): `Code/baseline/emergnn/*`, `Code/my_code/models/screen_s2_v2_meetnode/mnah_trainer.py`, `Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py`. D2 inherits from `_PerModeEmerGNN_V2I4` like D1 did.

### 3.2 Builder (`precompute_meet_mediators.py`)

**Input**.
- `Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet` (verified 7,099,528 edges 2026-05-30 in deviation note)
- `Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet` (178,029 nodes, 8 main kinds — verified)
- `Code/data/coldddi_legacy/800drug/seed42.pkl` (800-drug split universe)

**Process**.
1. Build per-drug 1-hop and 2-hop non-drug neighbor sets (reuse `precompute_meet_features.py:81-117` logic copied into the D2 builder; do NOT modify that file). **Hard-asserted filter**: emit only `(non-drug, kind)` into n1/n2 (matches the explicit `src_is_drug and not dst_is_drug` filter at `precompute_meet_features.py:97-104`). The D2 builder asserts this at every emit.
2. **Cache universe matches MNAH's** (CP-1 round 1 minor fix): per `precompute_meet_features.py:167-185`, the cache must cover all drug pairs in train + val_s2 + test_s2 + val_s2-negatives + test_s2-negatives + train_negatives epoch_0. For the 800-drug regime the Cartesian product (≈320k pairs) is feasible — emit the FULL canonical-ordered drug-pair grid so epoch-regenerated train negatives can be looked up at runtime without a cache miss.
3. For each canonical pair (drug_a_id, drug_b_id), compute mediator set `M_ab = (n1[a] ∪ n2[a]) ∩ (n1[b] ∪ n2[b])`. This is the **union+cross-hop** definition (codex CP-1 round 1 clarification): it includes the 22-d cache's `n1∩n1` and `n2∩n2` PLUS the new cross-hop terms `n1(a)∩n2(b)` and `n2(a)∩n1(b)`. The expanded universe means D2-specific leakage + cardinality audits are mandatory; the 22-d cache audit does not transfer.
4. For each pair, output the mediator KG-string-ids as `mediator_kg_ids: list[str]` and cardinality `n_mediators: int`. Trainer will translate to entity-index at lookup time (matches the canonical-pair pattern from `meet_feat` cache at `mnah_trainer.py:159`).
5. **Run D2-specific leakage + cardinality audit before writing cache** (CP-1 round 1 fix). Assertions:
   - `assert not any(m in drug_set for pair_mediators in cache.values() for m in pair_mediators)`. No mediator is a drug.
   - Report per-pair `n_mediators` distribution stats: `mean, median, p95, p99, max`, and `% pairs with n_mediators = 0`.
   - Cross-check expected union+cross-hop cardinality vs sum of 22-d count cache row (which captures `n1∩n1 + n2∩n2` after kind aggregation). The D2 cardinality should be ≥ 22-d sum, but ≤ a sane multiple (≤ 5×) — fail loudly if the median ratio exceeds 5× (likely a builder bug).
   - Reference prior: 22-d count cache mean total mediators per test_s2 pair = **33.83** (verified `refine-logs/LEAKAGE_AUDIT.txt:33`). D2 mean expected ≈ 33.83 – 100 range. **Single threshold contract** (aligned with §3.2 step 4 output spec and §6 R1): R1 (diffuse-mask risk) triggers if `mean > 50` OR `p95 > 200` OR `max > 1000`. On trigger, the §6 one-sweep allowance is committed to top-K filtering at builder time before CP-2 PASS.

**Output**.
- `Code/data/_cache/meet_mediators/meet_mediators__seed42_drugbank.parquet`
  - cols: `drug_a_id, drug_b_id, mediator_kg_ids: list[str], n_mediators: int`
  - canonical-pair order (a ≤ b lexicographically)
- `Code/data/_cache/meet_mediators/_summary__seed42_drugbank.json`
  - Per-pair cardinality stats: `mean / median / p95 / p99 / max`. **Sanity check at build time** (single threshold set, aligned with §6 R1 and §3.2 step 5): if `mean n_mediators > 50` OR `p95 > 200` OR `max > 1000`, R1 triggers and the §6 one-sweep allowance is committed to top-K filtering. CP-2 will inspect.

**Determinism**. Pure function (no randomness). Builder runs through with `numpy.random.seed(0)` set just in case.

**Cold-start safety** (CP-1 round 2 reworded). The builder reads (a) the merged KG edges parquet (DDI masked at upstream build per paper_writeup.md §3.Y audit; the residual 4855 het:CrC drug-drug edges are filtered by the n1/n2 builder per C2), and (b) the split-tables (`train.parquet`, `val_s2.parquet`, `test_s2.parquet`, plus their negative parquets) ONLY to enumerate the pair-universe for the Cartesian-style cache. The split tables are NOT consulted for label / interaction information — only for `drug_a_id` and `drug_b_id` column reads — so they cannot contribute to leakage at the builder layer.

### 3.3 EmerGNN subclass (`emergnn_meet_mask.py`)

New class `EmerGNNWithMeetMask(EmerGNN)`. Does NOT modify `Code/baseline/emergnn/model.py`. Subclasses cleanly.

```python
class EmerGNNWithMeetMask(EmerGNN):
    def __init__(self, *args, alpha_meet_init: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # One scalar per layer L. init=0 → byte-equivalent to parent at step 0.
        self.alpha_meet = nn.Parameter(
            torch.full((self.L,), float(alpha_meet_init), dtype=torch.float32)
        )

    def _propagate(self, source_idx, source_embed, ht_embed,
                   edge_src, edge_dst, edge_rel, meet_mask=None):
        # Mirror parent _propagate (model.py:116-180) line-by-line, then
        # add meet-mediator bonus AFTER the per-layer activation (model.py:178).
        ...
        for l in range(self.L):
            # parent message passing as-is (model.py:139-178), producing new_hiddens
            new_hiddens = self.act(self.linear[l](new_hiddens))
            if meet_mask is not None:
                # meet_mask: (n_ent, B) {0, 1}
                # bonus: (n_ent, B, n_dim) = alpha_meet[l] * mask.unsqueeze(-1) * new_hiddens
                bonus = self.alpha_meet[l] * meet_mask.unsqueeze(-1) * new_hiddens
                new_hiddens = new_hiddens + bonus
            hiddens = new_hiddens
        return hiddens

    def forward(self, head, tail, edge_src, edge_dst, edge_rel, meet_mask=None):
        # Same as parent forward (model.py:182-210) but threads meet_mask through
        # both _propagate calls (u→v and v→u).
        ...
```

**Why bonus added AFTER linear+act, not BEFORE**. Two reasons. (1) The activation makes the bonus more interpretable as "boost the post-message-passing representation of meeting nodes, not the raw incoming messages". (2) Pre-activation injection would interact with the linear layer's bias absorption in non-obvious ways.

**Trainable parameter count added by D2**: `self.L` floats = 3 floats (for L=3). 4 orders of magnitude smaller than D1's 10 new relation embeddings × n_dim × L=3 layers × 2 directions (forward+reverse) + 7600 new entity embeddings. D2's capacity addition is essentially zero — this is critical to defuse the capacity-confound risk that killed D1.

### 3.4 D2 trainer (`v3_meet_mask_trainer.py`)

`class _PerModeEmerGNN_V3MeetMask(_PerModeEmerGNN_V2I4)`. Inherits v2i4's count + i4 readout heads. Overrides `fit()` (since the model construction is inline in mnah_trainer.py:332-336, we can't just override `_build_model`; we override `fit` and duplicate the relevant body). Overrides `_combined_logit` to thread meet_mask through.

**Mediator-mask construction**. At `_combined_logit`, for each batch row look up `(drug_a, drug_b)` in the cached parquet, translate `mediator_kg_ids → entity_indices` via `self._entity2id`, build sparse `meet_mask` of shape `(n_ent, B)` with 1.0 at mediator positions and 0.0 elsewhere. Move to device. Pass into the new forward.

**Mediator IDs not in entity vocab**. Defensive skip with logged count. Some 2-hop mediators may be entities not part of the trainer's entity universe (e.g., if a drug-uncovered KG region is reachable). Drop those mediators from the per-pair set.

### 3.5 Run entrypoint (`run_v3_meet_mask.py`)

Mirrors `run_v3_llm_edge.py`. CLI flags (CP-1 round 2 fix: synced with §4 control definitions; the OLD "on-average cardinality" K1 and "uniform random" K2 single-flag semantics are dropped):

- `--d2-disable`. Skip mask construction; mathematically equivalent to v2i4 (K4 structural parity).
- `--d2-freeze-alpha-meet`. Freeze α_meet[l] = 0 at all layers + requires_grad=False (K3 mathematical pass-through).
- `--d2-shuf-mediators`. K1 — **cardinality-bucketed pair-matching shuffle** (per §4 K1 definition): pairs bucketed by exact `n_mediators`, within each bucket a random derangement (no pair maps to itself when bucket > 1); singleton buckets fall back to ±1 cardinality bucket with a logged note. Per-pair cardinality EXACTLY preserved post-shuffle for buckets with size > 1 (where derangement applies); for singleton-bucket pairs fallback-merged to ±1 cardinality, the post-shuffle cardinality may differ by 1 — fallback count logged at builder time. Mutates ONLY the binding, not the set-size.
- `--d2-rand-mediators-kind-matched`. K2a — **degree+kind-matched random replacement** (per §4 K2a): for each mediator `v`, sample a random non-drug entity `v'` of the same kind (e.g. gene→gene) and within ±20% of `v`'s degree, from a per-kind degree-bucketed entity pool. Preserves kind composition and degree shape; isolates "identity within proper structure".
- `--d2-rand-mediators-uniform`. K2b — **uniform random non-drug sampling** (per §4 K2b): for each pair, sample `n_mediators` entities uniformly without replacement from `_kg_entity_set - drug_set`. Kind/degree NOT matched. Harshest control.
- `--d2-alpha-init`. Default 0.0. Sweep candidate 0.1 if main run shows α_meet stuck at 0.
- **(codex Q8 process improvement)** `--deterministic`. Calls `torch.manual_seed(args.seed)`, `np.random.seed(args.seed)`, `torch.cuda.manual_seed_all(args.seed)`. Default True for D2 (unlike v2i4/D1 which were non-deterministic, see `Notes/Log/round4_d1_k4_observation.md`).
- `--d2-shuffle-seed`. Seed for K1/K2a/K2b shuffles. Default 12345 (matches D1 convention).

Old CLI removed (CP-1 round 2; do NOT bring back):
- ~~`--d2-rand-mediators`~~ (replaced by `--d2-rand-mediators-kind-matched` / `--d2-rand-mediators-uniform`).

Channel reporting (mirrors run_v3_llm_edge.py):
- `auc_combined`, `auc_emergnn`, `auc_count_only`, `auc_i4_only`.
- Mediator-coverage breakdown: per-pair `n_mediators` quartile bucket (Q1: 0 mediators, Q2: 1–3, Q3: 4–10, Q4: >10) and per-bucket AUC. Lets CP-3 see whether D2's lift concentrates on pairs with rich mediator structure.

---

## 4. Controls — pair-specific by design

Per codex Q7: "D2's controls can test mediator identity versus cardinality." Three controls + one parity check, all baked in from the start.

**Control K1 — shuf-mediators (CP-1 round 1 fix: STRICTLY cardinality-preserving via paired-matching)**.

Codex CP-1 round 1 flagged that "approximately preserved cardinality" leaves binding-vs-cardinality conflation. Fixed: implement K1 as a **cardinality-bucketed pair-matching shuffle**. Algorithm:

1. Compute per-pair cardinality `k_p` from the canonical D2 cache.
2. Bucket pairs by exact `k_p` (so all pairs with mediator count 3 are in one bucket, all with 7 in another, etc.).
3. Within each bucket with size > 1, apply a **random derangement** (a permutation with NO fixed points) — `seed=12345` for reproducibility, retry up to 10× if first sample has a fixed point. This guarantees every non-singleton-bucket pair gets a DIFFERENT pair's mediator set.
4. Pairs in singleton buckets (their k_p is unique in the dataset) cannot be deranged within their bucket. Fallback: merge them with the adjacent ±1 cardinality bucket and derange within the merged bucket. Log the singleton-bucket count and the post-merge derangement result.
5. Final report: total pairs shuffled, total fixed points (must be 0 for non-singleton-merged pairs), total ±1-cardinality fallback pairs. Log all three.

Per-pair cardinality is now **exactly** preserved (post-permutation). Mediator IDENTITY is destroyed but cardinality, mediator-degree-distribution, and per-pair set-size are intact. This is the clean "binding-only" falsifier.

Expected: combined drops ≥ 1.5 pp from D2 main. emergnn drops at least back to v2i4 anchor 0.7405. If K1 does NOT drop, mediator IDENTITY doesn't matter; cardinality alone is the signal (still scientifically interesting, just a weaker claim).

**Control K2 — rand-mediators (CP-1 round 1 fix: degree+kind-matched, uniform-random as harsher fallback)**.

Codex CP-1 round 1 flagged that uniform random sampling can hit unreachable / wrong-kind / degree-mismatched entities, making K2 trivially-low without separating "mediator identity matters" from "any biomedical node would do".

Fixed: K2 implemented as **two-tier**:
- K2a (degree+kind-matched): for each pair's mediator set `M_ab`, replace each mediator `v` with a random non-drug entity `v'` of the same kind (e.g. gene→gene, pathway→pathway) and within ±20% degree of `v`. Use the per-kind degree-bucketed entity pool from the merged KG. **Fallback contract** (CP-1 round 2 add): if no same-kind ±20%-degree candidate exists, widen to same-kind ±50% degree; if still none, fall back to same-kind any-degree; if still none (kind has fewer than 2 entities), fall back to K2b uniform random for that mediator slot. Each fallback level is counted and logged separately. This preserves the structural shape — same kind composition, similar degree — and isolates "mediator IDENTITY matters" cleanly.
- K2b (uniform random, harsher): for each pair, sample `n_mediators` random non-drug entity indices uniformly from `_kg_entity_set - drug_set`. Kind/degree mismatched. This is the strict ablation that probes whether any sane mediator set suffices.

Expected: K2a combined ≈ K1 (both isolate "identity matters" given matched structure). K2b combined ≤ K2a (harshest control). If K2a ≈ D2 main but K2b drops, "identity matters but only within proper kind/degree" — a richer paper story. If both K2a and K2b ≈ D2 main, K2 confirms the capacity story.

**Control K3 — α_meet=0 frozen (the trivial backbone-pass-through check)**.
`--d2-freeze-alpha-meet`. α_meet[l] = 0 at all layers, requires_grad=False. The bonus term `α_meet[l] * mask * hiddens` is zero. Forward is byte-identical to parent EmerGNN.

Expected: combined ≈ v2i4 anchor 0.7804 (no signal injected). This is the architectural-equivalence sanity. If K3 differs by > seed-noise (~0.5pp), there is a hidden non-α_meet path of D2 information leakage that we didn't intend.

**Control K4 — d2-disable parity smoke (CP-2-time only)**.
`--d2-disable` skips mask construction in `_combined_logit`. Mathematically equivalent to v2i4. CP-2 will run a 5-epoch K4 with `--deterministic` and compare to a v2i4 `--deterministic` reference; mean abs diff per-epoch loss must be ≤ 1e-4 (this time achievable thanks to deterministic seeding, unlike D1 K4 where we had to withdraw the claim).

---

## 5. Hard-stop and expected lift

Per round4 plan §6:
- combined < 0.775: STOP, open failure note.
- 0.775 ≤ combined < 0.785: run K1 + 1 hyperparam sweep (likely α_meet_init=0.1 or per-layer α_meet sweep).
- combined ≥ 0.785: run K1+K2+K3 full controls, then CP-3 → multi-seed.

Expected (per round4 plan §4 D2 + the lessons-learned):
- combined ≥ 0.785 (target). 0.79+ ideal.
- emergnn-branch ≥ 0.755 (the architectural-difference signal). Note D1 achieved 0.7533 via capacity-only; D2 needs to be cleanly above that AND drop under K1/K2.
- K1 drop ≥ 1.5 pp. K2 drop ≥ K1.

---

## 6. Risks (specific to D2)

**R1**. **Mediator-set is too diffuse (CP-1 round 2 single threshold contract)**. Verified prior: 22-d count cache mean total mediators per test_s2 pair = 33.83 (`refine-logs/LEAKAGE_AUDIT.txt:33`). D2 union+cross-hop will EXPAND this — possibly to mean 50–100 per pair. **SINGLE THRESHOLD CONTRACT (aligned with §3.2 step 4 and §3.2 step 5)**: R1 triggers if `mean > 50` OR `p95 > 200` OR `max > 1000`. When triggered, mask becomes diffuse and pair-specificity dilutes (the bonus boosts ~10% of all entities for some pairs, indistinguishable from a global per-pair learning-rate bump). **CP-1 → CP-2 hand-off contract**: builder MUST report mean / median / p95 / p99 / max + zero-fraction. On trigger, the §6 one-sweep allowance is pre-committed to top-K cap by 22-d count (CP-2 will surface concrete K candidate; preliminary K = top 20 mediators by count-cache contribution). Below all three thresholds, proceed without filtering.

**R2**. **Mediator-set is too sparse**. If many test S2 pairs have `n_mediators = 0` (no shared 1-hop or 2-hop non-drug neighbor), D2 cannot help those pairs at all. Mitigation: builder reports the fraction of pairs with n_mediators=0. We expect this to be small because the merged KG is dense.

**R3**. **α_meet saturation**. If α_meet[l] explodes to large values, the bonus dominates and destabilizes training. Mitigation: clip via softplus(raw_α_meet) - 1.0 (so α_meet ∈ (-1, ∞)) — but actually for the simple `bonus = α * mask * hidden` form, large α just amplifies a learned signal; clipping is not needed if Adam's adaptive learning rate handles it. CP-1 should comment.

**R4**. **Capacity confound re-emergence**. Adding L=3 floats can't be capacity — but if D2's lift is again invariant to K1/K2, the issue is the bonus is acting as a node-level dropout-like regularizer that helps the backbone in some content-agnostic way. Just like D1 we'd see a result that "looks like" mediator-aware learning but isn't. CP-1 should flag this as a critical risk.

**R5**. **forward-signature change is invasive**. We override `forward(meet_mask=...)` but call sites in v2i4_trainer / mnah_trainer expect the old signature. We solve this by overriding `_combined_logit` in the D2 trainer to call the new forward with the extra kwarg. We do NOT modify any existing call site. CP-2 will trace.

---

## 7. File-by-file change-set (zero deletion, all new files)

```
NEW Code/my_code/models/screen_s2_v3_multimodal/precompute_meet_mediators.py
NEW Code/my_code/models/screen_s2_v3_multimodal/emergnn_meet_mask.py
NEW Code/my_code/models/screen_s2_v3_multimodal/v3_meet_mask_trainer.py
NEW Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py
NEW Code/data/_cache/meet_mediators/meet_mediators__seed42_drugbank.parquet  (builder output)
NEW Code/data/_cache/meet_mediators/_summary__seed42_drugbank.json
```

Unchanged
- `Code/baseline/emergnn/*` (all files).
- `Code/my_code/models/screen_s2_v2_meetnode/*` (entire folder).
- `Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py` (and other v2i4 files).
- D1's files (`v3_llm_edge_trainer.py`, `precompute_llm_edges.py`, `run_v3_llm_edge.py`) — kept as-is for negative-result reference.

---

## 8. CP-1 review questions for codex

Q1 (C1). Is D2 genuinely pair-conditional, OR could shuffle/random controls fail to drop (D1-style failure) because the mask is "approximately the same shape" on average regardless of identity? Walk through the mask-construction logic. Argue why K1-shuf-mediators should hurt; if it shouldn't, redesign.

Q2 (C2). Cold-start leakage. Does the meeting-mediator-set construction transitively inherit the merged-KG leakage audit, or does the 2-hop extension expose a leakage path that the 22-d count cache didn't (because counts aggregate per-kind, while D2 uses per-entity-id information)?

Q3 (C3). Empirical worst-case. Is the α_meet=0 frozen state genuinely byte-equivalent to v2i4 (or just within seed-noise)? Walk through `EmerGNNWithMeetMask._propagate` with `α_meet = 0` and confirm the forward is bit-identical to parent `EmerGNN._propagate` (assuming the same RNG).

Q4 (controls, CP-1 round 2 wording aligned with §4). Are K1 (cardinality-bucketed derangement) / K2a (degree+kind-matched) / K2b (uniform random) / K3 (α_meet=0 frozen) / K4 (--d2-disable + deterministic parity smoke) the right falsifier suite? Are there ANY missing controls that would matter for CP-3?

Q5 (mediator-set size). Without running the builder, can you estimate (or flag the need to verify) typical mediator-set cardinality from the existing 22-d count cache `Code/data/_cache/meet_feat_drugbank_seed42_kgonly_v1.parquet`? If sum(22-d feature row) is consistently > 100 for test pairs, R1 (diffuse mask) is a real risk.

Q6 (relation to v2i4 readout). v2i4 has a 22-d count head reading the SAME mediator information. D2 reads the same set at the propagation level. Will the v2i4 count head become redundant under D2 (similar to D1's i4 readout collapsing to 0.5301)? Is there a worry that the v2i4 backbone-vs-readout fusion saturation that hurt D1 will recur?

Q7 (capacity confound re-emergence — the D1 lesson). D1 added 7,600+ entities and 10 relations. D2 adds 3 learnable scalars. So D2 cannot have D1's capacity confound. But the mask `meet_mask` is a runtime input that effectively boosts certain hidden values — could this act as node-level dropout/regularization (content-agnostic) instead of pair-specific evidence routing? How would CP-3 detect that?

Q8 (deterministic seeding). The decision to set `torch.manual_seed + np.random.seed + cuda manual_seed_all` for D2 (vs the non-deterministic D1) lets us make tighter K4 parity claims. Any downside (e.g., scientific generalizability concern from single-trajectory training)?

Q9 (hyperparameter sweep). If main D2 hits 0.775-0.785 band, the §6 one-sweep allowance is best spent on: (a) α_meet_init=0.1 (escape zero-init), (b) longer length L=4, (c) post-vs-pre-activation bonus placement. Rank these.

Q10 (paper-story alignment). Does D2 support the round4 plan §9 paper story bullet about "meeting-node-anchored evidence injected into propagation"? Does the D1 negative result help the D2 story (shows we tried global edge addition, found it shuffle-invariant, then made it pair-conditional)?

---

## 9. Post-CP-1 followup tracking

Design doc + codex CP-1 verdict + critical/major resolutions will be archived at `Code/my_code/models/screen_s2_v3_multimodal/_reviews/<YYYY-MM-DD>__d2_design__round1.md` per round4 plan §8.5. Primary = Claude opus-4-7, Independent = codex MCP default.

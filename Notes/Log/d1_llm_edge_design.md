# D1 Design — LLM Mechanistic Fields as New KG Edge Types Injected into EmerGNN Backbone

**Round**. Round 4, direction D1 (per `Notes/Log/round4_backbone_diff_plan.md` §4).
**Goal**. Push at least one readout-side signal (LLM `i4_typed_sets.json`) into the EmerGNN path-flow backbone so that the architecture differs from EmerGNN in a non-readout-only way, while preserving combined AUROC ≥ 0.78 vs the v2i4 anchor 0.7804 (`Code/runs/2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42/results.json:15`).
**Gate**. CP-1 codex design review must PASS (critical=0, major=0) before any code is written. Report archives to `Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-05-30__d1_design__round1.md`.

This design assumes the KG-source deviation already recorded in `Notes/Log/round4_d1_kg_source_deviation.md`: D1 extends the **drugbank 5-bucket KG** (the substrate of the 0.7804 anchor, verified from `run_v2i4.py:85` + that run's results.json config), not the merged 178k KG.

---

## 1. Three claims this design makes (for CP-1 verification)

**Claim C1 — Backbone is meaningfully changed, not readout-only**. D1 enlarges the KG with 10 new relation types (`llm:cyp_substrate`, `llm:cyp_inhibitor`, `llm:cyp_inducer`, `llm:transporter_substrate`, `llm:transporter_inhibitor`, `llm:therapeutic_class`, `llm:primary_targets`, `llm:pd_effects`, `llm:toxicity_mechanisms`, `llm:clearance`) and ~7800 new node IDs (one per unique LLM-distilled token). The EmerGNN `path-flow GNN, length=3` (`baseline/emergnn/_per_mode.py:138-186` defaults: `length=3`) is then trained without any change to its propagation code — but its input KG (edge list and relation embedding table) is genuinely different. Reviewer must verify that this **is** a backbone change (KG input space changed) and **is not** a hidden readout-side hack (no path of the new LLM info reaching the model except through the propagation).

**Claim C2 — No cold-start leakage from the new edges**. The LLM-distilled tokens are per-drug intrinsic properties only (prompt forbids any other-drug mention; two sanitizers strip residual interaction phrases — see `Code/my_code/models/screen_s2_v3_multimodal/distill_llm_pharmacology.py` and `v2i4_trainer.py:128-141`). The new edges are drug → token, never drug → drug. Two drugs sharing a token can route information through it via path-flow (this is the desired evidence-routing mechanism, the same lift principle as the readout-side i4 head), but this routing is built only from per-drug intrinsic properties, not from any DDI label. Reviewer must verify that no LLM-cache field encodes pair-relational information, and that the D1 builder re-applies the exact sanitizer list from `v2i4_trainer.py:128-141`.

**Claim C3 — There is an empirical worst-case guard against falling below 0.78, not an architectural guarantee.**

(Updated 2026-05-30 after CP-1 codex round 1 NOT_PASS feedback. Original C3 incorrectly claimed an architectural worst-case bound.)

Growing `n_base_rel` from 5 to 15 is not a no-op even for the existing 5 relations: EmerGNN at `model.py:59` derives `self.all_rel = 2 * n_base_rel + 1` (forward + reverse + self-loop slots), and `model.py:78` allocates a per-layer relation embedding table `self.rel_kg = ModuleList([Embedding(all_rel, n_dim) for _ in range(L)])`. Growing `all_rel` from 11 to 31 changes the random-init draws for the OLD relation rows (across all L=3 layers). The relation attention output also scales with `all_rel` (`model.py:84` `attn_relation = Linear(5, all_rel)`), and the self-loop relation id assignment in `kg_builder.py:175` depends on `n_base_rel`. So D1 cannot architecturally guarantee combined ≥ 0.7804.

What D1 does instead:
1. Keeps the v2i4 readout-side i4 head + 22-d count head intact (no removal). They give an empirical floor through the inherited additive logit fusion, learned end-to-end alongside the modified backbone.
2. Introduces a new **K4 drop-new-edges parity smoke test** (see §3): instantiate the D1 trainer with the LLM-edge parquet pointed at an empty parquet OR with `--d1-disable` (a no-op flag). With this flag, `_inject_llm_edges` is bypassed, no relation/entity expansion happens, and the trainer should be byte-identical to v2i4's `_PerModeEmerGNN_V2I4`. Seed42 smoke run (5 epochs) must reproduce v2i4's per-epoch loss trajectory to within numerical tolerance — this verifies the override is non-invasive when disabled.
3. The hard-stop rule §4 then enforces the empirical floor at run time: `combined < 0.775` halts the experiment before sweep/multi-seed. So D1 cannot stealthily ship a regressed model.

Reviewer must verify that the K4 smoke test is implementable (the override mechanism cleanly turns off) and that the §4 hard-stop is binding before any paper-claim use.

---

## 2. Architecture and data flow

### 2.1 Components and files

| Component | Type | New file path | Modifies existing? |
|---|---|---|---|
| LLM-edge KG-extension builder (precompute) | CLI script + library | `Code/my_code/models/screen_s2_v3_multimodal/precompute_llm_edges.py` | No |
| LLM-edge cache (output of builder, on-disk) | Parquet | `Code/data/_cache/llm_edges/llm_drug_edges__seed42_drugbank.parquet`<br>`Code/data/_cache/llm_edges/llm_token_nodes__seed42_drugbank.json` | No |
| D1 trainer (extends `_PerModeEmerGNN_V2I4`) | Library | `Code/my_code/models/screen_s2_v3_multimodal/v3_llm_edge_trainer.py` | No |
| D1 run entrypoint | CLI script | `Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py` | No |

**Files that stay untouched** (per round4 plan §6 禁止改 list): `Code/baseline/emergnn/*`, `Code/my_code/models/screen_s2_v2_meetnode/mnah_trainer.py`, `Code/my_code/models/screen_s2_v3_multimodal/v2i4_trainer.py`.

### 2.2 Builder (`precompute_llm_edges.py`)

Input
- `Code/data/_cache/llm_pharma/i4_typed_sets.json` (1530 drugs, 10 typed fields each — verified schema 2026-05-30).
- 800-drug split universe from `Code/data/coldddi_legacy/800drug/seed42.pkl` (verified 800 drugs in the split union).

Process
1. Load `i4_typed_sets.json`. Apply the same 13-phrase sanitizer used at `v2i4_trainer.py:128-141` (BAD list = `("interact","coadminist","co-administ","combined with","combination with","concomitant","avoid with","contraindicated","with inhibitor","with inducer","increase levels","decrease levels","co-medic","co-prescri")`, applied to free-text fields `FREE = ("therapeutic_class","primary_targets","pd_effects","toxicity_mechanisms","clearance")`). Log `[d1-builder] sanitizer-2 dropped <n>` matching the v2i4 log style.
2. For each field f in the 10 typed fields, gather all unique tokens across all drugs after sanitization. Construct a new node ID per token: `llm:<field>:<token_slug>` where `<token_slug>` is the lowercased token with whitespace collapsed to underscore and non-`[a-z0-9_-]` characters dropped.
3. Emit drug→token directed edges with relation `llm:<field>`. **Direction**: drug as head, token as tail (matching the convention of `kg_builder_merged.py:101-103` "drug always as head" — drugbank 5-bucket likewise uses drug-as-head edges in `build_kg_from_kb`). Skip drugs not in the cache (their slot is just absent — no zero-degree token node is created on their behalf).
4. Output two artifacts:
   - `llm_drug_edges__seed42_drugbank.parquet` with schema `[drug_id: str, token_node_id: str, relation: str]` — one row per (drug, token, field) triple, deduplicated.
   - `llm_token_nodes__seed42_drugbank.json` with schema `{token_node_id: {field: str, raw_token: str, n_drugs: int}}` — the full token-node manifest for inspection and shuffle controls.

Determinism. Builder is pure (no randomness). Re-running on the same input MUST produce byte-identical output (the builder writes deterministic row order: sort by `(drug_id, relation, token_node_id)`).

Cold-start safety. By construction the per-drug LLM fields are intrinsic-only (prompt + double sanitizer). Builder does not consult any DDI table.

### 2.3 D1 trainer (`v3_llm_edge_trainer.py`)

Class: `class _PerModeEmerGNN_V3LLMEdge(_PerModeEmerGNN_V2I4)`. Inherits the v2i4 architecture (path-flow backbone + 22-d count head + 13-d i4 head fused at the logit), so all v2i4 readout-side guarantees are preserved.

Hook point for KG extension. Override `_setup_graph(train, kg)` (from `baseline/emergnn/_per_mode.py:218`). Pseudocode (file-line-grounded):

```python
def _setup_graph(self, train, kg):
    morgan_mat, drug_id_list = super()._setup_graph(train, kg)  # parent does drugbank 5-bucket
    if self.d1_enabled:
        # parent has populated: self._entity2id, self._n_ent, self._n_base_rel,
        # self._kg_triplets, self._kg_entity_set, self._edge_src/_dst/_rel.
        self._inject_llm_edges(drug_id_list)
        # morgan_mat is per-entity (n_ent, 1024). After injection n_ent grows,
        # so we zero-pad morgan_mat to the new size (LLM token nodes have no SMILES).
        new_morgan = np.zeros((self._n_ent, morgan_mat.shape[1]), dtype=np.float32)
        new_morgan[:morgan_mat.shape[0]] = morgan_mat
        morgan_mat = new_morgan
    return morgan_mat, drug_id_list
```

`_inject_llm_edges(drug_id_list)` does
1. Load `llm_drug_edges` parquet, drop rows whose `drug_id` is not in `self._entity2id` (drugs outside the trained drug pool — silent skip with count log).
2. Optionally apply control mutations (see §3 below).
3. Assign each new token node ID a fresh entity index, starting at `self._n_ent`, and extend `self._entity2id`.
4. Assign each of the 10 LLM field names a fresh relation index, starting at `self._n_base_rel`. Extend `self._n_base_rel += 10`.
5. Build a new (head, tail, rel) array of length n_llm_edges with the new indices, concatenate it to `self._kg_triplets`, dedupe at (head, tail, rel), and refresh `self._kg_entity_set`.
6. Rebuild `self._edge_src/_dst/_rel` via `build_sparse_adj(...)` + `edges_as_dense_lists(...)` on the new triplets, matching `_per_mode.py:298-301`.

Then in `fit()`, when the parent EmerGNN allocates `EmerGNN(n_ent=self._n_ent, n_base_rel=n_base_rel_with_ddi, n_dim=self.n_dim, length=self.length, feat=self.feat, morgan_features=morgan_mat if self.feat=="M" else None)` (`_per_mode.py:366-373` / `mnah_trainer.py:332-336`), the model automatically gets the enlarged relation table and entity table. **No propagation-kernel change is needed.**

`shuffle_train` (`_per_mode.py:466-475`) only swaps DDI edges in/out of the per-epoch KG; it ignores non-DDI edges like our LLM edges. So LLM edges are always present at training and at eval — matching the spec that this is a stable per-drug intrinsic substrate.

### 2.4 Run entrypoint (`run_v3_llm_edge.py`)

Mirrors `run_v2i4.py` CLI exactly, with these added flags:
- `--d1-shuf-token`. Activates `D1_CONTROL_SHUF_TOKEN` (see §3).
- `--d1-rand-token`. Activates `D1_CONTROL_RAND_TOKEN`.
- `--d1-drop-i4-head`. Sets `β_i4 init = 0` and freezes `raw_β_i4` (forces backbone to carry the i4 signal alone).
- `--d1-edges-parquet PATH`. Override default builder-output path.

Channel reporting (mirrors run_v2i4.py:52-57):
- `auc_combined` (the canonical metric).
- `auc_emergnn` (backbone-branch only, computed by zeroing β_count and β_i4 at predict time using the existing `predict_channels` pattern from v2i4_trainer.py:235-256).
- `auc_count_only`, `auc_i4_only` — preserved from v2i4.
- **(CP-1 round 1 add)** Covered-vs-uncovered S2 split metrics. For each test_s2 pair (drug_a, drug_b), tag the pair as:
  - `both_covered` if both drugs appear in `i4_typed_sets.json` keys,
  - `one_covered` if exactly one does,
  - `neither_covered` if neither does.
  Report `auc_combined / auc_emergnn` per tag (with n_pos / n_neg per bucket). Purpose: detect the case where lift is concentrated in `both_covered` pairs (which is what we'd expect if the LLM substrate is doing its job), and flag if lift inexplicably appears in `neither_covered` (which would suggest a capacity / KG-rebuild artifact).

Expected D1-success signature (see §4 hard-stop logic):
- combined ≥ 0.785, and emergnn-branch lifts from 0.7405 (v2i4 baseline) to ≥ 0.755 (this is the direct evidence that the backbone, not just the readout, absorbed the new LLM signal).
- both_covered AUC > one_covered AUC > neither_covered AUC ≈ v2i4 anchor (no lift on neither_covered).

---

## 3. Controls (P2 falsification ladder)

D1 only earns the novelty claim if its lift is attributable to the LLM-derived edge semantics, not to (a) extra parameters/capacity, (b) extra graph density, or (c) spurious training-time advantage. Four controls (K1, K2, K3 main + K4 parity smoke), in roughly increasing destructiveness for K1–K3 plus the disabled-injection sanity K4:

**Control K1 — shuf-token (the load-bearing falsifier)**.
Before injection in `_inject_llm_edges`, permute the `drug_id` column of the LLM-edge parquet within the set of drugs that have **at least one** LLM edge (so coverage cardinality is preserved). Edge counts per relation, per-relation token vocabulary, and the n_ent / n_base_rel counts are byte-identical to the canonical run. The only thing destroyed: the binding between a drug and its specific LLM tokens. Expected:
- combined AUROC drops by ≥ 1.5 pt below the canonical D1 run.
- emergnn-branch AUC drops at least back to the v2i4 anchor's 0.7405 (or lower).
- If shuf-token does NOT drop, the canonical D1 lift was likely capacity-driven, not semantics-driven → NOT_PASS at CP-3.

**Control K2 — rand-token (collision-free construction; CP-1 round 1 fix)**.
For each drug-token edge in the canonical D1 parquet, **replace the token node ID with a globally unique synthetic ID `rand_<edge_idx>`** where `edge_idx` is the row index in the sorted parquet. Each replacement is unique, so **no two drugs ever share a synthetic token**. Edge count is preserved (1:1 row mapping); total token-vocab size is `n_total_edges` (~23k) instead of the canonical ~7800; relation IDs are preserved. The shuf-token control K1 verifies "does the binding matter"; K2 verifies "do the bridges (shared tokens between two drugs) matter, controlling for raw edge count". Expected: emergnn-branch returns to ≈ v2i4 anchor; combined drops more than K1.

(Original design proposed sampling 23k edges independently from a 8000-token pool, which would generate many collisions and preserve some bridges. Codex CP-1 round 1 flagged this as a major issue; fix above guarantees zero collisions.)

**Control K3 — drop-i4-head**.
Set `β_i4 init = 0, frozen` so the readout-side i4 head can't contribute. Forces the model to demonstrate that backbone integration alone carries the lift. Expected: combined ≥ v2i4 baseline (0.7804) if backbone genuinely absorbed the LLM signal; combined < 0.7804 means the backbone failed and the lift in canonical D1 was just the readout head doing its v2i4 job (D1 then offers no architectural novelty).

**Control K4 — drop-new-edges parity smoke (CP-1 round 1 add)**.
Activated via `--d1-disable`. Skips `_inject_llm_edges` entirely so the trainer is structurally identical to v2i4 (`n_ent / n_base_rel` unchanged). 5-epoch smoke run on seed42 must produce per-epoch loss trajectory matching v2i4 main run to numerical tolerance (mean abs diff per-epoch loss ≤ 1e-4). Purpose: empirical sanity that the override mechanism is non-invasive when disabled, complementing the K3 head-drop check. This is the empirical replacement for the original (incorrect) architectural worst-case claim in C3.

All four controls share the canonical D1 hyperparameters (length=3, n_dim=64, bs=32, lr=1e-3, wd=1e-8, S2 shuffle_train, ratio=0.8, 100 epochs unless K4, seed=42, β_count init=1.0, β_i4 init=0.5 unless K3). K4 is 5 epochs only.

---

## 4. Hard-stop rule (round4 plan §6, restated for this design)

Apply after the seed=42 100-epoch main D1 run completes:
- **combined < 0.775**. STOP. Open `Notes/Log/round4_d1_failure.md`, summarize observations, do not sweep, do not start K1/K2/K3.
- **0.775 ≤ combined < 0.785**. Run K1 (shuf-token) + one hyperparam sweep (probable axis: D1 token edges per drug capped at top-N most frequent for that field, to reduce open-vocab noise). Do not multi-seed.
- **combined ≥ 0.785**. Run K1+K2+K3 in parallel. If all three give the expected directional response (K1 down ≥1.5pt, K2 down ≥K1, K3 still ≥0.78), then trigger CP-3, and on CP-3 PASS, multi-seed seed=43, seed=44 same setting + K1.

---

## 5. Expected lift and risks

**Expected (hopeful)**.
- canonical combined: 0.788 – 0.793 (vs 0.7804 baseline).
- emergnn-branch: 0.755 – 0.770 (vs 0.7405). This is the primary architectural-diff evidence.
- count-branch (i2): roughly unchanged at 0.66 – 0.68 (LLM edges don't replace the 22-d count cache).
- i4-branch (readout): roughly unchanged at 0.59 – 0.62 in the K3=off setting (LLM edges don't replace the 13-d pair feature).
- K1 shuf-token: combined back to ~0.770 – 0.778 (≥1.5pt drop from canonical).

**Risks (rank-ordered, mitigation plan stated)**.

R1. **Vocabulary explosion in open-vocab fields** (`therapeutic_class`/`primary_targets`/`pd_effects`/`toxicity_mechanisms`/`clearance` together contribute ~7800 tokens with long tail). If most tokens appear for ≤1 drug, those tokens carry no inter-drug routing and just inflate n_ent. Mitigation: log per-field per-token degree at builder time; in CP-1 we will NOT yet cap the vocab, but if CP-3 result is borderline (0.775 – 0.785), the hyperparam sweep in §4 step (b) is to cap each open-vocab field to top-K most frequent tokens (K=200 default candidate).

R2. **Token-string normalization collision with existing 5-bucket nodes**. The 5-bucket KG node IDs are `db:enzyme:<HGNC>`-style structured identifiers (to be confirmed at builder time by inspecting `build_kg_from_kb` output via the existing kg_builder.py logic). The LLM tokens are free-text strings prefixed `llm:<field>:`. Prefix makes accidental collision impossible by construction — explicitly checked in the builder (assert `not any(node.startswith(("db:","het:","prime:")) for node in token_nodes)`).

R3. **Cold-start leakage via LLM tokens that name another drug** (R3 source-file + audit-strength fix, CP-1 round 1).
The two sanitizers (the prompt design in `distill_llm_pharmacology.py` + the 13-phrase sanitizer at `v2i4_trainer.py:128-141`) already strip interaction phrasing for the readout-side i4 head. For D1, backbone routing can amplify a leak into a long-distance influence (length=3 path: drug A → LLM-token-containing-drug-B-name → drug B), so a fresh audit is needed.

**Audit source-of-truth**. `Code/data/KG/drugbank/filtered/id2name.json` (1900 drugs → primary canonical name, verified 2026-05-30: `id2name n=1900`, sample entries `DB00006: Bivalirudin, DB00014: Goserelin, DB00027: Gramicidin D`). This replaces the previous incorrect citation of `drug_smiles__seed42.csv`, which has columns `drugbank_id, smiles` and **does not** carry a name column.

**Audit logic** (stronger than v1's equality-only check, per CP-1 round 1 codex feedback). For every LLM token (in every typed field, after sanitizer-2 strips interaction phrasing), the builder must:
1. Build a normalized-name set `N = { norm(name) : name in id2name.values() }` where `norm(s) = s.lower().strip()` with non-alphanumeric collapsed to single space.
2. For each token `t`, compute `norm(t)`. Reject (drop edge + log) the token if any of these hold:
   - exact equality with any element of `N`,
   - `norm(t)` contains a single-word drug name that is ≥ 6 characters AND `norm(t)` is itself ≤ 4× that name's length (avoids false positives on common short English words like "do" but catches e.g. "warfarin metabolism" or "co-administered with warfarin").
   - any element of `N` (length ≥ 6) is a substring of `norm(t)`.
3. Builder emits per-field rejected-token counts in the log and saves the full rejected token list to `Code/data/_cache/llm_edges/_audit_rejected_tokens__seed42_drugbank.json` for post-hoc inspection.

**v1 audit scope limit (explicit caveat)**. id2name.json contains only the primary canonical name. DrugBank brand names / synonyms (e.g. "Coumadin" for warfarin) are NOT in this audit's name set. If CP-3 result review suspects residual leakage, v2 of the audit must source synonyms from `Code/data/KG/drugbank/drugbank_with_mechanisms.csv` (141 MB) or from a DrugBank synonyms extract. For round 4 first pass we accept primary-name-only audit and flag the synonym gap to codex CP-2.

R4. **n_ent / morgan_mat size mismatch on save/load**. The base `_PerModeEmerGNN.save/load` (`_per_mode.py:620-734`) persists `n_ent`, `n_base_rel_with_ddi`, `entity2id`. We must store the same — extended — values. Inheritance via `_PerModeEmerGNN_V2I4 → _PerModeEmerGNN_MNAH → _PerModeEmerGNN` carries this for free as long as `_setup_graph` updates the same attributes. CP-2 will verify by running a save → load → predict_proba round-trip in a smoke test before the main run.

R5. **Edge count near-doubling slows training**. drugbank 5-bucket edge count to be measured at builder time (estimated ~10k–50k). LLM adds 23,227 directed drug→token edges (per builder-time projection from `i4_typed_sets.json` sums). Reverse-edge addition by `build_sparse_adj` will roughly double this. Per-epoch wall time may increase by 50%–2x. Acceptable as long as the seed42 main run completes in < 4 hours (v2i4 main was 1848.35s = ~31 min per `results.json:12`, so D1 budget is ≤ ~62 min target, ≤ 4 hours hard-stop).

R6. **Builder bug producing different output on re-run**. Mitigated by determinism contract (sort by `(drug_id, relation, token_node_id)` before parquet write, and seeded-only-when-control random). CP-2 will additionally hash the builder output (sha256) and store the hash in builder log so re-runs can be byte-compared.

---

## 6. File-by-file change-set (zero deletion, all-new-files)

```
NEW Code/my_code/models/screen_s2_v3_multimodal/precompute_llm_edges.py
NEW Code/my_code/models/screen_s2_v3_multimodal/v3_llm_edge_trainer.py
NEW Code/my_code/models/screen_s2_v3_multimodal/run_v3_llm_edge.py
NEW Code/data/_cache/llm_edges/llm_drug_edges__seed42_drugbank.parquet   (builder output)
NEW Code/data/_cache/llm_edges/llm_token_nodes__seed42_drugbank.json     (builder output)
NEW Code/my_code/models/screen_s2_v3_multimodal/_reviews/                 (created on first CP-1 archive)
```

Unchanged
- `baseline/emergnn/*` (all files, including kg_builder.py, _per_mode.py, model.py, shuffle_utils.py).
- `mnah_trainer.py`.
- `v2i4_trainer.py`.

---

## 7. CP-1 review questions for codex

Reviewer should give a verdict ∈ {PASS, PASS_WITH_NITS, NOT_PASS, FAIL} and call out each issue as critical / major / minor with file:line where relevant.

Q1 (C1). Does this design actually change the backbone, or is it a disguised readout-side hack? Specifically, is the only path from the new LLM tokens into `combined_logit` the EmerGNN `path-flow GNN` propagation? Verify against the trainer hook plan in §2.3.

Q2 (C2). Cold-start leakage. Are the per-drug intrinsic fields truly intrinsic, given the two sanitizers + the new R3 token-vs-drug-name audit? Are there any other leakage paths (e.g., a token like "metabolized by CYP3A4" that effectively encodes "this drug is a CYP3A4 substrate" — already in the data design as a feature, but is there any token that could encode pair-specific info)?

Q3 (C3, updated). The original architectural worst-case bound was wrong (codex CP-1 round 1). New C3 relies on (a) the inherited v2i4 readout head as soft floor, (b) the K4 drop-new-edges parity smoke that empirically asserts the override is non-invasive when disabled, and (c) the §4 hard-stop at combined < 0.775. Verify that those three together are sufficient empirical guard, and that the K4 implementation plan (5-epoch seed42 smoke with `--d1-disable`, mean abs diff per-epoch loss vs v2i4 ≤ 1e-4) is realistic given that v2i4 uses the same seed RNG order.

Q4 (controls). Is K1 shuf-token truly capacity-preserving? Is K2 rand-token's collision-free per-edge unique construction correct (zero cross-drug bridges by design)? Is K3 drop-i4-head the right ablation for "did backbone really absorb the signal"? Is K4 parity smoke the right empirical replacement for the old architectural worst-case bound? Is there a stronger / cheaper control I'm missing?

Q5 (data integrity). Drug-pool coverage gap = 19.4% (155/800 split drugs uncovered, 30/159 G2 drugs uncovered). Is this acceptable for a single experiment, or does this materially weaken the D1 evidence base?

Q6 (file independence). Does the `_PerModeEmerGNN_V3LLMEdge` inheritance plan violate the round4 plan §6 禁止改 list? Specifically, does overriding `_setup_graph` count as "modifying" the parent, or is it a clean override?

Q7 (numbers). Are all numbers / paths / file:line citations in this doc correctly verified, or have I made any unverified claims? (My own self-audit pass already corrected the merged-vs-drugbank-5-bucket KG claim from the round4 plan; see `Notes/Log/round4_d1_kg_source_deviation.md`.)

Q8 (hyperparameters). Should `length=3` be re-swept for D1? The added KG depth may benefit from longer paths; the v2i4 family already tried `--length 4` and `--length 5` (see runs `2026-05-30_01-54-08__run_v2i4__v2i4_len4__seed42` and `2026-05-30_02-33-58__run_v2i4__v2i4_len5__seed42` per existing log listings). Recommend whether D1 should fix length=3 (apples-to-apples vs 0.7804) or sweep.

Q9 (paper-story alignment). Does this design support the round4 plan §9 paper story bullet "我们识别了 LLM-distilled mechanistic edges 作为 KG 上 missing 的 evidence layer，把它当成新 relation type 直接进 path-flow", or do I need to add scaffolding now to make that claim later (e.g. pre-vs-post per-relation attention weights from EmerGNN, for the interpretability angle)?

Q10 (anything else). Catch-all for things I missed.

---

## 8. Post-review followup tracking

The design doc + the codex CP-1 verdict + every critical / major resolution will be archived to `Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-05-30__d1_design__round1.md` per round4 plan §8.5. Both Primary and Independent reviewer fields will be populated (Primary = Claude opus-4-7, Independent = codex gpt-5-codex).

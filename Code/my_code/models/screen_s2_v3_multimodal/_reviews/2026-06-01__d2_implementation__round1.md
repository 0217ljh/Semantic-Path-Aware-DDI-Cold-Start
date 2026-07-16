# CP-2 Implementation Review — D2 (Meeting-Node-Aware Propagation)

## Metadata

- **Date**. 2026-06-01
- **Scope**. CP-2 implementation review for Round 4 direction D2. CP-1 PASS_WITH_NITS at `_reviews/2026-06-01__d2_design__round1.md`.
- **Primary reviewer**. Claude (claude-opus-4-7).
- **Independent reviewer**. codex (codex MCP default model). Codex thread: `019e84b0-ba23-7142-b1eb-cc6c1b64f089`.
- **Final verdict**. **PASS_WITH_NITS** at codex round 4 (rounds 1–3 were NOT_PASS for save/load roundtrip completeness).
- **Decision**. Proceed to D2 seed42 main run with `--deterministic`.

## Critical CP-1 → CP-2 design deviation: Path B vocab pivot

Recorded in full at `Notes/Log/round4_d2_vocab_mismatch_discovery.md`.

CP-1 design §3.2 step 3: mediator(a, b) = `(n1∪n2)(a) ∩ (n1∪n2)(b)` over merged KG (178k entities, design §3.2 step 1).

CP-2 smoke test discovered:
- 98% of merged-KG mediator IDs are NOT in the trainer's drugbank 5-bucket entity vocab (~5,633 entities). The two ID namespaces don't overlap except for drugs.
- Path A (filter merged-KG mediators to drugbank vocab) yields zero non-drug mediators after Compound-leakage fix.
- **Path B (implemented)**: build `n1` directly over the drugbank 5-bucket bipartite KG via `build_kg_from_kb(...)`. `n2` is empty (bipartite schema). mediator(a, b) = `n1[a] ∩ n1[b]` over 5-bucket only.

Path B builder result (sha16=ac2b6b345f4264dd):
- 319,600 pairs, 0 leakage violations
- Cardinality: mean=0.57, median=0, p95=3, p99=5, max=26→20 post-topK
- zero_fraction=70.2% (70% of pairs have no shared 5-bucket entity)
- R1 not triggered post-topK

Codex CP-2 confirmed Path B is "acceptable as a CP-2-time disclosure, not a mandatory CP-1 redo, provided the CP-2 report clearly says CP-1 §3.2 merged-KG mediator universe was abandoned due to vocab mismatch and Path B is the implemented hypothesis."

The design intent — pair-conditional propagation bonus on shared mediators — is preserved. Only the mediator universe definition changes.

## Reviewed code (4 new files; no edits to forbidden files)

| File | Purpose |
|---|---|
| `Code/my_code/models/screen_s2_v3_multimodal/precompute_meet_mediators.py` | Path B builder. Output `Code/data/_cache/meet_mediators/meet_mediators__seed42_drugbank__topk20.parquet` (sha16=ac2b6b345f4264dd). |
| `Code/my_code/models/screen_s2_v3_multimodal/emergnn_meet_mask.py` | `EmerGNNWithMeetMask(EmerGNN)` subclass. Adds per-layer `alpha_meet` scalar + `meet_mask` kwarg to `_propagate` and `forward`. |
| `Code/my_code/models/screen_s2_v3_multimodal/v3_meet_mask_trainer.py` | `_PerModeEmerGNN_V3MeetMask(_PerModeEmerGNN_V2I4)`. Override `_combined_logit`, `fit()`, `save()`, `load()`. Controls K1 (cardinality-bucketed derangement), K2a (currently falls back to K2b uniform), K2b (uniform random non-drug), K3 (alpha_meet=0 frozen), K4 (--d2-disable). |
| `Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py` | CLI entry point. Adds `--deterministic` flag for torch+np+cuda seeding (CP-1 round 2 process improvement). |

## Smoke test (1 epoch, Path B cache, --deterministic)

Run dir: `Code/runs/2026-06-01_15-35-30__run_v3_meet_mask__d2_smoke_pathB__seed42`.

- mediator-cache summary: `n_dropped_mediators_out_of_vocab=0` ✓ (vocab aligned).
- cardinality_mean_post_mutation = 0.57 (matches builder summary).
- 1-epoch val_combined=0.6909, val_emer=0.6794, test_s2 AUC=0.7009.
- alpha_meet end of ep1 = [0.022, 0.006, 0.0] — learning non-zero.

## Codex independent verdict — round-by-round timeline

### Round 1 (NOT_PASS): inherited load() hardcodes EmerGNN, not subclass

> Major: `save()` will include `alpha_meet`... but inherited `load()` hardcodes `EmerGNN`, not `EmerGNNWithMeetMask`, at `Code/baseline/emergnn/_per_mode.py:714`. Then `load_state_dict(..., strict=True)` should fail with unexpected key `alpha_meet`.

### Round 2 (NOT_PASS): load() rebuilds backbone only, not v2i4 aux+beta+norm stats

> The new `load()` reconstructs only `_model`; it never rebuilds or loads `_aux_mlp`, `_raw_beta`, `_feat_lookup`, `_feat_matrix`, `_feat_mean/_feat_std`, or `_i4_mean/_i4_std`. `predict_proba()` inherited from MNAH will fail.

### Round 3 (NOT_PASS): persisted aux+beta+norm-stats but missed _feat_lookup + _feat_matrix

> `_lookup_features()` immediately dereferences `self._feat_lookup.get(...)` and then indexes `self._feat_matrix`... still leaves `_feat_lookup` and `_feat_matrix` as `None` from `__init__`. The first `predict_proba()` call should therefore fail.

### Round 4 (PASS_WITH_NITS)

> The roundtrip blocker is fixed. `predict_proba()` after load now has the required pieces: `EmerGNNWithMeetMask` + `alpha_meet` state, eval edges, aux head state, `_raw_beta`, count feature lookup/matrix, count normalizer, and i4 normalizer. K3 freeze also survives because `freeze_alpha_meet` is passed into the subclass constructor before strict `load_state_dict()`.
>
> Nits only:
> - `d2_config.json` is written but not directly read; the manifest hparams are the actual source of load config.
> - The loaded model still depends on `d2_mediator_parquet` existing at the saved path unless `d2_disable=True`. Artifact portability limitation.
> - K2a remains K2b semantics by disclosure, not a true kind-matched control yet.
>
> No critical or major issues remain for CP-2.

## Issues + resolutions

| Severity | Issue | Resolution | Round |
|---|---|---|---|
| major | `load()` hardcoded `EmerGNN`, strict load_state_dict fails on `alpha_meet` | Override `load()` to construct `EmerGNNWithMeetMask` with `alpha_meet_init` + `freeze_alpha_meet` | R1 |
| major | `load()` didn't restore v2i4 aux head + raw_beta + normalizer stats | Override `save()` to persist `aux.pt`, `beta.pt`, `normalizer.pkl`; override `load()` to restore | R2 |
| major | `load()` still missed `_feat_lookup` + `_feat_matrix` | Add `feat_lookup` (dict) and `feat_matrix` (numpy, restored to device tensor) to normalizer.pkl | R3 |
| minor | `d2_config.json` written but not used in load() | Documented as redundant artifact; manifest hparams are the source of truth | R4 |
| minor | Loaded model depends on `d2_mediator_parquet` path | Documented as portability limitation | R4 |
| minor | K2a fallback to K2b semantics | Already disclosed in trainer log via `print("[d2] *** K2a RAND-KIND-MATCHED *** CP-2 round 1 falls back to uniform...")` | R1 disclosed, accepted at R4 |

## Unresolved / followup flagged

- **K2a kind+degree matching deferred**. CP-2 confirmed acceptable for round 1; CP-3 will only interpret K2a runs as "K2b-equivalent harsher control". Real K2a requires wiring id2kind cache into trainer.
- **Path B sparse mask reality**. 70% zero-mediator pairs is a real cold-start sparsity. CP-3 will need to report cardinality-stratified AUC to discriminate "where D2 has signal" vs "where mask is null".
- **K4 numerical parity**. Codex CP-2 round 1 noted determinism is "for reproducibility attempts, not a strict <1e-4 CUDA guarantee". Will operate in practice but not bit-perfect.

## Next step

Launch seed42 100-epoch D2 main: `python -u Code/my_code/models/screen_s2_v3_multimodal/run_v3_meet_mask.py --epochs 100 --tag d2_main_seed42 --seed 42 --deterministic`. Hard-stop check + controls + CP-3.

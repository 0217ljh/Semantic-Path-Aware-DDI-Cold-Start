# NBFNet v1.72 — dual-source signed-bilinear interference: implementation review

## Meta
- **Date**. 2026-06-06
- **Primary reviewer**. Claude (claude-opus-4-8) — designed (with codex), implemented, verified.
- **Independent reviewer**. codex (gpt-5.4, xhigh, MCP) — design rounds (threads `019ea444`/`019ea581`/`019ea597`) + implementation review (thread `019ea8ab`).
- **Trigger**. User. build the simplest dual-ignition interference NBFNet, following the codex-review workflow.

## What v1.72 is
v1.71 backbone (NBFNet on merged KG; AMP/TF32/eval-cadence; dual-source batched BF) +
**signed fields** + **action-signed edges** + **endpoint Hadamard interference readout**.
The interference core = signed+hadamard; a built-in 2×2 (activation × combine) with
action held ON enables clean attribution. v1.7 / v1.71 untouched.

## Files
- `Code/my_code/models/nbfnet_v1_72/{__init__,nbfnet_model,nbfnet_trainer}.py`
- `Code/scripts/run_nbfnet_v1_72.py` (incl. action edge-sign builder + PD-slice eval)

## Design (4 necessary additions, codex-blessed)
1. Edge-sign canonicalization: inverse edges inherit SAME sign; drug-target conflicting actions → neutral(+1); missing/neutral → +1.
2. Shared endpoint RMSNorm for u and v.
3. Signedness diagnostic (neg-coord fraction of u,v on eval).
4. Eval = overall + PD slice + non-PD; PD slice strictly eval-only.

## Verifications (mine, before review)
- Sign builder (merged KG): 319,303 edges, 27,521 neg-signed; **DB00006→Prothrombin(inhibitor) = −1** ✓; CuG=+1/CdG=−1; 235 conflicts→neutral; PD pairs=21,686. id-alignment needed stripping the `db:target:` prefix (KG dst `db:target:BE…` vs drug_targets `BE…`); after strip, (drug,target) pairs align 1:1 (7292).
- Toy 2×2 (all 4 cells) run end-to-end; signedness diag signed≈0.5 / relu=0.

## Codex implementation verdict (verbatim, key parts)
> **NEEDS-FIX** — The only real blocker is the `relu` control no longer being architecturally non-negative after the shared endpoint norm. `endpoint_norm` is a shared `nn.RMSNorm(d)`; its learnable weight can flip signs, so `diag=0` is empirical, not enforced. If the 2×2 attribution depends on "relu = non-negative control", fix this first.
> - Addition 1 correctly wired (missing/neutral→+1, conflict→neutral; inverse inherits same sign in model and scorer).
> - Addition 2 implemented structurally (one shared endpoint_norm for u and v); caveat is its sign behavior.
> - Addition 3 implemented for validation-time eval (reset+print in `_validate`); not printed at final test predict_proba, but eval path correct.
> - Addition 4 implemented; fit/early-stop uses only overall val; PD/non-PD post-fit; no PD-slice tuning leakage.
> - `_build_epoch_kg` sign-alignment assumption is SAFE: parent eval KG order `[train_ddi, base_kg]`; `shuffle_train` returns `[fact, train_kg]`; parent preserves row order when tensorizing.
> - `_cur_training` routing correct; validation and inference call `_score_pairs(...,training=False)` → pick `_eval_edge_sign`; no stale-sign path.
> - No cold-start node-id leakage (no node-id embeddings; only relation/query params).
> - "For the full 2×2 attribution run, I would fix the relu-cell non-negativity issue first, then call it GO."

## Fix applied
`endpoint_norm = nn.RMSNorm(d, elementwise_affine=False)` (drop learnable γ).
Affine-free RMSNorm = x/rms preserves input sign → relu fields (≥0) stay ≥0,
signed fields keep signs. Verified: relu-input neg_frac=**0.0**, signed-input ~0.5.
→ relu 2×2 control now architecturally non-negative. **Post-fix verdict: GO.**

## Open / not-blocking
- Signedness diag is val-only (not printed at final test predict_proba). Acceptable; val diag confirms sign use.
- Amortized fallback path (`_score_pairs_amortized`) is v1.7 additive/non-signed; v1.72 raises if a maskable batch is seen (never in S2). Documented.
- Scope: this tests the interference OPERATOR (signed×hadamard vs additive) WITH action; it does NOT by itself prove "captures synergy/antagonism" — that needs the PD-slice gain to concentrate + the action on/off ablation.

## Reproduce / run (training = user-run)
- signed stability smoke (AMP off, 5 epoch): `run_nbfnet_v1_72.py --epochs 5 --no-amp --activation signed --combine hadamard --tag v172_smoke`
- 2×2 cells: `--activation {relu,signed} --combine {additive,hadamard}` (action ON).

# NBFNet v1.7 — Paper-Faithful Trainer Implementation Review

## Meta

- **Date**. 2026-06-05
- **Primary reviewer**. Claude (claude-opus-4-8) — drafted this report, applied the Round 6 fixes, wrote + ran the smoke test.
- **Independent reviewer**. codex (gpt-5-codex, MCP) — Round 7 verdict (verbatim below). The original Round 1-6 thread `019e95e4-...` had expired ("Session not found"), so Round 7 ran in a fresh codex thread `019e990a-8fe2-79d0-9904-6cb169030ef9` with the full patched code pasted verbatim.
- **Trigger**. User task. complete NBFNet v1.7 trainer per `Notes/Log/nbfnet_v1_7_handoff.md` (apply Codex Round 6 Fix 1-5, add union-mask model method, smoke test, Round 7 review).
- **Scope**. trainer + one additive model method. The model module `nbfnet_model.py` already held Round 5 GO and was NOT otherwise modified.

## Target

Vanilla NBFNet (Zhu et al., NeurIPS 2021, generalized Bellman-Ford) for cold-start
S2 binary DDI on the 800-drug DrugBank split. Independent branch (NOT integrated
with PMP C1/C2/C3).

## Files reviewed / changed

- `Code/my_code/models/nbfnet_v1_7/nbfnet_model.py` — **one additive method** `build_union_query_edge_mask` (no existing Round-5-GO function touched).
- `Code/my_code/models/nbfnet_v1_7/nbfnet_trainer.py` — Fixes 1-5 applied (file under review).
- `Code/scripts/run_nbfnet.py` — new CLI runner (KG build + RunLogger + eval).

Reference files read for plumbing correctness:
- `baseline/emergnn/shuffle_utils.py:27` — `shuffle_train(train_ddi, train_kg, setting, *, ratio, rng, extra_kg_ent)`, `[h, t, r]` triplets.
- `baseline/emergnn/kg_builder.py:59` — `build_kg_from_kb` returns `triplets` as `(h, t, rel)` with `N_BASE_REL=5` (rels 0-4: target/enzyme/transporter/carrier/pathway).
- `baseline/emergnn/_per_mode.py:347` — EmerGNN's shuffle_train integration + eval KG = train_ddi + base_kg (the reference this trainer mirrors).

## Round 6 fixes — applied status

| Fix | What | Where (trainer) | Status |
|---|---|---|---|
| 1 | `_build_epoch_kg` calls `shuffle_train`, returns `(epoch_edges, epoch_targets)`; `setup_graph` stores `base_kg_triplets` + `train_ddi_triplets`; does NOT use `build_edge_lists_from_triplets` | `setup_graph`, `_build_epoch_kg` | ✅ |
| 2 | `_train_epoch` per-source amortization (`forward_cache`/`reverse_cache`) via `build_union_query_edge_mask` | `_train_epoch`, `_score_pairs_amortized` | ✅ |
| 3 | rename `score_pairs_from_source` → `encode_targets_from_source` + corrected docstring (returns hidden states, not logits) | `encode_targets_from_source` | ✅ |
| 4 | `_should_early_stop` with `best_epoch` tracking | `fit`, `_should_early_stop` | ✅ |
| 5 | checkpoint persistence to `run_dir/best_model.pt` + negatives sampled against epoch_targets emerging-drug pool | `fit`, `_sample_negatives` | ✅ |

Plus: new additive model method `build_union_query_edge_mask` (vectorized union of
per-target `build_query_edge_mask`; user approved adding it to the otherwise-frozen
model module).

## Real-run KG config (verified on seed42 dataset)

`n_ent=5633`, `n_base_rel=6` (5 DrugBank KG buckets + 1 DDI slot), `ddi_rel_id=5`,
model doubles to 12 relations for inverse edges. `n_base_kg_edges=18792`,
`n_train_ddi=53743`. (Printed by `run_nbfnet.py` / `setup_graph`.)

## 8 architectural invariants — held

1. No drug embeddings in score head — head = `mlp_head(cat([h_sym, query]))`. ✅
2. drug_a = INDICATOR anchor (`h_a^0=q`) — model unchanged. ✅
3. Boundary reinjection every layer — model unchanged. ✅
4. Per-(layer, relation) transforms — model unchanged. ✅
5. Inverse-edge augmentation (not generic self-loops) — `augment_inverse_edges`, and Fix 1 deliberately avoids `build_edge_lists_from_triplets`. ✅
6. Query-edge masking relation-aware (only `(a, r_ddi, b)` + inverse) — `build_union_query_edge_mask` restricts removal to `ddi_rel_id` and `ddi_rel_id + n_base_rel`. ✅
7. Symmetrize at representation level before MLP — `h_sym = forward_cache[a][b] + reverse_cache[b][a]`. ✅
8. MLP input `concat([h_q_sym, q])`. ✅

## Smoke test (toy 14-node KG, d=8 L=3, GPU) — ALL PASS

1. forward + backward finite (logit=0.1161, loss backprops).
2. eval determinism — two identical eval calls `allclose`.
3. `shuffle_train(S2)` — per-epoch targets differ (epoch1=2, epoch2=3).
4. union mask removed exactly 4 edges for src=0 tgts={1,3} (2 fwd + 2 inv) = intersection of per-target masks.
5. neg sampling — 5 negs, none overlap positives.
6. fit — runs, early stop fired at ep3 (best ep1, patience2: 3−1≥2), `best_model.pt` persisted.
7. predict_proba in [0, 1].

(Smoke test was run as a throwaway script in `/tmp`-equivalent and removed; not a repo artifact.)

## 1-epoch sanity (real 800-drug seed42, GPU) — passes

`val_auc=0.6950 val_ap=0.7370` after epoch 1; `test_s2 AUC=0.7069 AUPRC=0.7444
NLL=0.6086`; per-epoch wall ≈ **321s** (so 100 epochs ≈ 9h, longer than the
handoff's 3-5h estimate; early stop patience 10 will likely cut it).

## Codex Round 7 independent verdict (verbatim)

> **Verdict**
>
> `GO` from the provided excerpts. I do not see a blocking faithfulness or correctness defect in the trainer path or in the additive `build_union_query_edge_mask` method.
>
> **Critical Review Result** — No critical items found that would corrupt a `seed=42`, `100`-epoch, `S2` cold-start run under the stated setup.
>
> **Fix 1-5 Status** — Fix 1 correct (`setup_graph`/`_build_epoch_kg`, avoids `build_edge_lists_from_triplets`). Fix 2 correct (`_train_epoch`/`_score_pairs_amortized`, augment once per epoch, per-source caches, training-time union mask, eval `training=False`). Fix 3 semantically correct (trainer consumes hidden states from `encode_from_source`). Fix 4 correct (`best_epoch` updated on improvement; `(epoch - best_epoch) >= patience`). Fix 5 correct (best ckpt persisted; negatives from `pool = np.unique(epoch_targets[:, :2])`, excludes both directions).
>
> **build_union_query_edge_mask** — correct and equivalent to the intersection of single-target `build_query_edge_mask` results. Edge cases checked: `ddi_rel_id is None` → all-True; targets device/dtype via `torch.as_tensor`; empty targets → `isin` all-False → keep all; duplicate targets harmless; `torch.isin` semantics correct.
>
> **8 Invariants** — no violation visible.
>
> **Seed42/100ep/S2 risks** — none blocking. Eval KG = train_ddi+base_kg correct; scheduler step placement correct; best-ckpt persistence correct; gradient flow through amortized caches correct (cached tensors retain graph; single `loss.backward()` propagates through reused source encodings); masking not applied at eval (correct).
>
> Non-blocking notes: (a) `_sample_negatives` may under-fill if pool tiny/dense — changes class ratio, not correctness (we already log a warning). (b) cache is memory-heavy (each unique source holds `(n_nodes, d)` with grad until backward) — scaling concern, not a bug. (c) verify `best_state` is a deep copy.
>
> Overall: `GO`.

## Resolution of Codex non-blocking note (c)

Confirmed directly: `fit` stores `best_state = {k: v.cpu().clone() for k, v in
self.model.state_dict().items()}` — `.clone()` is a true deep copy, and the run
also persists `best_model.pt`. Safe.

## Open / future items

- Per-epoch wall ~321s → full 100ep ~9h. Consider a smaller eval cadence or larger
  batch if iteration speed becomes limiting (does not affect correctness).
- Training logging is manual print (per-step rolling mean + per-epoch summary),
  matching the skeleton; not routed through `my_code.utils.train_progress.TrainProgress`.
  Acceptable for this model branch; flag if standardization is desired later.
- `load_state` does not restore `_eval_edges` (inference after a cold `load()`
  without re-`setup_graph` would lack the eval KG). Not exercised by the first run
  (predict happens in-process right after fit). Documented.

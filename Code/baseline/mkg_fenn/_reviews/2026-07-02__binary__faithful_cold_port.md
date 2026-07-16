# MKG-FENN binary (DDI-existence) — faithful cold-start port acceptance (Case-B)

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8)
- **Independent reviewer**: codex (gpt-5.2-codex; review thread 019f247f; ambiguity thread 019f2479)
- **Trigger**: user — every baseline needs all 3 tasks; MKG-FENN binary after multiclass.
- **Scope**: NEW `binary_cls/{baseline.py, baseline_unified.py, __init__.py}` + additive runner
  registration. Reuses the multiclass port's infrastructure. multi_cls/*, model.py, kg_builder.py,
  dead top baseline.py UNTOUCHED.

## Task framing (Case B)
MKG-FENN paper = 65-event multiclass. Binary DDI-existence is a Case-B adaptation: reuse the paper's
ALGORITHM CORE (4-channel warm / 3-channel cold + nearest-seen imputation) UNCHANGED; swap only the
task surface (2-way head, per-epoch negatives, AUPRC). The cold-start mechanism is task-agnostic and
is KEPT for binary cold (CLAUDE.md case-B "不允许借口新任务省略 paper 核心") — the dead ColdDDI
binary adapter WRONGLY used the 4ch warm model for cold; this port does NOT repeat that.

## Design + implementation
- `MKGFENNBinaryBaseline` (register("mkg_fenn_binary")): REGIME-AWARE — S0 (transductive) ->
  `baseline.mkg_fenn.model.MKGFENN(event_num=2)` 4ch; S1/S2 (inductive) ->
  `baseline.mkg_fenn.multi_cls.model_cold.MKGFENNCold(event_num=2)` 3ch + the SAME drug_sim1..4 +
  test_adj nearest-seen imputation (baseline.py:264-283 cold plumbing, :285-297 model select).
- REUSE (import, not duplicate) the multiclass pure helpers find_dif / jaccard /
  _build_feature_matrices / _build_test_adj / _ghost_pad_kg1 / _drug_smiles_dict from
  multi_cls/baseline.py (codex confirmed no import-time side effects / global state). Model classes
  MKGFENN / MKGFENNCold + kg_builder.build_all_kgs reused unchanged. Single source of truth, no drift.
- Binary head: event_num=2; CrossEntropyLoss on {pos=1, neg=0} in the core (:304, raw logits);
  predict_proba -> (n,) = softmax(logits)[:,1] once (:451). No double-softmax.
- Negatives: per-epoch DETERMINISTIC via leaf adapter get_train_negatives(epoch, regenerate=True)
  (design B), re-fetched each epoch (:328); symmetric aug (a,b)+(b,a) applied to BOTH pos+neg (:331-340).
- Best-ckpt: VAL AUPRC over {val pos + val neg}, labels aligned pos-then-neg (:416-427). OOV/missing-
  drug rows: predict_proba(_fill_oov=np.nan) drops NaN before AUPRC at val (no filler pollution,
  :420-426); _fill_oov=0.5 at test time so the runner scores every row (:433-454).
- Kept task-agnostic paper protocol: deterministic seeding + determinism flags, per-epoch shuffle +
  fresh negatives, Adam+CE no scheduler, TrainProgress. save/load persist dict1/kgs/event_num=2.
- Wrapper: register_unified("mkg_fenn"), task="binary"; KG1 from native Code/data/KG/drugbank/filtered
  via KnowledgeGraph.from_filtered_dir (baseline_unified.py:88-94, NOT resources.kg.source); cold flag
  from resources.meta["regime"] (:97-99); make_dataset(task="binary") (:133-135); predict -> (n,)
  aligned (:137-140).
- Device safety inherited from the fixed model_cold (.to(embed.device), idx.cpu().numpy()) + binary
  eval moves batches to self.device (:449-451). No new CPU-tensor/CUDA-numpy bug.

## Codex verdict: APPROVED (no findings)
codex independently verified all 7 checkpoints (regime switch, binary head/loss, per-epoch
deterministic negatives actually refreshed, val-AUPRC OOV NaN-drop + aligned labels, device safety
intact, native KG1 + (n,) shape + KG3 self-loop for unseen, import isolation with no import-time
mutation). "The agent's claims about cold-core reuse, epoch-wise negative refresh, val-AUPRC NaN
dropping, and test-time 0.5 OOV fill are all true in the checked source." Source review only, no run.

## Verification
- py_compile OK; import+registry smoke: get_unified("mkg_fenn","binary") -> MKGFENNUnifiedBinary.
- Offline cold-path plumbing (agent, no training) on deng cold-S2 fold0 (subsampled 55 drugs/15 unseen/
  200 pos): KGs built, test_adj (4 ch, 15 unseen keyed), deterministic negs (200 for epoch 0 and 1),
  one cold eval forward -> (20,) finite float32 in [0,1]. (No training: GPU busy; deferred.)

## Decision: ACCEPTED (simplified gate) — core idea + cold mechanism preserved
MKG-FENN: multiclass + binary done. Multilabel (TWOSIDES 200-label) still TODO.

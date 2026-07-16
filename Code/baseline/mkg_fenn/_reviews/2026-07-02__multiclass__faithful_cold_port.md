# MKG-FENN multiclass (65-event) — faithful cold-start port acceptance

- **Date**: 2026-07-02
- **Primary reviewer**: Claude (opus-4.8) — verified the cold mechanism against official source directly
- **Independent reviewer**: codex (gpt-5.2-codex; design 019f23fe, review 019f2461, ambiguity 019f2455/019f245c)
- **Trigger**: user — every baseline needs all 3 tasks; MKG-FENN's paper task (multiclass) first,
  user chose the FAITHFUL cold-start path ("忠实方案").
- **Scope**: NEW `multi_cls/{model_cold.py, baseline.py, baseline_unified.py, __init__.py}` + additive
  runner registration. Dead top `baseline.py` (binary, coldddi imports), `model.py`, `kg_builder.py`
  UNTOUCHED. Full design spec: `Notes/Log/mkg_fenn_faithful_cold_port.md`.

## Task framing (case A — same task as paper)
MKG-FENN paper = 65-event MULTICLASS softmax CE (task1.py:39 event_num=65, :175 CrossEntropyLoss,
:206 softmax@eval). task1/2/3 = warm / one-unseen / both-unseen SPLIT scenarios. Our unified S0/S1/S2
map to those, and — critically — the official cold models DIFFER from the warm model.

## The faithfulness crux (verified from official source, not from the migrated repo)
The three official models have distinct md5. diff modeltask1 (warm) vs modeltask3 (cold):
1. **Cold = 3 channels, not 4.** Cold net = `nn.Sequential(GNN1, GNN2, GNN3, FusionLayer)` — GNN4
   (molecular-property) NOT in the chain (task3.py:400-403). FusionLayer concats gnn1/gnn2/gnn3 ->
   `embedding_num*3*2 -> *3 -> *2 -> event_num` (modeltask3.py FusionLayer).
2. **Cold has nearest-seen-neighbour embedding imputation** (warm has none): for each unseen test
   drug, per channel, `drug_f[i] = mean(drug_f[nearest_seen])` when train_or_test==1.
   Channel->test_adj index: GNN1<-test_adj[0] (modeltask3.py:53), GNN2<-test_adj[1] (:120),
   GNN3<-test_adj[3] (:187). (GNN4<-test_adj[2] but GNN4 unused.)
3. **Per-channel similarity** (task3.py:291-319): fm1<-dataset1(KG1), fm2<-dataset2(KG2),
   **fm3<-dataset4(KG4 property)**, **fm4<-dataset3(KG3 DDI)** — the upstream fm3<->KG4 / fm4<->KG3
   SWAP is real. drug_sim1/2/4=Jaccard, drug_sim3=find_dif. test_adj[k] built from drug_sim{k+1}.
   The double-swap (fm3/fm4 AND test_adj[2]/[3]) cancels so each GNN imputes from its OWN modality.
The repo `model.py` is ported from modeltask1 (4ch, no impute) -> faithful for S0 ONLY.

Primary reviewer (Claude) INDEPENDENTLY re-verified points 1-3 against modeltask3.py + task3.py
(task3.py:400-403 net; :291-319 fm/sim; :344-395 test_adj; modeltask3.py:53/120/187 impute indices).
An earlier truncated-diff misreading by the primary reviewer was WRONG; the agent's port was RIGHT.

## Design + implementation
- `model_cold.py` — cold 3-channel model ported verbatim from modeltask3 (GNN1/2/3 + FusionLayer, no
  GNN4 in chain; mean-of-nearest-seen imputation gated on train_or_test; GNN attention body verbatim,
  only 572->n_drug and 65->event_num parametrized per CLAUDE.md case-A).
- `baseline.py` `MKGFENNMulticlassBaseline` — REGIME-AWARE: S0 uses the existing 4ch
  `baseline.mkg_fenn.model.MKGFENN` (imported unchanged); S1/S2 uses `MKGFENNCold` + drug_sim1..4
  (fm3<->KG4/fm4<->KG3 swap preserved, baseline.py:128) + test_adj. event_num = #train-observed
  ddi_type (re-vocab, NOT 65). CrossEntropyLoss. symmetric (a,b)+(b,a) aug. per-epoch shuffle.
  best-ckpt by VAL macro-F1 (deliberate deviation from task3.py's TEST selection, to avoid leakage).
  TrainProgress per-epoch loss+eval. predict -> (n, K_train). Reuses kg_builder.build_all_kgs +
  _build_dict1 (all drugs reachable). KG1 ghost-pad for empty-annotation drugs applied to the model's
  KG ONLY; similarities built from UN-padded KG1 (codex-confirmed Option A, thread 019f245c).
- `baseline_unified.py` — register_unified("mkg_fenn"), task="multiclass". KG1 from
  `KnowledgeGraph.from_filtered_dir(Code/data/KG/drugbank/filtered)` (native, NOT resources.kg.source).
  make_dataset(task="multiclass") (ddi_type=y_cls). cold flag from resources.meta["regime"]
  (inductive->cold, transductive->warm). predict scatters (n,K_train)->(n,n_labels_global) via
  _idx_to_ddi_type (EmerGNN pattern); OOV-in-train gold -> ~0 mass -> runner oov_target_rate.

## Codex verdict: CHANGES_REQUIRED -> fixed -> APPROVED_WITH_NITS
codex independently confirmed faithful: 3-channel chain (task3.py:400), impute indices
(modeltask3.py:53/120/187), fm3<->KG4/fm4<->KG3 swap (baseline.py:128), CE + sym-aug + per-epoch
shuffle + val-macro-F1 (baseline.py:417/430/451/478), native KG1 + regime flag + scatter
(baseline_unified.py:88/98/136/143), import isolation (top __init__ __all__=[] safe).
- **FINDING 1 (CHANGES_REQUIRED, real bug) — FIXED**: model_cold.py used `torch.LongTensor(...)`
  (CPU) at GNN forwards + `idx.numpy()` in FusionLayer -> GPU predict crash (core moves eval batch to
  self.device). Fixed: index tensors `.to(self.drug_embed.weight.device)` (3 GNN forwards) +
  `idx.cpu().numpy()` (FusionLayer). Device-only fix, algorithm unchanged.
- **NIT 2 — FIXED**: `_seed_all` lacked upstream's determinism flags (task3.py:54-57). Added
  `torch.use_deterministic_algorithms(True, warn_only=True)` (warn_only = faithful-equivalent; hard
  True can crash on this torch/data or need CUBLAS_WORKSPACE_CONFIG) + cudnn.deterministic=True /
  benchmark=False / enabled=False. Per-leaf runner = one process per baseline, so no global-state leak.
- Re-smoke after fixes: py_compile OK; `get_unified('mkg_fenn','multiclass')` -> MKGFENNUnifiedMulticlass.

## Verification
- Offline plumbing (agent, no training) on real S2 leaf (ddi_full, 1900 drugs, 760 unseen, 215 global
  classes): KG1 loaded; build_all_kgs tail_len {ds1:2904, ds2:513, ds3:1900, ds4:6}; fm1..4 shapes
  ok; sims (1900,1900); test_adj nearest-seen excludes self+unseen; ghost-pad KG1 tail 2904->2905;
  cold forward train (6,K) + impute-eval (1,K) all finite. (No training run: GPU busy; deferred.)
- Caveat (upstream-faithful): Jaccard 0/0->NaN for all-zero feature rows is upstream's unguarded
  behavior (task3.py:247-251); NaN rows never selected as nearest-seen (NaN fails v>max_v). Preserved.

## Decision: ACCEPTED (simplified gate) — core idea + cold mechanism preserved
MKG-FENN multiclass done (S0 4ch / S1S2 3ch+impute, faithful). Binary + multilabel still TODO.

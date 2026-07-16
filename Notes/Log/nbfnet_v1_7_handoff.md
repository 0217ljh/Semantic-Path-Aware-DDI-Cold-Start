# NBFNet v1.7 Handoff Spec (2026-06-05)

**Purpose**. Self-contained handoff for new session to complete NBFNet v1.7 paper-faithful implementation. Previous session completed model module (codex Round 5 GO) + trainer skeleton (codex Round 6 NEEDS REFACTOR). This doc captures all codex review intelligence + prescribed fixes + remaining work, so new session接手时不需要重新 derive.

## Quick context

- **Target**. v1.7 = vanilla NBFNet (Zhu et al. NeurIPS 2021, arxiv 2106.06935) for cold-start S2 binary DDI on DrugBank 800-drug
- **Independent branch**. NOT integrated with PMP (C1/C2/C3) for v1.7
- **Expected AUC**. 0.772-0.783 (codex 保守-乐观估计), 不期待破 v2i4 0.7804 ceiling
- **Reference baselines**. EmerGNN 0.7458, MNAH 0.7670, v2i4 0.7804, v1.5A 0.7764

## Current state of code

### ✅ Completed (codex Round 5 GO)

**`Code/my_code/models/nbfnet_v1_7/__init__.py`** — package init, exports NBFNetDDI / NBFLayer / PNAAggregator

**`Code/my_code/models/nbfnet_v1_7/nbfnet_model.py`** — paper-faithful model module. **Do NOT modify without re-passing codex faithfulness review (codex thread 019e95e4 Rounds 1-5).**

Faithful elements (all verified):
- L=6 layers, d=32 hidden, mlp_hidden=64 (paper defaults)
- `PNAAggregator` (per-layer, NOT shared across layers) with 4 aggs × 3 scalers
- `NBFLayer` with per-(layer, relation) `W_r^(t) ∈ (n_rel, d, d)`, `b_r^(t) ∈ (n_rel, d)`
- `NBFNetDDI` with:
  - INDICATOR boundary `h_v^(0) = q if v=source else 0`
  - DistMult MESSAGE `m = h_x * w_q(r)` where `w_q(r) = W_r^(t) @ q + b_r^(t)`
  - PNA AGGREGATE with **boundary reinjection at every layer** (h0 appended as self-message via scatter)
  - Inverse-edge augmentation `(h, r, t) -> + (t, r+n_base_rel, h)`
  - Query-edge masking (relation-aware, removes only DDI relation edges between (a, b))
  - Representation-level symmetrization `h_q_sym = h_q(a,b) + h_q(b,a)`
  - MLP head input `concat([h_q_sym, q])` (official-repo convention)
  - ReLU after each layer aggregate
  - `encode_from_source(source, ..., already_augmented, edge_keep_mask)` — clean API for trainer
  - `score_pair(...)` returns **logit** (not sigmoid probability), apply BCE-with-logits externally

### ⚠ Skeleton with critical placeholders (codex Round 6 prescribed fixes)

**`Code/my_code/models/nbfnet_v1_7/nbfnet_trainer.py`** — fresh trainer (NOT _PerModeEmerGNN subclass per codex Round 5 recommendation).

**Sound parts**.
- Architecture skeleton (setup_graph → init_model → fit)
- `score_pair_symmetric` correctly implements 2-pass BF + symmetrize + MLP head
- `init_model` creates NBFNetDDI + Adam + ReduceLROnPlateau correctly
- `setup_graph` stores edge tensors on device

**5 critical placeholders that MUST be fixed before training run**.

### Codex Round 6 prescribed fixes (verbatim)

#### Fix 1. `_build_epoch_kg` must integrate `shuffle_train` and return epoch_targets

**Current**. Returns static KG, no per-epoch resampling. **Breaks S2 faithfulness** because epoch positives stay static instead of using shuffle_train's emerging-drug resampling.

**Required change**.

```python
# 1. setup_graph should store BOTH torch tensors AND raw numpy triplets:
#    self.kg_edge_src/dst/rel (torch)            # for model forward
#    self.base_kg_triplets (np.ndarray, m x 3)   # for shuffle_train (non-DDI facts)
#    self.train_ddi_triplets (np.ndarray, n x 3) # for shuffle_train (train DDI positives)

# 2. _build_epoch_kg:
epoch_kg_triplets, epoch_targets = shuffle_train(
    train_ddi=self.train_ddi_triplets,
    train_kg=self.base_kg_triplets,
    setting=self.shuffle_train_mode,
    rng=np.random.default_rng(self.seed + epoch),
)

# 3. Convert epoch_kg_triplets to torch edge tensors directly:
edge_src = torch.from_numpy(epoch_kg_triplets[:, 0]).long().to(self.device)
edge_dst = torch.from_numpy(epoch_kg_triplets[:, 1]).long().to(self.device)
edge_rel = torch.from_numpy(epoch_kg_triplets[:, 2]).long().to(self.device)

# 4. Return (edge_src, edge_dst, edge_rel), epoch_targets
#    Caller MUST use epoch_targets as positives for this epoch (NOT static train_pos_pairs)
```

**Critical**. `epoch_targets` replaces static `train_pos_pairs` for this epoch. Negatives must be sampled against `epoch_targets`'s drug pool, not the full original training pool.

**Do NOT use** `build_edge_lists_from_triplets` — it adds self-loops + EmerGNN relation convention that conflicts with NBFNet's already-handled inverse-edge augmentation.

#### Fix 2. `_train_epoch` must use per-source amortization

**Current**. Naive per-pair loop: 64 BF passes per batch (32 pairs × 2 directions). **Too slow** at 178k node graph scale.

**Required pattern**.

```python
forward_cache = {}   # source -> h_from_source tensor
reverse_cache = {}   # source -> h_from_source tensor (for swap-direction)

# Group batch pairs by source-a
grouped_by_a = defaultdict(list)
for (a, b, label) in batch:
    grouped_by_a[a].append((b, label))

# Forward BF: one per unique source-a, with union mask covering all that source's targets
for src, target_label_list in grouped_by_a.items():
    targets = [t for (t, _) in target_label_list]
    # Build UNION mask: remove (src, ddi_rel, t) for all t in this source's targets
    mask = build_union_query_edge_mask(aug_src, aug_dst, aug_rel, src, targets)
    forward_cache[src] = model.encode_from_source(src, aug_src, aug_dst, aug_rel,
                                                   already_augmented=True,
                                                   edge_keep_mask=mask)

# Group batch pairs by source-b (for reverse direction)
grouped_by_b = defaultdict(list)
for (a, b, label) in batch:
    grouped_by_b[b].append((a, label))

# Reverse BF: one per unique source-b
for src, ... in grouped_by_b.items():
    targets = [t for (t, _) in ...]
    mask = build_union_query_edge_mask(aug_src, aug_dst, aug_rel, src, targets)
    reverse_cache[src] = model.encode_from_source(src, ..., edge_keep_mask=mask)

# Compose pair logits from cached encodings
logits = []
for (a, b, label) in batch:
    h_sym = forward_cache[a][b] + reverse_cache[b][a]
    logit = model.mlp_head(torch.cat([h_sym, model.query], dim=-1)).squeeze(-1)
    logits.append(logit)
logits = torch.stack(logits)

loss = F.binary_cross_entropy_with_logits(logits, labels)
```

**Need to add**. `build_union_query_edge_mask(edge_src, edge_dst, edge_rel, source, targets)` method on NBFNetDDI model. Same logic as `build_query_edge_mask` but vectorized over multiple targets, returning intersection of per-target masks.

**Expected speedup**. From 64 BF passes per batch (32 pairs × 2 directions) to ~10-30 BF passes (unique sources in batch × 2), 2-6× faster.

#### Fix 3. Fix `score_pairs_from_source` docstring

Current docstring says "FORWARD-ONLY logits" but returns hidden states `(T, d)`. Either:
- Rename to `encode_targets_from_source` and clearly document returns `h_q_st` hidden states
- OR fix to actually compose logits + reverse pass

Recommended: align with `encode_from_source` and rename to make it clearly an encoder.

#### Fix 4. Implement `_should_early_stop`

```python
def __init__(self, ...):
    ...
    self.best_epoch = 0

# In fit() validation loop:
if val_auc > self.best_val_auc:
    self.best_val_auc = val_auc
    self.best_epoch = epoch
    self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

def _should_early_stop(self, current_epoch: int) -> bool:
    return (current_epoch - self.best_epoch) >= self.early_stop_patience
```

#### Fix 5. Persist checkpoints to run_dir + concrete negative sampling

**Checkpoint persistence**.
```python
# In fit(), after best update:
if val_auc > self.best_val_auc:
    ...
    self.save_state(run_dir / "best_model.pt")
```

**Concrete negative sampling**. Sample negatives against epoch_target's drug pool, not full original training pool:
```python
def sample_negatives(epoch_pos_pairs, all_drugs_in_epoch, n_negatives, seed):
    """Random pair sampling from drug pool, excluding positives."""
    pos_set = set((a, b) for a, b in epoch_pos_pairs)
    pos_set.update((b, a) for a, b in epoch_pos_pairs)  # symmetric
    rng = np.random.default_rng(seed)
    neg = []
    while len(neg) < n_negatives:
        a = rng.choice(all_drugs_in_epoch)
        b = rng.choice(all_drugs_in_epoch)
        if a != b and (a, b) not in pos_set and (b, a) not in pos_set:
            neg.append((a, b))
    return neg
```

## Remaining work checklist

| # | Work | Spec | Estimated |
|---|---|---|---|
| 1 | Apply Codex Round 6 Fix 1 (shuffle_train integration) | Above | 2-3 days |
| 2 | Apply Codex Round 6 Fix 2 (per-source amortization) | Above | 2-3 days |
| 3 | Apply Codex Round 6 Fix 3-5 (docstring + early stop + checkpoint + neg sample) | Above | 1 day |
| 4 | Write CLI runner `Code/scripts/run_nbfnet.py` | Pattern from existing `Code/my_code/models/screen_s2_v3_multimodal/run_v2i4.py` | 1 day |
| 5 | Smoke test on toy 10-node KG | Below | 1 day |
| 6 | Codex Round 7 review of patched trainer | thread 019e95e4 | 1-2 rounds |
| 7 | Single seed42 first full run | shuffle_train(mode='S2') + 100 epoch | wall ~3-5 hours |
| 8 | Write `_reviews/<date>__paper_faithful_implementation.md` | Per project _reviews/ convention | 0.5 day |
| 9 | Write `_results/<date>__seed42_first_run.md` | Per project _results/ convention | 0.5 day |

**Total estimated. ~2 weeks for paper-faithful first verified result**.

## Smoke test plan (Codex Round 6 prescribed)

Before full training, verify pipeline on toy KG.

```python
# Toy setup
n_nodes = 10
n_base_rel = 3   # 1 = DDI, 2 = some other relation
ddi_rel_id = 1

# Tiny triplets (head, tail, rel) format expected by shuffle_train
base_kg_triplets = np.array([
    [0, 5, 0], [1, 5, 0], [2, 6, 0], [3, 6, 0], [4, 7, 2],
    # ... a few more non-DDI facts
])
train_ddi_triplets = np.array([
    [0, 1, 1],  # drug 0 - DDI - drug 1
    [2, 3, 1],  # drug 2 - DDI - drug 3
])

# 1. Model-only forward check
trainer.setup_graph(...)
trainer.init_model()
logit = trainer.score_pair_symmetric(0, 1, epoch_edges, training=True)
assert torch.isfinite(logit)
loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(1.0))
loss.backward()  # no shape/device errors

# 2. Symmetry check
logit_ab = trainer.score_pair_symmetric(0, 1, epoch_edges, training=False)
# Note: same model state, just different input order — should be IDENTICAL (within numerical precision) 
# because score_pair_symmetric does both directions internally
logit_ab_2 = trainer.score_pair_symmetric(0, 1, epoch_edges, training=False)
assert torch.allclose(logit_ab, logit_ab_2)

# 3. shuffle_train integration
epoch_edges, epoch_targets = trainer._build_epoch_kg(epoch=1)
epoch_edges_2, epoch_targets_2 = trainer._build_epoch_kg(epoch=2)
assert not np.array_equal(epoch_targets, epoch_targets_2)  # different per epoch

# 4. Mini train loop (2 epochs, ~5 pairs)
metrics = trainer.fit(toy_train_pos, toy_neg_callable, toy_val_pos, toy_val_neg, ...)
# Verify: loss decreases, val_auc computes, best ckpt updates, early stop triggers if patience exceeded

# 5. Only then move to amortized _train_epoch implementation
```

## Codex review thread reference

**Thread ID**. `019e95e4-c847-72e1-87a2-3fe702ee21c6`

**Round summary**.
| Round | Topic | Verdict |
|---|---|---|
| 1 | Design faithfulness check (initial) | 6 deviations |
| 2 | Revised design with fixes | 3 remaining issues |
| 3 | Final design after Round-2 fixes | **GO** |
| 4 | Model code review | 2 critical + 1 doc fix |
| 5 | Model post-fix verify | **GO model module** |
| 6 | Trainer code review | 4 critical + 1 misleading docstring |
| 7 (TODO) | Trainer post-fix verify | (after Round 6 fixes applied) |

**To resume codex review in new session**. Use `mcp__codex__codex-reply` with `threadId="019e95e4-c847-72e1-87a2-3fe702ee21c6"`, ask Round 7 to verify Round 6 fixes applied correctly.

## Anti-fabrication reminders (CLAUDE.md §核心规则)

- Any concrete number / file path / code line / API behavior must be verified by Read / Grep / Bash before citing
- "I haven't checked / let me look" 是 OK 的开头, 凭印象说事实不 OK
- Verify chain: `model.py` line 116 (chunked propagation), `_per_mode.py` line 218 (`_setup_graph`), line 347 (`shuffle_train` integration), `shuffle_utils.py` line 27 (shuffle_train signature)

## WSL conda env (CLAUDE.md §强制规则)

**任何 Python 命令必须用 WSL conda env**.

```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_nbfnet.py [args]"
```

不能用 agent 默认 Windows Python (`torch 2.8.0+cpu`, 无 CUDA), 否则 100 epoch silent 跑 CPU 10+ 小时.

## Architectural invariants (DO NOT break)

1. **No drug embeddings in score**. drug_a/drug_b 不应有 learnable embedding 直接进 MLP score head
2. **drug_a = INDICATOR anchor**. `h_a^(0) = q`, drug_b 起 0, 通过 propagation 接 evidence
3. **Boundary reinjection EVERY layer**. `AGGREGATE({messages} ∪ {h_v^(0)})`
4. **Per-(layer, relation) transforms**. NOT shared across layers, NOT shared across relations
5. **Inverse-edge augmentation**. KG case, 不用 generic self-loops
6. **Query-edge masking relation-aware**. 只 mask `(a, r_ddi, b)` 和 inverse, 不 mask 其他 relation 在 (a, b) 之间的 edges
7. **Symmetric at representation level**. `h_q_sym = h_q(a,b) + h_q(b,a)` 在 MLP 之前, 不在 sigmoid 后
8. **MLP input concat([h_q_sym, q])** — official-repo convention, do not drop the query concat

## Hand-off prompt for new session

```
请按 Notes/Log/nbfnet_v1_7_handoff.md 的 spec 完成 NBFNet v1.7 trainer 的 paper-faithful 实现.

当前状态.
- Model module `Code/my_code/models/nbfnet_v1_7/nbfnet_model.py` 已 paper-faithful, 通过 codex Round 5 GO. 不要修改.
- Trainer `nbfnet_trainer.py` 有 skeleton, 但 5 个 critical placeholders 等修复 (见 handoff doc).

任务.
1. 应用 Codex Round 6 prescribed fixes (handoff doc 详列):
   - Fix 1: _build_epoch_kg integrate shuffle_train(mode='S2'), return epoch_targets
   - Fix 2: _train_epoch per-source amortization (forward_cache + reverse_cache)
   - Fix 3: score_pairs_from_source docstring 修正 (rename to encode_targets_from_source)
   - Fix 4: _should_early_stop with patience tracking
   - Fix 5: Checkpoint persistence + concrete negative sampling
2. 加 model.build_union_query_edge_mask 方法 (handoff doc Fix 2 需要)
3. Smoke test on toy KG (handoff doc 详列)
4. Codex Round 7 review of patched trainer (thread 019e95e4)
5. 写 CLI runner Code/scripts/run_nbfnet.py (pattern from run_v2i4.py)
6. 单 seed42 first full run (shuffle_train mode='S2', 100 epoch)
7. 写 _reviews/<date>__paper_faithful_implementation.md
8. 写 _results/<date>__seed42_first_run.md

约束.
- Anti-fabrication 强制: 任何 number/path/line 必须 verify
- 所有 Python 命令必须用 WSL conda env project_1
- 不修改 nbfnet_model.py (已 paper-faithful)
- 保 8 个 architectural invariants (handoff doc 列出)
- 每个 fix 应用后 sanity check 别 break model module

Codex thread to continue. 019e95e4-c847-72e1-87a2-3fe702ee21c6
```

# Effect-Neighbor Selection & Pair-Conditioned Aggregation — Follow-up Improvement Area

**Status**: tested in ONE configuration(top-K=32 by drug_deg, kind ≤ 2,learned attention only AFTER static prefilter); 整个 channel 在该配置下 dead at CHANCE
**Logged**: 2026-05-30
**Cache**: `Code/data/_cache/effect_neighbors_pubmedbert.npz`(7754 drugs CSR + kind + drug_deg)
**Used by**: v2 effect-channel cross-attention(`v2_trainer.py::V2Head.effect_logit`)
**Status of v2 effect-channel**: **dead** in all tested configs(pure-effect PK 0.572 / PD 0.588 = chance)

This log is about a critical question: was v2 effect-channel's failure due to the **mechanism not existing in this KG**, or due to the **fixed prefilter rules that pre-determined what the model could see**?

---

## Why this matters

**Counterintuitive observation**: We claim "v2 effect-channel has learned pair-conditioned attention" — but the truth is the attention only operates on **a pre-selected 32-neighbor window per drug**, picked by **a static rule** (top-K=32 by drug_deg ascending, restricted to PD-composable kinds). The model never sees neighbors outside that window.

So "the effect channel is at chance" could mean:
- (a) PD-composable effect signal genuinely doesn't exist in this KG → confirms codex's redundancy thesis
- (b) The signal exists but our static prefilter throws away the relevant neighbors → we've never actually tested whether learned pair-conditioned selection over the full neighbor pool would work

**We don't know which one is true.** v2 effect-channel failure cannot be attributed cleanly to (a) until we've at least once given the model access to all 95-150 effect neighbors per drug and let it learn the selection.

---

## Locked-in static choices(none isolation-tested)

```
Cache content(per drug):
  nbr_rows[i:j]    全部 1-hop effect-layer 邻居的 PubMedBERT row index
  nbr_kind[i:j]    每个邻居的 kind code(0=side_effect, 1=phenotype, 2=symptom, 3=disease, 4=anatomy)
  nbr_drugdeg[i:j] 每个邻居被多少 drug 链接(反向频率)

Runtime filter chain(v2_trainer.py):
  step 1: keep only kind ≤ 2(PD-composable subset)              ← FIXED RULE
  step 2: sort by drug_deg ASCENDING(rarest mediator first)     ← FIXED RULE
  step 3: take top K=32 / pad to 32                              ← FIXED K
  step 4: cross-attention over a_top32 × b_top32 = 1024 pairs    ← only here is "learned"
  step 5: softmax with frequency-debias bias 1.0                 ← FIXED bias coefficient
  step 6: optional top-k pair sparse selection                   ← codex r019e6251 patch, never run as main
```

Five static rules in the pipeline. **Learned attention sits only on top, operating on a window the rules have already chosen.**

---

## Two specific user concerns(2026-05-30)

### Concern 1 — 缺自适应加权 / pair-conditioned weighting

Same theme as feedback on `kg_typed_aggregation.md`: stop using fixed rules where learned, pair-conditioned weighting could work. Specifically here:

- **Top-K=32 by drug_deg is the same regardless of partner drug.** Drug A's prefiltered top-32 looks identical when paired with B vs with C vs with D. No partner conditioning.
- **drug_deg as saliency proxy is one of many possible heuristics**(could be wrong — never validated against alternatives)
- **PD-composable kind ≤ 2 restriction is asymmetric and pre-decided** — codex's recommendation, but never sanity-checked against {0..4} full set

**True pair-conditioned design**:

```
Inputs:  drug_a's all_eff_neighbors(可变长度,up to 1201)
         drug_b's all_eff_neighbors

Pair-conditioned attention(no prefilter, or larger K with mask):
  scores = MLP(neighbor_emb_a, partner_drug_b_context, kind_a, drugdeg_a)
  weights = softmax(scores, mask=valid_mask)
  selected_a = weighted_sum(weights, neighbor_emb_a)

  对称地 selected_b = ...

  Then cross-attention between selected_a and selected_b
  → pair representation
```

This gives the model **full access** to the neighbor pool and lets it learn:
1. Which neighbors are relevant for this specific pair(pair-conditioned)
2. What saliency proxy actually matters(replacing the hand-coded drug_deg ranking)
3. Whether kind restriction matters(could include kind=3,4 if they have signal)

### Concern 2 — 维度的问题

cache 自己只存 indices,但有几个跟维度直接相关的下游选择都没扫过:

| 选择 | 当前 | 未测替代 |
|---|---|---|
| K(预筛大小) | 32 | 8 / 16 / 64 / 128 / 256 / unlimited |
| Per-neighbor 嵌入维度 | 768(PubMedBERT 原始) | 预投影到 256/128/64 减算力,配合下游 cross-attn |
| K × K cross-attn 空间 | 32×32 = 1024 | 64²=4096(需 efficient attention)/128²=16384(需 sparse) |
| 输出 pair representation 维度 | 64 | 32 / 128 / 256 |
| Storage strategy | runtime gather(节省内存) | pre-gather + 落盘(显存大,index 快 — 必要 if K > 64) |

---

## Other untested static rules(完整 inventory)

| 维度 | 当前 | 未测替代 |
|---|---|---|
| **K size** | 32 | 8 / 16 / 64 / 128 / 256 / all |
| **Kind subset** | {0,1,2}PD-composable | {0,1}更窄 / {0..4}全部 / **learned/pair-conditioned kind weights** |
| **Saliency proxy** | drug_deg ascending | (a) PubMedBERT embedding cluster rarity (b) MeSH 层级深度 (c) **semantic similarity to partner drug**(pair-conditioned) (d) GO/ontology 信息熵 (e) **learned saliency**(per-neighbor importance MLP) |
| **Sort direction** | ascending(rarest first) | descending(generic first,逆向假设)/ unsorted / by relation type |
| **Neighborhood scope** | 1-hop only | 2-hop / 结构相似 drug 的 effect 邻居(transitive)/ path-based |
| **Per-neighbor metadata** | (row, kind, drug_deg) | + relation_type(edge label)+ source_KG(Hetionet vs PrimeKG)+ pubmed mention count + node centrality |
| **Cross-attention bias** | -1.0 × log1p(drug_deg)的 freq-debias bias | bias 系数 sweep / 学到的 bias / 删掉 bias |

---

## 设计草案: pair-conditioned learned-saliency effect channel

不预筛,让模型自己选:

```
Inputs(per pair (a, b)):
  Effect_a = [emb_neighbor_i for i in drug_a's all effect neighbors]
             shape (Na, 768), Na ∈ [0, 1201]
  Effect_b = [emb_neighbor_j for j in drug_b's all effect neighbors]
             shape (Nb, 768)
  + kind_a, kind_b, drugdeg_a, drugdeg_b as side info

Step 1: pair-conditioned saliency
  For each neighbor i in Effect_a:
    s_i = MLP([emb_i, kind_one_hot_i, log1p(drugdeg_i), drug_b_context])
  weight_a = softmax(s_i over i) with mask
  Select(via Gumbel-Top-K, or weighted-sum, or top-K hard with K learnable)

  对称地 weight_b

Step 2: cross-attention on the selected sets
  (same as v2 effect-channel cross-attn, but on learned-selected subset)

Step 3: pool to pair representation
  eff_logit = MLP(pooled)
```

**关键改变**:
- Step 1 是 **pair-conditioned learned selection**,替代当前的 "top-32 by drug_deg" 静态规则
- 全部可见的 neighbor pool 进入决策,不预筛
- saliency 是学的,不是固定的 drug_deg

**两种实现深度**:
1. **Soft 版本**: weight_a 是 softmax 后的连续权重,直接加权 sum 进 cross-attn → 可微但 attention 候选还是大
2. **Hard top-K 版本**: 用 Gumbel-Top-K 或 straight-through 选 top-K 个,K 可以小很多 → cross-attn 候选少但需要 trick 处理梯度

---

## 工程级风险

- **可变长度处理**: 1-hop effect neighbor 数量 Na ∈ [0, 1201],差 4 个数量级。padding 到 1201 浪费严重,需 dynamic padding 或 batch-by-length sampling
- **GPU memory**: 当 Na > 256 时,attention 矩阵在 768d 上膨胀。需要 efficient attention(Linformer / Performer / FlashAttention)
- **数据稀疏度**: 77% test drug 有 effect 邻居,但 23% 没有 → 这些 drug 的 effect 通道完全无 input,模型必须 graceful fallback
- **重新评估"effect 通道死"是否成立**: 如果 learned pair-conditioned selection 仍然让 channel 在 chance 附近,**这才是 effect-neighbor pathway in this KG 是 dead end 的强证据**。当前的证据(top-32+drug_deg 失败)不足以下这个结论

---

## 跟其他 follow-up logs 的关系

| Log | 关系 |
|---|---|
| `language_encoding.md` | 上游变量。每个 effect 邻居的 PubMedBERT 嵌入来源 — 如果换 encoder(MedCPT / BGE / + UMLS definition),neighbor embeddings 整体升级,这个 cache 的 row indexing 不变但底层 embedding 变 |
| `molecular_alignment_design.md` | 同设计哲学。Cross-attention 是文献新标准,这个 cache 是它的天然消费者(分子 ↔ KG 邻居 cross-attn 的 KG 端就是 effect 邻居序列) |
| `kg_typed_aggregation.md` | 姊妹设计。typed_kg 是邻居 → 桶聚合,effect_neighbors 是邻居 → cross-attention pair。两者用同样的"pair-conditioned learned aggregation"主题但 grain 不同(bucket vs neighbor) |
| `kg_molecular_redundancy.md` | 最上层判决。如果 R1 KG-neighborhood ablation 显示 PD-composable 邻居确实是冗余的,这个 cache 的所有改进收益封顶。但如果 ablation 揭示 PD 邻居在 partial KG 下是有用的 → 这个 cache 升级反而是验证 redundancy hypothesis 的 partial 性的关键武器 |

---

## 优先级建议

| 优先级 | 改动 | 理由 |
|---|---|---|
| **P0** | **K sweep**(K ∈ {8, 32, 64, 128, all})with 当前静态 drug_deg sort | 测量"K=32 prefilter 太严"假设是否成立。便宜 — 不需要改架构,只改 v2_trainer 里 K 参数 |
| **P0** | **kind subset ablation**({0,1} 更窄 / {0..4} 全部) | 测"kind ≤ 2 restriction 是否过早决策"。同样不需要架构改动 |
| P1 | **pair-conditioned learned saliency**(替代 drug_deg sort) | 你的头号提议。需要 MLP saliency module + Gumbel-Top-K 或 soft 加权,工程量中等 |
| P1 | **per-neighbor 嵌入预投影**(768→128/256) | 配合 K 增大时控算力 |
| P2 | **edge relation type 加进 cache**(per-neighbor metadata) | 需重做 precompute_effect_neighbors.py,加 edge label 列 |
| P2 | **2-hop neighbor expansion** | 邻居池更大,但需 careful leakage 检查(2-hop 可能引入 DDI 间接路径) |
| P3 | **efficient attention** for large K | 当 K > 64 时才有必要 |

**最高 EV 入口**: P0 的 K sweep + kind ablation,因为它们能 isolated 测出"v2 effect-channel 失败到底是不是 prefilter 害的"。**不跑这个,我们的 effect-channel-dead 结论永远是 conditional 的**。

---

## 实施时的 run tag 约定(将来跑这套时用)

- `effneb_K{8,16,32,64,128,all}__seed42` — K sweep
- `effneb_kind{012,01,01234}__seed42` — kind subset ablation
- `effneb_learned_sal__seed42` — pair-conditioned learned saliency
- `effneb_preproj{64,128,256}__seed42` — embedding pre-projection

汇总到 `refine-logs/EFFECT_CHANNEL_REVISITED.md`,带 isolated-variable 表格。

# typed-KG Bucket Aggregation & Dimension — Follow-up Improvement Area

**Status**: not yet tested (single-shot design, never ablated)
**Logged**: 2026-05-30
**Cache**: `Code/data/_cache/kg_typed_target_pubmedbert.npz` (7754 drugs × 6144d = 8 buckets × 768d)
**Used by**: E-frag (shared/residual molecular alignment to typed-KG target);可被任何 KG-aware 多模态对齐模块复用

---

## Headline 用户提议(2026-05-30)

**Pair-conditioned attention over the 8 buckets, replacing the current concat + linear projection.**

核心想法:每个待预测 DDI pair `(a, b)` 应该有**自己的 bucket 权重**,而不是所有 pair 共享一个 linear projection。形式上:

```
对 pair (a, b):
  q_pair = f(drug_a_features, drug_b_features)          # pair-level query
  for each bucket b in {0..7}:
    weight_b = softmax_over_buckets(q_pair · bucket_emb_b)
  aggregated = sum_b weight_b · bucket_emb_b              # 加权聚合后的 768d
```

**为什么这是合理的**:
- 不同 pair 的判别证据可能在不同桶里(e.g. 一个 CYP 抑制-底物对应该看 protein_gene 桶;一个 PD 协同对应该看 phenotype + side_effect 桶)
- 当前 `proj_k(6144 → 128)` 是一个静态 linear,所有 pair 共用同一组桶权重 —— 这是最大的表达力浪费
- pair-conditioned attention 是 i1 / i3 的"机制按 pair 路由"思想在 KG aggregation 层级的实现

**实现选项**(从轻到重):
1. **Soft bucket gate**: `q_pair → MLP → 8 个 logits → softmax → 8 个权重`,然后对 8×768 加权求和 → 单 768d
2. **Cross-attention**: `q_pair` 作为 query,8 个桶嵌入作为 keys/values,multi-head 或 single-head attention
3. **Bilinear gate**: `weight_b = sigmoid(q_pair^T · W_b · bucket_emb_b)`(每桶独立 gate,不强制 softmax 加和=1)
4. **Mixture of experts**: 每个桶有自己的小 expert MLP,gating network 在 pair-level 路由

**`q_pair` 怎么算**(待决):
- 选项 A: 从 EmerGNN 输出的 (h_a, h_b) 拼接 → MLP
- 选项 B: 从 I4 13-d pair-feature 起,反推 pair-level 上下文
- 选项 C: 从两 drug 的 raw counts (22-d) 起
- 选项 D: 从两 drug 的 molecular fingerprint 起(更"前置"于 KG)

---

## 第二个用户提议 — 维度过大需进一步实验

**Issue**: 当前 6144d 输出,proj_k(6144 → 128) **直接丢掉 98% 的维度**,中间无非线性。

**待测的 dimension 实验**:

| Setup | Per-bucket dim | Total dim | proj_k | 备注 |
|---|---|---|---|---|
| 当前 | 768 | 6144 | 6144→128 | baseline |
| Per-bucket pre-reduce | 96 | 768 | 768→128 | 每桶先 reduce,proj_k 更紧凑 |
| Per-bucket pre-reduce(更狠) | 64 | 512 | 512→128 | 节省更多算力 |
| 重要桶不均匀分配 | 256/64 | ~1024 | 1024→128 | protein_gene/disease 给 256,sparse 桶给 64 |
| 用 SVD/PCA 离线压 6144 | 1024 | 1024 | 1024→128 | 无可学权重的对比 baseline |
| Attention-pool 后单 768 | — | 768 | 768→128 | 配合上面的 pair-attention 提议 |

**预期**:`per-bucket pre-reduce + pair-attention` 组合应该比当前 raw concat 强,因为:
1. 维度浪费消除
2. proj_k 的容量从"挑选 6144 维里哪些有用"变成"利用 attention 输出的紧凑表示"
3. pair-level 路由让模型对不同机制 pair 用不同桶组合

---

## 其余已知改进方向(从之前讨论搬过来,完整记录)

### 桶定义维度

- **drop 死桶**(P1,几乎肯定有收益)
  - bucket 2 (molfunc_bioproc) coverage = 0.00
  - bucket 6 (anatomy) coverage = 0.00
  - drop 这两个 → 6 桶 × 768 = 4608d,直接节省 25% 维度
  - 当前 InfoNCE 训练时 proj_k 仍处理这 1536 死维度,纯 noise

- **按 edge RELATION 而非 node KIND 分桶**
  - 当前 protein_gene 一个桶把"靶点 / 代谢酶 / 转运体"混在一起 → 机制上很不一样
  - 如果 merged KG 的 edges parquet 带 relation 标签(`drug-targets-protein` vs `drug-metabolized-by-enzyme` vs `drug-transported-by-transporter`),按关系细分能拆开这些子机制
  - 需要先 audit merged KG 的 relation 字段有多少种、覆盖如何

- **层级 PK/PD 大组**
  - 上层先聚合成 PK-group (protein_gene + pathway + molfunc_bioproc) vs PD-group (phenotype + side_effect + disease + anatomy)
  - 下层再保留细桶
  - 跟 i1 的 PK/PD 主题呼应

### 桶内聚合方法

- **IDF-weighted mean**:用 `nbr_drugdeg`(已在 effect_neighbors cache 里)反向加权,稀有 mediator 更重
- **桶内 attention-pool**(不只是桶间):桶内邻居各异,直接 mean 会稀释稀有但关键的 mediator(经典 case: CYP3A4 一个酶被几十个通用 GO term 淹没)
- **Top-K-by-saliency 截断**:每桶只保留 top-K 个最稀有/最相关的邻居
- **[mean, std, max] 拼接**:3 倍维度膨胀但信息更丰富(配合后续 dim reduce)

### proj_k 容量

- 当前 single Linear(6144 → 128) 无非线性
- 候选:Linear → LayerNorm → GELU → Dropout → Linear(2-layer MLP)
- 但 codex 当初要求 proj_k linear-only("stable target, not co-adapting lookup")—— 这个约束需要重新评估,因为它是在原始 k_u 设计下定的,不一定适用 pair-attention 版本

---

## 优先级建议

| 优先级 | 改动 | 成本 | 杠杆 |
|---|---|---|---|
| **P0** | drop 死桶 | 30 秒改 1 行 | **必有**:消除已知 1536d noise |
| **P0** | 实现 pair-conditioned attention(用户头号提议) | ~1 天 | 中-高;表达力跨越式提升 |
| **P0** | dimension 实验 sweep(per-bucket 96/64 + total 512/768) | ~1 天(并发跑) | 中:解决 98% 维度浪费 |
| P1 | edge-relation 细分(if merged KG 支持) | ~2 天(需 audit + 重 precompute) | 中-高,机制贴近 i1 |
| P1 | 桶内 attention-pool / IDF-weighted | ~半天 | 中:解决稀有 mediator 稀释 |
| P2 | 层级 PK/PD 大组(预聚合) | 简单 | 故事强,跟 i1 主题呼应 |
| P3 | [mean, std, max] / proj_k 改 MLP | 边际 | 工程性优化 |

---

## 推荐实验顺序(当这个 cache 被排上日程时)

1. **配置稳定的 baseline**:rebuild `kg_typed_target_drop_dead.npz`(6 桶 × 768 = 4608d),重跑 E-frag 等于现在的对照,看 drop 死桶能否独立带来 +0.3pt 级别提升
2. **加 pair-conditioned attention**(基于 P0 的 6 桶 cache):实现 SoftBucketGate,把 proj_k 替换成 (q_pair → 6 logits → softmax → weighted sum → 768d → small MLP)
3. **dim sweep**:在 attention 架构上扫 per-bucket dim ∈ {64, 96, 128, 256, 768},找性能-维度 Pareto 拐点
4. **桶定义升级**:audit merged KG 的 relation 类型,按 relation 重做桶。这个最重(可能要从 KG 边重新提取)但机制上最干净
5. **桶内 attention-pool**:替换桶内 mean-pool,看是否 stack on top of pair-attention 还有边际收益

---

## 关联其他 follow-up log

- `language_encoding.md` — encoder 和 input 是上游变量,典型实验应该先固定一个再扫这个 typed-KG 设计
- `kg_molecular_redundancy.md`(下个文件)— 是更上层的"这个对齐到底有没有用"问题。如果 redundancy 论证成立,本文件里的优化效益上限被压住;如果 redundancy 是 partial 的(per-drug 冗余但 pair-level 不冗余),这个 typed-KG 升级的 pair-attention 设计反而更重要

---

## 下游影响(如果改这个 cache)

`kg_typed_target_pubmedbert.npz` 现在被 E-frag(v2res_trainer.py 的 ResHead)消费。改 cache 格式 → 同步更新 v2res_trainer:
- `proj_k = nn.Linear(ktyped_dim, d)` 改成 pair-attention 模块
- 输入 `kt` 张量从 (B, 6144) 变成 (B, 8, 768) 或 (B, 6, 768)
- ResHead.mol_logit 签名可能需要扩展接收 `q_pair`

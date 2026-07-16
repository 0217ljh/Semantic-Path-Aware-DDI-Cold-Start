# Molecular↔KG Alignment Redundancy — Open Question / Potential Paper Core

**Status**: hypothesis with mixed evidence; never rigorously isolated
**Logged**: 2026-05-30
**Why this might be the paper's center**: if rigorously demonstrated, this reframes the contribution from "we built a method that gains +0.84pt" to "**we identify and characterize the cold-start ceiling caused by KG-pharmacology saturation, and show which kinds of features escape it**". That's a much stronger scientific claim than incremental numbers.

---

## The hypothesis(codex strategic verdict, thread `019e67af`)

> 在 S2 cold-start 这个 split 里,unseen 测试药仍然保留它们在 KG 里的生物医学邻居(靶点/酶/通路/表型/疾病) —— 只有 DDI 边被 mask 掉。
>
> 因此 EmerGNN path-flow 在测试时**已经直接传播过**这些邻居信息。
>
> 任何"per-drug 加新模态特征"(分子结构 / LLM 文本 / 化学指纹)本质上都是把 KG 已经经由 path-flow 传播的同一类信息**重新表达一遍** → 在端到端 DDI 监督下,这些重表达**冗余于 KG path-flow 已提供的信号**,所以加不出增量。

如果这个 hypothesis 成立,**它就是为什么 4 个不同 substrate 的 per-drug 增强都失败的统一解释**:

| Substrate | Form | Outcome | Consistent with hypothesis? |
|---|---|---|---|
| Per-drug k_u (KG-neighbor mean of PubMedBERT) | per-drug 768d | E-frag mol-head alone 0.577 | ✓ |
| Morgan / fragment | per-drug 2048d / 768d | E-frag 无 lift; BRICS oracle gain 0.000(pre-session) | ✓ |
| Free-text LLM CLS | per-drug 768d | E-llm shuffle = main 0.766(INERT) | ✓ |
| KG effect-neighbor cross-attention | per-drug 邻居集 | v2 effect-channel at chance(0.57-0.59) | ✓ |

四种来源、四种 representation,结果一致 → strong directional evidence for the hypothesis。

---

## The counter-evidence(I4 +0.84pt)

但 hypothesis 不是完全干净的 —— **I4 给了 +0.84pt 的真实 lift**,有干净的 shuffle/random 控制塌缩,且机制可解释(CYP inhibitor → substrate)。

I4 跟前 4 个失败 substrate 的**关键不同点**:

| 失败 substrate | I4 |
|---|---|
| **PER-DRUG** 聚合特征(每个药一个向量) | **PAIR-LEVEL** 机制特征(每对药 13 维 explicit 重叠) |
| 信息形式:邻居/结构的 distributional summary | 信息形式:两个药特定属性集合的 explicit set-overlap |
| 模型必须在 backbone 里学"a 的 x 跟 b 的 y 是否构成机制" | 这一步直接以 feature 形式 hand-craft 出来了 |

**精确化的 hypothesis**(I4 的存在迫使我们提精):

> **per-drug aggregation 形式的特征,在 S2 cold-start 中相对 KG path-flow 是冗余的**;但 **explicit pair-level mechanistic features**(尤其是 directed/asymmetric relations like inhibitor→substrate)**不冗余**,因为它们编码的是 KG 拓扑没直接传播的**关系组合**。

换句话说:冗余的不是"分子信息",而是"per-drug 表示形式"。

**(CORRECTION 2026-05-30,用户挑战这个 hypothesis)**:上面的 per-drug / pair-level 二分法
**也是 over-claim**。两个反例同时存在:

1. **v2 effect-channel 是 pair-level**(cross-attention 本质 pair-conditioned),但 chance 水平
   (0.57/0.58)→ pair-level form **不充分**
2. **MNAH 的 22-d meeting-node counts** 不严格是"explicit mechanism"(只是计数,没有方向性关系),
   但给了 +3pt → "必须是 explicit mechanism"也不必要

至少 **4 个竞争 hypothesis** 同时与观察一致:

- **H1**(本节最初的):pair-level 表示形式是必要的 — 被 v2 effect-channel 反例打脸
- **H2**:KG-complementarity 是必要的 — feature 要 encode KG 拓扑没传过的信息(I4 的 CYP 方向性关系是 EmerGNN 不显式编码的)
- **H3**:Representation 容量必须匹配训练样本量 — 53k 训练对撑不起 768d+ opaque embedding
- **H4**:Inductive bias 必须吻合 DDI = mechanism overlap 的任务结构

四个 H 大概率**同时部分成立**;**目前没数据 isolate 出哪个是 binding 约束**。下面的可证伪分析在
H2 框架下展开,但需理解它是 working hypothesis 之一,不是 confirmed claim。R1 ablation 跑出来后
才能 disambiguate。

---

## 这个精确化的可证伪性

**核心断言**:`pair-level mechanism` ⊥ `per-drug aggregation`(在带 KG-mask 的 S2 cold-start 下)

可测的逻辑后果:

1. **I4-without-CYP-inhibitor-substrate-cross**:把 13 维 pair feature 里那两个**非对称**的 cross 项(`cyp_inhib_subst`, `cyp_induc_subst`)拿掉,只留对称的 `shared_*`。如果 lift 大幅塌缩 → 非对称关系正是 I4 携带的非冗余信号
2. **Re-cast 任何 per-drug substrate 成 pair-level**:把分子片段从 per-drug 计数向量改成 `shared_fragments(a,b)` + `unique_fragments_a_only(a,b)` 这类 pair 形式 → 如果 pair-level molecular 突然有 lift,redundancy hypothesis 被进一步证实
3. **Inverse**:把 I4 的 13 维 set-overlap 重新表达成两个 per-drug 多 hot 向量(每个药一个 K-dim multi-hot 表示 cyp_sub/cyp_inh/...)→ 如果在 MNAH 之上无 lift,而 pair 版本有 lift,**直接证明 representation form 是关键变量**

---

## 三个直接的 confirmation 实验(codex 之前提议,这里完整记下)

### 实验 R1 — KG-Neighborhood Ablation(最关键的一个)

**做法**:对 unseen test 药,**逐层 mask 掉它们在 KG 里的不同 neighbor 类型**,重跑 MNAH:

- mask 0:no mask(baseline,~0.772)
- mask 1:remove targets(protein/gene)的 drug→target 边
- mask 2:remove enzymes(metabolic CYP/UGT 等)
- mask 3:remove transporters
- mask 4:remove pathways
- mask 5:remove side_effects / phenotypes / diseases
- mask 6:remove ALL non-DDI biomedical neighbors(unseen 药变成 isolated node)

**预测(if hypothesis is true)**:
- MNAH 在 mask 6 下应**大幅下降**(0.772 → 0.6x 甚至接近 EmerGNN 的 0.746 减去整个 meeting-node 贡献)
- 在 mask 6 下,**重新评测 E-frag / E-llm / 各 per-drug substrate** → 它们应该突然变得有用(因为现在 KG 没了,molecular/LLM 是唯一信息源)

**这是 redundancy hypothesis 的"金标准"测试**:如果重做这个实验,molecular feature 在 biomedical-cold-start 下能 lift 而在 DDI-edge-cold-start 下不能,redundancy claim 就被定量确认了。

### 实验 R2 — Mutual Information / CKA between Representations

**做法**:对每个药 d,计算:
- `repr_kg(d)`:它在 MNAH 学完之后的 drug embedding(EmerGNN 输出)
- `repr_mol(d)`:它的 Morgan / aligned z_m / ChemBERTa
- `repr_llm(d)`:它的 LLM-pharma sanitized text embedding

然后测**两两之间的 CKA / 线性可预测性 / mutual information**:

- 如果 `repr_kg ≈ linear(repr_mol)` → 分子结构在 KG embedding 里"等价线性可恢复" → 冗余
- 如果有显著残差 → 分子带的是 KG 没传播的信息(那为什么端到端无 lift?可能是模型容量/学习信号问题,需另查)

这个是 representation-level 的冗余度量,跟 R1 的 task-level 是互补的。

### 实验 R3 — Synthetic Biomedical-Cold-Start Split

**做法**:从 1900 个药里挑出**生物医学邻居稀疏**的子集(`n_neighbors < threshold`),把它们作为 G2(unseen)重新构造一个 split。

**预测**:在这个 split 下:
- MNAH 性能下降(KG 不给力了)
- 分子 / LLM substrate 性能相对上升 → "cold-start 的难度由 KG 邻居稀疏度决定,而不是 DDI-edge 缺失"

这个直接 operationalize 了 codex 那个 "DDI-edge cold-start vs biomedical cold-start" 的分类法 —— 让两类 cold-start **同时存在于同一个数据集的不同切分上**,显示 substrate 效用是 cold-start 类型的函数。

---

## 如果实验都跑出来支持 hypothesis,paper 的故事变成

**Title 候选**: "When does multimodal augmentation help cold-start DDI? A study of pharmacological knowledge graph saturation"

**核心 claim**:

1. 在带 biomedical KG 的 cold-start DDI 任务(S2 类型),per-drug substrate augmentation(molecular structures, LLM pharmacology, BRICS fragments, free-text embeddings)**系统性地** fail to add lift over a strong KG backbone(MNAH baseline 0.772)
2. 通过 R1 KG-neighborhood ablation 我们证明:这是因为 KG path-flow 已经传播了同一类信息 → "**pharmacological saturation**" of the KG view
3. 通过 R3 synthetic biomedical-cold-start split 我们证明:同样的 substrate 在 biomedical-sparse cold-start 下变得有用 → 任务难度由 KG 邻居稀疏度决定,not by DDI-edge masking alone
4. **唯一突破 saturation 的特征形式是 explicit pair-level mechanistic features**(I4 +0.84pt,clean shuffle separation)—— 因为它编码 KG 不传播的 directed relation composition
5. 提出 **PHARMACOLOGICAL SATURATION TEST**:对任意"multimodal cold-start DDI"方法,先做 R1 ablation 验证你的 cold-start 是哪一类,不能笼统称 "cold-start" 就声明 multimodal gain 是通用的

**这个 framing 的优势**:
- 把所有"看起来失败"的 substrate 实验(E-frag, E-llm, v2 effect)从 negative-result 变成**对一个 unifying claim 的 systematic evidence**
- 给出可重用的 diagnostic test(R1 ablation)和 split classification(R3),这是社区可以采用的方法学贡献
- I4 的 +0.84pt 从"小幅 lift"变成"hypothesis 的关键 falsifier 兼 constructive evidence"

**这个 framing 的风险**:
- 需要 R1 跑出来 *的确* 显示 mask 6 下 MNAH 大幅下降 AND substrate 变得有用 —— 如果 mask 6 下 substrate 还是无用,redundancy claim 就被反驳(可能 KG saturation 不是全部解释,某种 cold-start 永远难)
- 需要 mutual information / CKA 测出来确实有显著残差,否则 hypothesis 变成 trivial("KG 已经全有了" 几乎是个 truism)
- R3 synthetic split 可能选不出真的 biomedical-cold-start 样本数足够大(KG 太密,几乎所有药都有邻居)

---

## 优先级 / 时间预算

| 实验 | 难度 | GPU 时间 | 故事价值 |
|---|---|---|---|
| **R1 KG-neighborhood ablation** | 中(需写 KG-edge masking 逻辑) | 5-6 个 ablation × 35min = ~4hr | **核心**:hypothesis 的金标准测试 |
| R2 CKA / MI representation analysis | 低(post-hoc 分析,无需重训) | < 1hr | 量化补强,论文 method section 用 |
| R3 synthetic biomedical-cold-start | 高(需重构 split,可能样本不足) | 2-3hr 重训 | 故事最锋利,但实操风险 |
| 重要支持:**pair-level vs per-drug ablation** on I4(把 13 维 set-overlap 转回 per-drug 看是否 lift 消失) | 中 | 1 run × 35min | **直接 isolate "form"变量** |

**最高 EV 入门点**:**先做 R1 mask 6**(remove all non-DDI biomedical neighbors of unseen drugs) —— 一个实验就可以决定 hypothesis 是否值得继续深挖。

---

## 跟其他 follow-up logs 的关系

- `language_encoding.md`:encoder/input/aggregation 升级在 redundancy hypothesis 下的预期收益受限 —— 如果信息本身冗余,encoder 再强也没用。但如果 R1 显示 KG 没把全部信息传播好(部分 saturation),encoder 还是有空间
- `kg_typed_aggregation.md`:typed KG bucket-attention 升级同样受 redundancy 制约;但 pair-conditioned attention 这个 **representation form** 的改变,在 redundancy hypothesis 下反而是**有希望的**(它移动到 pair-level,跟 I4 同类)
- 隐含的统一原则:**break out of per-drug aggregation toward pair-level relational features** —— 这可能是所有后续改进的共同方向

---

## 实施时的命名约定

如果跑这套实验,run tag 用:
- `redundancy_R1_mask{0..6}__seed42`
- `redundancy_R2_cka__seed42`
- `redundancy_R3_bio_coldstart__seed42`
- `redundancy_pair_vs_perdrug_I4_ablate`

所有结果归到 `Code/runs/_logs/index.csv` 之外,额外汇总到 `refine-logs/REDUNDANCY_STUDY.md`,因为是 paper 主线证据。

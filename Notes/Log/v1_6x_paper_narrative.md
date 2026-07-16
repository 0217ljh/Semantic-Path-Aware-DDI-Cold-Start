# v1.6.x Paper Narrative — Locked v2 2026-06-08

**Status**: This document **supersedes** the initial 2026-06-08 narrative (canopy-vs-leaves hypergraph framing). The locked framework is now **3 → 1 + 2**: a path-aware multimodal alignment thesis (③), with drug embedding (①) and KG-side multi-granularity hyper-path (②) as the two implementations serving the thesis.

**Scope**: v1.6 and its derivatives (v1.6.x). Applies to ColdDDI AAAI 2027 submission.

---

## 一、核心论点 (一句话)

> 现有多模态 DDI 方法学的是 warm-start 训练分布上的统计对齐, 迁移到 cold start 时对齐空间坍塌、预测失效; 我们提出以**分子片段 ↔ KG 多粒度 hyper-path**为对齐目标, 因为这种**机制对齐**是药理化学定律的结构性反映而非分布特性, 从而对 cold start 具有可迁移性.

---

## 二、问题诊断: warm-start 多模态方法为何在 cold start 失败

**论证链**:

- **前提 1**: 现有多模态 DDI 方法在 warm-start 设定下设计, 通过 cross-modal attention 或 late fusion 学习分子表征与 KG 表征之间的对齐关系.
- **前提 2**: 这种对齐本质上是**统计共现** — 在训练分布中, 某种分子表征恰好与某种 KG 表征共同对应某种 DDI label. 它依赖训练分布, 不依赖底层物理化学机制.
- **前提 3**: 迁移到 cold start 时, 测试 drug 的两路表征都是分布外的; 统计共现学到的对齐没有可外推的结构, 对齐空间坍塌, 跨模态信号失去意义.
- **实证 (已有)**: 两个代表性 warm-start 多模态方法在 cold start 设定下分类 AUC 接近 0.5.

**诊断结论**: cold start 下需要的不是"在更难数据上重新拟合统计对齐", 而是**换一个不依赖统计分布的对齐目标**.

---

## 三、解法的药理学原理

DDI 的真正生成机制是:

$$
\text{drug 的分子片段} \;\longleftrightarrow\; \text{protein 的物理结合} \;\longleftrightarrow\; \text{KG 上的一条路径}
$$

这三个环节之间的对应关系来自**物理化学定律** — 分子片段决定能否与特定 protein 物理结合, 物理结合关系在 KG 上表现为特定的边, 因此每一次"片段-蛋白结合事件"在 KG 上对应一条具体路径.

这种**机制对齐**是**结构性**的 — 它的成立不需要训练分布的支撑, 只需要分子化学和蛋白生物学事实成立. 因此把对齐目标从**统计共现**换成**机制对应**, 得到的方法对 cold start 具有可迁移性.

---

## 四、关键概念: Hyper-Path 与中间 Common Neighbor

### 4.1 为什么不能做 path-level 一一对应

同一个 DDI 机制在 KG 上通常对应**多条同类型的 path**. 例如:
- 经过同一类 target protein 的多条不同路径
- 经过同一副作用节点的不同通路

如果坚持分子片段与具体 path 的严格一一对应, 模型会**过拟合到特定路径、失去机制层面的泛化能力**.

### 4.2 Hyper-path 的定义

我们将 DDI 机制定义为 **path 的等价类**, 而非单条 path:

> **Hyper-path = 所有"经过某类特殊类型节点"的同类型 path 的合集.**

- **特殊类型节点的集合** (target / side effect / pathway 等) 由 KG schema 中的药理学先验决定
- 每个 hyper-path 由一个**锚定节点** (或一组节点) 标识 — 这个锚定节点就是机制等价类的标识
- **粗层 hyper-path 类型有两种实现**:
  - **(a) Hand-defined**: 12 KG-KIND bucket (当前 v1.6 实现)
  - **(b) Learnable**: unsupervised cluster-like 方法自行学习不同的 hyper-path 族 (后续扩展)

### 4.3 中间 common neighbor: 可迁移性的真正来源

传统 KG-based DDI 方法使用的 common neighbor 是**端点的** common neighbor (两个 drug 共享的一跳邻居). 在 cold start 下, drug 端点邻域稀疏, 该信号失效.

我们的 hyper-path 设计的关键差异: **它捕捉的是路径中间的 common neighbor** — 两个 drug 的路径在中间汇聚到某个**非 drug 节点** (target / pathway 等).

**这个差异决定了可迁移性**:

| | 端点 (drug) | 中间 (target / pathway) |
|---|---|---|
| Cold start 下 | OOD, 表征不稳定 | 训练中已与大量 drug 共现, 表征稳定 |
| 信号利用率 | 失效 | 完整保留 |

即使两个 drug 都是 cold 的, 只要它们的路径汇聚到**训练中见过的中间节点**, 机制对齐就有锚点.

**这是整篇 paper 可迁移性的数学化来源**.

---

## 五、框架结构: 3 → 1 + 2

整体框架由三个组件构成, 逻辑关系是 **③ 是论点, ①② 是论点在两端的实现**:

```
                ③ Path-Aware Multimodal Alignment (论点)
              ╱                                    ╲
        ①  drug embedding                 ②  Hyper-Path on KG + NBFNet
        (alignment 的分子端 + KG 端)      (alignment 的 KG 端多粒度路径推理)
```

### Component ③ Path-Aware Multimodal Alignment (论点)

提出对齐原则: 把对齐目标从"统计共现"换成"**分子片段 ↔ KG hyper-path**". 这是框架的核心创新点, 所有其他组件为它服务.

**具体方法靠消融选**: contrastive / cross-attention / 其它. 同时需考虑**信息正交化或最大化**, 避免对齐过强带来的副作用.

### Component ① Drug Embedding (分子端 + KG 端的表征)

为对齐提供两端的输入表征:

- **分子端**: 从 SMILES 提取分子片段感知的 drug embedding
- **KG 端**: 从 protein 邻域提取 KG 感知的 drug embedding

这两路表征是后续机制对齐的两端. 本组件**已实验验证**, 在 known-known DDI 设定下达到相对 baseline (EmerGNN) 的涨点.

### Component ② Hyper-Path on KG + NBFNet 内部精细化 (KG 端多粒度路径推理)

实现 KG 端的多粒度 path 表征, 内部为两层结构:

**粗层 (hyper-path level)**:
- 对一个 query `(drug A, drug B)`, 识别它们对应的 hyper-path 集合 — 即两个 drug 的路径都经过哪些**特殊类型节点**
- 这一层是**离散的、机制等价类层面的**
- 两种实现 candidate: (a) 12 KG-KIND hand-defined, (b) learnable 类型

**细层 (NBFNet level)**:
- 在每个 hyper-path 内部, 用 **NBFNet 做 query-conditioned path propagation**, 学不同具体 path 在此 query 下的相对重要性
- 这一层是**连续的、可微的**
- 具体处理方法**靠消融来定** (NBFNet 嵌进每个 hyper-path 独立 vs 共享 NBFNet by hyper-path 条件化, 等等)

**两层职责互补**:
- 粗层提供**可迁移性** (经过稳定中间节点)
- 细层提供**表达力** (query 条件化的精细 path scoring)

两层都从机制对齐的需求推出, **不是独立堆叠**.

---

## 六、Hypergraph 的使用理由

分子-蛋白结合本身具有**多粒度**:
- 单 binding site
- 多 site 协同
- pharmacophore 通路效应

KG 端的 path 表达必须有**对应的粒度结构**, 因此采用 **hypergraph 表达不同粒度的 path**.

---

## 七、为什么这个框架的逻辑是贯通的

每个设计决策都从**同一根线 (机制对齐 + 可迁移性) 推出**, 没有冗余:

| 设计 | 来自哪一步推导 |
|---|---|
| 多模态对齐 | 前提 3: 机制涉及分子端 + KG 端 |
| 机制对齐 ≠ 统计对齐 | 前提 1-3 + cold start 失败诊断 |
| Hyper-path (等价类) | 同机制可由多 path 实现 |
| 中间 common neighbor | 端点冷但中间节点稳定 |
| NBFNet 在 hyper-path 内 | 等价类内部需要可微细粒度 |
| Hypergraph 表达多粒度 | 分子-蛋白结合本身有多粒度 |
| ①② 服务于 ③ | ③ 是论点, ①② 是两端的实现 |

---

## 八、关键实施决策 (verified with user 2026-06-08)

| 决策点 | 方案 |
|---|---|
| 粗层 hyper-path 类型 | 两个 candidate: (a) 12 KG-KIND hand-defined (当前 v1.6); (b) learnable unsupervised cluster-like. **写 paper 时双方案做对比**. |
| C2 细层 NBFNet 处理方法 | **靠消融决定** (内部独立 vs 共享 by query 条件化, 等等) |
| C3 多模态对齐方法 | **靠消融决定** (contrastive / cross-attention / ...). 同时**需要信息正交化或最大化**作防副作用 regularization |
| Multi-cls 任务怎么挂 | **外接 MLP 头转移**. 对任意 multi-cls 都成立, 只要有可微调的 label. 不嵌进核心架构 |

---

## 九、v1.6 当前在新框架里的映射

```
✅ Component ①  drug embedding (KG 端): v1.6 的 pair-conditional mediator aggregation 已实现
                drug embedding (分子端): 暂未明确, 待 verify 是否已有

🟡 Component ②  Hyper-path 粗层: v1.6 的 12 cluster ≈ 12 类锚定节点, 已实现但 hand-crafted
                Hyper-path 细层 (NBFNet): v1.7 paper-faithful NBFNet 独立存在, 尚未嵌进 hyper-path 内部

❌ Component ③  Path-aware multimodal alignment: 完全没做
                - 分子片段提取
                - Fragment ↔ hyper-path 对齐目标
                - 信息正交化 regularization
                - cold-start 评估

🟡 Multi-cls   80 类 mechanism prediction: v3.0 设计的 MLP 头思路保留, 后续做 external head
```

**已有涨点来源**: 主要来自 ① (drug embedding) + ② 粗层 (hypergraph), 相较 EmerGNN baseline.

---

## 十、术语统一 (写 paper 时严格)

| 旧术语 (v1.6 现实现) | 新术语 (paper 用) |
|---|---|
| Mediator | **Anchor node** (机制等价类的标识节点) |
| Cluster | **Hyper-path equivalence class** 或 **mechanism class** |
| Hyperedge | **Hyper-path** |
| Within-cluster pool | **Hyper-path aggregation** |
| Cluster meta-graph T | **Hyper-path inter-class transition** |
| Pair-conditional intersection `M_k^{a∩b}` | **Middle common neighbor of (a, b) anchored at hyper-path k** |
| Common neighbor of drugs (传统) | **Endpoint common neighbor** |
| 我们的 (区分用) | **Middle common neighbor** |

写 paper 时所有公式、figure caption、章节标题都用**新术语**.

---

## 十一、下一步 action items

| # | 行动 | 状态 |
|---|---|---|
| 1 | **数学形式化 hyper-path** (跟药理学锚定, 这是 paper 绝对核心) | 🔴 next step |
| 2 | 设计 v1.6.x 的下一步实施 — NBFNet 嵌进 hyper-path 内部 (= C2 细层) | ⏸ pending |
| 3 | 设计 ③ 的最小验证 — SMILES fragment + 已有 KG hyper-path 做对齐 | ⏸ pending |
| 4 | 设计 learnable hyper-path 类型 (= 粗层 candidate b) | ⏸ pending |
| 5 | 多 cls 外接 MLP 头 spec | ⏸ pending |
| 6 | Attention stability 实验 (多 seed v1.6) | ⏸ pending (Plan A figure) |
| 7 | v1.6.x narrative figure draft | ⏸ pending |

---

## 十二、Cross-references

**项目内代码**:
- `Code/my_code/models/pmp_v1/v1_6/__init__.py` — v1.6 PyG-form docstring
- `Code/my_code/models/pmp_v1/v1_5/__init__.py` — v1.5A docstring (mathematically equivalent)
- `Code/my_code/models/pmp_v1/v1_1/precompute_cluster_cache.py:87-100` — hand-crafted 12 KIND_GROUPS
- `Code/my_code/models/nbfnet_v1_7/nbfnet_model.py` — paper-faithful NBFNet (待嵌进 hyper-path)
- `Code/data/_cache/ddi_type_map_v3_0.json` — 80 mechanism vocab (multi-cls external head)
- `Code/baseline/emergnn/shuffle_utils.py:27-127` — shuffle_train(S2) cold-start protocol

**Reference paper**:
- MHGNN (Liang et al., IEEE TNNLS 2026): `C:\Users\27911\Downloads\MHGNN_Multiplex_Hypergraph_Neural_Networks_for_Predicting_HerbSymptom_Interactions.pdf` — multiplex hypergraph lineage, our hyper-path is a generalization (path equivalence vs static node hyperedge)

**Codex threads (storytelling history, archived)**:
- 019ea598-7167-7f80-bf6c-72cbc92b1bd6 — 4 rounds of narrative crafting (canopy-vs-leaves framing, superseded by this v2)
- 019ea581-c5ba-7cb1-8b05-f01534d9656d — MHGNN framework-fit analysis

**Previous narrative (superseded)**: This file's v1 version (also dated 2026-06-08) used the "canopy vs leaves" hypergraph framing as primary contribution. Replaced because the new framework cleaner positions hypergraph as the **coarse layer of C2 multi-granularity reasoning**, not the main thesis. The C3 thesis is now **path-aware multimodal alignment**.

---

## 十三、Status

| Item | Status | Date |
|---|---|---|
| Framework locked | ✅ v2 | 2026-06-08 |
| Math formalization of hyper-path | 🔴 next | TBD |
| Multi-seed v1.6 training (seed 42/43/44) | ⏸ pending | — |
| Attention-saving hook in trainer | ⏸ pending | — |
| NBFNet integration into hyper-path | ⏸ pending | — |
| Multi-modal alignment implementation | ⏸ pending | — |

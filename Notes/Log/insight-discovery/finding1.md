---
type: insight-finding
project: Semantic-Path-Aware-DDI-Cold-Start
finding_id: 1
title: "KG-based reasoning has structural gaps for B-class DDI prediction"
status: draft
created: 2026-05-12
related: [[01_kg_pd_path_audit.ipynb]]
dataset_anchor: insight-discovery/eval_200_PKB_PDB.parquet
---

# Finding 1 — KG-based Reasoning Has Structural Gaps for B-Class DDI

## Context

来自 `Notebooks/01_kg_pd_path_audit.ipynb` 的发现。原 insight 假设"KG 缺 PD path"经过数据检查后**只部分成立**——curated DrugBank KG 确实 0 PD 边，但 Hetionet/PrimeKG 有大量 PD-relevant 边。真正的问题不在"缺"而在三个更细致的结构性 gap。

本 finding 聚合 3 个 sub-finding（A/B/C），分别需要进一步澄清和量化证据。所有量化证据将在固定的 **200-pair 评估集**（100 PK-B + 100 PD-B，均带 DDInter 临床机制描述）上反复测量。

---

## Sub-finding A — Vocabulary Chain Break: 微观与宏观节点之间断层

### 现状

DDInter 临床机制文本描述的**因果链**形态是：

```
drug → 受体/通道（中观）→ 生理系统效应（中观）→ 副作用（宏观，临床观察）
```

KG 提供的节点是**两端**：
- 微观端（充足）：`Drug` / `Gene/Protein` / `Pathway` / `Biological Process`
- 宏观端（部分）：`Side Effect`（Hetionet 5701 个，SIDER 来源）/ `Phenotype`（PrimeKG 990 个）

KG 缺的是**中间一层**：

| DDInter 用的中观概念 | KG 节点状态 |
|---|---|
| "alpha-2 adrenergic receptor"（受体亚型） | 只作为 Pathway 名 / GO MF 术语，**不是可链接的实体节点** |
| "beta-2 adrenergic agonist"（药理活性类别） | ❌ 不存在 |
| "QTc prolongation"（药理副作用术语） | ❌ 连 Hetionet SE 都没收录 |
| "CNS depressant"（药物机制类别） | Hetionet 有 "Central Nervous System Depression" 当 Pharmacologic Class，但**没建跟具体 drug 的边** |
| "potassium / 电解质失衡"（电解质级机制） | 只匹配到 GO 的 "potassium ion transport"（基因层），**没有"electrolyte imbalance"概念** |

### 需要澄清

在 200-pair 数据集上系统量化：

1. **划定 4 层语义颗粒度**：molecular（gene/target）→ functional（pathway/MF）→ system/receptor（"alpha-2 receptor", "CNS"）→ clinical-observable（"sedation", "QTc"）。
2. 对每个 DDInter mechanism 文本，**抽取它涉及的实体**，标注每个实体所属层级。
3. 对每个层级，统计**该层概念在 Hetionet/PrimeKG 节点集里命中率**（match by name + alias）。
4. 假设要验证：**system/receptor 层命中率最低**（< 30%），形成 micro-macro 之间的断层。

### 关键问题

> "断层"具体是**哪两层之间**的断？怎样的描述能让方法层（接下来要设计的 LLM/KG 联合推理）知道"该补什么概念节点"？

---

## Sub-finding B — B-class Path Is Over-Abundant, Not Sparse (噪声主导)

### 现状

跟 insight 最初猜想"PD path 稀疏"相反，实际上 **B 类（PK-B / PD-B）的 2-hop path 在 Hetionet 上是过剩的**：

| 指标 | PD-B (n=83) | PK-B (n=63) |
|---|---:|---:|
| Median paths / pair | 11 | 37 |
| Mean paths / pair | 33.3 | 47.6 |
| Max paths / pair | 293 | 205 |

并且 case 1 (Tizanidine + Dimenhydrinate) 的 34 个 shared SE 里，**只有 ~40% 跟 DDInter 描述的机制相关**，剩下 ~60% 是**两药副作用谱广泛重叠的统计偶然**（Agranulocytosis / Dermatitis / Rash / Thrombocytopenia / Diarrhoea / etc.），不是因果机制。

### 需要澄清

在 200-pair 数据集上量化 **signal-to-noise ratio (SNR)**：

1. 对每个 pair，枚举 Hetionet/PrimeKG 上所有 2-hop path。
2. 对每条 path 的**中间节点**（specific SE / phenotype），用 DDInter 文本判断它是否跟该 pair 的机制相关（语义匹配，可用关键词 + 后续 LLM 校准）。
3. 计算 per-pair SNR：`relevant_paths / total_paths`，再算分布（mean、median、p25-p75）。
4. **预期**：PD-B 和 PK-B 的 SNR 都很低（< 30%），证明"path 过剩 + 噪声主导"。

### 关键问题

> 信号是 SE 节点本身的**语义内容**，不是 path 数量或拓扑。**怎么从一大堆 path 里只挑出语义相关的那 N 条？** 这一步是后续方法的核心。

---

## Sub-finding C — PD-B and PK-B Are Topologically Indistinguishable (待澄清)

### 现状

PD-B 和 PK-B 的 2-hop path 在 Hetionet 上**拓扑结构几乎一样**：

| 维度 | PD-B | PK-B | Diff |
|---|---:|---:|---:|
| 中间节点 = Side Effect | 93.2% | 93.5% | -0.3% |
| 中间节点 = Gene | 6.2% | 6.4% | -0.3% |
| Pattern `CcSE~CcSE` | 93.2% | 93.5% | -0.3% |
| Pattern `CbG~CbG` | 3.1% | 4.4% | -1.3% |

所有差异都 < 1.5%。**Subgraph / meta-path / GNN-based 方法看到的 PD-B 和 PK-B 是同型子图**——这解释了为什么这些方法在 B 类预测上 plateau。

### 需要澄清

我们目前**不确定**：PD-B 和 PK-B 是不是**应该**有 path 拓扑差异？

- 如果"是"（应该有但 Hetionet 这种 KG 没体现）：需要找有这种差异的 KG，或者构造新的 KG
- 如果"不是"（拓扑本来就该一样，区分靠 SE 节点的语义内容）：那直接放弃 path 拓扑维度，全力做语义分布

### 澄清思路

1. 在 200-pair 数据集上，对每个 pair 做**完整 subgraph 抽取**（drug-induced 2-3 hop ego graph）。
2. 用 **graph metrics**（直径、聚类系数、节点度分布、relation entropy）量化两类 subgraph 形状差异。
3. 用 **NLP metrics**（SE 名称集合的 TF-IDF / BERT embedding 中心距离 / 语义聚类簇数）量化两类 SE 集合的语义差异。
4. 用 **LLM 判断**：给定 drug-pair 的 SE 列表，LLM 能否反推 PK-B vs PD-B 标签？（与 sub-finding A 的 LLM 部分对齐做）

### 关键问题

> 区分 PK-B 和 PD-B 的信号到底在 **graph topology** 还是 **node semantic content**？这决定方法是 graph-side 还是 LLM-side 主导。

---

## 评估锚点

所有 sub-finding 的强化证据来自固定的 **200-pair 评估集**：

- **位置**：`Notes/Log/insight-discovery/eval_200_PKB_PDB.parquet`
- **组成**：100 PK-B + 100 PD-B，全部要求 `ddinter_mechanism` 非空
- **来源**：DrugBank `filtered/drugbank_with_mechanisms.csv` ∩ PK-B/PD-B 4-way 分类
- **抽样**：随机 + 固定 seed=42
- **冻结**：一旦生成不修改，所有后续分析复用同一份

---

## 下一步顺序

1. ✅ 写 finding1.md（本文档）
2. 生成 200-pair 数据集 → `insight-discovery/eval_200_PKB_PDB.parquet`
3. **Sub-finding A 强化**：4 层颗粒度划分 + 命中率分析
4. **Sub-finding B 强化**：per-pair SNR 计算
5. **Sub-finding C 强化**：graph metrics + NLP metrics 对比

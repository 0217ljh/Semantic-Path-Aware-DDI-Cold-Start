---
title: Idea2 — LLM 修复分层 KG + 动态邻域的冷启动 DDI 预测
status: design (pre-implementation)
created: 2026-06-28
tags: [idea2, cold-start-ddi, knowledge-graph, llm, dynamic-neighborhood]
---

# Idea2 设计文档

> 本文件夹 `Notes/Ideas/idea2/` 是 idea2 这条线的设计/想法主文档。
> 决策流水见 `Notes/Log/idea2/`，代码在 `Code/code-idea-2/`（与正在跑的另一条 idea 物理隔离）。

## 0. 一句话

冷启动 DDI 在 KG 上失效的根因不是 encoder，而是 **KG 的机制链在 PD 上被普遍破坏**（同实体跨命名空间碎片化 + 判别性 meso 汇聚链缺失/被 hub 淹没）。
我们用 **LLM 一次性修复一个分层生物 KG**，再在其上做 **动态、特异性驱动的邻域** 预测。PK/PD 与三个 DDI 任务统一在同一框架下。

## 0.5 KG 状态（2026-06-28，锁定 v2）

**canonical KG = `Code/data/KG/_merged_kg_dedup_v2/`**（`nodes__dedup.parquet` 141,381 节点 / `edges__dedup.parquet` 6,537,040 边）。弃用 v1 `_merged_kg_dedup` 和原始 `_merged_kg`。

- 新 schema：nodes `[id, kind, name, source_kg]`；edges `[src, src_kind, dst, dst_kind, relation, source_kg, directed]`。蛋白 id = `prot:<entrez>`，drug id = `DB...`。53 种 relation。
- ✅ **蛋白跨源合并完成**：`source_kg='merged'` 恰 = `gene/protein` 数 29774；SLC12A1 = `prot:6557` 单点，Furosemide(DB00695) 连它。**L1/PK 浅锚层干净**。
- ⚠ **仅蛋白合并**：meso/macro 层（GO/effect/disease/pathway/anatomy）**未跨源合并** → L2–L4/PD 深锚层仍源碎片化。kind 大小写重复（pathway/Pathway、disease/Disease、drug/Drug、BP/MF/CC 残留）即证据。
- ⚠ 残留 44 个 `db:target:BE` 蛋白 + 41 重名蛋白；`prime:anatomy_protein_present` 占 ~49% 边（疑垃圾边，待定去留）。

**分工**：kind→layer 折叠（大小写）在 idea2 loader 做；**meso 层跨源同义节点合并 = idea2 的 LLM T1 实体对齐范围**（KG agent 做蛋白硬合并，meso 合并留给我们的补全步，契合设计）。

**kind→layer 映射（v2 实际 kinds）**：
- DRUG = {Drug, drug, Compound}
- L1 protein = {gene/protein}
- L2 GO/function = {biological_process, molecular_function, cellular_component, pathway}（含 TitleCase 残留）
- L3 effect = {effect/phenotype, Side Effect, Symptom}
- L4 system/disease = {disease, anatomy, exposure}

**待量**：meso 层跨源同名重叠有多少（确认 LLM 合并工作量 + 它是否真存在）。loader 按新 schema 重写（旧 `MergedKG.from_parquet` 指向旧 KG，不可用）。

## 1. 已核实的诊断（verified，带出处）

| 事实 | 数值 | 出处 |
|---|---|---|
| 去流行度后真结构可分性 | AUROC ≈ **0.566** | `Code/scripts/analyze_pd_degmatch.py` |
| 随机负样本可分性（含流行度） | ≈ 0.60 | 同上 |
| PD 分子汇聚存在率 vs 随机 | **93% vs 86%**（都高 → 汇聚过剩、非判别） | `analyze_pd_convergence_kg.py` |
| 通用 hub 度数 | Nausea deg ≈ **932**，SE hub 中位 max deg ≈ 1190 | `analyze_pd_paths.py` |
| 同蛋白跨命名空间碎片化 | SLC12A1 = `het:Gene:Gene::6557`(name=SLC12A1, Furosemide 连) + `db:target:BE0000502`(空名, Furosemide 连) + PrimeKG 符号节点；三者**互不相连** | 本会话 KG 核查 |
| LLM 直接当预测器上限 | PD_C effect-conditioned AUC = **0.686**（仅 ~+0.06 over molecular 0.63） | `Notes/Log/.../llm_oracle_results_anon.json` |

**关键结论**：汇聚不是缺，是**过剩且非判别**（走通用 hub）；同时**判别性的特定链被碎片化/缺 meso 断开**。
molecular design 在 PD 上不行；引入文本的 **TextDDI 在 PD 上最好** → PD 的信息在"机制/语义"层，我们赌它其实在 KG 里、只是被埋了，从 KG 侧把它救出来。

## 2. 任务设定（locked）

- **冷启动**：drug pair **无 DDI history，但两药都有 protein 信息**，两药均 unseen。pair 分数只能是 `{p_A}` 与 `{p_B}`（靶点集）之间子图的函数。无 drug-id、无 DDI 邻域。
- **数据规模（verified）**：DrugBank filtered DDI 共 565,731 对，**仅 1,900 个不同药**。

| | pairs | unique drugs | distinct ddi_types |
|---|---|---|---|
| PK | 295,406 | 1,698 | 30 |
| PD | 270,241 | 1,783 | **184** |
| Mixed | 84 | — | — |

- **三个任务都要覆盖**：binary cls / multi-class cls / multi-label cls。不做机制生成等新任务。数据集/任务不创新。
- **目标**：S2 AUROC > 0.80（项目主目标）。

## 3. 分层生物树（骨架，固定，可扩展）

有向抽象阶梯（下→上）：

| Level | 生物层 | KG `kind` | 锚落此层 = |
|---|---|---|---|
| L0（未来） | molecule / substructure | 暂无，预留 | 分子共享（future work，HDN 那套可接入） |
| L1 | protein | `protein_gene`（target+enzyme+transporter+carrier） | **PK**：共享酶/转运体/靶点 |
| L2 | GO/function | `pathway` + `biological_process` | 通路/过程汇聚 |
| L3 | effect | `side_effect`（+phenotype?） | 表型/副作用汇聚 |
| L4 | system/disease | `disease`（+anatomy/system?） | **PD**：汇聚到系统/疾病 |

- L1–L4 的 kind 已核实存在（`analyze_pd_convergence_kg.py`）；L3 phenotype、L4 system/anatomy 是否单独成 kind **待核**（全 `KIND_ORDER`）。
- **汇聚 = 两条流沿树向上在最低公共层相遇**。相遇层 = 涌现的 PK/PD 信号，**不写死**。

## 4. 方法骨架（borrow + adapt，AAAI 不需重理论）

从 general/recent 工作借机制，适配到 DDI。三个位置各有借用源（见 §8）。

- **机制端点** `M_A` = target+enzyme+transporter+carrier 的并集（`drug_rel` 对应 buckets；具体 bucket 编号待核 `REL_BUCKET`）。
- **统一汇聚锚**：层无关，`{m_A_i, anchor@Lℓ, m_B_j}`；锚的 type/layer 当特征。PK 浅锚 / PD 深锚自然分。
- **核心组合（idea 1 + 6）**：
  - **(1) Posterior Quotient Relink**：per-pair 软激活候选边（跨命名空间合并 + 缺失层间链），得商图。
  - **(6) Ephemeral Mechanism Hypergraph**：在商图上诱导临时超边 `{m_A, anchor, m_B}` 打分；超边权 = 它捆绑边的 posterior 软与（1 直接参数化 6）。

### 双图架构（避免 per-unknown-drug LLM，已被数据坐实）

- **G_gen（通用、drug-agnostic 骨架）**：protein→GO→effect→system 节点 + 层内/层间边。**LLM 只增强这一次**（合并碎片 + 补缺失机制边）。
- **drug 挂载**：药 → 它的蛋白（G_gen 节点）。**unknown drug 推理时用蛋白挂到已增强 G_gen，零 LLM 调用**。
- LLM 成本 = O(G_gen 一次) + 至多 O(1900 药各一次)，**永不 per-pair、永不 per-unknown-drug**。
- **codex 警告**：全局 only 增强有 **over-densification** 风险（补一堆真但通用的边，连通涨判别不涨）。gate 偏弱 ≠ 否定 per-pair 修复，可能"收益在选择性激活(Stage2)"。诊断：密集补全 vs 高精度保守补全。

## 5. LLM 的角色（修复，不预测；label-blind）

- LLM 做：① 跨命名空间实体对齐（合并碎片）② 补缺失的 protein→GO→effect→system 机制边。**不当 per-pair 预测器**。
- **label-blind 硬规则**：补全只用生物学信息（名称/别名/功能/source/本体），**绝不注入 drug pair、DDI 是否存在、effect 类别**。
- **现有 `Code/runs/pd_mechanism/chains.csv` 泄漏**（per-pair + 注入了 effect + 临床文本）→ **gate 不能用，必须重建 label-blind 版**。
- prompt 模板（全 label-blind）：
  - **T1 实体对齐**：候选节点 name/别名/id/source/type → "哪些是同一实体？"
  - **T2 骨架补边**：蛋白 symbol/功能 + 候选 GO/effect 节点 → "经什么机制连到哪个终点？"
  - **T3 逐药链**（drug-specific 但 label-blind，≤1900 次）：单药靶点集 → 逐靶点摊 target→GO→effect→system。

## 6. 两阶段计划

### Stage 1 — GATE（不是模型，先验证假设）

label-blind 补全后，**初步**确认两件事，**不做 hub-handling、不做动态**：

1. **断链恢复率（连通）**：raw 上被碎片/缺 meso 断掉的真 DDI 链，补全恢复多少。注意测"断链恢复率"而非"绝对存在"（粗汇聚已 93% 会饱和）。
2. **初步预测准确率（探针）**：w/ vs w/o 加边，简单探针，degree-matched 负样本，按 PK/PD 分桶，三任务。
   - 探针阶梯（同 schema/容量/预算，只换图）：`logreg` + 深度3-4 `GBT` + 固定1-2步 typed-MP/Cleora 子图嵌入 + 线性头。LLM-judge 只当 oracle 上界，不进 gate 决策。
   - 防假阴：planted-edge 自检（删已知真边再补回，探针须能恢复增益）。
   - 防假阳：degree-only baseline + null-completion 对照（加类型/度匹配随机边）+ 窄 degree bin 内增益须存活。
   - **Stage 1 的"无 hub-handling 准确率" = 加边本身贡献的下界；它与上界的差 = 留给 Stage 2 hub-handling 的空间** → 两阶段增益可归因。

**Go/No-Go（codex）**：S2 + degree-matched + **PD 桶**上，completed 比 raw 稳定 **≥ +0.03 AUROC 或 +0.05 AUPRC**，≥ **2/3 探针**、95% CI 不含 0，增益**集中在 PD/effect 切片**，degree-only 与 null-completion 对照近 0。
边界：若只有固定嵌入探针涨、手工特征不涨 → 不杀项目（"信号在但非标量可解码"，正好论证需要动态模型）。

### Stage 2 — 动态模型 + hub-handling

- 动态 per-pair 邻域（特异性驱动），GNN 判别。
- **hub-handling**（两类 hub，处理相反）：
  - 通用 hub（Nausea，要压）：null-model/IDF 特异性 + bottleneck 路径打分（一个 hub 毁全链）。1/deg(Adamic-Adar) 是线性基线但不够（可加 + 度数≠功能hubness）。
  - 机制 hub（CYP3A4，不压要救）：relation-type/方向 typed motif（"A 抑制 X ∧ B 是 X 底物"），把巧合共享和致互作共享分开。**待核**：KG 是否保留 inhibitor/substrate/inducer 关系类型。
- 算力：邻域压缩成 token 喂 LLM（动态那步可选用 LLM，与 §4 一次性增强是两码事）。

## 7. 三任务统一（共享 backbone × 3 head；锚=标签）

- 共享 backbone：动态分层特异性锚选择（层无关 → 覆盖所有 DDI 类）。产出 = 带权选中锚 + pooled 向量。
- 三 head = 对同一"选中锚集"的三种读出：

| 任务 | 数据集 | head | loss |
|---|---|---|---|
| binary | DrugBank 派生 | "有没有高特异锚" → 1 logit | BCE + degree-matched 负样本 |
| multi-class | DrugBank（PD 184 类） | "哪个锚主导" → softmax 锚→类 | CE |
| multi-label | TWOSIDES | "哪些锚命中" → 多 sigmoid | BCE-multi |

- **锚=标签**：L3/L4 锚本身就是 effect/event → multi-class/multi-label 天然可解释，需学一个 `锚→event类` 软对齐，不 1:1 处回退 pooled head。
- 训练全任务一起；评测按 PK/PD 分桶。

## 8. Novelty 边界（只对 DDI 圈，competitor = ColdDDI baseline）

competitor 三族（`03-Projects/ColdDDI/Code-Released/baseline`）：

| 族 | 方法 | 软肋（对我们） |
|---|---|---|
| 分子子结构 | SSI-DDI / DSN-DDI / HDN-DDI（化学层级） | 无生物 KG，表示不了 PK 共享酶 / PD 生理汇聚 |
| 固定 KG+分子 | TIGER / MKG-FENN | KG 静态，吃 hub 饱和+碎片化病 |
| 文本+RL 冷启动 | TextDDI | 扁平文本+PPO，无结构化机制层 |

**delta**：三族要么没 KG、要么用静态 KG，没人做 **per-pair 动态修复 + 特异性汇聚锚选择**。
**撞词澄清**：HDN 的 hierarchy = 化学层级（≠ 我们生物层级，且分子是我们 L0 future）；TextDDI 的 LLM=编码器+文本RL（≠ 我们 LLM=图修复器）。
**AAAI 现实**：理论轻、empirics 不轻 → 必须 S2 上赢 6 baseline + 每个改动可 ablate。

### 可借用的 general 工作（三轴，年份按 arxiv YYMM 核实）

- **分层**：HyGRAG(WWW2026, Cleora+LSH)；OntoKG(`2604`, 关系路由)；Multi-Hop Pooling(Springer2026, 转移矩阵)。
- **动态**：Path-aware minimal-sufficient subgraph GraphRAG(`2603`)；Structural Adaptive RF via MI(WWW2021)；Customized Subgraph Selection for DDI(`2411`)。
- **LLM 整合**：GraphEdit(`2402`, LLM 裁边)；LLM Clinical Graph Refiner(`2604`)；Tree-Guided LLM GSL(`2503`, 树+LLM+GSL)；KG Denoising for RAG(`2510`, 合并实体)。
- **必读焊边界**：Tree-Guided LLM GSL(`2503`，最可能撞)、Customized Subgraph Selection for DDI(`2411`)。

## 9. Pending / 待核实

- [ ] KG 由另一个 agent 处理中，处理好后核对：全 `KIND_ORDER` → 层映射；L3/L4 kind 是否齐。
- [ ] `REL_BUCKET` 具体 bucket（target/enzyme/transporter/carrier 编号）。
- [ ] enzyme/transporter 边是否保留 inhibitor/substrate/inducer 关系类型（决定 Stage2 typed-motif 可行性）。
- [ ] 重建 label-blind 的链补全（弃用泄漏的 `chains.csv`）。
- [ ] 是否有现成 BE-code→symbol/Entrez 映射表（决定实体对齐工程量）。

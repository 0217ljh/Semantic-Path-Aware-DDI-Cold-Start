# Meeting-Node Anchor vs EmerGNN Flow — Boundary Analysis (v4 FINAL, post-codex-r3)

> 这份文档把我们的"meeting-node-anchored + PK/PD-aware perception"路线与 EmerGNN (Zhang et al. *Nat. Comp. Sci.* 2023) 的"Bellman-Ford path-flow"路线放在一起，逐点拆清楚边界，回答两个核心问题：
> 1. 我们的路线在何处真正不同于 EmerGNN？
> 2. 我们能否在它的基础上做出实质提升？
>
> **v2 修订**（post-codex-r1）：删 overstating；3 个 primary axes 替代 5 个；承认 EmerGNN 也用 intermediate 节点；加 CrC shortcut control 实验需求；ranking 改进方向。

## 核心定位

> **EmerGNN learns path-flow to endpoints; we test whether cold-start DDI benefits from explicit, pair-conditioned mediator perception with PK/PD structure.**

**不是**"我们打败 EmerGNN 因为它有缺陷"，**而是**：两种方法学**假设不同的解释单元**——EmerGNN 把 intermediate 节点当作**通道**（信号从 u 经过它们流向 v 终点），我们把 intermediate 节点当作**主要解释对象**（mediator 本身承载机制信号，我们直接在它身上读出）。

---

## 1. 两条路线的核心机制

### EmerGNN（flow-based）
1. 把 DDI 网络 + 生物医学 KG 合并成增广网络，加 inverse 边
2. 对 query pair (u, v) 抽取 ≤ L 的全部 path，union 成 subgraph G_{u,v}^L
3. Bellman-Ford 风格 flow GNN：让 drug u 的 fingerprint 沿路径流动 L 步直到 v，每步聚合邻居信息
4. Relation-type attention α_r 由 (fu, fv) 调制——同关系类型在该 pair 整个 path 共享
5. 同时跑 u→v 和 v→u，拼接读出
6. **Anchor**：drug 终点 v（flow 累积到 v 的 representation）

### 我们的路线（meeting-node-anchored）
1. 在合并 KG 上找 query pair (u, v) 的共同邻居集合 M = (N₁(u) ∪ N₂(u)) ∩ (N₁(v) ∪ N₂(v))
2. **浅 GCN** (K=2) 把 M 中节点的局部表征算出来，节点 init 用 PubMedBERT([node.name]) (i4)
3. **Pair-conditional perception**：对 M 中每个候选节点独立打分、加权（i3），按 kind 走 mol / effect 两条通道（i1）
4. **Anchor**：meeting node M（不是 u 也不是 v）

---

## 2. 三条 primary 边界差异（codex 收束后）

### 差异 1 · Readout target：endpoint flow vs mediator readout

**EmerGNN**：representation 在 drug v 终点累积。Intermediate 节点是 conduit，pair 表征语义是"u 的特征流过 KG 到达 v 时的累积态"。

**Ours**：representation 在 M 节点直接读出。Mediator 是 primary explanatory object，pair 表征语义是"u 和 v 共同关联的某个分子/效应实体的表征"。

这是结构性差异。但要注意：EmerGNN **也通过 intermediate 节点 reason**——它不是"忽略 intermediate"，而是把它们当作传输通道。**诚实表述**是"explanatory unit 不同"，不是"我们能 reason intermediate，他们不能"。

### 差异 2 · Selection granularity：relation-type vs node-instance

**EmerGNN**：
> α_r^{(ℓ)} = σ(w_r^{(ℓ)} · [f_u; f_v])

同关系类型 r 在层 ℓ 的所有边共享同一个 α（虽然 α 是按 (fu, fv) 调制）。意味着：层 ℓ 里**所有** `drug→gene` 关系的边权重一样，模型无法直接区分"这个 gene 是 CYP3A4，对 query 重要"vs"这个 gene 是无关 receptor"——区分只能通过下游 message content 实现。

**Ours**：M 中每个具体节点独立打分。"CYP3A4" 节点的注意力权重由 (fu, fv) + node-name 语义 + kind 类型先验**共同**决定，不依赖于它是哪种关系连接的。

**正确措辞**：
> EmerGNN conditions **relation importance** on the pair;  
> our perception conditions **individual mediator importance** on the pair.

### 差异 3 · Mechanistic decomposition：flat vs PK/PD dual-channel

**EmerGNN**：flat readout 预测 86 种 DDI 类型，没有 PK / PD 范式区分。Attention 学到的 top-5 关系（CrC / CrD / CbG / PCiC / CcSE）**混合**了 PK-relevant 和 PD-relevant 关系——单 attention 头同时处理两种 mechanism。

**Ours**：mediator kind 层面**显式分离**：
- Molecular-kind M（Gene / Protein / Pathway）→ PK channel
- Effect-kind M（Side Effect / Phenotype / Anatomy）→ PD channel
- 类型先验直接对应 i1 paradigm split
- PK-B 子类（无 mol mediator）可借 effect channel（E8.7-hetero 数据支持）

这是**最强的概念创新**（codex round 1：strongest conceptual addition）。

---

## 3. 实现细节差异（次要 axes）

以下两条**不构成独立 contribution**，但在实现/消融上有意义：

### 实现 axis A · 深度 vs over-smoothing

**EmerGNN**：L=3 最优（论文 Fig 5c），L>3 性能下降。这是 over-smoothing 拐点。

**Ours**：K=2 GCN + M-readout。不需要"u 走 L 步到 v"。

**修订措辞**（codex round 1 警告"K=2 仍可能 blur"）：
> 我们**不**是"避开了 over-smoothing"，而是**把负担从 long propagation 转移到 explicit mediator retrieval + shallow encoding**。
> 我们仍跑 GNN，K=2 在 hub-dense KG 上也可能 blur；但 retrieval（找 M）这一步是 deterministic 的 KG 集合操作，不受 GNN 深度的限制。

### 实现 axis B · Node semantic prior：ID vs PubMedBERT

**EmerGNN**：中间节点用 learned ID embedding。

**Ours**：M 节点 init 用 PubMedBERT([node.name]) 投影到 128d。

**注意**（codex round 1）：这是 **feature prior，不是架构创新**。一个 fair hybrid baseline 是"EmerGNN + PubMedBERT init"，如果它也获得提升，那 PubMedBERT 这条不是我们的主 contribution——只是合理的 feature 选择。**不应**作为主 selling point；应作为 E3 已验证的 design choice。

---

## 4. CrC shortcut 风险 + 必须的 control 实验

**Codex round 1 关键质疑**：EmerGNN 论文 Fig 3c 显示 #1 attended 关系是 `CrC`（Compound-resembles-Compound）。这暗示其成功**很大程度依赖 drug similarity shortcut**——"u 像 u₁，u₁ 与 v 互作 i，所以 u 与 v 也是 i"。

如果我们的 M 大多落在 compound-similarity 节点（即另一个 drug u₁ 作为 mediator），那我们**同样在用这个 shortcut**，只是换了个 readout 位置。

**必做的 control 实验**（i2 实证补强，post-codex-r2 扩充）：

1. **Mediator kind 分布统计**：M 节点的 kind 分布是什么？多少 % 是 Compound（drug-similarity）？多少 % 是真正的 mechanism-aligned mediator（Gene/Protein/SE）？
2. **CrC edges 移除消融**：去掉 KG 中 `het:CrC` 边后，再跑：
   - EmerGNN（按其原始 protocol）
   - Ours
   - 比较 PD / PK 两个 mechanism 类别的 AUC 衰减
3. **High-degree hub 过滤**：移除 top 1% degree 的 KG 节点，再比较两种方法
4. **PK / PD channel 在 suppressed CrC 后的有效性**：如果 CrC 移除后 PK channel 仍能 carry signal，说明 mol mediator 真正起作用（不是 CrC shortcut）
5. **Shuffled-within-bucket negative control** (codex r2)：把 mediator 特征在相同 degree 桶 / 相同 relation 桶内 shuffle；如果 AUC 没掉，说明所谓 "selected mediator" 只是高 degree node，不是 mechanism 载体
6. **CrC-only LR predictor** (codex r2)：仅用 compound-resembles-compound 边能预测多少？建立 shortcut upper bound
7. **Scaffold split / entity-disjoint split**：除了 pair-disjoint，加 endpoint-disjoint + scaffold-disjoint，验证我们的 contribution 不依赖弱 split

只有这组 control 跑下来证明：
- 我们**不是 CrC shortcut 的换皮**（control 1-4, 6）
- 我们**不是 high-degree hub 选择器**（control 3, 5）
- 我们的 mechanism-aligned channel **真在做 mechanism reasoning**（control 4, 5）

…i2 的方法学 claim 才稳。

---

## 5. EmerGNN 的核心缺陷（基于 i1-i4 重新诊断）

不再用 ranking "improvement potential"，改用**缺陷诊断**——4 个缺陷都对应 i1-i4 中一条 insight：

| 缺陷 | 命中 insight | EmerGNN 表现 | 我们的 structural fix |
|---|---|---|---|
| **A · Perception block** | **i3** | Relation-type α_r 对同 relation 下的所有边一致 → 无法在 type 内区分具体 mediator instance；同时 flow 顺序传播把信号沿 path 摊薄 → 关键节点感知不到 | Per-instance pair-conditional attention 在 M 上（i3 直接 unblock） |
| **B · 端点 anchoring** | **i2** | representation 在 drug v 处累积，需要 L 步 flow 把 u 的信号传到；长 path → over-smoothing（EmerGNN 自己的 Fig 5c 显示 L>3 性能下降）| Set-retrieval at M（i2 paradigm change）；K=2 GCN 在 M 局部够用，**深度与 KG 距离解耦** |
| **C · ID-embedded intermediate** | **i4** | KG 中间节点用 learned ID embedding，cold-start 下 unseen 节点是初始值（信息量为零）| PubMedBERT([node.name]) init（i4）；cold-start drug 的 name **永远 known**，第 0 层就有信号 |
| **D · Flat mechanism reasoning** | **i1**（次要）| 单 readout 处理全部 86 种 DDI 类型，无 PK/PD 范式区分 | Dual-channel routing；**但这是 task structuring，不算 paradigm-level 创新** |

**主战场是 A + B + C**——这三个是真正的 method-level 创新点。D 是 i1 在 DDI 任务上的具体应用，不算 paradigm novelty（per 用户判断）。

**核心诊断词："Perception block"**：EmerGNN 不是不能 reason intermediate，是**它的方法在识别 path 上关键信息时存在 block**——relation-type 共享 attention + sequential flow dilution 的双重机制使它**结构性地无法 selectively perceive** 哪个具体 intermediate 是 query pair 的核心。我们的 set-retrieval + per-instance perception **structurally 移除这个 block**。

---

## 6. Cold-start 上超越 EmerGNN 的 4 条 promising 说法

不要求严格证明，但 plausible + 启发方法设计：

### 6.1 · Depth-distance 解耦（i2 命中 B + 缺陷 B）

cold-start S2 下 **20% pair 需要 SSPL ≥ 3**（E1a）。

- EmerGNN：必须 L ≥ SSPL 才能让 u 的信号到 v；其自己 ablation 显示 **L>3 性能下降**（接近 over-smoothing 拐点，per Oono-Suzuki 理论 4-8 层指数衰减）
- Ours：K=2 GCN 在 M 的 1-hop / 2-hop 邻域就够；**深度与 SSPL 无关**

→ **Cold-start 上结构性优势**：对那 20% 远距离 pair，EmerGNN 必须冒 over-smoothing 风险，我们不受影响。

### 6.2 · Unseen-node 信号注入（i4 命中 C + 缺陷 C）

cold-start S2 下两 drug + 大量 KG 中间节点都是训练时未见过的。

- EmerGNN：unseen 节点的 ID embedding 是初始值 → flow 通过这些节点传播必然损耗；CoT signal-from-zero
- Ours：PubMedBERT init 让 unseen drug 和 mediator **从第 0 层就有 informative vector**——E3 已实证 specific name semantics +5.65pt（real vs shuffled）

→ **Cold-start 上结构性优势**：EmerGNN 的 cold-start 性能本质上受限于 KG 训练曝光率；i4 把这个限制解除。

### 6.3 · Per-instance perception unblock（i3 命中 A + 缺陷 A）

cold-start 下，pair 的关键 mediator 往往是**specific entity**（specific CYP isoform / specific phenotype），不是高 degree hub。

- EmerGNN：relation-type 共享 α 在这种 case 上严重失效——同 type 下两个具体节点权重一致
- Ours：per-instance pair-conditional attention 能区分。E8.6 LLM oracle (Sonnet 4.5) 用 per-instance selection +8.7pt PD AUROC vs bag attention；E8.7-hetero (Opus+GPT-4o) PD 0.737 跨模型稳定

→ **Cold-start 上结构性优势**：emerging drug 的关键机制信号往往落在 long-tail mediator 上，而非高 degree hub；per-instance perception 在 long-tail 区域优势放大。

### 6.4 · Retrieval-augmented compute 优势（i2 paradigm-level）

- EmerGNN：path-flow 需要给 G^L_{u,v} 整个 path-subgraph 学 embedding，每对 query 的 cost 是 O(|E_subgraph| × L)
- Ours：retrieval = set 操作 N(u) ∩ N(v) 是 O(|N(u)| + |N(v)|)；encoding 只在 M 局部 GCN K=2

→ **Compute 优势**在 cold-start scenarios 放大：cold-start = 大量 unseen drug 涌入 → query pair 数量爆炸；我们的 retrieval 每 query 是 lightweight set lookup，flow 则需 full subgraph processing。**适合规模化部署**。

---

## 7. 建立在 i1-i4 之上的 method design problems

不是 "failure modes we admit to"，而是 **i1-i4 联合定义的 design problems，我们的方法需要 explicit 解决**：

| 问题 | i1-i4 给的解决方向 |
|---|---|
| **P1 · M 集合在 1-2 hop 内为空 / 稀疏** | i2 + i3 + i4：extended **soft-intersection** 到 ≤ 3 hop（带衰减权重）；pair-conditional perception 在 extended candidate pool 上 selection；text-semantic neighborhood 提供 virtual mediator（即使 KG 没有显式连边，PubMedBERT 能找到语义近邻）|
| **P2 · Mediator 可用性依赖 KG schema 完备性** | i4 dominant：当 KG 关系类型 / 节点类型不完备时，**node-name text embedding** 是 schema-agnostic 的稳定 fallback。EmerGNN 的 relation-type attention 完全依赖 schema，**反而**是它的弱点 |
| **P3 · Multi-label polypharmacy (TWOSIDES 类)** | i1 extended to label-level：每个 SE label 有自己的 PK/PD 倾向（QT prolongation = PD，CYP-mediated 代谢变化 = PK）；dual-channel 扩展为 **per-label mechanism-routed prediction head** |
| **P4 · Mediator-level vs Path-level 可解释性** | 不是劣势是 framing 选择：mediator-level ("通过 CYP3A4 介导") 比 path-level ("Drug A → enzyme X → drug Y → Drug B") **更 clinically actionable**——药师关心的是机制，不是图论路径 |

**核心论断升级**：我们的方法不是"在 4 个 insight 上 build 一个新模型"，而是 **"i1-i4 联合定义了一个新的方法学问题域 (P1-P4)，我们的方法是这个问题域的第一个系统化解"**。

---

## 8. 必须的 fair hybrid + cheap baseline

Codex round 2 强调：要防 "EmerGNN + features" 攻击，还要防 "cheap baseline 也能 match" 攻击。

### 8a · Fair hybrid baselines（防 "我们就是 EmerGNN + features" 攻击）

| Baseline | 描述 | 检验什么 |
|---|---|---|
| EmerGNN 原版 | 论文 protocol | 主对手 |
| EmerGNN + PubMedBERT init | 中间节点 init 换 PubMedBERT | D 是否在 EmerGNN 框架下也 work |
| EmerGNN + PK / PD 双 head | 输出层分 PK/PD head | C 是否要在 EmerGNN 框架内独立验证 |
| EmerGNN + same node/text features | 直接给 EmerGNN 所有我们用的 features | 验证 features 不是核心 |
| Ours full | meeting-node + PubMedBERT + PK/PD | 完整方法 |
| Ours w/o PubMedBERT (use ID) | 验证 D 不是必需 |  |
| Ours w/o PK/PD（flat readout）| 验证 C 关键 |  |
| Ours w/o mediator readout（use endpoint）| 验证 A 关键 |  |
| Ours with random / top-k-degree mediators | 验证 selection 不是随机也能 work |  |

### 8b · Cheap baselines（防 "简单模型也能赢" 攻击）

Codex round 2 warning："如果一个 cheap feature model 能匹配你，architectural story 就崩了"。

| Cheap baseline | 描述 | 攻击面 |
|---|---|---|
| **LR on CrC features only** | 仅用 compound-resembles-compound 边的 1-hop/2-hop intersection count | 检验 EmerGNN 的 CrC shortcut 单独有多强 |
| **LR on 1-hop / 2-hop metapath counts** | 仅用 metapath count 特征（drug-protein-drug, drug-SE-drug, etc.）| 检验 hand-crafted topological 特征上限 |
| **MLP on drug fingerprints only** | 不用 KG，只用 SMILES fingerprints | cold-start 下 fingerprint 信号上限 |
| **Random mediator selection in our framework** | 验证 retrieval 不是随机 work | 已在 7a 里 |

如果 LR-CrC 或 LR-metapath 在 S2 上 AUC 接近 EmerGNN 或 ours，说明"我们的 mediator perception"本质上是"高级化的 hand-crafted topological feature"——不是真正的架构创新。

### 8c · 报告标准（codex r3）

所有 ablation 必须：
- **多 seed**（≥ 3 个 random seed）→ 报告 mean ± std 或 bootstrap CI
- **Parameter / compute matched**：EmerGNN + 我们的 features 时要保证总参数量 / FLOPs 可比，防"我们大模型当然好"反驳
- **Scaffold split 类型明确说明**：用 molecular scaffold split（按 Bemis-Murcko）还是 drug-combination split（按药物身份）——两者 cold-start 程度不同，论文里写清

---

## 9. 我们 claim 应该有多强（post-codex-r2 收紧）

**正确的 claim**：
1. **Mediator perception 是 cold-start DDI 信号的一个 plausible / high-yield explanatory unit**（i2 + E1b 拓扑 + E7 LR baseline + E8.6/E8.7 LLM oracle 实证）——避免 "natural" / "the only correct"
2. **PK / PD 双通道**对应实证的 **mechanism-aligned** asymmetry（E1b + E4 + E8.7 cross-class）——避免 "mechanistic" 这种 implies validated 的措辞
3. 我们**不**比 EmerGNN 在所有场景都好——在没有清晰 mediator 但有长因果链的 pair 上他们更优。这是 **boundary condition**，不是 apology
4. 我们**确实**需要 control CrC shortcut（control 实验跑完前不 claim 主因果归因）

**不应做的 claim**：
- "EmerGNN 不能 reason intermediate"（错——它通过 path-subgraph 可以）
- "我们完全避开 over-smoothing"（错——K=2 仍可能 blur）
- "PubMedBERT 是我们的核心创新"（错——是 feature 选择，需要在 EmerGNN 框架下验证）
- "Mediator explanation 是因果"（错——只能 claim mechanism-aligned，因果归因需要 stability + perturbation 测试）

---

## 9b. 必须的 crossed ablation（防 axes 1 和 2 塌缩）

Codex round 2 警告：如果不交叉 ablate"readout target × selection granularity"，reviewer 会说"你的 mediator readout 之所以好只是因为你顺便换了 retrieval"。需要 2×2 grid：

| 设计 | Selection | Readout | 检验 |
|---|---|---|---|
| Cell A (baseline EmerGNN-like) | unconditioned / topological | endpoint (drug v) | 主对手 |
| Cell B | **pair-conditioned mediator selection** | endpoint (drug v) | selection 单独 contribution |
| Cell C | unconditioned mediator selection (random / top-k degree) | mediator readout | readout 单独 contribution |
| **Cell D (Ours full)** | pair-conditioned mediator selection | mediator readout | 完整方法 |

**Separability 判别准则**（codex r3 修订，原表述 `D-A > (B-A)+(C-A)` 实际测的是 superadditive synergy，不是 independence）：
1. **`B > A`**：selection 单独有效
2. **`C > A`**：readout 单独有效
3. **`D > B` AND `D > C`**：crossed 时双方仍添加价值
4. **Interaction term `I = D − B − C + A`** 解读：
   - `I > 0`（明显正）→ synergy，两个 contribution 互相加强
   - `I ≈ 0` → additive independent，两条都 work 但互不放大
   - `I < 0` but `D` 仍最佳 → partial overlap，两个 axes 互相替代但各自仍贡献
   - **只有 `D > A` 而 `B ≈ A` 且 `C ≈ A`** → coupled mechanism，**不**能 claim 两条独立 axes

判别原则：**"We claim independent contributions only if each axis improves performance when added alone AND still contributes when crossed with the other; superadditivity is treated as stronger evidence of synergy, not required for independence."**

---

## 9c. Mediator stability + perturbation 测试（防 post-hoc 攻击）

**重要措辞**（codex r3）：Mediator 输出是 **predictive rationale**，**不是** causal mechanism。Paper 里需要明确写"selected mediators reflect predictive evidence patterns aligned with biomedical mechanisms; causal attribution requires curated benchmarks or wet-lab validation outside our scope."

Codex round 2 第二个 attack："Mediator 解释是 post-hoc rationalization"。要回应：

1. **Stability**：同 pair 在不同 seed / 不同子图采样下，选出的 M 是否一致？Jaccard overlap > threshold？
2. **Removal sensitivity**：把选出的 top-k M 节点从 KG 中去掉，AUC 是否大幅下降？（如果下降，说明 mediator 真的是 prediction-critical）
3. **Substitution test**：用随机选的同 kind 节点替换 M，AUC 是否下降？
4. **Mechanism plausibility audit**：在 30-50 个 high-confidence prediction 上人工 / LLM-as-judge audit，selected M 是否对应已知机制
5. **Selected mediator distribution vs high-degree hub distribution**：是否只是在 surface KG 上选 high-degree hub？

---

## 10. Compose 可能性

可以**hybrid**——EmerGNN path extraction + 我们的 mediator readout + PK/PD channel + PubMedBERT init。但作为论文论述，**主结构应当 claim mediator-anchored**（i2 + i3 + i4 闭环），flow 是 path-extraction tool，不是 readout 路径。

---

## 相关文件

- EmerGNN 完整 paper review：`Notes/Settings/Paper-Review/c2_path_multihop/zhang_2023_emergnn.md`
- 我们的 i2 文档：`Notes/Settings/Insights/i2.md`
- 我们的 i1（PK / PD 范式）：`Notes/Settings/Insights/i1.md`
- E1b 拓扑：`Notes/Log/insight-discovery/exp2-0513-test-insight/02_path_endpoint_layer.py`
- E2 over-smoothing：`Notes/Log/insight-discovery/exp2-0513-test-insight/04_depth_sweep.py`
- E7 meeting-node LR：`Notes/Log/insight-discovery/exp2-0513-test-insight/08_meeting_node_lr.py`
- E8.6 / E8.7：`Notes/Log/insight-discovery/exp2-0513-test-insight/08f_*.py`, `08h_*.py`

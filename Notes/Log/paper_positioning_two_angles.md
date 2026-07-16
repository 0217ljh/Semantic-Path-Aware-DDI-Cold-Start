# 两个 paper 定位 + 各自待实现清单(target: AAAI 2027, 投稿 2026-07-28)

2026-07-01。纯结构线(多模态 parked)。两个叙事角度并列,供选型。数字均本 session 已 verify。
共享前提见末尾。相关:[[metapath_discovery_analysis_design]] · [[spmn_v2_theory_anchor]] ·
[[spmn_v2_autoresearch_plus1pt]] · [[paper_locked_logic]]。

## 已有资产(verified,两角度共用)
- **PMP standalone(spmn_v2 aware, copath OFF)**:S2 AUROC **0.7765±0.020**(3-seed),AUPRC 0.7884。
- **对手**:EmerGNN ~0.7458(+3.07pt);NBFNet ~0.784/0.747/0.750(≈ 我们)。
- **消融梯**:naked AND 0.7658 → +bf 0.7729 → +OOD 0.7743 → +mse 0.7765(3-seed,落盘)。
- **EmerGNNFast**(本 session):CSR 传播 1.69x、数值等价——让 EmerGNN 能在更大/更公平 KG 上跑。
- **理论**:Claim ①(见证可达性)、Theorem 5.1(MPNN 证明不能保留 typed common-neighbor)。
- **粗+细**:仅单 seed(copath ON 0.779~0.7794),≈ 纯粗;**无干净 3-seed**;KG-only co-path 无净增益。

---

# 角度 A —— 通用结构"涨点" adapter(PMP)

## 一句话
> Pair-Mediator Pooling (PMP):一个近乎无参、backbone-无关的**结构 adapter**,插到纯结构 DDI
> backbone 上即在 cold-start 显著涨点;因为它显式保留传播式 GNN 会稀释的 typed common-neighbor 证据。

## 核心 contribution
1. PMP 原语(端点仅作选择器,打分只来自共享中介池)→ 对端点 embedding 质量**架构不变** = cold-start by design。
2. **通用性**:同一 adapter 插到 NBFNet / EmerGNN 都涨。
3. 理论:Theorem 5.1(MPNN 不保留 A_τ)+ Claim ①。

## Headline
"PMP 把 S2 AUROC 从 0.7458(EmerGNN)提到 [插件融合目标 0.79~0.80],且 control 证明来自共享中介语义而非容量。"

## 证据状态(have / need)
- ✅ PMP standalone 0.7765、胜 EmerGNN、消融梯、理论。
- ❌ **"通用涨点"核心实验没做**:PMP 作为**插件**同时抬升 ≥2 backbone(NBFNet+PMP、EmerGNN+PMP)。
  现有 PMP 是 standalone/对比,不是"挂上去抬 backbone"。
- ⚠ 融合 ~0.805 只是单处提及、**未 verify**(我这 session 没找到落盘)。

## 还需实现(A)
| # | 内容 | 类型 | 备注 |
|---|---|---|---|
| A1 | **插件融合实验**:frozen NBFNet + PMP fusion head;frozen EmerGNN + PMP fusion head → 各涨多少 | 训练(GPU) | "通用"卖点的命根;≥2 backbone |
| A2 | 验证/复跑那个 ~0.805 融合数字(3-seed) | 训练 | 现在是孤证 |
| A3 | baselines on 统一 S2 benchmark(EmerGNN/NBFNet/SumGNN/HDN-DDI/…) | 训练(用户在跑) | 主表 |
| A4 | (可选)跨任务通用:binary/multi-cls/multi-label 同 adapter | 训练 | 加"universal across tasks" |
| A5 | control:shuffle/random/mediator-drop(证明来自语义非容量) | 分析 | 部分已有 |

## 风险
- "通用"若只在 1 个 backbone 上涨,卖点塌 → A1 必须 ≥2 backbone 都正。
- 融合数字未证,可能达不到 0.80 → headline 要留活口(用实测值)。
- 跟 EmerGNN 太近的 reviewer 攻击 → 靠 Theorem 5.1 + 显式 A_τ 区分(已锁措辞)。

---

# 角度 B —— 纯粗+细的结构 meta-path 方法(HIN-DDI 的学习化增强)

## 一句话
> 一个纯结构、可归纳、cold-start 的 DDI 方法 = **粗粒度(高阶相遇家族)+ 细粒度(meta-path)**,
> 是经典手工 meta-path 相似度(HIN-DDI)的**学习化 + 冷启动**全面增强版;能**自动发现**手工目录里
> 没有的、生物有意义的新 meta-path,而传统 GNN 对此做不好。

## 核心 contribution
1. 粗+细两级结构方法(粗发现相遇家族、细内生还原具体 meta-path)。
2. **自动发现新 meta-path**(headline: 共享 biological-process 模块的高阶相遇),生物有意义、经因果删除确认、HIN-DDI 手工集无。
3. 理论:Theorem 5.1 + coarse-to-fine necessity(粗提取隔离结构信号、细浅传播只精化)。

## Headline
"我们的方法自动发现高阶 meta-path(两药经多跳汇聚到共享 biological-process 模块),HIN-DDI 手工短路径没有、传统 GNN 只能弱恢复;性能与最优纯结构方法持平(0.77~0.78)。"

## 证据状态(have / need)
- ✅ 粗 standalone 0.7765;理论;Exp 0 候选 schema(BP 相遇覆盖 0.705)。
- ❌ **细粒度组件没接进赢家模型**:0.7765 standalone 只有粗;§3.2 细路径编码器(NBFNet(a→m) in hyper-edge)未 wired → **当前内生还原不了高阶 BP 的具体 path**(2-hop 是占位)。
- ❌ 无干净 3-seed 粗+细(≥0.7765 待证)。
- ❌ meta-path 发现分析 Exp 1-5 未做。
- ❌ HIN-DDI 确切目录(S_HIN)+ HIN-DDI baseline 复现;外部生物验证集。

## 还需实现(B)
| # | 内容 | 类型 | 备注 |
|---|---|---|---|
| B1 | **接入细粒度组件**(§3.2 hyper-edge 内 path 编码 或 co-path),让方法**内生还原** meta-path | **开发** | 新变体,不改 0.7765 standalone;方案 1(粗GNN+细GNN)vs 方案 2(NBF)择一 |
| B2 | 3-seed 粗+细(copath/细 ON)确认 ≥0.7765 | 训练(GPU) | 夯"性能不损"地基 |
| B3 | Exp1 高阶相遇消融(1-hop vs +2-hop,证"高阶→涨点/不损") | 训练+分析 | 故事第一段地基 |
| B4 | Exp2 PMP 相遇家族**注意力加权**重要度(BP 是否真靠前) | 分析(只读,载模型) | 决定 headline 站不站 |
| B5 | Exp3 因果删除 vs matched control(BP 家族必要性) | 分析(只读) | 非循环关键 |
| B6 | Exp4 **内生**路径还原(BP 家族具体关系序列) | 分析(依赖 B1) | "发现具体 path"的证据 |
| B7 | Exp5 plain GNN 对照(归因保真更差) | 训练+分析 | "GNN 做不好" |
| B8 | HIN-DDI 目录 S_HIN + HIN-DDI baseline 复现 | 文献+开发 | novelty 基准 + 被增强对象 |
| B9 | 外部 PK/机制验证集(DDInter/FDA) | 数据 | 生物验证(避循环) |

## 风险
- B1 细粒度接入是真开发,且历史证据显示细在纯 KG **无净增益** → 细的存在理由=**可解释性/path 还原**,不是涨点(靠 coarse-to-fine "细是降级精化层"框架吸收)。
- 若 3-seed 粗+细明确 <0.7765 → "性能不损"塌,退回"性能持平、卖分析"。
- BP 高阶相遇的"具体 path 还原"依赖 B1;若 B1 做不出,headline 退到"相遇家族"(较粗)。
- novelty=中介 KIND+高阶(非关系-角色);比有向 CYP 弱,但真实数据只支持这个。

---

# 两角度对比 + 时间线(到 7/28 约 4 周)

| 维度 | A 通用 adapter | B 粗+细 meta-path 方法 |
|---|---|---|
| 脊椎 | **性能/通用涨点** | **meta-path 发现分析** |
| 主要缺口 | 插件×多 backbone 实验(训练重) | 细粒度接入(开发)+ 发现分析(只读多) |
| GPU 依赖 | 高(多个融合/backbone 训练) | 中(细变体 1 训 + Exp1/Exp5 训;分析多只读) |
| 开发量 | 低(PMP 已有,搭 fusion head) | 高(B1 细粒度组件) |
| 被 reviewer 攻击点 | "就是 EmerGNN+共邻" | "细粒度不涨点、发现是否 circular" |
| 差异化强度 | 中(通用性 + 理论) | 高(自动发现新 meta-path + 理论,故事独特) |
| 与已锁 §5(Thm5.1)契合 | 高 | 高 |

## 是否互斥?
不完全。可**一篇 paper 两 pillar**:主 = PMP(性能/通用),analysis section 放 meta-path 发现。但:
- 若 headline 押 A → B 的发现分析只需做到"相遇家族 + 因果删除"(Exp2/3/5),**可不接细粒度**(B1/B4/B6 省),分析讲粗层发现即可 → **省一大块开发,4 周更稳**。
- 若 headline 押 B → 必须接细粒度(B1)+ 全 Exp1-7,开发+实验都重,4 周紧。

## 我的初步判断(待用户定)
7/28 时间紧 + GPU 被 baseline 占。**A 作为 headline + B 的"相遇家族发现"作 analysis(不接细粒度)** 是**最省、最稳**的组合:PMP 性能已 3-seed 锁死,只差插件×backbone(A1/A2)+ 分析 Exp2/3/5(多为只读)。把"内生细粒度还原具体 path"(B1/B4/B6)列为**加强项/rebuttal 弹药**,行有余力再上。纯 B(必接细粒度)风险最高。

## ✅ 决定(用户 2026-07-01):走**合一**
A headline(通用 PMP 涨点)+ B 的**相遇家族发现**作 analysis,**暂不接细粒度**(B1/B4/B6 降级为
加强项/rebuttal 弹药)。理由:有 7/28 ddl,必须拿出能投的;这是最省最稳组合。
执行清单 = A1/A2/A3(需 GPU)+ 分析 Exp1(✅ 已做,模型无关前提)/Exp3/Exp4(需模型,等 GPU 重训)。
gating:0.7765 无 checkpoint → 模型相关分析要等 GPU 重训一次。

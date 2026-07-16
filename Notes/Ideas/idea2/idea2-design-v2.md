---
title: Idea2 v2 — 机制分型的冷启动 DDI：typed PK 路由 + PD 脱靶 liability 补全
status: design (locked 2026-07-01, supersedes idea2-design.md)
tags: [idea2, cold-start-ddi, mechanism-typed, pk-pd, liability, dynamic-routing]
---

# Idea2 v2 设计文档（机制分型版）

> 本文件是 idea2 的 **v2 锁定版**，不覆盖 `idea2-design.md`（v1，LLM-修复分层 KG 版，已被证据推翻其结构-修复前提）。
> ARIS auto-research 进度记录见 `Notes/Log/idea2/aris-progress.md`；代码在 `Code/code-idea-2/`。

## 0. 一句话贡献

**冷启动 DDI 是机制分型的**：PK 标签可从**现有酶/转运体 KG typed motif** 预测；而 PD 标签的因果**单药脱靶 liability anchor 在 KG 里系统性缺失、且被相关的 PK hub 混淆**——所以静态 KG 聚合必然失败，除非**先补 liability 层、再按机制类型路由 pairwise 证据**。

## 1. 已核实的诊断（本会话，v15，degree-matched 单 meta-path AUROC，215 类按描述分 PK/PD）

| 机制 | shared protein | shared enzyme(db:enzyme) | shared target | 判别性 |
|---|---|---|---|---|
| PK 总(21+ 类) | **0.825** | 0.79 | 0.52(惰性) | **强** |
| ├ absorption | 0.894 | 0.855 | 0.52 | 强 |
| ├ metabolism(CYP) | 0.839 | 0.798 | 0.51 | 强 |
| └ excretion | 0.740 | 0.749 | 0.51 | 强 |
| PD 总(184 类) | 0.604 | 0.55 | 0.55 | 弱 |
| ├ QTc/torsade(5.2万对) | 0.616 | 0.55 | 0.56 | 弱（最佳 gobp 0.668）|
| ├ BP/vascular | 0.53 | 0.51 | 0.53 | ~0.5 |
| └ hypersensitivity | 0.47 | 0.46 | 0.50 | **不判别** |

**Smoking gun（QTc = hERG/KCNH2 阻断引起）**：hERG = `prot:3757` 在 KG 里，但**只 76 个药连它**；1293 个 QTc-DDI 药中**仅 58 个(4.5%)连 hERG**。QTc 药真正共享的是 PK 代谢 hub：**CYP3A4 47% / ABCB1 28% / CYP2D6 26%**。
→ KG 对 PD 双重失败：**(i) 漏掉因果脱靶靶点**（liability，DrugBank 不标脱靶为 target，4.5% 覆盖）；**(ii) 被 PK 混淆误导**（共享 CYP3A4 与 QTc 药相关但非 QTc 之因）。泛化：PD 因果靶都是脱靶 liability（hERG→QTc、5-HT2/serotonergic→5-HT 综合征、muscarinic→抗胆碱、GABA/opioid→CNS 抑制），系统性缺席 drug-target 边。与项目旧发现一致：分子/靶点在 PD 上不行，TextDDI(临床文本) 在 PD 上最好。

## 2. 方法（2-step，codex 锁定 2026-07-01）

### Step 1 — 单药 liability 补全（label-blind，offline，推理零 LLM）
把 KG 系统性缺失的 PD 脱靶 liability 蒸馏成 **typed anchor 节点**：`hERG_blockade_liability`、`serotonergic_excess_liability`、`muscarinic_antagonism_liability`、`gabaergic_cns_depression_liability` …
- **硬约束**：纯单药任务。无 DDI 对、无 DDI 标签、无 pairwise prompt、推理无 LLM。
- **实现优先级**：hERG 先用**结构化化学预测器**（不是 LLM，更干净可辩护）；再对比 `Liab-Struct` / `Liab-LLM` / `Liab-Hybrid`。
- **独立评测**：liability 模块先对外部单药药理/脱靶证据评，再看下游 DDI 增益。论文里叫 **liability completion**，不叫 "LLM 药理发现"。

### Step 2 — pair-conditional 机制路由器（动态调控，3 头）
| 头 | 读什么 | 主用于 |
|---|---|---|
| **A. PK-shared** | L1 typed 酶/转运体 motif | PK 标签 |
| **B. PD-shared-liability** | Step1 插入的 liability anchor | PD 标签 |
| **C. PK→PD potentiation** | `A 抑制 CYP3A4 ∧ B 是 CYP3A4 底物 ∧ B 有 QT liability` | 混合因果 |

- **PD 标签从 B/C 读，不从 A 读**。"两药都走 CYP3A4" 本身不是因果，C 头（抑制+底物+liability）才是。
- **去混淆**：显式分头 + **轻量反事实正则**（PD 预测删掉 liability 边应崩，只留共享 CYP 不应撑住）。不用对抗 invariance（机制性混淆，显式分头更干净）。
- **硬规则**：macro(GO/pathway/anatomy) 只能**佐证** PD，不能当主驱动；PD 主 anchor 必须是插入的 liability 层。
- 承接 v1 的动态邻域 + 2D 路由（纵向 micro→macro、横向 drug-drug）+ hub-handling（通用 hub 压 / 机制 hub typed motif 救），但落到 3 头机制路由上。

## 3. 任务与目标
- 三任务全覆盖：binary_cls / multi_class_cls / multi_label_cls（数据 `Code/data/ddi_unified/`）。
- **目标：三任务都明显优于 baseline（baseline = EmerGNN 单个），差距不小。**
- 冷启动 setting（inductive S1/S2），KG = `_merged_kg_dedup_v15`。

## 4. 建造顺序（带 kill-test）
1. **纯 PK typed 模型**（v15、无 LLM、无 liability）→ 打干净 PK cold-start。
2. **最小 QTc kill-test**：只加一个 liability family（hERG/qt_liability）。
3. 对比：shared-PK baseline / liability-only / 3-头路由器。
4. **degree-matched + CYP-matched 负样本**；ablate liability 边 + PK→PD motif。
5. **杀标准**：QTc 在 CYP-matched 对照下打不过 shared-CYP baseline → 砍/重修 PD-liability；赢 → 扩到 5-HT/muscarinic/GABA/opioid。

## 5. Novelty 边界（competitor = EmerGNN；论文对标 ColdDDI 圈）
- 分子(SSI/DSN/HDN)：抓不到 PK 酶共享，也抓不到脱靶 hERG。
- 静态 KG(TIGER/MKG-FENN/**EmerGNN**)：吃 CYP 混淆、漏 hERG liability。
- TextDDI：文本拿 PD 但无结构化机制。
- **idea2 delta**：mechanism-typed —— PK 从 typed KG 路径恢复；PD 补缺失脱靶 liability + 分离 PK 混淆与 PD 因果 + PK→PD 增效。

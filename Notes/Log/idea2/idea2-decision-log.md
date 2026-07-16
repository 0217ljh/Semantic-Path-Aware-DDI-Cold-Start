---
title: Idea2 决策流水
created: 2026-06-28
tags: [idea2, decision-log]
---

# Idea2 决策流水（chronological）

设计主文档见 `Notes/Ideas/idea2/idea2-design.md`。本文件只记关键决策与"为什么"，按时间累积、不覆盖。

## 2026-06-28 初次成形

1. **方向锁定**：从 KG 侧解 PD 的普遍缺陷。依据：molecular 在 PD 不行、TextDDI(文本) 在 PD 最好 → PD 信息在机制/语义层，赌它在 KG 里被埋。
2. **competitor 缩到 DDI 圈**（ColdDDI 6 baseline），不与 general graph 方法比创新。general 方法"拿来+改"即新，AAAI 理论轻。
3. **三个发力位置**：分层 / 动态 / LLM 整合。HyGRAG(WWW2026) 的分层 + 动态感受野作主参考。
4. **分层 = 固定生物树** protein→GO→effect→system（L0 molecule 预留 future）。汇聚 = 最低公共层相遇，PK/PD 涌现不写死。
5. **核心方法 = idea1(posterior quotient relink) + idea6(ephemeral mechanism hypergraph)**，1 参数化 6。
6. **双图架构**：G_gen 一次性 LLM 增强 + drug 用蛋白挂载 → unknown drug 推理零 LLM。被数据坐实（565k 对但只 1900 药）。
7. **LLM = 修复器不是预测器**（oracle 仅 0.686）。补全必须 **label-blind**；现有 chains.csv 泄漏作废。
8. **两阶段**：
   - Stage1 GATE = 断链恢复率 + 初步预测准确率（探针阶梯 + degree-matched + 三对照），**不含 hub-handling、不含动态**。
   - Stage2 = 动态邻域 + hub-handling。
9. **争议点收敛**：
   - 我曾把 hub-handling 放进 Stage1 → 用户反对 → hub-handling 移到 Stage2（它是判别工具）。
   - 我曾把 Stage1 砍成纯连通性 → 用户反对 → **Stage1 保留"初步预测准确率检查"**；只有 hub-handling 那套机制移走。
   - 结论：Stage1 = 恢复率 + 初步准确率探针（无 hub-handling）；Stage2 = hub-handling + 动态。两阶段增益可归因。
10. **存放范式确立**：`Notes/Ideas/idea2/`、`Notes/Log/idea2/`、`Code/code-idea-2/`，与正在跑的另一条 idea 隔离。

## 2026-06-28 KG dedup 落地（锁定 v2）

- KG agent 完成去重，canonical = `_merged_kg_dedup_v2`（141,381 节点 / 6,537,040 边），用户指定只用 v2。
- 新 schema：nodes `[id,kind,name,source_kg]`，edges `[src,src_kind,dst,dst_kind,relation,source_kg,directed]`，蛋白 `prot:<entrez>`。
- ✅ 蛋白碎片解决（`merged` source 恰=蛋白 29774，SLC12A1=prot:6557 单点，Furosemide 连它）。
- ⚠ 只合蛋白；meso/macro 层未跨源合并（kind 大小写重复为证）→ PD 深锚层仍碎片化 → **划给 idea2 LLM T1 实体对齐**。
- ⚠ 残留 44 db:target:BE 蛋白 + 41 重名；anatomy_protein_present 占 ~49% 边（疑垃圾）。
- 下一步：量 meso 跨源同名重叠 + 新 schema loader。

## Go/No-Go（codex 定）

S2 + degree-matched + PD 桶：completed 比 raw ≥ +0.03 AUROC 或 +0.05 AUPRC，≥2/3 探针，95% CI 不含 0，增益集中 PD，degree-only/null-completion 对照近 0。

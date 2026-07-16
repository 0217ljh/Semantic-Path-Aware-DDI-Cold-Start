# Idea: Drug-as-wave-source 传播 for cold-start DDI (PD 攻坚)

**日期** 2026-06-07
**一句话** 从"drug 作为波源在 KG 上传播"的角度干净地建模 DDI，**借用 NBFNet 的部分算法（路径机制/快速检索/冷启动归纳）但不套它的壳**；用复值（幅值=结构、相位=语义）+ 干涉读出，专攻结构上难分的 PD 类。

> 这是高层设计 idea。关联实验日志见 `Notes/Experiments/`；相位/干涉(Strategy a)在另一 session 深挖。

## 1. 问题诊断（已实测，verified）
- 真实 NBFNet v1.7 cold-start S2，按机制切：**PK AUROC 0.855 vs PD 0.727**（PD 落后 ~13 点）。`runs/...nbfnet_merged_L3_dim16...`。
- PD 是**结构性缺陷**（质的，非量的）：连接边量 PK≈PD(~54/对)，但 PK 的共有 mediator 走**具体低度数通道**(共享酶 54%/转运体 31%)，PD 走**弥散 hub**(共享副作用 44%/contraindication 41%/共享靶点仅 7%)。
- **几何(距离)本身不分 PK/PD**；判别在 mediator 的**类型/语义**。判别信号集中在 confluent (1,1)；远端 (2,2) 是 hub 噪声。
- DDI **不在 base KG** 里（生物骨架），但 v1.7 把 train DDI 注入 vKG（标准 NBFNet），仍救不了 PD。

## 2. 设计方向（架构）
**幅值=结构 / 相位=语义 的复值波传播 + 中介处两波前干涉 + (节点×深度)注意力读出。**
- 边界仍 indicator（保 path-sum 局部性 → 快速检索/冷启动），语义经**门控相位印记**进相位（不进初始状态）。
- 关系算子 = RotatE 式旋转（兼当"位置"）。
- 两波前（A、B 各传播），在**共享中介**处干涉 `⟨z^A,z̄^B⟩`（相对相位=共享语义），保留端点项（NBFNet 兼容）。
- 防 washout：分 **concept-slot**（每槽=一个机制概念，可解释）+ 归一化相干池化。
- 复杂度：传播≈baseline，唯一瓶颈是逐对中介读出 $O(P|\mathcal M|L^2d)$，**top-K 中介**强制（K~64）→ 总 ~1.5–2× baseline，可跑。

## 3. 关键约束（守住的 / 可丢的）
- **守住 BF 局部性**（每步只依赖本地邻居+query）→ 快速检索、摊销传播、冷启动归纳、"NBF-based"框架。门 α 必须 local，读出独立阶段。
- **可丢 严格半环/path-sum 精确性**——baseline 的 PNA+ReLU 早已不是干净半环，不需要。
- 复值时 PNA(max/min/std 无定义)→复加/复均值；ReLU→modReLU。
- 节点语义只给**固定生物节点**（effect/gene/protein，seen），药 θ=0 → 冷启动安全。

## 4. 已关闭的分支（probe 否掉，省得白建）
- **跨长度 / RoPE 相对深度注意力：否。** Probe(`2026-06-07__PD-crosslength-effect-probe`) 显示 PD 跨深度收敛同一 effect 的 **OFF-ONLY gap=0%**，非 PD 特异 → 同深度覆盖已 97%，跨长度无净增益。**不作为重头戏。**
- **"加性 vs 拮抗"用相位编码：暂否。** 二元标签无梯度对齐方向(Codex)，需机制级监督才谈。先只主张"同效应→相长"。

## 5. 待跑/待定（go-no-go）
- **[最高优先] identity probe**：共享 effect 节点的 **identity** 比单纯"计数"对 pos-vs-neg(PD) 多带多少 AUROC。≥0.03 → KG 含 PD 语义、表示问题、值得建复值/语义注入；<0.01 → KG 无此信息、直接上分子/文本。**这是整条线的 go/no-go，不占 GPU。**
- 三种语义放置(语义入相位 / 结构入相位 / 三通道分离)实现后都试。
- top-K 中介选取（按 $|z^A_v||z^B_v|$）进读出公式。
- 模型消融：RoPE-相对位置 vs depth-blind 注意力 vs 无注意力（结构 probe 测不了，必须模型层）。

## 6. 新颖性定位（Codex 提醒）
复数/相位/旋转/干涉(RotatE/ComplEx/MGC/QWNN)都不新。立得住的只有：cold-start DDI 特定诊断 + 保 source-indicator 归纳性下只注入固定生物节点语义 + **干涉/语义要打过显式 identity baseline**（否则=换皮）。**别叫 RoPE，叫 RoPE-inspired 相对相位相干匹配。**

## 关联
- 诊断: `Notes/Experiments/2026-06-05__DDI-samples-merged-KG-structural-analysis.md`, `2026-06-06__DDI-mechanism-KG-separability.md`
- 跨长度否定: `Notes/Experiments/2026-06-07__PD-crosslength-effect-probe.md`
- PK/PD transfer 实验: `Code/experiments/pkpd_transfer/`
- 相关脚本: `Code/scripts/analyze_ddi_*` / `analyze_pd_crosslength_effect_probe.py`

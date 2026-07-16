# Context-Aware DDI LoRA (封版框架 2026-07-02)

> 本篇是主线方法的定稿框架。本质是 AND-intersection typed-mediator 方法的
> **低秩 OOD-迁移 + 语义增强**版本,但 paper 以"诊断优先"框定贡献,而非
> "给 AND 加语义"。关联 [[project_semantic_path_ddi]]。

## 0. 一句话

Cold-start DDI 里**可迁移的信号是低秩的**;据此用一个显式低秩残差 adapter
(`H = H_base + M_A·M_B`)高效抓取迁移信号,再用语义在类型内做判别 refinement。
可插拔、cold-start 安全 by construction、也能单独预测。

## 1. Framing(paper 地基:warm/cold OOD)

逻辑链(箭头中标注**待验假设**):
```
warm 好(监督近 ceiling) / cold 差(~0.78)
  → [THESIS] cold 能迁移的信息本身低秩(全表征高秩,但迁移子空间低秩)
  → 像 warm 那样高维处理 = 引入大量冗余
  → 显式低秩矩阵分解拿最大迁移信息量
  → 在低秩上引语义,增强"类型内"判别力(targeted disambiguation,非高秩通道)
  → 可插拔 adapter,低秩设计,对多数非低秩模型有增益 [待普适实验证]
```

**低秩 = 迁移性质,不是 encoder 塌**。关键区分实验见 §4-A1。

## 2. 方法:`H = H_base + ΔM`,`ΔM = M_A·M_B`

- `H_base` = 冻结的结构节点表征,来自基 GNN(谱=GCN / 空间=EmerGNN)。
- **端点仅作选择器**:两药只选出共享 typed 介导(共同邻居 ≤2 hop),
  **打分路径无任何可学 drug embedding** → cold-start 安全 by construction。
  (实测:满 per-node entity embedding 是**死重**。)
- 二分 LoRA:
  ```
  M_B ∈ R^{(T·λ)×dim} = [B_1;…;B_T],  B_t ∈ R^{λ×dim}   基(每类型 λ 个)
  M_A ∈ R^{N×(T·λ)}   块稀疏(m 只在类型 t 的 λ 块非0)     系数
  Δ[m] = M_A[m]·M_B    仅类型 t 的 λ 块参与
  N_rep[m] = H_base[m] + γ_t·Δ[m]
  ```
  - `T` = KG node type 数(12,固定);`λ` = 每类型基数(超参);`a = T·λ`。
  - 粗 = typed 低秩基(哪些类型相遇);细 = M_A 的类型内分配。
  - 跨类型交互 = W_T(pooling 后),**不在** M_A(块稀疏=组内)。

## 3. 语义 + relation(细粒度)

- **语义进 M_B 原型 + M_A 由 z_m 导出(冷启动红线)**:
  ```
  B_t 的 λ 行 = 语义原型(LLM 介导 emb z_m 的类型内聚类)
  M_A[m] 静态 α_m = softmax(−‖z_m − 原型‖)  = z_m 的函数(非自由 per-node!)
  pair 门:  M_A_uvm = softmax( log α_m + gate(r̃) )    relation 在这
  🚩 M_A 不许自由 per-node 表 = 死身份线
  ```
  语义是 **within-type refinement / disambiguation**(平均边际、局部决定),
  不是 "widening",不是高秩通道。

- **广义关系 r̃(2-hop/多-hop 修正)**,可微、有上界、极限回落 node-type:
  ```
  r̃_a(m) = e_type(t_m) + s_a(m)
  ρ_a(m) = LayerNorm( (1/l_a) Σ_i e_r(r_i) )       加性组合(TransE 式,位置无关)
  s_a(m) = Gumbel-Softmax( W_slot·ρ + b )          可微离散,上界 C(3~5),无码本
  floor: C=1 → node-type + 1 slot;  max: C=3~5
  ```
  (乘性/RotatE 无优势:对角旋转 = 加性相位,codex 已判)。

- **role-expert 路由**(把粗族从固定 type 变可学)= **可选消融**,非主线。
  codex 判"固定 type 地板 + role 残差"可辩护;纯替换 type = over-engineering。
  推理时可塌成 W_b(签名空间可枚举 → 离线查表)。

## 4. 三个 Assumption + 验证

- **A1(低秩迁移)**:对冻结结构表征的**任务相关修正是低维的、低秩饱和**。
  - 不说"图 DDI 信息本质低秩"(intrinsic,打得掉);说"充分性"。
  - **证据 bundle(核心)**:
    - ★ warm/cold **秩截断分叉**(warm 随秩升、cold 早 plateau)
    - ★ **容量对照**:高容量残差 warm 涨 / cold 平(容量真实但非迁移)
    - ≥2 backbone 的谱/有效秩;adapter 秩扫描 plateau;entity_embed 死
  - 排除 (b)encoder 塌(warm~100% 证)与 DDI-本高秩(cold plateau 证)。
- **A2(介导中心 + 位置无关)**:收窄为"本 benchmark 短走廊内,精确位置在
  mediator 类型/身份之外几无增量信号"。**需 ablation**(位置无关池化 vs 无序
  typed-set 几无差),不是靠 case。**cold-start 来自"选择器/无身份 + A1",
  不是位置无关**。
- **A3(语义=类型内 refinement)**:同类型内不同 meta-path-like role,语义门
  分辨。杀手 case:**case2≡case4**(肾病—ALB 同链、标签相反,结构分不开)。
  需**聚合分析**(结构近同、标签相反的 pair 上语义门是否涨)+ **leakage 控制**。

## 5. 实验 / 分析 / 图

- **实验**:核心 = A1 的 warm/cold 秩截断分叉 + 容量对照;
  + 三任务(bin/mc/ml)、不同 backbone 挂载、adapter 单独预测。
- **分析**:低秩→**冗余**(非"稀疏"!)讨论:entity_embed 全冗余、简单 adapter
  ≈/> 复杂 GNN、hub→图秩冗余(**本篇作补充猜想,主理论留下一篇**)。辅以 case。
- **图(intro 主图)**:warm/cold **秩截断分叉曲线**(左)→ adapter 低秩+语义(右)。
  codex 判此强于 2D 坍缩投影。术语用"低有效秩/谱集中",**非 sparsity**;
  "低维子空间",**非一个点**;"所测 backbone 上",**非 general GNN**。

## 6. 贡献定位(顶会向)

以**诊断**为先,非"改进 AND":
1. 诊断:cold-start DDI 的可迁移信号低秩(warm/cold 分叉证);
2. 方法:据此的低秩 typed 语义 adapter(cold-start 安全 by construction);
3. 通用:可插拔 + 单独预测,跨任务/backbone 增益。

## 7. 待落实的经验赌注(别当已证)

1. **warm-start ~100% 要实测**(可能要建 transductive split);
2. **秩截断分叉/容量对照要真跑出来**(warm 涨 cold 平);
3. 普适增益("对多数模型")要多 backbone 实验证。

## 8. 下一步

- `Code/scripts/analyze_rank_sufficiency.py`:warm/cold 秩截断分叉 + 容量对照
  + 按类型可分性,GCN 先跑。产出 A1 证据 + intro 图素材。
- 关联:[[project_semantic_path_ddi]] · [[project_pmp_v1_5_official]] · idea2 是另一条线。

## 9. codex readiness + novelty verdict(2026-07-02)

**Verdict**: 值得建,但作为"诊断驱动的 inductive DDI paper",不是"DDI 大理论"。
Novelty = 交叉处 real-but-narrow(每个 ingredient 都不新,组合+诊断新)。

**要 center 的最强 claim**:"cold-start DDI 中可迁移信号的秩显著低于同分布全信号;
mediator-only typed 低秩残差利用之,比 entity-centric 高容量模型更泛化到 unseen 药。"
**必须丢**:"首个低秩/LoRA/语义 DDI" · "DDI 是低秩的" · "DDI 的 OOD 迁移理论"。

**最近 prior art**:EmerGNN + emerging-drug/cold-start DDI;meta-path/HIN-DDI;
inductive KG link prediction(无 entity emb);R-GCN basis 分解;LoRA;GNN rank-collapse
(但那些说深层 GNN 低秩=坏事,没人说"cold-start DDI 可迁移信号本就低秩")。

**新增 readiness 要求(进实验计划)**:
1. 诊断要"稳定现象":≥2 backbone、多 split/seed、多种改秩方式、控制(排除塌/欠训/容量错配)。
2. ★ 参数配平 baseline(堵"低秩=普通正则"):dense residual / bottleneck MLP /
   spectral-norm / weight-decay / PCA 压缩特征 / **随机低秩基**——低秩要打赢这些。
3. 按介导邻域丰度**分层** + 薄/空交集 fallback(选择器故事在稀疏邻域会崩)。
4. inductive 设定 airtight:叫 inductive/emerging-drug,不是 zero-shot;明确 unseen 药带什么(KG 邻域)。

**⚠️ 投稿前硬 gate — 真 novelty search(4 线)**:2024-26 cold-start/emerging-drug
DDI+KG+text/LLM;inductive 生物医学 KG link pred 用 text prototype/frozen LM emb;
DDI GNN 里的低秩/basis/adapter;明确论证"entity emb 伤 cold-start"的 DDI 论文。
若已有"inductive DDI via typed mediator KG + text" → novelty 压缩到只剩 rank 诊断。

## 10. Novelty search 结果(2026-07-02,gpt-5.4 xhigh 交叉验证)

**结论:novelty 中心已移到"诊断",方法单独看偏薄。PROCEED WITH CAUTION,diagnosis-first 是必需的。**

**实读结论(2026-07-02,读了 arXiv 摘要原文;codex 早先"方法偏薄/territory 被占"
基于坏摘要,已作废)**。诊断 + 方法在本领域均 **unclaimed**:
- **GenRel-DDI**(2601.15771):诊断="embedding 相似度≠DDI 标签",方法=relation-centric、
  独立于 drug identity。**无低秩/无 warm-cold 秩对比/无 adapter**。与我们**机制完全不同**;
  仅共享"cold-start 少用 drug identity"这个宽泛前提。**非竞品**。
- **AIM-DDI**(2605.14327):architecture-independent **多模态融合模块**,both-unseen。
  非低秩/LoRA。仅共享"可插拔+unseen-drug"高层 framing,机制不同。**非竞品**。
- **K-Paths**(2502.13344):training-free 路径抽取(Yen)。无低秩/adapter。**非竞品**。
- **LR-GCL/GCL-LRR**(2402.09600):低秩正则用于 **transductive** 节点分类,非 OOD/cold-start,
  与 DDI 无关,无 warm/cold 分叉。**非竞品**(证明"低秩→泛化"在通用图 ML 有,但非我们的诊断)。
- EmerGNN/KnowDDI/SumGNN/R-GCN/DDIPrompt:KG-GNN/路径/basis,均无秩迁移诊断与低秩 adapter。

→ 我们独有(headline):**rank-transfer 诊断 + 低秩 typed-mediator adapter(LoRA-DDI)**。
→ 护栏(措辞非 novelty):两个宽泛前提别当 headline —— "cold-start 少用 drug identity"
  (GenRel 共享)、"可插拔模块 for unseen-drug"(AIM-DDI 共享)。
→ related-work 引用时按此实读定性(已非二手摘要)。

**诊断(A)= 真 novelty(未被占)**:找到 PTDNet / LR-GCL(图低秩)与图-OOD(invariance/
decorrelation),但**没有**我们这条:"warm 随秩/容量持续涨、cold 早饱和、控制排除塌/高秩"。
GenRel 只到"加容量不改善泛化"(motivation 级),**没有谱/秩诊断**。→ A 能扛 paper,
但**只有当 paper 明确是"诊断驱动设计"**:"cold-start DDI 迁移是低秩的,这是它蕴含的最小架构"。

**要 center 的最强 claim(gpt-5.4 版)**:
> "在 entity-disjoint cold-start DDI 中,可迁移信号集中在共享 typed mediator 的低秩子空间;
> 强制这个 mediator-only 低秩瓶颈,比高容量 dense relation head 更可靠地提升 cold-start 泛化。"

**要丢/降调**:丢"首个可迁移 relation DDI"(GenRel 占)、"无 per-drug emb"广义 claim、
"可插拔模块"(AIM-DDI)、"机制 KG 路由"(SumGNN/K-Paths);降调 LLM prototype→实现细节
(除非有决定性消融)、"entity emb 死重"→消融结果非 headline。

**★ 对 GenRel 的必跑差异化实验**:同一冻结上游特征、同一 split 上做**参数配平 head-swap /
秩扫描**:GenRel 式 dense relation head vs 我们 typed mediator-only 低秩 head(+ dense
mediator-only head 对照)。**一张图**:warm 随秩涨、cold 早饱和、我们的受约束 head 在低秩/
更少参数下 cold 相当或更好。**这张图出来 = paper 的确切 delta;出不来 = 方法塌成工程。**
(= 就是 A1 的 `analyze_rank_sufficiency.py`,新增 GenRel-style dense head 作对照。)

## 11. A1 pilot 结果 + 锁定决定(2026-07-04)

**pilot**:plain GCN(关系无关)over 完整 `_merged_kg`,ddi800 S2,binary,inductive(无 drug-ID)。
`train_gcn_rank_pilot.py` + `analyze_rank_sufficiency.py`(Code/scripts/)。
- **欠训是主因**:full-batch = 1 grad step/epoch → 60 epoch 只有 60 步。按**步预算**训(1000 步)→ warm 0.78。
- **★ 测量层 = 任务 scorer 之前的最后一层共享 pair 表征(pre-readout)**。post-scorer 被 binary 头压成 rank-1
  (`--key Z`);**pre-readout(`--key raw`)上分叉清晰**:warm r95=128 / cold r95=32(4×);低秩处两者
  ≈ degree-only(可迁移的底),warm 随秩爬到 0.79(specific),cold 卡 0.70。sanity:探针≈head、degree-only 不复现。
  → 即使关系无关 GCN(训够)都显分叉,现象稳。

**锁定设计(codex 2026-07-04)**:
- 编码器×任务网格;**原生模型** + 功能定义"最后共享 pair 表征"(**不强制 uniform head**;uniform head 仅作受控消融)。
- 任务:**binary 主** + **1 个 typed(multi_cls/label)robustness**;别说"binary 充分",说"scope-limited to binary"。
- node 代表:**R-GCN**(+ 附录 1 个 encoder)。测量:**layer sweep** + 完整恢复曲线 + warm/cold 绝对满秩 AUROC。
- 措辞:"**可恢复信号的低有效线性维**",非"模型本身低秩"。别叫"max-info",叫"最后共享 pair 表征"。

**R-GCN baseline 配方(codex)**:2 层 R-GCN over 全 KG;逆关系 ID;hidden 128~192;root/self 变换;dropout 0.1~0.2;
特征 type-onehot+log-degree(+ 可选 per-relation `log(1+deg_r)`);**主 baseline 不加非药节点可学 embedding**
(会被喷记忆 mediator 身份;要就作明标消融 `R-GCN+non-drug-ID`);basis(非 block)num_bases 4~8 仅作次要设定;
**按步预算训**(2k~10k 步,AdamW lr~1e-3,full-graph forward/step,混合精度,**别 per-pair-batch 重算 KG**);2 层。
CompGCN 不用(加 KGE 表达力,混淆"干净关系感知")。

**hub → 秩塌陷归因(supported hypothesis,要 earn)**:hub 把共享 common-mode 注入众多节点 → 表征相关 → 低**有效**秩;
契合 warm/cold(低秩底 = hub-generic-可迁移;高秩 = specific-warm-only)。**必须用干预实验挣到**(非断言):
(1) hub 降权/mask **vs 匹配对照**(随机/低度/同边预算)→ 有效秩↑+cold↑ 且**超过对照**;
(2) hub-only vs non-hub-only KG 视图;(3) top 奇异方向 ~ hub-exposure(带 degree/type 控制)。优先 (1)(2)。
**设计指导要软**:软**降权/capping** hub + 保留稀疏 specific mediator(**非硬删**;硬删掉有用全局语境,cold 更依赖)。
→ 正好**机制性地justify 现有 Adamic-Adar / inverse-log-degree 软加权**。措辞:"consistent with hub-driven
common-mode,major contributor not sole source"。这条也是那篇 hub-collapse 未来 paper 的种子。

**下一步**:实现 R-GCN(node 范式)+ 给分析加 layer sweep;然后 hub 干预实验(1)(2)。

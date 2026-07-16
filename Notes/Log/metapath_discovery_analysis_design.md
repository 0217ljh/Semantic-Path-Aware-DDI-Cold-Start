# Analysis design: neural discovery of new biologically-meaningful meta-paths

2026-07-01. codex thread 019f1fb8 (2 rounds). 必需分析:证明我们的方法**自动发现**
新的、生物有意义的 meta-path(经典 HIN-DDI 是手工设计),且传统 GNN **做不好**(非"不可能")。
这个分析**无论最终用什么 backbone 都必须有**。定位 = HIN-DDI(手工 meta-path 相似度)的
学习化 + 归纳化 + 冷启动增强版。

## 核心定义(codex,非循环)
一个 typed schema `s` 才算"**被发现的 meta-path**",当且仅当同时满足:
1. **可读**:能从训练好的模型直接抽成 typed schema;
2. **因果必要**:删掉它在"它 fire 的 pair"上的具体实例 → 预测显著下降;
3. **新颖/非平凡**:不在 HIN-DDI 手工集,且效应强于 matched control。
口号:**importance nominates; intervention validates.**

## Schema 空间
- **PMP schema(主)**:`(rel_a, τ, rel_b)`,τ=中介类型,rel_a/rel_b=两药到中介的边关系。
  (细粒度敏感性分析再加 `(d_a,d_b)`;主表用粗的,便于对标 HIN 手工 meta-path)
- **NBFNet schema(次/supplement)**:关系类型序列 `(r_1,...,r_L)`,更长、有方向。
- 主 vehicle = **PMP adapter**(最干净、最不易被 reviewer 当成 post-hoc 归因机器);
  NBFNet 作为"更长有向机制链"的补充。

## 关系词表(已验证,gating step)
merged KG 有 52–59 种关系,builder **不折叠**(kg_builder_merged.py:70)。
- ❌ `db:enzyme`/`db:target`/`db:transporter`/`db:carrier` 是**单一粗桶**,不分 substrate/inhibitor
  → **CYP 底物-抑制剂 star example 作废**。
- ✅ 有向非对称 drug→gene 关系存在:`het:CuG`(up-reg 18.8k)/`het:CdG`(down-reg 21.1k)/
  `het:CbG`(binds 11.6k)/`het:Gr>G`(有向基因调控 266k)。
- ✅ 丰富 typed 中介 + 跨源(het/prime/db)。

## Star schema(codex 定,措辞降级)
**`(het:CuG, Gene, het:CdG)`** = 两药对同一基因**相反调控**。措辞:**"调控冲突/汇聚模式,
在预测为 DDI 的 pair 中富集"**,**不**说"canonical DDI 机制"(那是过度承诺)。
备用更稳的:`(het:CbG, Gene, het:CuG/CdG)`(一药结合、另一药调控同基因 = 汇聚扰动)。
novelty 论点:HIN 手工是"共享基因 Drug-Gene-Drug"(仅中介类型);我们发现的是
**方向+关系精化**版 → 即使节点类型相同也算 novelty(relation-role refinement)。

## T_main:发现-schema 验证表(每行一个 schema,top 8–12 PMP + 3–5 NBFNet)
列:Schema · Family · Mean importance(alpha 聚合,3-seed mean±sd)· Coverage(active
pair 占比 %)· Novel vs HIN-DDI(Y/N)· Stability(% seeds in top-10;脚注 mean Jaccard@10)·
Causal Δscore(删 schema 后 f−f\s,median+IQR)· Matched-control Δ · **Δ advantage**(每-pair
配对差)· p(paired Wilcoxon,BH-FDR)· Mechanism label。
入选门槛:Coverage≥2–5% · Novel=Y · Δadvantage>0 · FDR p<0.05 · ≥2/3 seeds 进 top-K。

## 因果删除机制(PMP,非循环关键)
对 pair (a,b) 删 schema s=(rel_a,τ,rel_b):移除该 pair support 里匹配 s 的**全部中介**,
frozen 模型重打分 → Δ_s。**两重对照**(否则删高-alpha 中介必掉分 = 循环):
- **Control A(alpha-mass matched)**:同类型 τ、非-s、移除的 alpha 总质量与 s 尽量相等;
- **Control B(top-alpha wrong-schema)**:同类型最高-alpha 的非-s 中介(对抗性)。
主检验用**残差化**:`Δ*_s = Δ_s − Δ_(alpha-matched ctrl)`,直接减掉"你删了高质量证据"的反驳。
**抽取/评估分片**:一个 held-out 切片全局排 schema,另一个不相交切片做删除检验(防"在同批例子上既挑又验")。

## T_gnn:传统 GNN "做不好" 实验(可测,不吹"不可能")
训一个 plain relational GNN(RGCN/CompGCN + pair scorer,无显式相交模块),同 split/KG。
post-hoc 归因(integrated gradients / edge masking)映到同一 schema 空间。比:
- **Top-3 mass / schema entropy**(归因是否集中);
- **Deletion fidelity − control**(fidelity gap = 关键列);
- **Role separability**:非对称 schema `(CuG,Gene,CdG)` vs 其对称塌缩版,谁的归因质量高
  → PMP 高、plain GNN 低。
故事:plain GNN 能预测,但抽出的 schema 证据更**弥散、不稳、post-hoc、不辨方向**。
Theorem 5.1 只作**表示层动机**(typed common-neighbor 是 generic MPNN 的弱归纳目标)→
实验验证经验后果(更差的因果/可读发现)。

## Novelty 基准
- 主:vs HIN-DDI 确切手工目录(S_HIN);
- 次(robustness):vs 2–4 篇代表性手工 meta-path DDI 论文并集(S_prior),附录/脚注。
- canonicalization:方向/关系精化 vs 纯节点类型 → 前者算 novelty。

## F_case:单案例图(3 panel)
A 局部证据图(单 pair,只画匹配 schema 的活跃中介,按关系着色 + alpha)·
B schema 消融柱状(原分/删 schema/删 matched control)· C 生物注释小表(基因/功能/调控冲突标签,描述性)。
caption 最小统计:该 schema family 全部 active pair 上 删-schema vs matched-control 的配对 Wilcoxon p。

## gate 验证结果(2026-07-01,已查实)
1. **adapter 的 drug→mediator 关系是 11 粗桶**([retrieval.py:72-82](Code/my_code/models/spmn_v1/retrieval.py:72)),
   **line 75 把 het:CuG/CdG/CbG 折进同一 bucket 5** → 当前模型区分不了上/下调。原始 KG 里 CuG/CdG
   分开(18.8k/21.1k)→ 拆桶可复活。
2. **DrugBank action 数据在,没丢**(用户记忆正确):
   - 丢弃点 = [loader.py:66](Code/my_code/kg_lib/loader.py:66)(建 merged KG 时只留药+蛋白 id,
     relation 写粗桶 `db:enzyme`;跨 3-KG 统一 schema 的选择,非数据丢失)。
   - **filtered CSV 保留 action**:`drug_enzymes.csv` → **substrate 3335 / inhibitor 1559 / inducer 661**;
     `drug_targets.csv` → inhibitor 2841 / antagonist 1456 / agonist 902 / ...
   - **现成机制 ground-truth**:`enriched/ab.parquet`(565731 行)含 key_entity + action_drug_a/b +
     mechanism_chain + pk_pd_label;`action_pairs.parquet`(189k)。
   - `run_nbfnet_v1_72.py` 已用 action→sign(inhibitor=-1/agonist=+1)。

## LOCKED 设计(codex round 3,2026-07-01)
**Headline star 复活并升级为 CYP:`Drug -[substrate]-> Enzyme <-[inhibitor]- Drug`**(canonical PK-DDI
机制模板;措辞:"与 PK-DDI 强关联的机制模板",不说"每个这种 pair 都是 DDI")。直接对标
"HIN-DDI 手工粗 Drug-Enzyme-Drug → 我们发现方向/角色精化的真机制"。

**关系词表(小而有药理意义,按中介类型拆)**:enzyme {substrate, inhibitor, inducer, other};
target {agonist/activator, antagonist/inhibitor/blocker, binder/other};transporter 同理(若覆盖够);
其余保持粗桶。**只拆 enzyme(PK)必需,target(PD)可选,别全拆**(保统计功效)。

**非循环发现框架(关键)**:模型拿到的是带角色的**边**,发现的是 pair 级的**非对称合取**
(同一中介一臂 substrate + 另一臂 inhibitor)是预测单元 —— 不是"发现了 action 标签存在"。三层:
①输入可用(边带角色)②模型把该非对称合取权重抬过其它合取 ③因果删除 + 同输入下 plain GNN 无法同样直接/忠实地归因(逐节点聚合把两臂对称化)。

**生物验证(避循环)**:主 = **外部 curated PK-DDI 机制集**(FDA / DDInter / 已知 CYP substrate-inhibitor 表),
**不**能用训练所依赖的同源 action 派生的 ab.parquet 当主证据;ab.parquet 仅作 case study / 内部一致性(附录)。

**主挖掘模型 = action-refined PMP**(headline 必须是 adapter,否则 reviewer 觉得"有趣的可解释性来自 backbone");
NBFNet/v1_72 = 次要(更长有向链 + 更强的 plain-backbone 对照)。

**两层 claim**:
- **主(可行)**:action-refined PMP 发现非对称合取是预测单元(因果+novelty+vs plain GNN)。
- **补充(更强但高风险,bonus)**:**粗训 + 外部标注恢复** —— 只用粗桶 `db:enzyme` 训练(不给
  substrate/inhibitor 标签),训练后外部标注 shared enzyme 的角色,证明高重要度的 shared-enzyme 实例
  富集于 substrate/inhibitor 冲突模式 → "没被告知却恢复真机制"(最强,但粗 PMP 结构上分不清
  substrate+inhibitor vs substrate+substrate,可能弱/null → 只当 bonus,不 central)。

## 需用户拍板(动 pipeline,先报后做)
1. 建 **action-refined KG/retrieval 变体**(新 loader 变体保留 action → 拆关系;新 retrieval REL_BUCKET
   变体)——新文件,不改现有 loader.py/retrieval.py。
2. 在该变体上**重训 PMP**(真训练 run)。
3. 取**外部 PK-DDI 机制验证集**(DDInter / FDA CYP 表)——数据获取。

## 决策(codex round 4-5,2026-07-01):走 Option A,不改 adapter
用户定:**不改锁定 adapter**,故事优先。codex round 4 裁:方向**不需要**引入训练,除非
headline 那条 meta-path 本身必须有向(那才走 Option C = 并行有向分析变体,锁定的仍不动)。
→ **先走 A,失败再 C。**

## 真实候选挖掘结果(analyze_spmn_v2_schema_candidates.py,3-seed,11282 test_s2 pair,2.68M 中介)
- support **被 2-hop 中介主导**:`(2hop,τ,2hop)` 覆盖率 disease 0.853 / protein_gene 0.843 /
  anatomy 0.745 / side_effect 0.740 / molecular_function 0.722 / biological_process 0.705 / pathway 0.583。
- 干净 1-hop 对称 schema ≈ HIN-DDI 手工那些:(target,gene,target) 0.252、(enzyme,gene,enzyme) 0.238、
  (side_effect) 0.216、(gene-reg/bind[het:CbG/CdG/CuG],gene,同) 0.099。
- **1-hop×1-hop 非对称 schema 极稀**(最高 (indication,disease,contraindication) 0.018)。
- **桶合并了源**(bucket0={db:target,prime:drug_protein};bucket5={het:CbG,CdG,CuG})→
  codex round-4 设想的"跨源关系精化"**在锁定模型结构上不可表达**。
→ 诚实结论:novelty **不在关系-角色精化**,而在**中介-KIND + 高阶(indirect)**。
HIN-DDI 实体 = 药/蛋白/通路/副作用(2 次搜索一致;确切 meta-path list 仍待全文),
**没用 disease/GO/anatomy/biological_process** → 这些是我们的高覆盖 novelty 来源。

## HEADLINE 锁定(codex round 5)
**`(2hop, biological_process, 2hop)`** = 两药经多跳结构路由**汇聚到共享 biological-process 模块**。
- 不选 disease(易被驳为共病/适应症语义关联,非机制);BP 更机制化、更明显超出手工目录。
- backup:`(2hop, pathway, 2hop)`(更接近经典药理);fallback 直接 schema:`(gene-reg/bind, gene, 同)`。
- **路径还原必须内生(用户 2026-07-01 更正,codex round-5 的"NBFNet 外挂抽路径"作废)**:
  我们方法**本来设计**就是粗粒度(高阶相遇家族)+ 细粒度(meta-path),细路径还原是**方法自己的
  能力**,不用别的方法。对应 [paper_locked_logic:434](paper_locked_logic.md:434) §3.2:
  `h_τ = AttnPool_m φ(NBFNet(a→m), NBFNet(m→b), m, τ)` —— NBFNet(a→m) 是 hyper-edge **内部**的
  细粒度路径编码器 = 我们方法的一部分,非外部方法。
  ⚠ **诚实缺口(已查代码)**:跑出 0.7765 的 standalone `aware_core.py` **只有粗粒度那半**
  (type/rel-bucket/distance,无 §3.2 细路径编码器);2-hop 高阶中介的 rel_a/rel_b 是 `2hop` 占位
  → **当前 standalone 内生还原不了 BP 相遇的具体路径**。1-hop 的 `(rel_a,τ,rel_b)` 本身是内生
  meta-path ✅,2-hop ❌。→ 要"内生还原高阶 BP 路径",方法必须**真的带上 §3.2 细粒度组件**,
  这跟"不改 adapter"有张力,**待用户决定**。
- 一句话 claim(codex 原话):"Our model automatically discovers higher-order meta-paths in which
  two drugs converge on a shared biological-process module through multi-hop structural routes, a
  mechanism-level schema absent from HIN-DDI's hand-designed short protein/pathway/side-effect paths
  and only weakly recoverable from plain GNNs."

## 分析结构(Tier 0,不改 adapter、不重训)
1. PMP 排"相遇家族"重要度(注意力加权)→ headline = biological_process 家族。
2. 对该家族中介做因果删除 vs matched control(alpha-质量匹配 + top-alpha 错家族)。
3. **内生**路径还原:用我们方法自己的细粒度组件解析 BP 家族内主导的关系序列(Drug-…-BP-…-Drug)。
   ⚠ 前提 = 模型带 §3.2 细粒度路径编码器;当前 standalone 无 → 见上"诚实缺口",待定。
4. plain GNN 对照:同家族/序列空间,证明归因保真更差。

## 仍待
- HIN-DDI 确切 meta-path 目录(定 S_HIN;确认它确实没用 BP/disease/GO)。全文换源。
- 外部 PK/机制验证集(若最终要 case study 生物锚定)。

相关:[[spmn_v2_theory_anchor]](Claim ③ 显式 vs 隐式)· [[paper_locked_logic]](Thm 5.1)· [[project_semantic_path_ddi]]

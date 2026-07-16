	Baseline： EMERGNN~0.7458
	 数据处理：
##### 1 `molecular_mu_morgan.npz` ── 分子 Morgan 指纹
- **源**:`drug_smiles__seed42.csv` 的 `(drugbank_id, smiles)` 两列
- **处理**:rdkit `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` 对每个 SMILES 提指纹
- **输出**:1994 × 2048(全部 SMILES 解析成功,parse_fail=0)
- **对应**:**1994 split 全宇宙**(seen + unseen 都给,因为 unseen 测试时也要查特征)
##### 2 `kg_neighbor_target_pubmedbert.npz` ── KG 邻居均值 `k_u`(<font color="#ff0000">此处给出一个后续的改进log</font>)

- **源**:
    - `_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet`<font color="#ff0000">(DDI 边已 mask 的合并 KG)</font>
    - `_merged_kg/_cache/screen1_tag_init/d_name_only__pubmedbert.pt`<font color="#ff0000">(178029 节点 × 768 的 PubMedBERT 名字嵌入)</font>
    - `_merged_kg/nodes__drugbank_hetionet_primekg.parquet` 取 kind∈{Drug, drug} 的 <font color="#ff0000">7754 个药 ID</font>
- **处理**:对每个药,收集它在 KG 里所有非药 1-hop 邻居 → 取这些邻居的 PubMedBERT 嵌入 → mean-pool
- **输出**:7754 × 768,加 `n_neighbors`(每个药的邻居数)
- **对应**:**7754 KG 全宇宙**;本 split 的 1994 药全在其中(用 DrugBank ID 直接索引);6531/7754 有 ≥1 个邻居

##### 3 `kg_typed_target_pubmedbert.npz` ── relation-typed KG 目标 `k_typed`(<font color="#ff0000">此处给出一个后续的改进log</font>)

- **源**:同 #2(同一 KG 同一节点嵌入)
- **处理**:相同的邻居收集流程,但**按 node kind 分桶**:
    - bucket 0: protein_gene(gene/protein 类 kind)
    - bucket 1: pathway
    - bucket 2: molfunc_bioproc(molecular_function + biological_process + cellular_component)
    - bucket 3: disease
    - bucket 4: phenotype(effect/phenotype + symptom)
    - bucket 5: side_effect(Side Effect)
    - bucket 6: anatomy
    - bucket 7: other
    -<font color="#ff0000"> 每桶分别 mean-pool(768d)→ 8 个桶拼成 6144d</font>
- **输出**:7754 × 6144,加 `n_per_bucket` 矩阵
- **对应**:**7754**,bucket coverage 在 split 上:protein_gene 0.81 / disease 0.25 / phenotype 0.14 / side_effect 0.12 占主导

##### 4 `molecular_aligned_infonce.npz` ── <font color="#ff0000">InfoNCE 对齐</font>后的分子表示 `z_m`（能不能一起做？）

- **源**:#1 (Morgan m_u) + #2 (k_u)+ PKL 的 `ds.splits.train` 提取 SEEN 药列表
- **处理**:
    - SEEN 药 = 640;其中**同时**有效 Morgan(都有)且非空 k_u 的 = **615**
    - 留出 10% (61) 做 holdout 测对齐迁移
    - 训练 554 个,proj_m: 2048→256→128 + proj_k: 768→128 + InfoNCE + Barlow 去相关 + VICReg std-hinge
    - **最佳模型(early-stopped on holdout pairwise retrieval AUC)**:holdout AUC 0.68,top1 0.164,MRR 0.261
    - **关键**:训练只用 615 个 SEEN 药;然后对**所有 1994 个有效 Morgan 药**应用 proj_m → 给 ALL 包含 unseen 药输出 z_m
- **输出**:1994 × 128 z_m;加 residual(seen 药的 KG-对齐残差,用于探索性 self-gating)、is_seen 标志
- **对应**:训练子集 = **615 SEEN 药**(cold-start 安全);输出覆盖 = **1994 全部**(包括 159 个 test_s2 unseen 药,100% 都有 z_m,本会话验证过)

## #5 `effect_neighbors_pubmedbert.npz` ── 效应层邻居 CSR

- **源**:同 #2 的 KG,但只保留 kind 属于 effect-layer 的邻居
    - effect-layer kinds 映射:`side_effect`/`Side Effect` → 0, `phenotype`/`effect_phenotype` → 1, `symptom` → 2, `disease` → 3, `anatomy` → 4
- **处理**:per drug 存 CSR:`indptr[i:i+1]` 切出该药所有效应层邻居的:
    - `nbr_rows`(指向 `d_name_only__pubmedbert.pt` 的 row index,可以 gather 嵌入)
    - `nbr_kind`(0-4,主线只用 kind ≤ 2 的 PD-composable subset,占 82.4%)
    - `nbr_drugdeg`(这个 effect node 连了多少药 — 越大越通用,用作 top-K 反频率 saliency)
- **输出**:7754 药,235,869 总边,加 4 个 1D 数组
- **对应**:**7754**;本 1994 split:1505 (75.5%) 有效应邻居,test_s2 159 unseen 药里 123 (77.4%) 有

## #6 <font color="#ff0000">`molecular_fragments_brics.npz` ── BRICS 片段计数</font> （顺序）

- **源**:同 #1(SMILES csv)
- **处理**:rdkit `BRICS.BRICSDecompose` 每个 SMILES → 片段 SMILES 集合 → 跨所有药做 document frequency → 留 top-768 词表 → 每药 768-d 计数向量
- **输出**:1994 × 768,加 vocab(片段 SMILES 字符串数组)
- **对应**:**1994 全宇宙**;89.9% 药有 ≥1 片段(肽类/不可 BRICS 分解的不破裂)

## #7 `llm_pharma/llm_pharma.jsonl` ──<font color="#ff0000"> Claude 蒸馏内在药理</font> （导向驱动）

- **源**:
    - `id2name.json`(`Code/data/KG/drugbank/filtered/id2name.json`):1900 药的 `DBxxx → 名字`
    - `drug_smiles__seed42.csv`:1994 药的 SMILES
    - 二者交集 = **1897 药**(SMILES csv 里有但 id2name 没有名字的 97 个药被跳过)
    - Anthropic API key 从 `API-KEY/API-KEY.txt`(gitignored)读取
- **处理**:每个药调用 `claude-haiku-4-5-20251001`,严格 prompt:
    - 禁止提任何其他药/药类/相互作用短语
    - 只描述本药的内在性质:targets / MoA / class / 代谢酶 / 转运体 / PD effects / toxicity
    - 输出严格 JSON,含 `free_text`(≤120 词)+ structured 字段(`cyp_substrate/inhibitor/inducer`、`transporter_substrate/inhibitor`、`therapeutic_class`、`primary_targets`、`pd_effects`、`toxicity_mechanisms`、`clearance`)
- **Sanitizer**:扫整个 1900 药名字词典 + 14 个 DDI 短语黑名单(`interact`、`coadminist`、`combined with`、`contraindicated` 等)→ 在 free_text 上 word-boundary regex 命中就 redact 成 `[REDACTED_DRUG]` 并打 `leakage_flag=True`
- **输出**:JSONL 1897 records,每条 `{drug_id, name, raw, sanitized_text, structured{...}, leakage_flag, partner_hits, phrase_hits}`
- **统计**:**525/1897 (27.7%)** 被 leak-flagged → 这些 record 仍保留(只是 free_text 里有 [REDACTED_DRUG] 占位)
- **对应**:**1897 个 split 药**(seen + unseen 都查,因为 LLM 调用是 per-drug、无 pair 信息 → leakage-safe by construction)

## #8 `llm_pharma/llm_text_pubmedbert.npz` ── sanitized LLM 文本嵌入

- **源**:#7 jsonl 里**无 error 且 sanitized_text 非空**的 records
- **处理**:扔进 `screen1_tag_init/encoder.py::encode_pubmedbert(tag="llm_pharma_sanitized", max_length=256)` → PubMedBERT(microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)CLS token 取 768d
- **输出**:1530 × 768,加 `leakage_flag` 数组(1530 中有多少 flagged 可查)
- **副产物**:`screen1_tag_init/llm_pharma_sanitized__pubmedbert.pt`(encoder 内部 content-hash cache)
- **对应**:**1530 split 药**(1897 - 367 个有 error/空文本)

## #9 `llm_pharma/i4_typed_sets.json` ── 解析结构化字段

- **源**:同 #7 jsonl(只取 1530 个 valid records)
- **处理**:每条 record 的 structured 字段做归一化:
    - **CYP** 字段:regex `cyp[\s-]?(\d+[a-z]?\d*)` 抽 token,e.g. "CYP3A4 substrate" → 加入 `cyp3a4`
    - **Transporter** 字段:patterns 映射到 canonical token:`P-gp/MDR1/ABCB1 → pgp`,`BCRP/ABCG2 → bcrp`,`OATPxxx → oatpxxx` 等
    - **Free-text 字段**(class/targets/pd_effects/toxicity/clearance):lowercase + 去标点 + collapse 空白 + 留 ≥3 字符
- **输出**:JSON `{drug_id: {field_name: [tokens]}}`,1530 药 × 10 字段
- **field coverage(本会话亲测)**:CYP 字段比较稀疏(substrate 50%/inhibitor 27%/inducer 2%),transporter 也稀疏(18%/2%);free-text 字段密度高(class/targets/pd/tox/clearance ~80%)
- **对应**:**1530 split 药**

## #10 `ddi_type_map.json` ── pair → ddi_type → 215 类索引

- **源**:`Code/data/KG/drugbank/splits/seed42/*.parquet`(release split,**8 个 parquet 文件**:train/val_s0/val_s1/val_s2/test_s0/test_s1/test_s2 + 几个 negatives 子目录里没 ddi_type 所以跳过)
- **处理**:遍历所有有 `ddi_type` 列的 parquet → 收集 canonical-pair(按 lexicographic 排序 `DBxxx|DByyy`)→ `ddi_type` 字符串映射 → `type_to_idx` + `pair_to_idx`
- **输出**:JSON,565,731 pair 索引到 215 distinct 类型
- **对应**:**565k 全部 split pair**(覆盖所有 fold 的正样本对);本会话验证 test_s2 1919 个正样本对 100% 都有 type label
- **关键**:用 release split 不是 PKL 的原因 —— legacy PKL 不带 `ddi_type` 列,release split 才有




**Round 1 - 基于正则的方法: CACR
* 目的：让 cold-start 药的表示更稳
* 加入超参数$\lambda$, warm up, ddi-ratio, low-degree
* 结果：无明显影响，即单纯靠正则化不能构造稳定的cold start表示
 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

Round 2 - insight2 (以结构信号为主)
* ***Insight i2 (Notes/Settings/Insights/i2.md)**:DDI 的关键证据应该在**两个药的共享中介节点**(meeting nodes,例如共享的酶/通路/副作用)处汇聚;EmerGNN 的 drug-anchored path-flow 没充分利用 meeting-node 处的信号。
* Design
	* - 在 EmerGNN logit 之上加一个简单辅助头:`combined = emergnn_logit + softplus(β)·aux_mlp(meeting_count_features)`
	* - 22 维特征 = 11 个中介类型(protein_gene, pathway, side_effect, disease, ...) × 2 个 hop
	* - AuxMLP:22 → 32 → 1,极小
	* - 联合训练 EmerGNN + aux_mlp + raw_β
* Results: +3 pt
* Ablation：
	* **Shuffle control** (codex r13)
	* **Degree-only control**
	* **Leakage audit**
	* **seed43 generalization**
* 大致上，这个idea算是成功的

Phase 3 # PubMedBERT 中介节点文本特征
* **Insight i4**:meeting node 不仅是结构信号,**它们的 name(语义)**应该也带 DDI 相关信息(例如共享酶 CYP3A4 vs 共享副作用"恶心"的语义不同)。
* - 用 PubMedBERT 嵌入所有 KG 节点的 name (768-d) → 缓存到 `screen1_tag_init/d_name_only__pubmedbert.pt` (178029 节点 × 768)
- 对每个 drug pair,把共享中介节点的 PubMedBERT 嵌入做 mean-pool → 拼接到 22-d count 后面,得到 (22 + 64)-d 输入给 aux MLP
- Results: - **分支级别**:aux head 真实文本 vs shuffled 文本,n=3 → real 0.69 → 0.73,**有信号**
- **全模型级别**:combined 只 +0.5-0.8pt(部分跟 path-flow 冗余)
- Analysis: :meeting-node 名字语义携带真信号,但**在 readout 层加,大部分被 flow 重复**。i4 的"语义信号"是真的,但放置位置(readout vs init)影响其增益。 <font color="#ff0000">(此处改进为path的语义，而不是节点的语义)</font> （contribution2）

Phase 4： #  PK/PD 双通道分割(弱)
* insight 1: :PK 互作(药代,代谢/转运)经分子层中介(酶/通路);PD 互作(药效)经效应层(副作用/疾病)。E1b 拓扑统计 p<1e-300。
* Design: - 把 22 维计数按 mediator 类型软拆成 PK(分子层 10 维)+ PD(效应层 6 维)两个通道
- 两个对称 head + 两个 global softplus gather
- Result: - 通道路由**弱**(spec_PK 负)
- 但发现一个重要副产品:**PK pair AUC 0.836 vs PD pair AUC 0.702,差 +13.4pt** → **难度严重不对称**
- 这个难度差成为后来所有 i1 设计的核心驱动
- Analysis: :**对称软拆 + global gate 不足以路由 PK/PD**;但 PK/PD 的难度不对称是真实而稳健的。
- （切分是有效的，但不需要按照label硬切）


Phase 5: #  <font color="#ff0000">分子互补性 Gate(BRICS 模体)— 揭示分子↔KG 冗余</font>
* 既然 i1 说 PK 走分子层,那么**原始分子结构**(SMILES/BRICS)应该能补 KG?
* Design: - 提取每个药的 BRICS 模体 → 22-dim multiset 计数
- 训练一个 motif-only logistic head:S2 AUROC = **0.5565**(几乎是 chance!)
- Oracle ensemble:MNAH logits + motif logits → 上限组合
- 结果:**oracle gain = 0.000**(分子完全冗余于 MNAH 已有的 KG 信号)
- 分子结构信息(BRICS)对 cold-start DDI 在这个 split 上**完全可被 KG 替代**。

Phase 6: #  MNAH 5-run ensemble
* 略微提升：5 个不同初始化的 MNAH 平均(rank-avg):**0.7811**
* 仅作为辅助的side work


Phase 7： **保留多模态对齐 + i1 PK/PD 路由 + i2 backbone**
* Round1： PD 互作来自共享可组合副作用,需要 pair-conditional select-compose
	* design: molecular 通道(对齐后 z_m + 分子层 counts)+ effect 通道(对 N_eff(a)×N_eff(b) 做 cross-attention,sign-aware) + 学到的 soft gate
	* 几乎无提升
	* 诊断 dump:**effect 通道几乎是 chance(PK_only 0.572, PD_only 0.588)**
* Round2：  **Aux-BCE 强监督唤醒 effect 通道**
	* - 设计:loss = BCE(combined) + 0.3·BCE(eff_logit)
	* - 结果:**effect PD = 0.572(未唤醒)**,combined 0.7684
	* PubMedBERT-name 效应邻居 + 小 cross-attention + Hadamard 合成 = **死路**
* Round 3： **Shared/Residual 分子↔KG 对齐**
	* 设计:`m_shared`(对齐到 RELATION-TYPED KG 目标)+ `m_resid`(KG-正交残差,正交+去相关约束)
	* - 数据:BRICS fragment 替换 Morgan(更稀疏但更有结构信息);8-bucket relation-typed `k_typed`
	* - 结果:**combined 0.7546(同样丢了 count head;mol_head_alone 0.577)**
* Round 4：**MNAH + LLM-pharmacology Residual**
	* 引入 LLM 蒸馏作为新 substrate
	* design：MNAH(保留 count head)+ LLM 自由文本嵌入做对位 MLP → residual head
	* - 结果:**combined 0.7668(no lift)**;**shuffle ctrl 0.7658 ≈ main → LLM 通道完全 inert**
	* 自由文本 PubMedBERT CLS 不是有效的 cold-start DDI 信号载体
* Round 5：**结构事实**:S2 split unseen 药仍保留 KG biomedical 邻居,所以 KG 已通过 path-flow 传播了 molecular/LLM 想要补的药理信息 → 多模态附加冗余
	* - 用蒸馏 jsonl 里**没用过的 structured 字段**(cyp_substrate/inhibitor/inducer、transporter、therapeutic_class、primary_targets、pd_effects、toxicity、clearance)
	* 解析 → 13 个**可解释机制 pair 特征**(shared CYP、CYP inhibitor→substrate 非对称、shared class、shared pd_effect、shared toxicity 等)
	* - **首次真正的正提升**:
		* vs MNAH:overall **+0.84pt**, PK **+1.7pt**(经典 CYP 抑制-底物机制起作用), PD +0.4pt
		* vs shuffle:overall **+1.61pt**, PK +1.46pt, **PD +1.75pt**(机制信号被 shuffle 干净打掉)
		* i4_only 通道 0.600 vs shuffle 0.517 / random 0.510 → 通道带 ~10pt 真信号
* Round 6： **Hard-Negative Mining**
	* - codex 建议:cold-start 瓶颈不是缺 feature 而是判别能力 → 用 I4 重排训练负样本,top 50% 最难替换 bottom 50% 随机
	* 结果:**0.7537(-2.67pt vs MNAH, -3.67pt vs v2i4),HN 通道反而 SHUFFLE 控制 0.762 BEAT main**
	* 诊断:high-overlap 负对里可能存在 false negatives(其实有弱 DDI 但未标),模型反过来**学会压制 I4 信号**才能在训练集上降低误判 → 上集泛化失败
* Round 7： **Joint Binary + 215-Class**
	* 把多分类 ddi_type 作为辅助任务,讲"相辅相成"故事
	* - codex:shared backbone,binary head + 215-class type head(只对正样本算 CE),λ_type=0.3,I4 喂两个 head
	* 数据建设:`ddi_type_map.json`(565k pair → 215 type)
	* 结果:
	    - v2i4d λ=0.3:**binary 0.7792, macroF1 0.0225, microF1 0.287**
	    - v2i4d λ=0 sanity:**binary 0.7708**(噪声范围内 ≈ v2i4 0.7804)
	    - 解读:**架构内**(λ=0.3 vs λ=0)binary **+0.84pt** → 多分类辅助确实帮 binary,相辅相成成立;但跨架构与 v2i4 0.7804 持平,**不破天花板**
	    - 教训:type 头能学(macroF1 是 chance 的 4-5 倍,micro 是 57 倍),但绝对水平弱 + binary ceiling 未变
* Round 8— Backbone 加深 path-flow
	* - codex 称这是"唯一可能 +3pt 的杠杆":EmerGNN length 3 → 4 → 5
	* - 结果:
		* length=4:**0.7707**(-0.97pt vs length=3)
		* length=5:**0.7638**(-1.66pt vs length=3)
		* emergnn-only 通道下滑 0.74 → 0.67,count head 部分补偿但拉不回来
	- 教训:**过度平滑**;此 backbone 不接受更深传播



# 🧩 整体规律(可写进 paper)

1. **MNAH(EmerGNN path-flow + meeting-node counts)在这个 split 是天花板的下沿(0.77)**,大多数附加都打不破。
2. **唯一打破 +0.84pt 的杠杆**是**显式可解释的机制对特征 I4**(CYP inhibitor→substrate 这类),配合干净的 shuffle/random 控制 → 机制信号是真实可分离的。
3. **3/3 自由表示型新模态 substrate 全部冗余**(KG effect-emb、分子片段、自由文本 LLM 嵌入),因为 S2 是 DDI-edge cold-start,**unseen 药仍带 KG 邻居,KG 已传播了它们想补的药理信息**。
4. **HN 挖矿失败**揭示 cold-start 中的 false-negative / 标签噪声问题:很多机制可信的 pair 在 base set 里被标为负,模型被迫压制有用 feature。
5. **多任务监督(binary + 215-class)在架构内有用,但破不了 binary 天花板**,这本身是一个有意思的发现:类型监督起到 regularization 而非 information injection 作用。
6. **更深 path-flow 反伤**:length>3 over-smoothing,此 backbone 不收益于更深。



新的架构：
你抓到的这一点非常关键，可以直接 promote 成新架构的命名 motif。我把它从 "argument" 升级成一个具体的 architectural primitive。

## 1. 给它一个名字（建议）

**Pair-Mediator Pooling (PMP)**. 或者更长但更精确的版本叫 **Cold-Anchored Common-Neighbor Pooling**.

核心定义.

> 给定 query (a, b)，pair representation `z_{a,b}` 完全由**共邻 mediator 集合** `M(a, b) = N_≤k(a) ∩ N_≤k(b)` 中节点的 representation 池化得到，**端点 embedding `h_a, h_b` 不进 score function**.

数学形式.

```
M(a, b) = N_≤k(a) ∩ N_≤k(b)
φ(m) = MLP([ h_m, type_embed(m), rel_embed(a, m), rel_embed(m, b) ])
z_{a,b} = AttnPool({ φ(m) : m ∈ M(a, b) })
score(a, b) = σ( MLP(z_{a,b}) )
```

关键. `score` 函数里**根本不出现 `h_a` 和 `h_b`**。端点只作为"selector"决定 `M(a, b)` 是哪个集合，**不作为"被打分的对象"**。

## 2. 为什么这是 cold-start 专门设计的（论证 2 的 architectural 实现）

这就是把你说的"中介 m 在 G1 时被充分见过、端点 a/b 是 unseen"这个**信息不对称性**直接 hardcode 进 pooling.

|量|Cold-start S2 下的统计|论证 2 的对应|
|---|---|---|
|`h_a, h_b`|a, b ∈ G2，训练时 0 次 supervision，embedding 接近 init|不可信|
|`h_m` for m ∈ M(a, b)|m 是 KG 上的 CYP3A4 / pathway / side effect，在 G1 训练时见过 ~10²–10⁴ 次|充分训练|
|`M(a, b)` 集合本身|由 KG **静态结构**决定（a 和 b 各自的 KG 邻居取交），跟 a, b 是否 seen 完全无关|永远可计算|
|`rel_embed(a, m), rel_embed(m, b)`|边类型 embedding 在 G1 训练时充分见过|充分训练|

所以 `z_{a,b}` 的 every component 都是 well-trained 的；只有 `M(a, b)` 这个集合是 query-specific 的，而集合本身不需要训练，由 KG 直接给出。

PMP 把"端点冷 / 中介热"这个 cold-start 特有的不对称性直接做成 architectural prior. score 函数对端点 embedding **梯度为零**，意味着即使端点训练得再差，score 也是可信的。

## 3. 跟 MNAH 是什么关系（degenerate 版本）

MNAH 22 维就是 PMP 的最 degenerate 版本.

|MNAH 在 PMP 框架里的对应|
|---|
|`h_m` 不用，只用 `type(m)` 的 11 维 one-hot|
|`rel_embed(a, m), rel_embed(m, b)` 完全不用|
|`AttnPool` 退化成 `sum`|
|`φ(m) = onehot(type(m))`|
|`z_{a,b} = Σ_m one_hot(type(m))` ≡ "按 11 类 type 数共邻"，concat 1-hop 和 2-hop = 22 维|

MNAH +3 pt 是 PMP 在**最 lossy 的 pooling formulation** 下的下界 lift。如果换成 full PMP（用 `h_m` + relation context + attention pool），lift 应该显著更大，因为.

- `h_m` 比 type one-hot 信息密度高 64 倍（64-d KG embedding vs 11-d type 1-hot）
- relation context 区分 "a 是 m 的 substrate" vs "a 是 m 的 inhibitor"，这是 PK 机制的核心区别，MNAH 完全没用到
- attention pool 区分 "PKx 关键的 CYP3A4" vs "PKy 不相关的 hub enzyme"，sum 给它们一样的权重

## 4. 跟 SEAL / Labeling Trick / Distance Encoding 的区别

PMP 不是这些方法的简单变体，是一个 cold-start-targeted 的不同 design.

|方法|端点 embedding 进 score 吗|Common neighbor 怎么处理|Cold-start 鲁棒性|
|---|---|---|---|
|标准 GNN + dot product|是（`σ(h_a · h_b)`）|不显式|弱（依赖端点）|
|SEAL (Zhang & Chen 2018)|是（subgraph 包含 a, b 端点 readout）|在 enclosing subgraph 里隐式|中（端点权重稀释了一些）|
|Labeling Trick (Zhang 2021)|是（pair-label 后 GNN readout 仍用端点）|pair-conditional label 让 GNN 能区分|中|
|Distance Encoding|是（`score(h_a, h_b)` 加距离 feature）|distance feature 隐式包含|中|
|**PMP（本文主张）**|**不进**|**first-class 显式 pooling anchor**|**强（架构隔离端点 unseen 问题）**|

PMP 的 novelty 点是把"端点 representation 从 score 函数里**架构性地剥离**"作为 cold-start 设计原则。SEAL / Labeling 都没做到这一步，它们的端点 embedding 仍然影响 score。

## 5. 几个 design detail 必须考虑

**Detail 1. M(a, b) 为空的 fallback**. 如果 a 和 b 真的没共邻（KG 上完全 disconnected），PMP 退化到没信号。需要 fallback. (i) 默认 vector `z_default`；(ii) 退到 path-flow 分支；(iii) 用 hop 半径扩到 3-hop。我倾向 (i) + (ii) 联合，让 fallback rate 在训练时可观测。

**Detail 2. M(a, b) 太大时的 hub mediator 问题**. CYP3A4 几乎邻接每个药，pooling 时它会出现在几乎所有 pair 的 M 里。如果不加 weighting，CYP3A4 的 embedding 几乎 dominate 所有 pair 的 `z_{a,b}`，pooling 失去区分度。两个解法.

- **IDF-style weighting**. 类比 TF-IDF，给 m 一个 `1 / log(degree(m))` 的 prior weight，hub mediator 权重低
- **Attention 自己学**. 让 attention 在 pair-specific context 下决定 m 的权重，hub mediator 在多数 pair 下被 attention 主动 down-weight

**Detail 3. 端点 KG context 完全丢掉浪费**. a 和 b 的**非共邻**邻居（`N(a) \ M` 和 `N(b) \ M`）也有信息，只是不通过共邻反映。可以加一个 side branch 用 `N(a) \ M` 和 `N(b) \ M` 做 pooling，作为 "exclusive context"，跟 PMP 的 "shared context" fuse。但这会重新 introduce 端点依赖，要小心。我倾向 v1 不加这个。

**Detail 4. 跟 path-flow 的 fusion**. PMP 是 subgraph-centric，path-flow 是 walk-centric。理论上是 complementary 的（PMP 给共邻的 conjunction 信号，path-flow 给端点周围的 sequential 信号）。Fusion 公式.

```
final_logit = α · path_flow_logit(a, b) + β · PMP_logit(a, b)
α, β = pair-conditional gate(h_a, h_b 是否 G2 unseen)
```

理想结果. 在 cold pair 上 β 接近 1（PMP dominate）；在 warm pair 上 α 接近 1（path-flow dominate）。这个 gate 可以做成 paper 的另一个 finding（learned routing 卖点也落地了）。

## 6. 跟 MNAH 一起跑能直接验证的 ablation

如果实现了 PMP，paper 的 ablation table 自动就是.

|模型|Pooling content|Pooling weight|S2 AUROC 预期|
|---|---|---|---|
|EmerGNN anchor|端点 readout|n/a|0.7458 ✓ verified|
|MNAH (degenerate PMP)|type 1-hot|sum|0.767 ✓ verified|
|**PMP-counts**（中间步）|type 1-hot + rel context|sum|预期 0.775 左右|
|**PMP-embed**（full PMP）|h_m + type + rel|attention|**预期 0.79+**|
|**PMP-embed + path-flow fusion**|上面 + path-flow|gated|预期 0.80+|

这是一个很干净的 ablation ladder，每一步都对应 paper 里一个 design choice，每一步都可以 isolate 出贡献。Reviewer 一眼能看出 MNAH 是 lower bound、PMP-embed 是核心 contribution、fusion 是 cherry on top。

## 7. 我的建议

**Paper story 主线**应该变成.

> Cold-start link prediction on knowledge graphs suffers a representational mismatch. Endpoint embeddings are under-trained (unseen drugs have near-init representations) while shared mediator embeddings are well-trained (CYP enzymes, pathways, and side effects appear in many training-time pairs). Existing path-flow GNN methods (EmerGNN, SumGNN, KGNN) compute pair scores from endpoint representations, propagating endpoint information through path attention. We propose **Pair-Mediator Pooling (PMP)**, an architectural primitive that computes pair representations exclusively from the common-neighbor mediator set, with endpoints serving only as set selectors. This makes the score function architecturally invariant to endpoint embedding quality, addressing the cold-start regime by design. PMP subsumes typed common-neighbor counts (Adamic-Adar, MNAH) as a degenerate sum-pooling case. On S2 cold-start DDI, PMP raises AUROC from 0.7458 (EmerGNN) to [target 0.79+], with controls confirming the lift is attributable to shared-mediator semantics, not capacity.

这个 framing 三条好处.

1. **PMP 是新概念 + 新命名**，paper 有清晰的 architectural contribution 名词
2. **跟 link prediction 经典文献（共邻 / SEAL / labeling trick）严密接上**，但又跟它们 architecturally 区分
3. **Cold-start 的"端点冷 / 中介热"不对称性变成 paper 的 motivation 锚**，不是事后 rationalization

**实现路径**（建议替换 round4 plan 的 D1 为这个）.

|Step|内容|Combined AUROC 期望|
|---|---|---|
|**PMP-v0**|Backbone 完全替换 = PMP-counts only（MNAH 22-d 当 score function 主体，不再加 EmerGNN）|sanity check，应该 ≥ 0.69（degenerate 版本下界）|
|**PMP-v1**|Pool over `h_m`，sum pooling，无 relation context|≥ 0.77|
|**PMP-v2**|+ relation embed context `rel_embed(a, m), rel_embed(m, b)`|≥ 0.785|
|**PMP-v3**|+ attention pooling（替换 sum）+ IDF-like hub weighting|≥ 0.79|
|**PMP-v4 (final)**|+ path-flow fusion with learned gate|≥ 0.80|

每一步都有 isolate 出来的 ablation，每一步都对应 paper 一个 design choice，每一步的 lift 都跑独立 control（shuffle / random / mediator-drop）。

要不要把这个 PMP 框架替换掉原来 D1（LLM 当 edge type），作为 Round 4 的 primary 方向？我认为这个比 D1 强很多.

- D1 主要在做"加新 KG edge"，仍然是给 EmerGNN 喂更多输入，backbone 哲学没变
- **PMP 是换 backbone 哲学**——从 "endpoint path-flow" 切到 "mediator pooling"，paper-grade 的 architectural shift
- PMP 自带 SEAL / labeling trick 的文献接入，story 更厚
- MNAH 在 PMP 框架下自然变成 degenerate baseline，paper ablation 直接成形

确认一下要不要把 round4_backbone_diff_plan.md 改成 PMP 为主线，D1（LLM edges）降到 D3 作为后续 enhancement？






# Molecular ↔ KG Alignment Design — Joint Training & Objective Choice

**Status**: only one configuration tested (frozen, InfoNCE + Barlow + VICReg std-hinge)
**Logged**: 2026-05-30
**Cache**: `Code/data/_cache/molecular_aligned_infonce.npz` (1994 drugs × 128d aligned z_m)
**Used by**: v2 / v2llm / v2i4 / Phase D / Phase C-a — all consume frozen z_m as drug feature

This log captures two open questions about this cache:
1. **Frozen precompute vs joint-trained alignment** (with downstream DDI decoder)
2. **Is InfoNCE actually the right alignment objective?** Untested alternatives exist.

---

## Current setup

```
Source:    Morgan fingerprint m_u (2048-d, frozen by rdkit)
Target:    k_u 768-d (mean of PubMedBERT[CLS] over drug's non-DDI 1-hop KG neighbors)

Projectors:
  proj_m: Linear(2048→256) → LayerNorm → GELU → Dropout(0.25) → Linear(256→128)
  proj_k: Linear(768→128)                     -- linear-only (codex r2 "stable target")

Loss(applied per training step, batch≤256, 615 SEEN drugs only):
  L = InfoNCE(zm, zk, T=0.10)                 -- bidirectional, in-batch negatives
    + 0.005 * Barlow_offdiag(zm, zk)           -- decorrelation
    + std_hinge(zm)                            -- VICReg-style anti-collapse

Training: 200ep, AdamW lr=3e-4, wd=1e-3, early-stop on holdout pairwise retrieval AUC
Best checkpoint metrics on 10% holdout(61 SEEN drugs):
  pairwise retrieval AUC = 0.68
  top1 retrieval         = 0.16
  top5 retrieval         = 0.33
  top10 retrieval        = 0.44
  MRR                    = 0.26

After training: proj_m is applied to ALL 1994 drugs(including the 159 G2 unseen test drugs)
to emit frozen z_m. All downstream models freeze z_m and only learn how to consume it.
```

---

## Question 1 — Frozen precompute vs joint training

### 当前实际测过的(混乱,不可干净归因)

| Setup | Alignment trained | Result | Notes |
|---|---|---|---|
| v2 / v2llm / v2i4 / Phase D / Phase C-a | **FROZEN** | best = 0.7804(v2i4) | 都使用同一个 frozen z_m |
| E-frag(`v2res_trainer.py`) | **JOINT**(shared/residual,InfoNCE + 正交 + decorr + std-hinge 都和 DDI BCE 一起反传) | 0.7546 | 同时换了:source(Morgan→BRICS)、target(k_u→k_typed)、加 residual 分支、joint。**4 个变量同时变**,无法归因 |

**结论**: joint vs frozen **没有干净的 isolated 对比**。E-frag 的失败不能说明 joint 不好,因为同时换了太多变量。

### 理论上 frozen 的优势(codex round 2 当初的理由)

- ✓ **Clean cold-start 故事**:对齐和 DDI labels 解耦 → 可以独立 claim "我们做了一个无监督多模态对齐"
- ✓ **Reusable**:一次训练,所有下游模型复用 → 工程上经济
- ✓ **Holdout 验证可量化**:retrieval AUC / top-k / MRR 可以独立报告
- ✗ **次优**:对齐目标(预测 KG 邻居)和下游目标(DDI 判别)不直接耦合 → z_m 可能编码了不少 DDI 无用的信号

### 理论上 joint 的优势

- ✓ **任务信号塑造对齐**:proj_m 的梯度同时来自 InfoNCE 和 DDI BCE → 学到的 z_m 直接服务下游
- ✓ **避免 frozen 的 surrogate-target 问题**:不再被"预测 k_u"这个 proxy 限制
- ✗ **故事不那么干净**:alignment 不再是独立 contribution,而是 method 的一部分(可写,但 framing 不同)
- ✗ **风险:对齐 loss vestigial**:如果 λ_align 太小,DDI 梯度淹没 InfoNCE 梯度 → 等价于关掉对齐
- ✗ **每次跑 backbone 都要重训对齐**,工程上更贵

### 第三种中间形态(从未试过)

**Frozen pretrain + fine-tune 后续 epoch**: proj_m 用 frozen z_m 初始化下游模型,前 N 个 epoch 冻结 proj_m,从 N+1 起解冻让 DDI signal 微调。两全方案。

### 应当测的 isolated 对比

| Setup | proj_m 训练 | 描述 |
|---|---|---|
| **A. Frozen baseline** | 冻结(当前) | 等于现在的 v2i4 0.7804 |
| **B. Joint from scratch** | DDI + InfoNCE 同时反传,from random init | 看 joint 是否能补 frozen 的次优 |
| **C. Frozen init + late unfreeze** | 前 50 ep 冻,后 50 ep 解冻 | 中间路线,理论上最优 |
| **D. Frozen + small DDI-only fine-tune** | 冻结 InfoNCE,只对 proj_m 用 DDI gradient 微调 | 只让 DDI 调,不再做对齐 |

**关键**: 这 4 个对比**所有其他变量必须固定**(source = Morgan,target = k_u,alignment loss = InfoNCE+Barlow,backbone = MNAH+I4),否则又陷入归因混乱。

---

## Question 2 — Is InfoNCE the right alignment objective?

**完全没测过任何替代。** InfoNCE 是 codex round 1 的默认选择,后续只对它**打补丁**(加 Barlow,加 VICReg std-hinge,调温度),从未换过损失家族。

### InfoNCE 在当前数据规模下可能不是最优的具体理由

1. **615 训练药、batch ≤ 256**:in-batch 负样本只有 batch_size - 1 个,远不如 CV/NLP 的大规模设定。InfoNCE 的下界紧密度依赖于负样本质量和数量。
2. **k_u 是噪声 target**(KG 邻居均值,稀疏度极不均匀):InfoNCE 假设(m, k)是干净的"正配对",但实际 k_u 是聚合特征,匹配不一定 deterministic → InfoNCE 的 contrastive 假设被违反
3. **holdout top1=0.16 / AUC=0.68 远低于 codex 期望的"信号充分"**:暗示要么 source/target 信息不足,要么损失函数不匹配这个数据形态

### 替代损失家族(每个的适用场景)

#### 2a. 不需要 negatives 的方法(直接解决上面理由 1)

- **VICReg-only**:invariance(MSE m ↔ k)+ variance(每个 dim std ≥ τ)+ covariance(off-diag decorr)。我们已经在用 Barlow+std-hinge 作为 InfoNCE 的辅助项 — **InfoNCE 项可能本身就是多余的**,把它去掉做 VICReg-only 的对比应该跑
- **BYOL / SimSiam**:predictor MLP + stop-gradient on target side,无负样本依赖。615 药 + 小 batch 下理论上更稳
- **Barlow Twins (full)**:跟 VICReg 类似但用 cross-correlation 矩阵的 on-diag/off-diag 拆分
- **JEPA**(LeCun 阵营):latent-space prediction,无 contrast,无 reconstruction

#### 2b. 改变 alignment 形式(直接解决理由 2 — 不是一对一匹配)

- **Optimal Transport / Sinkhorn**:在 batch 内做软匹配,允许 m_i 和 k_j(i≠j)有部分关联 → 对噪声 target 更鲁棒
- **Knowledge distillation**:把 k_u 当 teacher,proj_m 只预测 proj_k_target(温度化的"软"目标),把 alignment 从"对齐两个 embedding"变成"一个学另一个" → 不假设两边等价

#### 2c. 不再用 in-batch contrast(直接解决理由 3 — top1 验证太严)

- **MSE / L2 regression**:直接 ‖proj_m(m) - proj_k(k)‖²,最简单,常作为对比 baseline
- **Cosine similarity loss**:1 - cos(z_m, z_k),不进 softmax
- **MI 的其它下界**(NWJ, MINE, IS-MI, alpha-divergence variants):InfoNCE 是 MI 的一个 lower bound,但不是最紧

#### 2d. 改变架构(不只是改损失)

- **Cross-attention alignment**:不投到 shared 128d 后对比,而是让分子 token 序列和 KG 邻居序列在 token-level 互相 attend(transformer encoder-decoder 风格)。表达力远高于 dual-encoder + contrast
- **Multi-view alignment**:同时对齐 m ↔ k 和 m ↔ LLM-pharma 和 k ↔ LLM-pharma(三视图)。codex 在更早 review 里提过一次,从未实现
- **Hyperbolic alignment**:在双曲空间对齐,适合层级关系(KG 邻居有 ontology 层级)

### 应当测的最小对比

| Loss | 实现成本 | 期望 |
|---|---|---|
| **A. InfoNCE-only** | 0(现在的 baseline 去掉 Barlow + std-hinge) | 当前的 ablation 基准 |
| **B. VICReg-only** | 改 loss 计算几行 | 检验 InfoNCE 项是否多余 |
| **C. MSE-only** | 改 loss | 简单 baseline |
| **D. BYOL** | 加 predictor MLP + stop-gradient | 看小数据 + 小 batch 下是否更稳 |
| **E. Sinkhorn OT** | 中等(需 entropy regularizer) | 看是否对噪声 target 更鲁棒 |
| **F. Knowledge distillation** | 简单 | 把 k_u 当 teacher 的 unidirectional 实验 |

**第二级**(如果 A-F 找到 winner,再叠加):
- 加 / 不加 Barlow off-diag
- 加 / 不加 VICReg std hinge
- 加 / 不加 reconstruction term(autoencoder-style)

---

## 联合二维设计空间

这两个问题是**正交的**,展开来是个矩阵:

|   | Frozen | Joint(λ_align=high) | Joint(λ_align=low) | Frozen init + late unfreeze |
|---|---|---|---|---|
| InfoNCE(current) | ✓ tested(v2i4 0.7804) | (E-frag 失败,但混了其他变量) | 未测 | 未测 |
| VICReg-only | 未测 | 未测 | 未测 | 未测 |
| BYOL | 未测 | 未测 | 未测 | 未测 |
| Sinkhorn OT | 未测 | 未测 | 未测 | 未测 |
| Cross-attn alignment | 未测 | 未测(架构差异大) | — | — |
| Knowledge distillation | 未测 | 未测 | 未测 | 未测 |

可测的格子: ~24,实际跑全要 ~24 × 35min ≈ 14 GPU 小时。**最高 EV 的子集**:
1. InfoNCE-frozen(已有,baseline)
2. VICReg-only-frozen(测 InfoNCE 项是否冗余)
3. InfoNCE-joint(明确 isolated joint vs frozen 对比)
4. BYOL-frozen(测小数据下 no-negatives 方法是否更好)
5. MSE-frozen(最简单 baseline,设地板)
6. Cross-attn alignment(架构跳跃,可能是真正解锁的钥匙)

**~6 × 35min ≈ 3.5 GPU 小时**,可在一个工作日内完成。

---

## 关联其他 follow-up logs

- `language_encoding.md`: 上游变量(encoder + input)— 如果 source/target 信息不足,任何对齐损失都救不了
- `kg_typed_aggregation.md`: KG target 的 aggregation 形式 — 如果 k_u 本身是噪声 mean,InfoNCE 就违反了"clean pair"假设(OT/distillation 可能更鲁棒)
- `kg_molecular_redundancy.md`: 最上层问题 — 如果分子↔KG 在 task 层冗余,这些 alignment 升级的天花板可能也就 +1pt;但**如果 redundancy 是 partial 的**(per-drug 冗余但表示形式可以解锁),换 alignment 架构反而可能是绕过 saturation 的方法

实际上**这四个 log 是层层嵌套的**:
- 最深:redundancy 问题 — 决定整个 alignment 方向是否值得追
- 中:alignment 架构 / objective(本 log)— 决定怎么对齐
- 中:KG aggregation(typed_kg log)— 决定对齐目标的形态
- 浅:language encoding — 决定对齐源/目标的原始表示

实验设计上应该 **outside-in**:先做 redundancy 的 R1 实验(`kg_molecular_redundancy.md` 里的 KG-邻居消融)确认对齐方向值不值得追,再决定其他三层的优先级。

---

## 推荐顺序(当这个 cache 排上日程时)

1. **VICReg-only-frozen vs InfoNCE-frozen**(同一个 backbone v2i4): 测 InfoNCE 项是否冗余。30 分钟训对齐 × 2 + 35min 跑下游 × 2 = ~2 hr
2. **InfoNCE-joint vs InfoNCE-frozen**(isolated joint 对比): 把 v2i4 改成 joint-train proj_m。35min × 1 = 35min(对齐损失加进 fit() 即可)
3. **BYOL-frozen**(小数据 no-neg 候选): 30 分钟训 + 35min 跑下游 = ~1 hr
4. **Cross-attention alignment**(架构跳跃,如果 1-3 没 break out): 这个工程量较大,单独做

---

# 用户方向锁定(2026-05-30): 新设计 alignment network,Cross-attention 是必须

更新方向: 不只是 loss 换,而是要**专门 design 一个新的对齐网络**,**building on prior similar work**。Cross-attention 已经是这个领域的事实标准,**必须用**。

## 文献现状: cross-attention 在 DDI / molecular-KG / multimodal 对齐里几乎是默认配置

下面这份清单从 `Notes/Settings/Paper-Review/i1-routing-v2-lit-backing.md`(我们 wave-1 lit search 时的产出)+ 后续讨论里整理出来,**这些工作的共性是: 配对实体不再各自 pool 成一个向量再做点积/InfoNCE,而是在 token / substructure / neighbor 序列上做 cross-attention**。

### Bucket A: DDI 任务内的 cross-attention 主流(我们直接对位)

| 工作 | 模态 | Cross-attention 形式 | 跟我们的关系 |
|---|---|---|---|
| **SSI-DDI**(Nyamabo et al., 2021, *Briefings in Bioinformatics*) | substructure ↔ substructure(单模态,两个 drug) | 学到的 substructures 间 co-attention,让 drug a 的每个 substructure attend over drug b 的所有 substructures | **最直接的 architectural template**,但是同模态(分子↔分子);我们要做的是跨模态(分子↔KG) |
| **DSN-DDI**(Li et al., 2023, *Briefings in Bioinformatics*) | dual-view: intra-view(同 drug 内部)+ inter-view(两 drug 之间) | inter-view 是显式的 cross-drug substructure attention | 双视图思想可以借: intra-view = drug 自己的 substructure / KG neighbor;inter-view = 跨 drug 的 cross-attention |
| **SA-DDI**(Yang et al., 2022, *Chem. Sci.* / PMC9337739) | size-adaptive substructure ↔ substructure | substructure-substructure interaction module,scoring cross-drug substructure pairs | 给"什么是机制上的 composable pair"提供了实现 template |
| **Multi-Scale Graph Neural Process w/ Cross-Drug Co-Attention**(2025, arXiv 2509.15256) | multi-scale graph rep ↔ multi-scale | cross-drug co-attention dynamically fuses multi-scale reps;explicitly framed as "beyond late-fusion co-attention" | 最 recent,可以直接 cite + "go-beyond" |
| **Taco-DDI**(2025, *Neural Networks*) | graph-transformer + dynamic co-attention matrices | dynamic co-attention 矩阵 | 跟 SSI-DDI / DSN-DDI 同源,2025 版 |
| **CASTER** | substructure + co-attention | 同上 | 早期工作,可以 cite |

**这条线索的共同点**: drug 不再被 pool 成单一向量,而是保持为 substructure 序列;两个 drug 之间通过 cross-attention 来做配对 → **跟我们 i1 的"select-compose"机制完全对位**,但我们的 v2 effect-channel 在 KG 邻居层级上试过,失败了(因为 PubMedBERT name + small attn 太弱)。**在分子↔KG 跨模态层级用这套架构,从未试过**。

### Bucket B: Vision-Language 里"fuse-before-contrast"的范式(最值得借鉴的核心思想)

这是 cross-attention 在 multimodal alignment 里**改变了 SOTA 范式**的根本原因。值得逐字消化:

| 工作 | 关键改变 | 与我们的关联 |
|---|---|---|
| **CLIP**(Radford et al., 2021) | dual encoder + InfoNCE(pooled embedding 点积)| 我们的 InfoNCE-on-z_m 就是 CLIP 范式,SOTA-ish 但有上限 |
| **ALBEF**(Li et al., 2021, NeurIPS) | **加一层 cross-attention encoder 做 fuse,在 contrast 之前 + 之后** | 这是关键转折点。fuse-before-contrast 通常 +2-5pt over CLIP 范式 |
| **BLIP / BLIP-2**(Salesforce, 2022/23)| Q-Former 用 cross-attention 把视觉 feature 蒸馏成 language-aligned queries | 直接架构 template: 一组 learnable queries 在 KG 邻居序列上 cross-attend,输出对齐到 molecular 序列 |
| **CoCa**(Yu et al., 2022)| dual encoder + multimodal text decoder + 三种 loss(contrastive + captioning + cross-attn fusion) | 多任务多 loss 组合,跟我们后面想叠 alignment + DDI + 多分类是同样思路 |
| **FLAVA, ALIGN, Florence** 等 | 各种 cross-attention 变体 + contrastive | 范式已 saturate;关键就是 fuse-before-contrast |

**Vision-Language 经验直接迁移过来的关键启示**:

> **InfoNCE on dual-encoder pooled vectors 是 baseline,fuse-before-contrast(cross-attention)几乎总是更好。** 在 multimodal alignment 里,2021 之后 SOTA 论文几乎都用 cross-attention 而不是纯 CLIP 范式。

### Bucket C: 分子 multimodal(SMILES + graph + KG + text)里的 cross-attention

| 工作 | 模态 | 关键点 |
|---|---|---|
| **AdaptMol**(2025, arXiv 2505.11878) | SMILES + 分子图 | adaptive multi-level attention fusion;显式说 naive concatenation 损害性能 |
| **UniMAP**(2023, arXiv 2310.14216) | SMILES + 分子图 | 深度 cross-modality fusion via transformer |
| **MoMu**(2022)| 分子 + 文本 | molecule-text cross-attention |
| **KV-PLM**(2022)| 分子 + 文本 | knowledge-aware molecular model + text |
| **MolFormer / MolBERT + KG**(各种 follow-up)| SMILES + KG | KG embedding 拼到 SMILES,通常用 cross-attention |

**这个 bucket 直接说明: 在分子领域里,跨模态融合用 cross-attention 已经是 2023+ 的 standard practice。** 我们用 InfoNCE-on-pooled 是 2021 范式。

---

## 新对齐网络的设计草案(基于上面文献综合的架构)

下面是**具体的实现 sketch**,不是抽象建议。这是个 "ALBEF-style fuse-before-contrast + SSI-DDI-style substructure-level cross-attn + KG-typed-neighbor sequence" 的混合:

```
Inputs(per drug d):
  S_d      = SMILES token / atom / substructure sequence,  长度 L_s,  feature d_s
             (e.g. from ChemBERTa CLS-level tokens, or 分子图 atom-level GNN outputs)
  K_d      = KG neighbor sequence,                          长度 L_k,  feature d_k
             (per drug 1-hop non-DDI neighbors,每个保留 PubMedBERT[CLS] 768d,
              不再 mean-pool;按 typed bucket 排序或加 type token)

Alignment encoder(cross-attention block × N):
  for layer l in 1..N:
    S_d_proj = MultiHeadAttn(query=S_d, key=K_d, value=K_d)  # 分子 attend KG
    K_d_proj = MultiHeadAttn(query=K_d, key=S_d, value=S_d)  # KG attend 分子
    S_d = LN(S_d + FFN(S_d_proj))
    K_d = LN(K_d + FFN(K_d_proj))

Pooled representations:
  z_m = pool(S_d)      # 例: CLS token or attention-pool
  z_k = pool(K_d)      # 同上

Losses(可以叠加,跟 ALBEF/CoCa 同思路):
  L_align = InfoNCE(z_m, z_k)              # contrastive head(传统对齐)
  L_recon = MSE(z_m, sg(z_k_target))        # optional: 直接预测
  L_decorr = Barlow_offdiag(z_m, z_k)       # 保留我们现有的
  L_DDI_optional = ... (if joint)           # 如果 joint training
```

**这个设计相对当前 frozen+InfoNCE-on-pooled 的改进**:

1. **不在 fuse 前 pool**:KG 邻居保留为序列,分子保留为 substructure 序列,避免一开始就 mean-pool 的信息损失(对应 `kg_typed_aggregation.md` 里 "uniform mean pool 稀释稀有 mediator" 的问题)
2. **cross-attention 是 architectural**:让模型自己学习哪些 substructure 配哪些 KG 邻居最相关(对应 i1 select-compose 的 cross-product 选择思想,但跨模态)
3. **pair-conditioned 自然成立**:cross-attention 的 attention weights 本质就是 pair-conditioned,跟 typed_aggregation log 里你提的 "pair-level attention" 设计一致
4. **可叠加多任务**:CoCa 范式直接支持 contrastive + DDI + type CE 多损失,跟我们 Phase D 的 multi-task 兼容

## 设计变量(实施时要决策)

| 维度 | 选项 | 默认建议 |
|---|---|---|
| 分子 sequence 来源 | (a) ChemBERTa token-level (b) 分子图 GNN atom-level (c) BRICS substructure-level (d) Murcko scaffold-level | (c) BRICS,跟我们已有 cache 兼容,机制可解释 |
| KG sequence 来源 | (a) 1-hop neighbors 全集 (b) typed bucket 内 top-K 排序 (c) 按 relation 类型分桶后展平 | (b) 跟 `kg_typed_aggregation.md` 的 typed 设计协同 |
| 序列长度 / padding | 固定 K_max 还是动态? | 固定 max 32(跟 effect-neighbors 当前的 top-K 一致) |
| Cross-attn block 层数 N | 1 / 2 / 4 / 6 | 从 2 开始(ALBEF 用 6,我们数据小,容易过拟) |
| Attention heads | 4 / 8 | 4(d_model=128 时) |
| FFN expansion | 2x / 4x | 2x(节省参数) |
| Pooling 方式 | (a) CLS prepended (b) attention pool (c) mean | (b) attention pool,跟 BLIP 的 Q-Former 思路一致 |
| 总损失组合 | (a) InfoNCE only (b) +Barlow (c) +reconstruction (d) +DDI joint | (b) 作为对齐独立训练的基线;(d) 作为 main method |
| Frozen vs joint | 见上文 Question 1 | frozen 训对齐 → 解锁后期 DDI 微调(C 选项) |
| Backbone 初始化 | random / pretrained ChemBERTa / pretrained PubMedBERT | pretrained,只学 cross-attn 层 |

---

## 实施的工程级风险(必须知道)

- **参数量爆炸**: 2-layer cross-attn block 在 d=128, heads=4 配置下,加上 FFN 大概 +0.5M 参数。MNAH 整体大约几 M 参数,加上 cross-attn 不夸张但**确实改变了 model size**,需要 cofounding-aware 对比(对照组也 +0.5M random params)
- **小数据训不动**: 615 SEEN 训练药 × 53k pairs = 任务样本不大;cross-attn 是 over-parameterized 架构,容易过拟。强烈建议先在 ChemBERTa pretrained backbone 上做 frozen + 只学 cross-attn 层
- **序列长度 mismatch**: 分子 substructure 平均 3.7 个(BRICS), KG typed neighbors 平均 ~95 个(全部) / 32(top-K),长度差异大 → cross-attn 一边是稀疏一边是密集。要么 KG 也做 top-K 一致,要么用 cross-attn 的 padding mask 严格处理
- **codex 反复警告"cold-start 故事干净"**: 如果 cross-attn 块是 joint 训的,alignment 不再纯无监督 → 论文 framing 需要调整。可考虑两阶段: 先无监督 frozen pretrain(对齐 loss 单独),再 joint fine-tune(对齐 + DDI 同时)

---

## 更新后的实验优先级

| 优先级 | 设计 | 理由 |
|---|---|---|
| **P0** | **ALBEF-style 2-layer cross-attn alignment**(frozen training, BRICS substructure ↔ typed-KG-top-32) | 文献已经 saturate 在这个范式 → 我们应该至少打个对比 |
| **P0** | **同设计的 joint variant**(对齐 + DDI 一起反传) | 配合 Question 1 的 isolated 测试 |
| P1 | **BYOL-style fuse-before-predict**(无 negatives, 配合小数据) | 615 训练药 + 小 batch 下可能比 InfoNCE 更稳 |
| P1 | **Q-Former 风格**(BLIP-2): 一组 learnable queries 在 KG 序列上 cross-attend,蒸馏成与 molecular 对齐的 fixed-size representation | 灵活性最高 |
| P2 | **Sinkhorn OT alignment after cross-attn fuse** | 跟 cross-attn 互补,处理 fuzzy 配对 |

第一个实验 (P0) 是必须的 — **不跑这个,我们就还在 2021 dual-encoder + InfoNCE 范式上,跟 2023+ 文献完全脱节,paper 故事很难讲**。

## 这一段总结(给将来回头看)

**结论**: 当前 alignment cache 用 InfoNCE-on-pooled-dual-encoder 是 2021 范式;2023+ 的 multimodal alignment 文献(vision-language + DDI substructure + 分子 multimodal 三个 bucket)已经把 cross-attention-based fuse-before-contrast 做成新标准。**设计新对齐网络时,cross-attention 不是 nice-to-have,而是 must-have**。

具体 architectural template 走 ALBEF / SSI-DDI / BLIP-Q-Former 的混合:**保持分子和 KG 为序列(不预先 pool),多层 cross-attention 互相 attend,在 fused representation 上做 InfoNCE/BYOL/OT 对齐**。

跟 `kg_typed_aggregation.md` 的 pair-conditioned bucket attention 设计**自然 unify**: cross-attention 的 attention weights 本身就是 pair-conditioned。两个 log 在架构上指向同一个目标。

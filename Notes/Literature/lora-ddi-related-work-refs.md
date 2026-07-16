# LoRA-DDI Related Work — 文献调研清单(2026-07-05, v2 深化版)

> 供 §2 related work 引用。**⚠️ 大部分为摘要/标题/搜索级信息,正式 cite 前必须读原文核准
> 作者/年份/venue/精确贡献。** 置信度:[C]=较确信(知名或已核) [A]=摘要级 [T]=仅标题/搜索级待查。
> arXiv 号:凡标 `arXiv:XXXX.XXXXX` 的,ID 均直接出现在检索结果链接里(ID 可信,内容仍需核)。
> 标 `(ID待查)` 的是知名 paper 但 ID 我没在链接里直接看到,cite 前查。
> **2025/2026 的 arXiv(25xx/26xx)一律 [T],太新,先别引,只作 landscape 参考。**
>
> §2.2 结构:Para 1 = inductive KG link pred(结构化迁移);Para 2 = 两条低秩流
> (流 A = OOD/transfer 低秩「可迁移信息是低秩的」;流 B = LoRA/内在维「低秩因子化这个 FORM」)
> + 理论桥(隐式低秩偏置 / 泛化界)+ 翻转框(GNN rank collapse 无意 vs 我们有意)。

---

## §2.1  Graph-Based DDI Prediction(node / path / subgraph + molecular + cold-start)

### node-based
- KGE 类(TransE/DistMult/RotatE/ComplEx over DDI-KG)—— entity/node 表征 [T,通识]
- R-GCN / GCN over biomedical KG —— 节点表征 + 边预测 [T,通识]

### path-based
- **EmerGNN** —— pair 间路径/flow 推理,emerging-drug DDI(我们的 baseline)[A] arXiv:2311.09261
- **K-Paths** —— training-free 路径抽取(Yen 最短路)[A] arXiv:2502.13344
- meta-path / HIN-DDI —— 手工 meta-path 相似度 [T,待查具体篇]

### subgraph-based
- **SumGNN** —— 子图汇总多关系 DDI [A/T] arXiv:2010.01450(待核)
- **KnowDDI** —— knowledge subgraph DDI [A/T] arXiv:2311.15056(待核)

### molecular / 子结构(§2.1 Para 2「分子内在特征」一路)
- **SA-DDI / MGDDI / MSDF / Taco-DDI** —— fragment/substructure-level,泛化未见 chemotype [A,综述提及]
- **MolBridge** —— atom-level joint graph refinement [A] arXiv:2510.20448

### cold-start / emerging / inductive DDI(§2.1 Para 2「KG 邻域归纳」一路)
- **PEB-DDI** —— inductive 协议,整药 held-out [A,综述提及]
- **GenRel-DDI** —— "embedding 相似度≠DDI",relation-centric(**非竞品**,机制不同)[C] arXiv:2601.15771
- **AIM-DDI** —— architecture-independent 多模态,both-unseen(**非竞品**)[C] arXiv:2605.14327
- **Dual-Pathway (EHR+KG)** —— unseen DDI [A] arXiv:2511.06662
- **DDIPrompt** —— graph prompt learning DDI event [A] arXiv:2402.11472
- **LLM-DDI** —— GPT emb 进 message-passing GNN [A] PubMed:40601466
- **KITE-DDI** —— KG+transformer, SMILES+KG [A] arXiv:2412.05770
- 综述:molecular/KG/cold-start DDI [A] PubMed:40187181 · S0010482525004731
- 综述:computational DDI (data/models/eval) [A] Frontiers fphar.2026.1816394

---

## §2.2 Para 1  Inductive KG Link Prediction(结构化迁移,无 entity emb 也能泛化到未见实体)
- **GraIL** —— Inductive Relation Prediction by Subgraph Reasoning [C,ICML 2020] Teru/Denis/Hamilton · arXiv:1911.06962
- **NBFNet** —— Neural Bellman-Ford Networks,path-based inductive link pred [C,NeurIPS 2021] Zhu et al. · arXiv:2106.06935
- (可选)inductive KG emb via subgraph + type [A] Nature SciRep s41598-023-48616-1

> 与我们关系:它们证明「不依赖实体身份、靠结构/路径也能对未见实体推理」这条 cold-start 前提可行;
> 但它们不问「什么样的交互结构在两端都未见时仍可迁移」,更没有低秩诊断。我们从这里接力。

---

## §2.2 Para 2 · 流 A  OOD / Transfer / DG 的低秩(核心论点:可迁移信息低秩;表征/协方差/子空间层面)

### A-1 transfer subspace learning via low-rank(源→目标子空间对齐,低秩滤噪传结构)
- **Generalized Transfer Subspace Learning Through Low-Rank Constraint** [C,IJCV 2014] Shao/Kit/Fu · Springer 10.1007/s11263-014-0696-6 —— 低秩约束对齐 source/target,传共享结构、滤噪
- Discriminative Transfer Subspace Learning via Low-Rank+Sparse [A,TIP] PubMed:26701675
- Transfer subspace learning: joint low-rank + feature selection [A,MTAP 2022] Zhou&Yang · Springer 10.1007/s11042-022-12504-z
- Non-Convex Transfer Subspace Learning for Unsupervised DA [A,IEEE] doc/8784937
- Transfer subspace learning via low-rank + discriminative reconstruction [A,KBS] S0950705118304222
- **Low-Rank Subspace Override for Unsupervised DA** [A,ECCV 2020] arXiv:1907.01343 —— 闭式单一域不变低秩子空间,盖过域结构
- **SALT: Subspace Alignment as Auxiliary Learning Task for DA** [A] arXiv:1906.04338
- Subspace-based Feature Alignment for UDA [A,IEEE] doc/9981324
- **Matrix Rank Embedding (MRE)** —— 同类跨域恢复低秩、异类最大分离 → 对齐 class-conditional [A,搜索确认,篇名待查]

### A-2 domain generalization 的低秩约束
- **Deep Domain Generalization With Structured Low-Rank Constraint** [A,DG] NSF par 10065328
- Local Domain Generalization with low-rank constraint (EEG emotion) [A] PMC10662311

### A-3 invariant-feature subspace 的低维/低秩恢复(可证明 DG)★新增·强相关
- **Provable Domain Generalization via Invariant-Feature Subspace Recovery (ISR-Mean / ISR-Cov)** [C,ICML 2022] · arXiv:2201.12919(扩展版 arXiv:2311.00966)
  —— 不变特征落在**低维子空间**,ISR-Cov 达 O(1) environment complexity;可证明 DG。**直接支撑「可迁移=低维」**
- Domain Generalization via Invariant Feature Representation (DICA) [C,ICML 2013] Muandet et al. · PMLR v28
- (landscape)Subset-Shared Invariances for DG with MoE [T,2026] arXiv:2606.25665 —— 提「不变子空间随域数单调收缩」的张力,别引只作参考

### A-4 effective rank / 表征秩 与迁移性★新增·非常贴题(warm/cold 秩发散的直接语言)
- **RankMe** —— 用 features 的 effective rank 评估自监督表征在下游的质量 [C,ICML 2023] Garrido et al.(ID待查)
- 冻结 ViT 知识本质低维:无任务适配时性能在 low rank r∈{2,4,8} 峰值;fine-tune 把最优 rank 推到 {8,16,32} 并改变 peak-to-tail [T,2025/2026 搜索级,别引,**但这正是我们 A1 warm/cold 秩发散现象的同构描述**]
- Ansuini et al. —— 内在维随层变化、与泛化相关 [C,NeurIPS 2019](ID待查)
- **The Tunnel Effect** —— 深层网络分「提取层 + tunnel 压缩到低维」两段 [T] arXiv:2305.19753
- Feature Contamination —— 网络学到无关特征、低秩化后仍难泛化 [T] arXiv:2406.03345

### A-5 OOD detection 的低秩/谱方法(注意:检测 ≠ 迁移预测,delta 要划清)
- **RankFeat: Rank-1 Feature Removal for OOD Detection** [C,NeurIPS 2022] Song/Wang/Sebe · arXiv:2209.08590 —— OOD 特征主导 rank-1 奇异值更大,去 rank-1 提检测;扩展 RankFeat&RankWeight [T-PAMI] arXiv:2311.13959
- **SeTAR** —— Selective Low-Rank Approximation 做 OOD 检测(training-free)[T] arXiv:2406.12629
- **GradOrth** —— 梯度正交投影做 OOD 检测 [T] arXiv:2308.00310

### A-6 OOD 泛化的低维子空间视角★新增
- **OOD Generalization of In-Context Learning: A Low-Dimensional Subspace Perspective** [T,2025] arXiv:2505.14808 —— 别引(太新),但论点同源:OOD 泛化靠低维子空间

### A-7 (GNN 侧低秩,transductive)
- PTDNet [A] arXiv:2011.07057 · LR-GCL / GCL-LRR [A] arXiv:2402.09600

> **流 A delta**:它们在特征/协方差/子空间层面做【对齐 / 检测 / 恢复不变子空间】,多为 warm/warm 域迁移;
> 我们把「可迁移信息低秩」搬到【残差交互函数 ΔM=M_A·M_B】层面,做【两端都未见的 cold-start DDI 预测】,且【诊断驱动】(A1 实测 warm/cold 秩发散)。

---

## §2.2 Para 2 · 流 B  LoRA / 内在维 的低秩(提供低秩因子化这个 FORM;参数效率 / 内在维度)

### B-1 内在维度(LoRA 之根)
- **Measuring the Intrinsic Dimension of Objective Landscapes** [C,ICLR 2018] Li/Farkhoor/Liu/Yosinski · arXiv:1804.08838(ID待查确认)—— 网络可在随机低维子空间里训练达标 → adaptation 内在维度低
- **Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning** [C,ACL 2021] Aghajanyan/Zettlemoyer/Gupta · arXiv:2012.13255 —— 低内在维 → 更新低内在秩(LoRA 直接理论根据)
- **Fine-tuning Happens in Tiny Subspaces** [A] arXiv:2305.17446
- **Exploring Universal Intrinsic Task Subspace via Prompt Tuning** [A] arXiv:2110.07867

### B-2 LoRA 本体与变体谱系(cite LoRA 本体即可,变体作 landscape)
- **LoRA: Low-Rank Adaptation of Large Language Models** [C,ICLR 2022] Hu et al. · arXiv:2106.09685 —— 冻结预训练权重,注入低秩因子化更新
- **AdaLoRA** —— importance-aware rank 分配 [C,ICLR 2023](ID待查)
- **DoRA: Weight-Decomposed Low-Rank Adaptation** [C,ICML 2024] · arXiv:2402.09353
- LoRA+ / ReLoRA / SARA(SVD-based adaptive rank, arXiv:2408.03290)/ NLoRA(arXiv:2502.14482)—— 变体,别逐个引
- 综述:**A Survey on LoRA of Large Language Models** [A,2024] arXiv:2407.11046
- 综述:**Low-Rank Adaptation for Foundation Models: A Comprehensive Review** [A,2025] arXiv:2501.00365

### B-3 LoRA 理论(为什么低秩够用)★新增·可支撑我们的「上界/迁移偏置」论证
- **The Expressive Power of Low-Rank Adaptation** [C,ICLR 2024] Zeng&Lee · arXiv:2310.17513 —— 给出 LoRA 逼近目标模型所需的秩下界;"task adaptation 落在低内在维子空间"
- LoRA vs Full Fine-Tuning: A Theoretical Perspective [T,2026] arXiv:2605.19018 —— 别引(太新)

### B-4 LoRA ∩ OOD(桥接两流:低秩适配在分布移位下的鲁棒性)★新增·连接 A 和 B 的关键
- **Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape** [T,2024] arXiv:2409.14396 —— flat minima 更能容分布移位 → 低秩适配的 OOD 泛化
- Bayesian-LoRA —— OOD 下更稳的 acc / 更低 NLL/ECE [T,2026] arXiv:2601.21003(别引)
- 注:landscape 一致提到「domain gap 大时低秩是瓶颈,rank 增大才回升」→ 呼应我们 warm 需高秩、cold 饱和于低秩

### B-5 图上的低秩适配(把 LoRA 搬到 GNN/KG)★新增·最近邻工作,delta 要清楚
- **GraphLoRA: Structure-Aware Contrastive Low-Rank Adaptation for Cross-Graph Transfer** [A,2024] arXiv:2409.16670 —— 对预训练 GNN 参数低秩分解做跨图迁移(**参数层低秩,非交互函数层**)
- GLoRA / IncLoRA(KG entity/relation emb 增量低秩适配)[T,搜索级]

> **流 B delta**:它们的低秩是为【参数效率 / 拟合已知任务的内在维】;
> 我们的低秩是【匹配两端未见时仍可迁移的预测结构】(是迁移归纳偏置,不是省参技巧)。
> GraphLoRA 等最近邻:低秩加在【预训练权重/参数】上做迁移;我们加在【mediator 残差交互 ΔM】上,且面向 cold-start 诊断。

---

## §2.2 理论桥  隐式低秩偏置 + 低秩泛化界(支撑「adapter 秩 a 决定泛化 gap 上界」)★新增
- **Implicit Regularization in Deep Matrix Factorization** [C,NeurIPS 2019] Arora et al. · arXiv:1905.13655 —— 加深度增强隐式低秩倾向
- **Towards Resolving Implicit Bias of GD for MF: Greedy Low-Rank Learning** [C,ICLR 2021] · arXiv:2012.09839
- **Gradient Descent for Deep MF: Dynamics and Implicit Bias towards Low Rank** [A] arXiv:2011.13772
- **From Low Intrinsic Dimensionality to Non-Vacuous Generalization Bounds in Deep Multi-Task Learning** [T,2025] arXiv:2501.19067 —— 低内在维 → 非空泛化界;"学子空间用全任务数据,子空间内学 per-task"(**与我们 M_B 共享基 + M_A per-pair 系数结构同构**)。别引正文,作理论叙事参考
- Meta-learning shared low-dim linear representation [A,2025] arXiv:2501.18975 / Sample-Efficient Subspace Meta-Learning arXiv:2102.07206 —— 共享低维子空间降低新任务样本需求

## §2.2 桥接:生物医学交互预测里的低秩因子分解(但 transductive,delta = 我们 cold-start)
- **统一 DTI 框架 = KG + recommendation(低秩矩阵分解)** [C,NatComm? 2021,你给的] —— transductive(两端已见)
- **DTI via low-rank matrix completion / nuclear-norm** 家族:Doubly Graph Regularized MC(bioRxiv 455642)、Multi-Graph Reg Nuclear Norm Min(PLOS One 2019)、Dual Laplacian Graph Reg MC(PMC6304580)、low-rank matrix projection(arXiv:1706.01876)、coupled matrix/tensor completion(Briefings in Bioinformatics 2021)
  —— **共识:DTI/DDI 交互矩阵本身低秩**,用 nuclear-norm/SVT 补全;但都 transductive、无 KG 语义路径、无 warm/cold 迁移诊断

---

## 翻转框(可选保留)GNN rank collapse:无意 collapse(坏)vs 我们有意低秩结构(好)
- **Oono & Suzuki, GNNs Exponentially Lose Expressive Power** [C,ICLR 2020] · arXiv:1905.10947
- **Rank Collapse Causes Over-Smoothing and Over-Correlation in GNNs** [C,LoG/TMLR] Roth · arXiv:2308.16800
- Li/Han/Wu, Deeper Insights into GCNs [C,AAAI 2018](ID待查)
- 用法:它们说深层 GNN 特征塌到低秩子空间 = **无意的、坏的**表达力损失;
  我们讲**有意的**残差交互低秩 = 匹配可迁移结构的归纳偏置。同一个「低秩」,一贬一褒,对比出我们的 novelty。

---

## 理论桥的隐式低秩偏置 ↔ neural collapse(可选,支撑「训练自发走向低秩」)★新增
- **Neural Rank Collapse: Weight Decay + Small Within-Class Var → Low-Rank Bias** [T] arXiv:2402.03991 —— weight decay 增大,各层秩按前层类内方差比例下降
- **Neural Collapse versus Low-rank Bias: Is Deep Neural Collapse Really Optimal?** [T] arXiv:2405.14468 —— 顶层低秩偏置可推更紧泛化界
- 用法:佐证「深网训练本就隐式偏向低秩解」,我们只是把这偏置**显式化**成 rank-a 的 ΔM。

---

## 引用时的 delta 速查(防 over-claim,一句话钉死每条对比)
```
vs Para1 inductive KG (GraIL/NBFNet): 它们做【无实体依赖的结构推理可行性】;
                                       我们问【两端未见时哪种交互结构可迁移】+ 低秩诊断
vs 流A A-1/A-2 transfer/DG 低秩:       它们在【特征/子空间】层面做【对齐】(warm↔warm 域);
                                       我们在【残差交互函数 ΔM】层面做【cold-start 预测】
vs 流A A-3 ISR(不变子空间):          它们证明不变特征低维;我们把它落到 DDI 交互 + KG 语义路径
vs 流A A-4 effective rank/RankMe:      它们观测表征秩与迁移相关;我们【实测】warm/cold 秩发散并【设计】利用它
vs 流A A-5 RankFeat/SeTAR:            它们做【OOD 检测】(主导 rank 是分布特有);我们做【可迁移预测信号】(低秩=共享)
vs 流B LoRA/内在维:                   它们低秩为【效率/内在维】;我们低秩为【可迁移预测结构】(迁移偏置)
vs 流B B-5 GraphLoRA:                它们低秩加在【预训练 GNN 参数】做跨图迁移;我们加在【mediator 残差交互】做 cold-start
vs 理论桥 隐式低秩/多任务界:         它们是【自发/无意】的低秩偏置;我们【显式】设成 rank-a 并给上界叙事
vs DTI 低秩矩阵补全:                 它 transductive(两端已见)、无 KG 语义;我们 cold-start + 语义路径 + 迁移诊断
```

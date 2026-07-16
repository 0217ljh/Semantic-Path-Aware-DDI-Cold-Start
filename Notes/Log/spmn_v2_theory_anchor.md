# spmn_v2 / PMP 理论锚点 (角度层,已锁 2026-06-28)

AND 相交 adapter (= PMP, Pair-Mediator Pooling) 相较 DDI GNN 的内生优势。**角度层定稿**;
调制层 (density/length-aware) 与对应实验、case studies 待 aware 部分做完再补。
唯一理论+实验对手锁 **EmerGNN**(path-based L-hop flow pairwise GNN,Zhang et al. 2023
Nat Comp Sci;contributions 见 `Code/reproductions/EmerGNN/_reviews/2026-05-18__paper_faithful.md` C1-C3)。
pair-GNN(SEAL/NCN/BUDDY)是通用 LP,非 DDI 竞品,写作时再于 related-work 处理。

经 4 轮 codex 对抗硬化(2026-06-28)。校准 = AAAI **解释级**,不追求强定理,够圆回
"为什么我们的 adapter 有效、胜过 EmerGNN" 即可。

## 设定

drug a 的 ℓ-hop 中介可达画像 r_a∈R^N;**AND/相交** j_ab = r_a ⊙ r_b(逐中介共可达,
带两端距离);打分 s = Σ_m w(m)·r_a[m]·r_b[m],w 按中介 type/relation/degree/distance。
EmerGNN = 在 a→b 间传播 L-hop flow 的 transport 式 pair 编码。

## 锁定的 3 条 claim(按承重排序;对手 = EmerGNN flow)

**① 见证可达性(主).** 共享中介 m 在两药各自够到它($\max(d_a,d_b)$ 半径)即成证据;
EmerGNN flow 要沿长度 $d_a+d_b$ 的 a→…→m→…→b 路由才用得上 → 非对称/远见证需更深 flow、
更易被过度扩张稀释。*成立*:flow 是 transport 式,完成一个 $(a,b)$ 见证须跨完整路径长度
合成;相交有显式 meet,$\max$ 即可。

**② shift-稳定 overlap 投影.** S2 里不稳的是单臂邻域质量,可迁移的是"两药都够到哪些中介";
相交显式投到 overlap,flow 把 overlap 与会漂移的单臂路径质量混在一起。

**③ 显式中介级加权(已软化).** 相交把每个共享中介保留到打分前 → 可直接按 hubness/type/
距离调权;EmerGNN 只能从 pooled path message **间接**恢复 → 更弱、更不省样本的 bias。
(措辞:**显式 vs 隐式**,非 "flow 不能"。)

*scope 脚注*:关于本文 S2 regime 的任务级解释,非不可能性定理。

## Argue(可近似直接用)

> EmerGNN 通过在两药间传播 L-hop flow 编码药对——transport 式:(i) 证据须沿长度 $d_a+d_b$ 的
> 完整路由,远/非对称联合见证需深 flow 且被稀释;(ii) 池化 path message,把可迁移 overlap 与
> 漂移的单臂路径质量混在一起;(iii) 中介级效应只能间接恢复。我们的 adapter 把药对投到其共享
> 见证集(相交):(a) 共享中介在 $\max(d_a,d_b)$ 即可用,无需深 flow;(b) 丢掉携带 shift 的
> 单臂质量、保住可迁移 overlap;(c) 显式暴露每个共享中介,压 hub、抬升判别信号所在的稀有特异
> 见证。这解释了为何近乎无参的相交 adapter 已追平强传播 baseline、并胜过 EmerGNN 的 flow。

## 措辞护栏(写作时锁死)

- ❌ "flow 无法表达合取/中介效应" → ✅ "显式 vs 隐式,更弱、更不省样本的 bias"
- ❌ "中介对 EmerGNN 在 max 处不可见" → ✅ "未经跨完整 $d_a+d_b$ 合成,无法作为完成的 $(a,b)$ 见证使用"
- ❌ "pairwise 胜单节点"(EmerGNN 本就 pairwise) → ✅ "overlap 投影 胜 transport 式 pair 编码(S2 shift 下)"
- ❌ "可迁移 Bayes 规则只依赖 j_ab" → ✅ "S2 下主导可迁移信号集中在 j_ab"

## promiscuity 张力 + 直接掉出的设计

promiscuity(单臂度数)是已知 DDI 预测因子 → 不能说"单臂全噪声"。吸收法 = **分解 scorer**
$s_{ab}=b(p_a,p_b)+\sum_m w_m r_a[m]r_b[m]$:$b$ 吃低维稳定边际协变量,第二项扛机制证据。
表述:"冷启动失败源于过度依赖单臂邻域**高维组成**;低维边际(promiscuity)可作协变量保留。"

## 支撑证据(已 verify)

- 域漂移:train vs val/test 支撑集域分类器 AUC ~0.76(`analyze_spmn_v2_domain_shift.py`,
  见 [[spmn_v2_adapter_asym_ab]])→ 支撑 claim ②(shift 在单臂)。
- 非-hub 判别信号:最稀疏档 Q1 正样本共享非-hub 中介 69.2 vs 负 24.9(2.8x),overlap-coef
  0.245 vs 0.091,过密度归一化 + max-arm 混淆对照(`analyze_spmn_v2_reach_density_control.py`)
  → 支撑 claim ②③ 的"判别信号在共享特异见证"。
- 洪水:ℓ≥3 共享集爆到 ~40% 图(`analyze_spmn_v2_reach_vs_density.py`)→ 支撑 claim ①
  (深 flow 被稀释)。
- standalone:adapter(PMP)3-seed AUC 0.7844/0.7472/0.7657 ≈ NBFNet 0.7841/0.7472/0.7495;
  已胜 EmerGNN(~0.7458,`Exps-loop-1.md:317`)。

## 待补(aware 部分完成后)— 实验账单

- **Claim ①**:固定/小容量 j_ab scorer,G1×G1 训→G2×G2 测,ID→OOD gap 明显小于 learned
  EmerGNN;+ pair 表征域分类器(EmerGNN 更可分域、j_ab 更不可分域但仍可分标签)。**这同时验证
  "裸 AND 内生优势 = shift-稳定"。**
- **Claim ②**:冻结 reach 编码器,只换 pair 算子 ⊙ vs +/concat/min/单臂;**marginal-matched
  负样本**上只有 ⊙ 还强;witness-deletion case(删 top 共享中介→塌,删等量单臂中介→不动)。
- **Claim ③**:按 $\max(d_a,d_b)$/非对称 gap 分层 + 扫深度;相交在 ℓ=max 即开,flow 拖到 $d_a+d_b$。
- 调制层方法本身(density 条件化 hub 抑制,固定浅 ℓ + Jaccard/overlap 特异性)、分解 scorer、
  case studies(CYP3A4/CYP2D6/CYP2C9 等,待查在不在 seed42 G2×G2)。

相关:[[project_semantic_path_ddi]] · [[project_pmp_v1_5_official]] · [[spmn_v2_adapter_asym_ab]]

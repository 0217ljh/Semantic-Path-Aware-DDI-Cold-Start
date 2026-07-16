# ColdDDI v1.6.x — Paper Frame (Locked 2026-06-08)

**Status**: Companion to `v1_6x_paper_narrative.md` (the lock doc).
**This doc** = the **executable paper frame**: the three concrete pieces (Part 1 / Part 2 / Part 3) the user asked to assemble into one document for paper-writing reference.

**Locked invariant** (cannot be modified):
> 存在一个结构变量, 它能让 cold-start DDI 变好.
> 基于端点 (drug) 的 GNN 在堆叠 / 平滑过程中会丢失这个结构变量.
> 我们新的 **hyper-edge + NBFNet** 框架可以保留它.

Everything in this document serves this single invariant.

---

# Part 1: Common Neighbor 作为 KG 结构变量

## 1.1 形式定义

**KG 设定** (跟 narrative doc §四 一致):

```
G = (V, E, R, K)
  V_drug ⊂ V       drug 节点子集
  V_drug = V_drug^seen ⊔ V_drug^unseen     训练时见过 vs cold-start 未见
```

### 1.1.1 Endpoint common neighbor (经典定义)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   N_endpoint(a, b)  =  N(a) ∩ N(b)                              │
│                                                                 │
│   where   N(v) = { u ∈ V : ∃r, (v, r, u) ∈ E or                 │
│                                  (u, r, v) ∈ E }                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

经典 link prediction 信号. 包含 drug + 非 drug 节点.

### 1.1.2 Middle common neighbor (我们的结构变量)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   N_middle(a, b)  =  (N(a) ∩ N(b)) \ V_drug                     │
│                                                                 │
│   分层按 anchor 类型:                                           │
│                                                                 │
│   N_middle^τ(a, b)  =  { m ∈ N_middle(a, b) :  κ(m) = τ }       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键意义**: `N_middle^τ` 是**纯结构对象** — 只由 KG 拓扑 + 节点类型决定, 不依赖任何学习参数.

### 1.1.3 推广到长 path (= hyper-path anchor node set)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   A_τ(a, b)  =  { m ∈ V :  κ(m) = τ,                            │
│                  ∃ path  a → ... → m → ... → b  in G }          │
│                                                                 │
│   特殊情况:  A_τ(a, b) | L≤2  ≡  N_middle^τ(a, b)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

→ **A_τ 是 middle common neighbor 在 L≥1 任意长度下的推广**.

→ Paper 写作时统一术语: **"generalized middle common neighbor"** = A_τ.

### 1.1.4 结构变量的核心性质

| 性质 | 内容 |
|---|---|
| **拓扑性** | A_τ(a, b) 只由 G 的边集 + κ 节点类型函数决定 |
| **零学习参数** | 计算 A_τ 不需要任何 trainable parameter |
| **Cold-start invariant** | 即使 a, b ∈ V_drug^unseen, A_τ(a, b) 完全可计算 |
| **可观察** | 给定 a, b, A_τ 是一个具体的节点集合, 可枚举 |

**这就是我们要 argue 的"结构变量"** — 它**先于任何学习**就存在, 是 KG 自身性质.

---

## 1.2 端点 GNN 为什么会丢失这个结构变量

### 1.2.1 端点 GNN 的工作方式 (formal)

经典 KG-based DDI 方法 (EmerGNN, SkipGNN, KGDDI, etc.) 都遵循:

```
1. Drug 端点表征:   e_a ∈ R^d, e_b ∈ R^d
                    通过 KG 上的 message passing 更新

2. Pair score:      f(a, b)  =  MLP( [e_a; e_b] )
                    或       =  e_a · W · e_b
                    或       =  similar bilinear form

3. Training:        最小化 train DDI labels 的 BCE / CE loss
                    learnable params 包括 e_a, e_b (or feature → e_a)
```

**Message passing 更新规则** (典型形式):
```
e_a^(l+1) = AGG( {message(e_x^(l), e_a^(l), r) : (x, r, a) ∈ E} )
```

### 1.2.2 失败模式 1: Over-smoothing 抹平 common neighbor 信号

**论证**: 假设 a, b 共享 middle common neighbor m. 一般 GNN 用 L 层传播:

```
Layer 0:  e_a^(0), e_b^(0) 各自独立 (随机 init 或 feature → init)

Layer 1:  e_a^(1) ← AGG over {m's neighbors of a, m, ... }
          e_b^(1) ← AGG over {m's neighbors of b, m, ... }
          ↑ 此时 e_a, e_b 都收到 m 的 message, 但 m 的贡献
            被 a / b 各自其他邻居稀释

Layer L:  e_a^(L), e_b^(L) 都已经经过 L 次平均-类型聚合
          ↑ Over-smoothing 现象: 远距离节点的信号被极度稀释
            m 在 a 和 b 上的贡献 完全混在所有其他邻居信号里
```

**关键问题**: 经过 L 层 GNN, `e_a, e_b` 各自被全 neighborhood "平均化". m 是 N(a) ∩ N(b) 的事实**没有被显式标记**, 只是混在 noise 里. 当 pair score `f(a, b) = MLP([e_a; e_b])` 计算时, **MLP 无法判断 `e_a` 和 `e_b` 是否共享 m 这个 source**.

### 1.2.3 失败模式 2: Drug 端点对 cold-start 是 out-of-distribution

```
训练: 学到的 message passing 参数 + e_a^seen 表征
       — 跟 V_drug^seen 的 neighborhood 模式 entangled

测试 (cold-start):
       a' ∈ V_drug^unseen, 没有 e_a'^trained
       即使做 inductive GNN, a' 的 neighborhood 模式可能跟 V_drug^seen 不同
       → 学到的 message weights 在新分布上不再 optimal
```

→ 端点 GNN 的 **信号被两次损耗**:
1. Architecture 不显式 reify "shared neighbor as feature"
2. Drug 端点本身在 cold-start 上 OOD

### 1.2.4 一句话总结端点 GNN 的损失

> 端点 GNN 把 **A_τ(a, b) 这个 KG 上可观察的结构变量, 通过 L 层 message passing 平滑成两个 drug 端点向量**.
> 平滑后, "a 和 b 共享类型-τ 中间节点" 这个**结构事实**不再可读 — 它被 noise (其他邻居信号) 淹没.
> Cold-start 时, drug 端点本身就 OOD, 失去恢复结构事实的最后机会.

---

## 1.3 Hyper-edge + NBFNet 为什么能保留这个结构变量

### 1.3.1 我们的架构 (recap from narrative)

```
Coarse layer (hyper-edge):
   对每个 anchor type τ ∈ T_anchor:
     A_τ(a, b)  ←  COMPUTED EXPLICITLY (set intersection)
     ↑ 这一步**直接 reify 了结构变量**, 不学

Fine layer (NBFNet within hyper-edge):
   对 m ∈ A_τ(a, b):
     h_τ_m  =  φ( NBFNet(a → m), NBFNet(m → b), m, τ )

Hyper-edge aggregation:
   h_τ(a, b)  =  AttnPool_{m ∈ A_τ(a, b)}  h_τ_m

Cross-hyper-edge:
   z(a, b)  =  ⊕_τ  attn_τ · h_τ(a, b)        (Meta-graph T)
   
Score:
   f(a, b)  =  MLP(z)
```

### 1.3.2 关键: A_τ 是显式输入, 不经过 smoothing

**对比端点 GNN**:
- 端点 GNN: A_τ 信息**通过 L 层 message passing 间接传到 e_a, e_b**, 沿途被平滑
- 我们: A_τ **直接作为索引集**, 由 model 显式枚举, 没有任何信息损失

**Architecturally**: 我们的模型在 forward pass 里**有一行代码** `compute A_τ(a, b)`. 这一行不学, 不传播, 不平滑. 它是**结构性算子**, 不是学习算子.

### 1.3.3 NBFNet 在内部不引入新的 smoothing

NBFNet 的 propagation 是 source-conditioned:
- 从 source `a` 出发, propagate L 层, readout 在 `m`
- 跟 endpoint GNN 不同: NBFNet **不需要学 drug 端点 embedding**
- v1.7 paper-faithful 设计: drug 节点 boundary = INDICATOR(v=a) · q, 没 learnable embedding

→ NBFNet 在每个 anchor m 周围算 `h(a→m), h(m→b)`, 这两段都是**结构 conditioned, 不依赖 drug-id**.

### 1.3.4 最终: A_τ 这个结构变量从输入到输出全程保留

```
输入:    A_τ(a, b) 结构集合
              ↓ (无 smoothing, 仅做集合操作)
中间:    {h_τ_m : m ∈ A_τ(a, b)}    显式按 anchor 分开
              ↓ (within-set attention pool)
输出:    h_τ(a, b)   per hyper-edge
              ↓ (cross-hyper-edge meta-graph)
最终:    z(a, b)    pair representation
```

**结构变量 A_τ(a, b)** 在 forward 全过程**始终 explicit available**, 没有被 over-smoothing 平掉.

### 1.3.5 数学陈述

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   命题 1 (我们的方法保留结构变量):                              │
│                                                                 │
│   存在一个 KG 上的结构变量 σ : V_drug × V_drug → 2^V × T_anchor │
│   定义为 σ(a, b) = ⋃_τ (A_τ(a, b), τ),                          │
│                                                                 │
│   使得我们的预测函数 f_ours 可以显式表达为                      │
│                                                                 │
│       f_ours(a, b)  =  g( σ(a, b),  θ_propagation )             │
│                                                                 │
│   其中 g 对 σ 的每个 (m, τ) 组件 explicit access, θ 与 drug-id │
│   解耦.                                                         │
│                                                                 │
│   而端点 GNN 方法 f_endpoint 满足                               │
│                                                                 │
│       f_endpoint(a, b)  =  g'( e_a, e_b,  θ_endpoint )          │
│                                                                 │
│   其中 e_a, e_b 是 σ(a, b) 经过 L 层 smoothing 后的 lossy summary, │
│   且 θ_endpoint 与 V_drug^seen 的分布 entangled.                │
│                                                                 │
│   命题: f_ours 的 cold-start invariance 由 σ 的纯结构性保证,    │
│   而 f_endpoint 在 cold-start 下因 σ 信息丢失 + drug-id OOD     │
│   而失效.                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.4 Figure 1 Spec — 结构变量在端点 GNN 中被 smooth 掉 vs 在我们框架中被保留

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Figure 1: Common Neighbor Structural Variable Under            │
│            Endpoint GNN Over-Smoothing vs Hyper-Edge Preservation│
│                                                                 │
│  Layout: 3 panels stacked vertically (each ~equal height)       │
│                                                                 │
│  ──────── Panel (a): The structural variable ────────           │
│                                                                 │
│   左侧: KG fragment showing a, b, and shared neighbor m (red    │
│   color, label "m: enzyme/pathway/target...").                  │
│   中间: 显式数学 box:                                           │
│         A_τ(a, b) = N_τ(a) ∩ N_τ(b) = {m}                       │
│         "structural variable, KG-intrinsic, no learning."       │
│   右侧: 文字 "This is what cold-start DDI prediction needs."    │
│                                                                 │
│  ──────── Panel (b): Endpoint GNN — over-smoothing loss ──     │
│                                                                 │
│   左侧: Same KG fragment, but now show L=3 GNN layers operating │
│         on drug nodes a, b. Each drug node has multi-hop        │
│         neighbors with arrows showing aggregation.              │
│                                                                 │
│   中间: Two side-by-side embedding vectors e_a^(L), e_b^(L),    │
│         shown as colored bars where each bar's color encodes    │
│         contribution source. m's contribution (red) is a TINY   │
│         slice diluted in many other neighbors (blue, green,     │
│         yellow). Visual: red ~ 5% of total bar, other colors ~  │
│         95%.                                                    │
│                                                                 │
│   右侧: MLP([e_a^(L); e_b^(L)]) → f(a, b)                       │
│         Caption sub: "MLP cannot determine if e_a and e_b       │
│         share m as a source — m's signal is smoothed away."     │
│                                                                 │
│   下方文字: "Endpoint GNN: structural variable A_τ(a,b) is      │
│             implicitly carried through e_a, e_b, but over L     │
│             layers of smoothing, it becomes indistinguishable   │
│             from background noise."                             │
│                                                                 │
│  ──────── Panel (c): Hyper-edge + NBFNet — preservation ─       │
│                                                                 │
│   左侧: Same KG fragment, but now:                              │
│         - Drug nodes a, b shown with NO learnable embedding     │
│           (gray, dashed border)                                 │
│         - A_τ(a, b) = {m} highlighted as a YELLOW SET BOX       │
│         - Arrow: "compute by set intersection — no learning"    │
│                                                                 │
│   中间: NBFNet propagates from a to m, then m to b, with        │
│         the path scoring shown explicitly. Two arrow chains:    │
│         a → m → b, the m node is highlighted as an EXPLICIT     │
│         anchor.                                                 │
│                                                                 │
│   右侧: h_τ(a, b) box, computed as AttnPool over m ∈ A_τ.       │
│         The structural variable A_τ is shown as INPUT to the    │
│         box, not buried in any embedding.                       │
│                                                                 │
│   下方文字: "Our framework: A_τ(a,b) is computed structurally   │
│             as a set operation and remains explicit through the │
│             entire forward pass. No smoothing erases it."       │
│                                                                 │
│  ──────── Bottom caption (figure-wide) ────────────────────     │
│                                                                 │
│   "Cold-start DDI requires preserving the KG structural        │
│    variable A_τ(a, b) — the typed middle common neighbors of    │
│    the drug pair. Endpoint GNNs smooth this variable into       │
│    learned drug embeddings, where it becomes inaccessible to    │
│    the pair classifier. Our hyper-edge + NBFNet framework       │
│    reifies A_τ as an explicit structural input, preserving      │
│    it throughout the forward pass and ensuring cold-start       │
│    transferability."                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**绘图实施 tips**:
- Panel (b) 的 "信号稀释" 是核心 visual — bar chart with red sliver vs other colors
- Panel (c) 的 yellow set box 跟 panel (b) 的 dilution 形成强对比
- Drug 节点 a, b 在 (c) 用虚线 / 灰色, 强调 "no learnable embedding"
- 用 same KG fragment 三个 panel, 这样读者**看到的是同一对 drug 在三种不同处理下的命运**

---

# Part 2: 多模态对齐失配的论证 (MKG-FENN / TIGER 攻击目标)

## 2.1 MKG-FENN 和 TIGER 的对齐方式 (paper 复习)

(待 Explore agent 返回 baseline 代码后补全具体 file:line 引用. 当前以方法层面论述.)

### 2.1.1 共同的 design pattern (基于已公开论文 + 项目内已知)

```
两者都依赖 "k-NN drug retrieval" / "similarity-based aggregation":

  step 1.  对每个 drug d, 学一组多模态 embedding (KG + 分子)
  step 2.  Cross-modal alignment: KG embedding ↔ molecular embedding
           via cross-attention / late fusion / contrastive loss
  step 3.  在 train 集 (known drug) 上学这套 alignment

  Inference for unknown drug d_new:
  step 4.  查 d_new 跟 known drug 的 top-k 相似 drug { d_1, ..., d_k }
  step 5.  Aggregate known drugs' multi-modal embeddings 作 d_new 的伪 embedding
  step 6.  用伪 embedding 进 DDI 预测头
```

### 2.1.2 已 verified 的失败现象

**MKG-FENN 和 TIGER 在 cold-start S2 上 AUC ≈ 0.5**.

- 项目背景: ColdDDI 已经验证过 (用更小 KG), 大 KG 消融也验证过, 数字稳定.
- AUC ≈ 0.5 意味着**完全无判别能力, 等同随机猜测**.

## 2.2 为什么对齐**不可迁移**

### 2.2.1 论证

**Argument 1: 对齐被锚定在 known drug**

两者的多模态对齐 (KG embedding ↔ 分子 embedding) 都是在 known drug 上学的:
```
Loss_align = Σ_{d ∈ V_drug^seen}  L( e_KG(d), e_mol(d) )
```

学到的 alignment function 把 known drug 的 KG embedding **投影到**它的 molecular embedding 邻域, 实现"两种模态对**同一个 drug** 描述一致". 这是经典 multi-modal 训练目标.

**但**: 这个 alignment 学的是 **point-to-point** 的具体对应 (e.g. drug X 的 KG embedding 在 R^d 哪一区, 该区跟它分子 embedding 哪一区 align).

```
对齐空间几何:
   {(e_KG(d_1), e_mol(d_1)), (e_KG(d_2), e_mol(d_2)), ...}
                                       ↑
                            point cloud on V_drug^seen
                            
对齐函数 f_align: f_align(e_KG(d)) ≈ e_mol(d) ∀ d ∈ V_drug^seen
```

**Argument 2: Cold-start drug 不在 alignment manifold 上**

```
对 d_new ∈ V_drug^unseen:
   e_KG(d_new), e_mol(d_new) 都是新分布的点
   
   f_align 是在 V_drug^seen 的 point cloud 上拟合的, 对未见点
   没有外推保证
   
   两者的实际 manifold 几何可能完全不重合:
   
            KG side                     molecular side
       ●     ●                          ●     ●
        ●  ●  ●   ← V_drug^seen ?       ●  ● ●   ← V_drug^seen
                                         f_align fits here
       ●  ★         ← d_new       ●           ★ ← d_new
                                        (in totally different region)
```

**Argument 3: k-NN retrieval 也修不了**

```
"用 k 个相似 known drug 做 d_new 的 surrogate" 的 implicit 假设:
  - d_new 应该跟某些 known drug 有 high similarity
  - 那些 known drug 的对齐学得对
  
但对 cold-start, 这两个假设都可能失败:
  - d_new 可能跟所有 known drug 都不太像 (OOD)
  - 即使找到 top-k, 它们的对齐 manifold 不能外推到 d_new
```

### 2.2.2 总结失配的本质

> **对齐对象错了**. MKG-FENN / TIGER 做的对齐是 **drug-to-drug 的 statistical alignment**: 学 "drug X 的 KG 视角 ↔ drug X 的分子视角". 这种对齐**绑定在 drug 身份上**, drug 身份 OOD 时, 对齐空间坍塌, alignment **失效**.

**对比我们的对齐 (preview)**: **fragment-to-hyperedge 的 mechanism alignment** — 不绑定 drug 身份, 绑定 **机制单元** (fragment 和 hyperedge), 这些机制单元在 V_drug^seen 训练时充分见过, 对 cold drug 仍可用.

## 2.3 我们的对齐对象为什么不同

### 2.3.1 核心切换: drug identity → mechanism unit

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  MKG-FENN / TIGER 的对齐空间:                                   │
│                                                                 │
│       e_KG(drug) ↔ e_mol(drug)                                  │
│                                                                 │
│       对齐单元 = drug identity                                  │
│       cold-start failure: drug identity OOD                     │
│                                                                 │
│  我们的对齐空间:                                                │
│                                                                 │
│       fragment(drug) ↔ hyperedge h_τ(drug, .)                   │
│                                                                 │
│       对齐单元 = mechanism (fragment-binding-anchor)            │
│       cold-start invariance: mechanism units 训练充分            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3.2 跟 hyperedge 的逻辑连接

**为什么这跟 hyperedge 是同一件事**:

- **Fragment 对应 mechanism action**: 一个分子片段 (e.g. β-lactam ring, benzene ring with halogen) 决定了它会跟哪类 protein 物理结合. 这是物理化学性质, 不是统计共现.

- **Hyperedge 对应 mechanism class**: 我们的 hyperedge τ 把"通过这类 anchor 节点 (target / pathway / etc.) 的 path" 聚成机制类. 这也是机制对应, 不是统计共现.

- **二者对齐 = 对齐机制**: fragment(d) ↔ h_τ(d, .) = "drug d 的某个片段 ↔ drug d 通过 τ-类机制起作用的 hyperedge 表征". 这种对齐**在物理化学定律保证下成立**, 跟 drug 是否见过无关.

### 2.3.3 可迁移性陈述

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   命题 2 (机制对齐可迁移性):                                    │
│                                                                 │
│   设 f_align: Fragment → HyperedgeEmbedding 学到 known drug      │
│   上的 fragment-hyperedge correspondence.                       │
│                                                                 │
│   对于 d ∈ V_drug^unseen, 因为:                                  │
│                                                                 │
│   (i) fragment(d) 由 d 的 SMILES 决定, 是分子的内禀化学性质,    │
│       不是 drug-identity 的函数                                 │
│                                                                 │
│   (ii) hyperedge h_τ(d, .) 由 A_τ(d, .) 决定, 而 A_τ 在 cold-   │
│        start invariant (Lemma 1)                                │
│                                                                 │
│   (iii) f_align 学的是 fragment 跟 hyperedge 类型的 mapping,    │
│        训练时见过的 fragment-hyperedge pair 在 cold-start 也      │
│        可能出现 (因为 fragment 是分子级别的, 远比 drug-level    │
│        更稠密)                                                  │
│                                                                 │
│   ∴ f_align 可外推到 V_drug^unseen, 对齐空间不会坍塌            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 2.4 Figure 2 Spec — 模态对齐在 known drug 上工作 vs 在 cold-start 失配

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Figure 2: Multi-Modal Alignment Failure on Cold-Start Drug     │
│           (Endpoint Alignment vs Mechanism Alignment)           │
│                                                                 │
│  Layout: 2 columns × 2 rows = 4 panels                         │
│                                                                 │
│  ──────── Top row: Endpoint alignment (MKG-FENN / TIGER) ──     │
│                                                                 │
│  (a) Top-left: "Known drugs — alignment learned"               │
│                                                                 │
│      2D scatter plot of drug embeddings, with two clouds:      │
│      - KG side (left): blue dots labeled "e_KG(d_i)"           │
│      - Mol side (right): red dots labeled "e_mol(d_i)"         │
│      - Arrows connecting same-drug pairs across sides          │
│      - All known drugs (~30 dots) shown with clean arrows      │
│      - Text: "f_align fitted: e_KG ↔ e_mol for d ∈ seen"      │
│                                                                 │
│  (b) Top-right: "Cold-start drug — alignment collapses"         │
│                                                                 │
│      Same 2D scatter, but now:                                  │
│      - Known drugs shown in gray (background)                  │
│      - One new drug d_new shown:                               │
│        ★ on KG side at position (5, -2)                        │
│        ★ on mol side at position (-3, 4)                       │
│      - Dashed arrow attempting alignment but pointing into     │
│        empty space (no clear known drug to anchor to)          │
│      - Red X over the connecting arrow                          │
│      - Text: "d_new lies OFF the learned manifold; f_align     │
│              has no valid extrapolation."                       │
│      - Caption sub: "k-NN retrieval also fails: top-k similar   │
│              known drugs are not actually close in either       │
│              modality."                                         │
│                                                                 │
│  ──────── Bottom row: Our framework — mechanism alignment ──     │
│                                                                 │
│  (c) Bottom-left: "Mechanism units — alignment by construction"│
│                                                                 │
│      Different visualization. Show fragments on left, hyperedge │
│      anchors on right. Each fragment ↔ hyperedge pair shown    │
│      as a labeled box:                                          │
│        β-lactam ring  ↔  hyperedge with anchor type="enzyme"   │
│        halogen group  ↔  hyperedge with anchor type="target"    │
│        amide bond     ↔  hyperedge with anchor type="pathway"   │
│      - Note: each pair is many-to-many (a fragment can map to  │
│        multiple hyperedge types)                                │
│      - Text: "Alignment unit = mechanism (fragment ↔ anchor    │
│              type), not drug identity."                         │
│                                                                 │
│  (d) Bottom-right: "Cold-start drug — alignment preserved"      │
│                                                                 │
│      Show new drug d_new with its fragments extracted (e.g.,   │
│      Mol → β-lactam + amide + halogen). Each fragment maps to  │
│      hyperedges that were trained on OTHER drugs that share    │
│      the same fragments.                                        │
│                                                                 │
│      Arrows: d_new's β-lactam → enzyme-hyperedge (anchored on   │
│      previously seen CYP enzyme nodes). Arrow is CLEAN, no X.   │
│                                                                 │
│      Text: "d_new's fragments map to mechanism units already   │
│              well-trained — alignment transfers without OOD    │
│              collapse."                                         │
│                                                                 │
│  ──────── Bottom caption (figure-wide) ────────────────────     │
│                                                                 │
│  "Multi-modal DDI methods like MKG-FENN and TIGER align KG      │
│   and molecular views at the drug-identity level (top row).    │
│   Under cold-start, the new drug's identity is OOD, and the    │
│   learned alignment manifold cannot extrapolate. Our framework  │
│   aligns at the mechanism level (bottom row): molecular         │
│   fragments map to hyperedge mechanism units, and because       │
│   fragments and mechanism units transfer beyond seen drugs,    │
│   alignment survives cold-start."                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# Part 3: 整体方法逻辑链

## 3.1 论文 narrative arc (一气讲完)

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  目前的多模态 DDI 预测在 cold-start 失效 (AUC ≈ 0.5).        │
│       ↓                                                      │
│  原因: 在 known drug 上学的多模态对齐 (drug-identity 锚定)   │
│        不能迁移到 unknown drug — 分子 ↔ KG 对齐失配          │
│        ↓                                                     │
│  我们提出: 分子 (SMILES / 分子图) ↔ KG hyperedge 对齐         │
│            — 在 mechanism 层面对齐, 不在 drug-identity 层面   │
│            ↓                                                 │
│  Hyper-edge 实现: 粗粒度 anchor-typed hyperedge              │
│                  +  细粒度 NBFNet within hyperedge           │
│                  ↑ (cold-start invariant by construction)    │
│       ↓                                                      │
│  实验: 多任务结果 (binary / multi-cls) — 未来 run           │
│        ↓                                                     │
│  分析: hyperedge 规避了不可靠的 unknown drug 端点            │
│        — 在 path 中间保留 common neighbor 这个结构特征       │
│        — 这是 cold-start 可迁移的根本原因                    │
│       ↓                                                      │
│  结论: 框架在 cold-start 上稳定可迁移                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 3.2 每个章节服务的核心 invariant

> **存在一个 KG 结构变量, 让 cold-start DDI 变好. 端点 GNN 会丢失它, 我们的 hyperedge + NBFNet 保留它.**

| 章节 | 核心 invariant 服务方式 |
|---|---|
| Introduction | 提出 invariant, 给 Figure 1 直观对比 |
| Problem (Sec 2) | 用 MKG-FENN/TIGER 失败论证 (Figure 2) 给问题加深度: 不只是 GNN 失效, 多模态对齐也失效, 都是同一根本原因 — 锚定在 drug-identity 而非结构变量 |
| Method (Sec 3) | 数学定义 A_τ, h_τ, fragment-hyperedge alignment. 全部围绕 "如何 reify 这个结构变量并保留之" |
| Experiments (Sec 4) | binary / multi-cls / cold-start S2 / ablation, 验证: 保留了结构变量, 因此性能 transfer |
| Analysis (Sec 5) | 显式 verify: 我们的方法在 forward pass 中 A_τ 始终 explicit (visualize attention on anchor nodes), 而 baseline 在 e_a/e_b 里看不到. 这是 paper 数学根基的实证 |

## 3.3 Multi-modal 对齐方法的具体设计 candidate (留作消融)

| Candidate | 实现 | 信息正交化 |
|---|---|---|
| **A. Contrastive** | InfoNCE on (fragment, h_τ) pairs | implicit via negatives |
| **B. Cross-attention** | fragment attends to {h_τ}_τ, fuse | explicit fragment-hyperedge attention |
| **C. Joint embedding** | project fragment + h_τ to shared space, distance loss | orthogonality penalty on shared space |
| **D. Mutual information max** | I(fragment; h_τ) maximization with MI estimator | explicit MI objective |

→ Paper main result 用 A (最简单), B/C/D 作 ablation. **信息正交化 / 最大化** (你提的) 进入 C/D candidate, 防止 fragment 信号 dominate 或 hyperedge 信号 dominate.

## 3.4 实验 ablation 结构 (留作 paper Sec 4)

```
Ablation 1 (cold-start invariance):
  - Endpoint GNN baseline 类 (EmerGNN, HDN-DDI, SkipGNN, ...)
    Drop ≥ 10 pt vs warm-start
  - Our framework on cold-start
    Stable performance
    → 验证 A_τ 保留

Ablation 2 (multi-modal alignment object):
  - MKG-FENN, TIGER (drug-identity alignment)
    AUC ≈ 0.5
  - Our fragment ↔ hyperedge alignment
    AUC ≈ 0.8+
    → 验证 mechanism alignment 可迁移

Ablation 3 (hyperedge granularity — coarse + fine):
  - Coarse only (Π_τ, hand-defined): some gain
  - + Fine NBFNet (variant C): bigger gain
  - + Cross-hyperedge meta-graph (T): full gain
    → 验证 multi-granularity 必要

Ablation 4 (anchor type — hand-defined vs learnable):
  - 12 KG-KIND fixed
  - Learnable soft assignment
  - Compare cold-start AUC
    → 验证 learnable upgrade 是否真有 gain

Ablation 5 (alignment objective): A/B/C/D 上述 candidate
  → 找最稳 objective
```

## 3.5 Cross-references for paper writing

**项目内文档**:
- `Notes/Log/v1_6x_paper_narrative.md` — v2 narrative anchor (3 → 1+2 framework)
- `Notes/Log/v1_6x_paper_frame.md` — this doc (Part 1/2/3 deliverable)

**项目内代码**:
- `Code/my_code/models/pmp_v1/v1_6/cluster_head.py` — current v1.6 implementation
- `Code/my_code/models/nbfnet_v1_7/nbfnet_model.py` — paper-faithful NBFNet (will integrate into hyperedge)
- `Code/baseline/mkg_fenn/` — TO BE MIGRATED (Part 2 deliverable)
- `Code/baseline/tiger/` — TO BE MIGRATED (Part 2 deliverable)

**Reference papers**:
- MHGNN (IEEE TNNLS 2026) — multiplex hypergraph lineage
- NBFNet (NeurIPS 2021) — fine layer path reasoning
- (MKG-FENN, TIGER) — multi-modal baselines to beat

---

# Status

| Deliverable | Status |
|---|---|
| Part 1: Common neighbor 形式化 + over-smoothing argument | ✅ Done |
| Part 1: Figure 1 spec | ✅ Done |
| Part 2: MKG-FENN/TIGER misalignment argument | ✅ Done (待 baseline 代码 verify 后补 file:line) |
| Part 2: Figure 2 spec | ✅ Done |
| Part 2: Baseline code migration | 🟡 In progress (Explore agent 进行中) |
| Part 3: Integrated method logic chain | ✅ Done |
| Part 3: Ablation structure | ✅ Done |
| 未来: figures 实际绘制 | ⏸ Pending |
| 未来: paper §3 数学小节正文写作 | ⏸ Pending |

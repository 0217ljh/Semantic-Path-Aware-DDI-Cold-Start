# Theory Framework v2 — Deeper Mathematics

> v1 (`theory-framework.md`) 保留作为 baseline。本文档是寻找**更深刻数学视角**的探索日志——v1 的工具（Ben-David HΔH、Le Cam two-point、Blackwell sufficiency、margin condition）都是 statistical learning theory 标准工具箱，框架的 novelty 在 *formulation* 不在 *mathematics*。本轮目标：找到能描述 cold-start **固有缺陷**的、有真正数学创新潜力的视角。

## v1 的 specific 缺陷

1. **数学工具全部 borrowed**：HΔH-divergence 是 2010 Ben-David；Le Cam 是 1970s；sufficiency 是 1940s；margin condition 是 Mammen-Tsybakov 1999。组合方式新，工具不新。
2. **"σ-field sufficiency under cold-start" 的统一概念虽然 clean，但只是 measure-theoretic 包装**——本质上还是 covariate-shift learning theory 的 specialization。
3. **没给 cold-start 一个 *本体论* 上的刻画**——cold-start 跟 IID、跟 covariate shift、跟 OOD generalization 的 *本质区别* 是什么？仅靠 support-disjointness 描述太弱。
4. **C2 的 path-signature 不变性**虽然是漂亮的架构定理，但本质是 WL/MPNN expressivity argument 的 specialization，工具不新。
5. **没有挖出"identity vs type"这一层**——cold-start 的 essence 是新 *个体* 出现，而 ML 框架习惯处理"同一 generative process 下的新 *样本*"。这是 token vs type 的哲学/数学 distinction，没被 v1 触及。

## 本轮要解决的核心问题

> Cold-start 的固有缺陷不能被现有 statistical learning theory 框架描述，**因为这些框架默认 entities 是 i.i.d. samples from a population distribution**——但 cold-start 处理的是 *individual entities with their own identity*，每个 entity 是一个 token 而非 type 实例。能否找到一套数学描述：

1. 把 entity 的 *identity* 作为 first-class object 而非 i.i.d. sample 处理？
2. 把 cold-start 的"未见过"形式化为 **structural-categorical** 而非 statistical 概念？
3. 给出一套能 reason about "extension across identity boundary" 的形式工具？
4. 揭示 anchor-dependent vs stable-feature 模型在这一层意义上的根本差别？

## 可能的数学视角候选

（不预设答案，留给 codex 评估和反提）

| 候选 | 直觉适配度 | 难点 |
|---|---|---|
| **Sheaf theory / sheaf cohomology** | 局部信息全局一致性 / cold-start 作为 obstruction to extension | 工具门槛高 |
| **Category theory / Kan extension** | model as functor; cold-start = functor extension along inclusion | 抽象，难落到 theorem |
| **Information geometry** | entities on statistical manifold; cold-start = Fisher-singular submanifold | Riemannian 张量计算 |
| **Persistent homology / TDA** | DDI 图拓扑随实体加入的变化 | 与 prediction risk 怎么挂钩不明显 |
| **Topos theory** | classifying topos of entity context | 离实证太远 |
| **Operator algebra / Koopman** | prediction as operator; cold-start = domain extension | DDI 不像 dynamical system |
| **Tropical geometry on path algebra** | EmerGNN flow 本质是 (max, +) 半环上的运算 | 创新但难 connect 到 cold-start |
| **Stochastic process / Polya / Hewitt-Savage** | exchangeability 在 entity level 下的破缺 | 适配中等 |
| **Algebraic statistics** | DDI graph as variety; new entity = new variable | 工程难 |
| **Non-commutative probability** | entities 不可交换的代数结构 | 抽象过头 |

## 与 codex 的讨论目标

- 让 codex 评估这些候选并推荐 1-2 个真正适配的
- 每轮我对 codex 的提议给 substantive critique（不是简单 "好" 或 "再深一点"）
- 目标是产出一个 **v2 theory section structure**，数学工具至少要让 v1 的反对者承认是真新东西

---

## Codex thread

`019e25eb-9c8c-7cc3-a250-d6c8d595eb82` (gpt-5.4)

---

## Round 1 — 开场 + codex 评估 10 候选

**我给 codex**: 列 10 个候选 (sheaf / category / info-geom / TDA / topos / tropical / operator / point process / algebraic stat / non-commutative)，要求 commit 一个，工具能 frame cold-start 为 *非 distributional* 概念。

**Codex 回**:

### 诊断 10 候选

真正有潜力的 3.5 个：
- `Category / Kan extension`：方向对但太语法化，会塌回 covariate shift——除非对象定义成 "有限 identity-context"
- `Sheaf theory`：可作几何外壳不能作 core；sheaf 本身不自动表达 "fresh identity"
- `Point process / exchangeability`：精准打到 (cold-start ≠ shift, 是 "新 atom 出现")，但**适合负面 / demolition tool**，不够 constructive
- `Operator`：appendix material

直接判掉:
- `Info-geom`：仍 statistical manifold，没跳出 distributional
- `Persistent homology`：装饰
- `Topos`：太高，不会比 nominal/presheaf 多给可证内容
- `Tropical`：与核心矛盾无关
- `Algebraic stat`：变量增加 ≠ identity emergence
- `Non-commutative`：buzzword

### Codex commit

**`Nominal semantics: entities as atoms, cold-start as fresh-name extension across finite contexts`**

来自 Pitts/Gabbay-Pitts 的 PL semantics 工具（fresh names, α-equivalence）。核心:
- Entity 先是 `atom`，然后才携带特征/关系
- 模型不是 `x → y` 而是随有限 context `S` 变化的 `P(S)`，对注入 `S ↪ T` 有 restriction/extension 规则
- Cold-start 本体定义：**沿 fresh inclusion `S ↪ S ∪ {a}` 是否存在 canonical、equivariant 的扩张？**

### Codex 的 anchor-dep vs stable-feature 区分

- `anchor-dependent`：support 绑在特定旧 token，fresh extension 不 canonical (需选 anchor/embedding/reference frame)，违反 naturality/equivariance
- `stable-feature`：只依赖重命名下不变的结构量，support 绑在结构 witness；fresh extension canonical

**Meeting-node 的真正意义**：**support reduction mechanism**——把预测从 "依附旧实体名的锚定推断" 改写成 "经由结构性中介对象的关系推断"，把预测从 non-equivariant anchor semantics 拉回 equivariant structural semantics。

### Codex 论证 nominal vs Kan

Kan 只说"怎么延拓函子"，nominal 先把对象定为"有限 name-context"再谈 extension。Theorem 新意:
- cold-start 一开始就是 `freshness/extension` 问题，不是 `shift` 问题
- i.i.d./exchangeability 不是 merely false，而是**表达错了对象**
- anchor failure 不是 estimator weakness，而是**non-canonical dependence on names**

辅助副武器: Hewitt-Savage / marked point process 写引言反面: "new entity appears at time t" 在纯 exchangeable 语义里**不是内部可表达的对象**。

---

## Round 2 — 我的 5 个 critique + codex 回应

### Q1 — Nominal vs Set / permutation equivariance 的 distinction

**我**: Deep Sets/Maron 已把 permutation invariance 做透。"finite permutation of atoms" 看上去就是 S_n-equivariance 换皮。**具体说出** nominal 比 set-equivariance 多出来的数学结构。

**Codex**: 三个结构同时存在，前两个最关键。

**(1) Finite support**: nominal object `x` 不只对所有置换等变，还要求存在 `supp(x) ⊂ A`，使任何固定 `supp(x)` 外的有限置换 `π` 满足 `π·x = x`。Deep Sets 没有 support 概念——元素是 anonymous。但 nominal 可以问 set-equivariant 问不了的：
- 预测 `F_S(a, b)` 的 support 是否包含某些训练原子 `c_1, ..., c_k`？
- 当 `a` fresh 时，`supp(F_{S∪{a}})` 是否仍必须绑住旧名字？

**这就是 anchor-dependence 的可定义 locus**。

**(2) Freshness as primitive relation** `a # x` (atom `a` is fresh for `x`, ≡ `a ∉ supp(x)`)。Set 语言里 `X ⊂ X ∪ {u}` 的 `u` 没逻辑地位；nominal fresh quantifier `Νa.P(a)` 表达"对**足够新**的 `a`，`P(a)` 成立且与选哪个 fresh `a` 无关"。把 **generic new name** 当语义原语，这正是 cold-start 要表达的 genericity。

**(3) Presheaf over injections, not S_n on fixed carrier**: 底座是 `Set^{Inj_f^{op}}`。对象随有限 name-context `S` 变化，态射是注入 `i: S ↪ T`。研究的是**跨上下文 extension/restriction 的相容性**。Cold-start 的核心方块:

```
C(S) ──F_S──→ P(S)
 │              │
i_*             i_*
 ↓              ↓
C(S∪{a}) ─F_{S∪{a}}→ P(S∪{a})
```

Deep Sets 只控横向"换标签不变"；nominal/presheaf 还控纵向"扩张上下文后如何一致延拓"。

**审稿人最硬一句**: "Deep Sets-style permutation invariance quotients away identity; nominal semantics retains identity as atoms with finite support and freshness. Cold-start is not symmetry over a fixed carrier, but extension along fresh-name inclusions in the category of finite contexts."

### Q2 — Qualitative impossibility 够不够 / 选 A/B/C

**我**: 给三个选项 (A 纯 qualitative / B hybrid 用 obstruction invariant / C 直接 lift 到 risk bound)。Commit 哪条？

**Codex commit**: **B (Hybrid)**

- 主定理: 结构性 impossibility / uniqueness（qualitative）
  - anchor-dependent predictor **cannot** admit support-bounded natural extension along fresh inclusions
  - stable structural predictor admits canonical natural extension
- 定义 quantitative obstruction:

  `δ_F(i, c) = d(F_T(i_* c), i_* F_S(c))`

  其中 `i: S ↪ T` 是 fresh inclusion，`d` 是 prediction 上的伪距离。聚合:

  `Δ_F^fresh = sup_{i, c} δ_F(i, c)` 或 `E[δ_F(i, C)]`

- 主线**停在 "structural defect invariant" 而非 generalization error bound**。实验证 `Δ^fresh` 与 cold-start degradation 相关。

理由: 保住 nominal 本体优势，给 reviewer 数字，留口子做 C 在 supplement。

### Q3 — Anchor-dep 的精确 locus (EmerGNN 哪一行 break naturality)

**Codex**: **不在 message passing per se；在第一个把 "训练期出现过的 drug token" 参数化进表示的部位。最直接是 learned drug-ID embedding。**

逐项判:
- **Morgan fingerprint init**: **不违反**。是 SMILES 的函数，与实体名 rename 无关。
- **learned drug-ID embedding**: **明确违反**。`e_u` 是按名索引的自由参数。Cold-start `a ∉ V_train`: `e_a` 未定义；对 `i: S ↪ S∪{a}` 不存在唯一的 `i_*(e_u)` 使扩张自然。**主破坏点**。
- **relation-type attention**: 单独不违反 (relation 在 signature 不是 atom)。但若 `h_u, h_v` 含 `e_u, e_v`，attention 继承并放大破坏。
- **source-conditioned readout-at-v**:
  - 通过结构 source descriptor (路径类型/子图签名) → 可 equivariant
  - 通过 ID-tied coordinates → 不自然

**论文 diagnosis 措辞**: "The nominal defect is not 'message passing' per se. It occurs at the first point where entity names are turned into free parameters. In EmerGNN-style architectures this is the learned entity-ID lookup; downstream attention/readout then operate in a coordinate system already anchored to training identities."

### Q4 — 跨域 instantiation

**Codex** 统一模板:
- `Atom`: 持久身份单位
- `Context S`: 有限 atoms + 可观察结构
- `Prediction object`: 对 fresh/existing atoms 上关系/行为的断言

**Inductive KGC**: atoms = entities (relation 通常**不是** atom，除非研究 new-relation gen)。Context = `S` + 三元组事实 `E_S ⊂ S × R × S` + 属性。Cold-start = `S ↪ S∪{a}` 后含 `a` 的三元组 scoring 是否 canonical extend。

**Cold-start recsys**: **two-sorted nominal** (`A_U`, `A_I` atoms)。Context = 有限用户/物品 + 交互边 + 侧信息 + 时间戳。双边 cold-start 是两类 atom 的上下文扩张，**不是同分布采样**——nominal 优势更明显。

**Few-shot relational**: episode-local nominal context。Few-shot = cold-start 局部版本: context 很小但问题仍是"给定有限 identities 如何对新 identity-combination 扩张"。

统一一句: "The framework does not care whether the observable structure is a KG, a bipartite interaction log, or a relational episode; it only needs a finite identity-context and a prediction rule whose behavior under fresh-name extension can be studied."

### Q5 — Reviewer hazard

**Codex**: **可管理的 hazard，不是致命**。具体策略:

**(1)** Main text 只保留最小 nominal core (4 个 primitive): atom / finite context, support, freshness `a # x`, fresh inclusion + natural extension square。**正文不出现 Schanuel topos**。

**(2)** 一张对比表区分:
- Set/perm view: anonymous elements, fixed-carrier symmetry, no support, no freshness, no context extension
- Nominal view: persistent atoms, finite support, freshness, injections between contexts, canonical extension question

**(3)** Supplement 放语义学血统 (nominal sets equiv to presheaves on `Inj_f^{op}`、fresh quantifier、categorical restatement)

**(4)** Framing 上**不要把 "nominal" 放标题主位**。优先用:
- identity-aware cold-start
- fresh-entity extension
- support-stable predictors

正文里再说 "we formalize this using nominal semantics"。让 reviewer 先接受问题对象，再接受这门数学。

---

## Round 3 — 3 个 operational hard problem + codex 回应

### O1 — Support 的可计算定义

**我**: 给 4 个候选 (counterfactual / gradient / declarative / functorial)。Commit 默认定义。

**Codex commit**: **`support = functorial (iv)` 作定义；architecture-level certificate (iii) 作可证上界；counterfactual masking (i) 作实验近似**。三层不混用。

形式定义:
```
supp_F(S, c) := ∩ { S' ⊆ S : c ∈ C(S'), F_{S'}(c) = res_{S', S} F_S(c) }
```
= **最小子上下文使输出已被决定**。语义性、架构无关、与 fresh extension 直接联通。

为什么不用 (i)/(ii) 当定义:
- (i) 需阈值，不该一开始数值化
- (ii) 只捕捉参数敏感性，对 non-smooth aggregation / 离散 retrieval / attention saturation 不稳

论文措辞: "We define support semantically, as the least subcontext on which the prediction is already determined."

**架构 certificate**:
- 纯结构模型 `F^{struct}`: 证 ∃ 局部结构决定的 `K(c)` 使 `supp_F(S, c) ⊆ K(c)`，且 `K(c)` 不依赖任意训练 token name
- 含 ID lookup 模型 `F^{id}`: 证存在 query `c` 使 supp 不能被任何 name-free structural witness 上界

Reviewer 问"怎么真算 support"答: "定义是语义的；对关键架构类证明可由结构 witness 上界 / 不可上界；实验用 counterfactual masking 作 operational probe，不作定义。"

### O2 — Naturality square 的可操作化

**我**: 单模型/双上下文 (A) vs 双模型 (B) vs 架构 by-construction (C)，commit 哪条。

**Codex commit**: **理论走 C，实验走 A，绝不走 B**。

**关键澄清**: `F_S` 不是"在 `S` 上单独训练出的模型"，而是**固定参数 `θ` 的预测程序 `F_θ` 在不同上下文 `S` 上的实例化**:
```
F_{θ, S}: C(S) → P(S)
```
同一个 `θ`，不同 `S` 只是输入 graph/context 不同。GNN 训完本来就这样工作。

Fresh inclusion `i: S ↪ T` 的 naturality square 说:
```
F_{θ, T}(i_* c) =? i_* F_{θ, S}(c)
```
**没有第二次训练，没有训练噪声**。

**理论定理形式**:
- Name-free structural architectures: 证对任意 `θ`、任意 fresh inclusion，新增 atom 不进 query 结构 support 时严格 commute；更强地证整个类的 canonical extension
- ID-anchored architectures: 证 by construction 存在上下文+查询使 commutative square 不可能 universally hold

**类级别 theorem，不需要数值 δ**。

**实验层定义 inference-time defect**:
```
δ_θ(i, c) = d(F_{θ, T}(i_* c), i_* F_{θ, S}(c))
```
- `S` = 原始 query context
- `T = S ∪ U_fresh`，加入与 query 无关 / weak bridge 的 fresh entities
- 参数固定不重训
- 比较 query score/representation 变化

关键句: "Our naturality defect is an inference-time contextual stability functional, not a retraining-based generalization gap."

### O3 — PubMedBERT(name) 怎么算

**我**: 给 3 个 reconciliation (α: name-preserving permutation / β: attribute-bearing atom / γ: name 是 structure 函数)。Commit 一个。

**Codex commit**: **(β)，吸收一点 (α) 的技术表达，不用 (γ) 主辩护**。

**核心立场**: atom 不是 bare identifier，而是带内禀属性的实体位点；name string 是该实体在上下文中的可观察属性，不是 atom 本身。PubMedBERT(name) 作用在属性上，因此不是 nominally illicit。

形式化用 **labeled nominal context**:
- 原子集 `A` 承载持久 identity
- 每个上下文 `S` 伴随属性映射 `name_S: S → Σ*`、`smiles_S`、`text_S` 等
- 预测器允许依赖属性，但**不允许以 atom 裸名作为参数键**

**Admissible renaming** `π: S → T` 必须保结构:
```
name_T(π(u)) = name_S(u)
smiles_T(π(u)) = smiles_S(u)
... etc.
```

等变性对象是 **atoms-with-attributes 的有限上下文同构**，不是裸 atoms 的 permutation。

- `lookup_table[u]` 非法（atom 为索引访问自由参数）
- `BERT(name_S(u))` 合法（可观察属性的函数）
- Fresh atom `a` 只需给出 `name(a)` 即可计算 PubMedBERT，无需新增参数

回应 reviewer "ad hoc 设计 atom 类型": "We do not treat every string attached to an entity as a renameable identifier. The nominal action is on persistent entity identities; textual names are observed attributes that travel with the entity under context isomorphisms. This is the standard notion of isomorphism for relational structures with labels, not an ad hoc exception."

更硬版本: **many-sorted labeled nominal structures** — sort 1: entity atoms; sort 2: attribute values (`Σ*`, fingerprints, text embeddings); morphisms act on entity sort, preserve attribute interpretation。

不用 (γ) 主辩护因为"drug names encode chemistry prior"太经验化，dataset-specific story。(β) 是结构上稳的解释: **name string 是 attribute，不是 atom identity itself**。

### Codex preempt 的两个 hidden landmines

1. **Query 含 fresh atom 的 typing**: `c = (a, v, r?)`, `a` fresh — `c` 属于哪个 context、restriction 如何作用、`a` 不在子上下文时 `c` 是否无意义。Support/naturality 的 formal typing 不能到定理时补。
2. **Canonical extension 的 uniqueness**: 在 labeled contexts 里"唯一"只能 "up to context-preserving isomorphism / observational equivalence"，不能写 strict uniqueness。

Codex 总评: landmine 已排 80%，剩下决定成败的是下一轮骨架能否把 5 个接口 (`context / query / support / extension / attribute-preserving isomorphism`) typed cleanly。

---

## Round 4 — 骨架 (semi-formal skeleton)

**我**: 要求按 §1-§9 顺序输出 rigorous skeleton，每个 §3-§5 定理给 precise statement + proof idea + fallback。

**Codex 修正**: §1 的 context 必须加入 `E_S` (有限关系事实)，不只 atoms+attributes，否则 query 类型不严谨。§4 Theorem 1 拆成 1A (old-query stability) + 1B (fresh-query canonicality)。

### §1 — Primitives

- 可数无限 `A`, atoms `a, b, u, v ∈ A` 是 persistent identities，不是 feature space
- 属性 schema `Σ` 有限 (`name: Σ*`, `smiles: Σ*`, `fp: R^d`, `text: R^{d_t}`)
- 关系签名 `R` 有限带 arity/type
- **Finite labeled context**: `S = (S_A, λ_S, E_S)`
  - `S_A ⊂ A` 有限
  - `λ_S: S_A → ∏_{σ ∈ Σ} cod(σ)`
  - `E_S ⊆ ⊔_{r ∈ R} S_A^{arity(r)}`
- **Context morphism** `i: S → T`: injection 保 attributes 与 relations
- **Fresh inclusion**: `j: S ↪ S[a, α, Δ]`，`a ∉ S_A`，`α` 是 `a` 的属性，`Δ` 涉及 `a` 与旧 atoms 的新增 facts

### §2 — Predictor as labeled-context-indexed functor

- Query presheaf `Q(S)`: DDI 用 `Q_DDI(S) = S_A × S_A × R_ddi`，KGC 用 `Q_KGC(S) = {(h, r, t)}`
- `i_*: Q(S) → Q(T)` 按 atom 替换作用
- Prediction object `P`: 主文 commit `P(S) = Y` (with `Y` independent of `S`，logit/probability/scalar)，naturality 简化为 `F_T(i_* c) = F_S(c)`
- **Naturality 用在 cold-start 上分两类 typed 情形**:
  1. Old-query stability: `c ∈ Q(S)`，比较 `F_S(c)` 与 `F_T(i_* c)`
  2. Fresh-query canonicality: `c ∈ Q(T)` 含 fresh `a`，要求 `F_T(c)` 只依赖结构 support 不依赖 anchor choice
- Support 用 O1 的 functorial 定义
- `Aut(S)` 作用是 baseline sanity，**不是** cold-start 完整条件——cold-start 关键是跨 context 的 fresh inclusion

### §3 — Architecture classification

按"是否通过 atom 本身访问自由参数"分类:

- **Class N (name-free structural)**: `F_S(c)` 只依赖 `c` 中 atoms 的 attributes、`S` 的关系结构、共享参数 `θ`、relation-type parameters。对任意 context iso `π: S ≅ T`，`F_T(π_* c) = F_S(c)`。允许 `BERT(name(u))`, `g(fp(u))`, `W_r`；**不允许** `u ↦ e_u` lookup
- **Class A (anchor-dependent / name-keyed)**: 至少一次 `u ↦ e_u(θ)` atom-keyed parameter retrieval，`e_u` 不能由 `λ_S(u)` 共享函数计算
- **Class M (mixed)**: 既含 N 又含 A 组件；继承 A 的 obstruction (除非 name-keyed 部分对目标 query 完全不可达)

**模型定位 (重要修正)**:

- **EmerGNN**: 按 published methods 文字，初始 input 是 fingerprint feature，message function 用 relation-type weights + drug fingerprints + attention，**不含 ID lookup**——按文字属 **Class N**，不是 Class A。这是 v1 C2 错误归类的修正！
- **NBFNet**: paper 明说 "若 parameterize with learnable entity embeddings 则 cannot solve inductive setting"——标准 inductive NBFNet 属 **N**，entity-embedding variant 属 **A**
- **Meeting-node + PubMedBERT(name)**: 按 O3 的 attribute view，属 **N**
- **KnowDDI / DeepLGF**: 若含 learned drug embeddings 则 **M/A**

Reviewer 反例 fallback: §3 论证对象收缩到 "all name-keyed architectures + specific inspected baselines that instantiate them"，不是"所有 baseline"。

### §4 — Lead theorem (qualitative)

**Theorem 1A (Old-query stability for Class N)**: 设 `F ∈ N`，`j: S ↪ T` fresh，`c ∈ Q(S)`，则 `F_T(j_* c) = F_S(c)`。

更一般地: `c ∈ Q(T)` 且 `supp_F(T, c) ⊆ j(S)` ⇒ `F_T(c) = F_S(j^{-1} c)`。

**Proof idea**: Induction on Class N computation graph layers——每层是 labeled-structure homomorphism compatible；新增 atom 不入 support ⇒ messages 不入 query computation cone。

**Break**: 全图归一化让 fresh atom 通过 denominator 影响旧 query。**Fallback**: 限 `N` 到 query-local / masked-global normalizations。

**Theorem 1B (Canonical extension for fresh queries in Class N)**: `F ∈ N`，`j: S ↪ T = S[a, α, Δ]` fresh，`c ∈ Q(T)` 含 `a`，则存在由 `(α, Δ, c|_{supp(c)})` **唯一决定**的 `Ext_F(S; a, α, Δ; c) ∈ P(T)` 使 `F_T(c) = Ext_F(...)`，**up to observational equivalence**。

**Proof idea**: 不是"从 `S` 推出 `F_T(c)`"，而是"一旦给定新实体 attributes + relations，Class N 程序有唯一结构性运行结果"。`F_T(c)` 本身就是该 canonical extension；要证它不依赖 anchor choice。Uniqueness 必须 up to context-preserving iso (latent gauge 不强证)。

**Theorem 2 (Obstruction for Class A)**: 设 `F ∈ A`，存在 atom-keyed access `u ↦ e_u` 对某 query family 真正 sensitive。则存在 `S, j, c` 使**不存在**任何 universal canonical extension rule `F̃` 满足 (1) `F̃|_S = F|_S` (2) 对所有 attribute-preserving isos natural (3) 通过同类 atom-keyed scheme 实现。

**Proof idea**: Non-functorial parameter indexing——fresh atom `a` 若无独立 `e_a` 则模型不封闭；若引入则不 canonical（`e_a` 选择不由 `(α, Δ)` 决定）；若借旧药 embedding 代 `e_a` 则引入 anchor choice。**类别论意义的 "no canonical lift"，不是统计意义的 risk bound**。

**Break**: 某些 A 模型在特定 query 上碰巧不用 lookup。**Fallback**: 写"there exists a context/query family exhibiting obstruction"而非 universal failure。

### §5 — Quantitative obstruction Δ^fresh

固定 `θ`，inference-time defect:
```
δ_θ^old(j, c) = d(F_{θ,T}(j_* c), F_{θ,S}(c))     [c ∈ Q(S), old query]
δ_θ^new(c; T_1, T_2) = d(F_{θ,T_1}(c_1), F_{θ,T_2}(c_2))   [c 含 fresh a, two completions]
Δ_θ^fresh = sup{δ_θ^old, δ_θ^new}
```

**Theorem 3 (Defect = 0 for Class N)**: `F_θ ∈ N` + 局部性 ⇒ `Δ_θ^fresh ≡ 0`。直接由 §4 推出，d 选为 observational metric。

**Theorem 4 (Positive defect for generic Class A)**: `F_θ ∈ A` + atom-keyed param 对 query active ⇒ 存在 contexts 使 `Δ_θ^fresh ≥ γ > 0`。**不是"所有 A 统一常数下界"**——是 context-specific positive lower bound。两种 witness:
1. Old-query witness: fresh atom 改变 anchor set / normalization cone / path competition
2. New-query witness: 同 fresh atom 两个 canonical-looking completions 触发不同 name-keyed parameter path

**Break**: 距离 `d` 选错 (无关 gauge)。**Fallback**: task-relevant observational metric (predicted distribution distance / rank inconsistency / margin flip rate)。

### §6 — DDI worked example (重要 honest correction)

按 EmerGNN published methods 字面意思，**EmerGNN 更接近 Class N**，不是 v1 中我们以为的 Class A。

分两个对象:

**(i) Meeting-node + PubMedBERT(name) ∈ N**: drug atom `u` 属性 `λ(u) = (name, smiles, fp, ...)`。`z_u^name = BERT_φ(name(u))`。Meeting-node 派生结构中介 `m(u, v)` 或 `M_{u,v}`。预测 `F_S(u, v) = g_θ(z_u^name, z_v^name, Agg(M_{u,v}; E_S, λ_S))`。不访问 `u ↦ e_u`，属 N。Theorem 1A/1B/3 直接适用。Meeting-node 的作用是 **support 从 anchor-tied dependence 收缩到 witness subgraph dependence**。

**(ii) Name-keyed DDI baseline ∈ A/M**: 若 `h_u^(0) = φ(fp(u), e_u)`, `e_u = lookup_θ[u]`，break point 就在 `e_u`——后续 attention/aggregation 即便表面只读 `h`，已继承 name dependence。

**EmerGNN 诚实定位**:
- `EmerGNN-as-published`: **N-leaning** structural flow model
- `EmerGNN + learned drug-ID table` 变体: **M/A**，break 点在 lookup init，不在 Eq.(2)/(3) attention 本身

v1 C2 改写: 不是"EmerGNN impossibility"，而是 **"source-conditioned pair-flow architectures become nominally problematic exactly when source identity is parameterized by atom-keyed embeddings, not when source conditioning is implemented structurally"**。

### §7 — Cross-domain instantiation

**Inductive KGC**: atoms = entities (relation 通常**不**当 atom)。NBFNet 自证: paper 明说 entity-embedding variant 不能 inductive，与 nominal obstruction 同向。

**Cold-start recsys**: two-sorted `A_U, A_I`。双边 cold-start 是两类 identity sort 的 context extension，**不是 covariate shift**。N: content-based; A: ID-lookup-dominant.

**Few-shot relational**: episode 内 entity tokens，support set 诱导小 finite context。Few-shot = cold-start 的局部化。N 通过 support-induced witness 扩张，不通过 token keys。

### §8 — v1 vs v2 关系 (codex commit)

**v2 deprecates v1 as the ontological top layer, but v1 survives as the statistical/architectural corollary layer**:

- **v1 C1** → v2 的统计后处理层: "Once a predictor admits canonical fresh extension, one may still study its statistical stability by σ-field / transfer tools"
- **v1 C2** → 改写为 "source-conditioned pair-flow 中 structural vs name-keyed source 区分" 的 specialization。Path-signature invariance 仍 useful，但只在 "条件化变量本身是结构性 attribute" 时是 v2 的 positive specialization
- **v1 C3** → 自然变成 v2 下的 **support reduction theorem family**: meeting-node / selector 的本质是把 support 从 anchor-tied 压缩到 structure-witness。Blackwell lower bound 保留为 quantitative supplement

总结: v1 不再扛"cold-start 是什么"，而扛"在 canonical-extension regime 下还能给什么 quantitative guarantees"。

### §9 — Scope / non-claims (6 条)

1. **不 claim 所有 Class A 都实际失败**——Theorem 2/4 只说 lack of canonical extension / existence of positive defect witness，benchmark 上某些 A 可能 attribute path dominates 表现好。Nominal theory 判 structural admissibility，不是 benchmark 必败。
2. **不 claim 所有 Class N 都成功**——canonical extension 只解决"对象定义对了"，不保证 margin / sample complexity / optimization / label noise / support coverage 都足够。N 是 cold-start meaningfulness 的**必要结构条件**，不是**充分性能条件**。
3. **不 claim risk lower bound**——`Δ^fresh` 是 structural defect functional 不是 test risk。可 empirically correlate，但本稿不提供 `R ≥ f(Δ^fresh)`。
4. **不 claim 所有 attribute 都该 nominally preserved**——morphism 选 attribute-preserving 是 modeling choice，适用于"属性是 entity semantic annotations"的场景。
5. **不 claim strict uniqueness of latent representations**——up to context-preserving iso / observational equivalence。
6. **不 claim full category-theoretic maximality**——主文只用 finite contexts/morphisms/support/freshness/extension。Schanuel topos / presheaf equivalence 放 supplement。

### Codex 提的 7 个 uncertain items

1. Theorem 1B canonical extension operator 是否显式 functorial 定义
2. Support lattice 是否限到 admissible subcontexts (typed induced subcontexts)
3. Class N 局部性假设强度 — 是否引入 `N_loc` 子类排除全图归一化
4. Theorem 2 obstruction 对架构类 vs 对实现 family
5. DDI 里 meeting-node 的 typed status (derived object / hyperedge / query-specific artifact?)
6. Empirical `Δ̂^fresh` 的可执行 protocol (fresh extension / old-query witness / new-query two-completion witness 怎么采样)
7. 正文是否显式用 nominal terminology

Codex 建议下一轮: (1) 把 Theorem 1B 的 canonical extension operator 写成可执行定义 + (2) 把 meeting-node 单独 typed 成 query-derived structural witness 然后把 C3/support reduction 接上。

---

## Round 4 总结 / 决定

**收获**:
- 完整骨架成形，§1-§9 全部 typed
- §3 重要修正: **EmerGNN-as-published 更接近 Class N，不是 Class A** — v1 C2 必须改写
- §8 v1 vs v2 关系定调: v1 作为 statistical/architectural corollary 保留，不是被推翻
- 理论 commit 全部到位: support functorial 定义 + 类级别 theorem + inference-time Δ^fresh quantitative supplement

**未尽事项**:
- Theorem 1B canonical extension operator 显式化 (Round 5 可补)
- Meeting-node typed status (Round 5 必补，否则 §6 worked example 撑不起)
- Empirical protocol (后续工程阶段)

---

## Round 5 — 接受 EmerGNN ∈ N，另开角度打

**Strategy thesis**: v2 Class N 是 *必要条件* (判 cold-start admissibility)。Class N 内部仍可区分 functor 形式 sub-classes，flow-style N 和 meeting-node-style N 即便都 name-free structural，在 cold-start DDI mediator-semantic targets 上有结构性区别。

### R5-1 — Class N 的 sub-classification (codex commit)

**统一三段式程序**:
```
F_S(c) = Readout(Prop_S(Seed_S(c), Sup_S(c)), c)
```

按 query 如何选 support / 信息怎么汇聚 / 输出从哪读，4 类:

1. **`N_path`** (source-conditioned path-flow with endpoint readout): `Sup_S(u,v) = PathSubgraph_L(S; u, v)`，`Seed` 端点，`Prop` 沿有向路径，`Readout` 端点。**EmerGNN, NBFNet inductive 属此**。
2. **`N_meet`** (query-derived witness retrieval with witness readout): `Sup_S(u,v) = Wit_S(u, v)`，witness 是共享 mediator subgraph。**我们方法属此**。
3. **`N_local`** (radius-local): `Sup_S(c) = B_r(c; S)`，architecture radius 决定。粗类，可作辅助标签。
4. **`N_global`** (full-context pooling): 冷启动不主战。

**主类: `N_path` vs `N_meet` 不可比 as raw architecture forms**:
```
N_path ⊄ N_meet, N_meet ⊄ N_path  (as architecture classes)
```
但在 `T_med` (mediator-instance-sensitive target family) 上:
```
Realize(N_path, T_med) ⊊ Realize(N_meet, T_med)
```
**严格 containment 只在 target-restricted realize 上 hold**——这是更稳的 separation 表述。

### R5-2 — Functor-level separation axis (两者协同, primary = (a))

**(a) Path-signature collapse (主轴, expressivity)**: `N_path` 的输出仅依赖 `L`-bounded relation-labeled path invariant `Sig_L(S; u, v)`。任何依赖 witness-instance semantics 但不由 `Sig_L` 决定的 target 对 `N_path` 不可表达。

**(b) Support locality (辅轴, stability)**: `N_path` 的 support 随 `L` 扩张到 `L`-hop path cone (architecture radius 决定)；`N_meet` 的 support 是 query-induced witness (query 决定，不随 architecture)。Fresh atom 落入 `L`-hop cone 但不改 witness ⇒ `N_path` 漂移 / `N_meet` 稳定。

**协同表述**:
> (a) separates adequacy; (b) separates stability. Together: admissibility (∈ N) 是必要不充分，witness-based functor 在 mediator-sensitive cold-start DDI regime 上 dominate path-flow functor。

若只能选一: (a)——representation theorem / impossibility 比 (b) 难被 heuristic 反驳 (reviewer 不能说"tune L or prune subgraph"绕过)。

### R5-3 — Meeting-node typed status (codex commit)

**Meeting-node = query-derived structural witness; NOT atom / NOT persistent entity / NOT prediction-space black-box**。

**Witness functor**:
```
Wit_S: Q(S) → SubCtx(S)
```

DDI 基本 typed definition:
```
Wit_S(u, v) = S ↾ (W_S(u, v) ∪ {u, v})
W_S(u, v) = {x ∈ S_A : Ψ_S(u, x) ∧ Ψ_S(v, x)}
```
`Ψ_S` 是 task-chosen witness predicate (shared 1-hop neighbor / shared typed mediator / shared relation-constrained connector / meeting-path core)。

更一般 (relation-aware):
```
W_S^τ(u, v) = {x : ∃ typed connections u ⇝ x, v ⇝ x satisfying τ}
```

**Meeting-node 不是新 atom**，是 `Wit_S(u, v)` 的 query-specific summary object。模型在 witness subcontext 上 readout:
```
F_S(u, v) = ρ_θ(u, v, Agg_θ(Wit_S(u, v)))
```

**Witness naturality**:
```
Wit_T(i_* c) ≅ i_* Wit_S(c) ∪ NewWit_{T/S}(i_* c)
```
Fresh extension 不改变 witness predicate truth set ⇒ `Wit_T(i_* c) ≅ i_* Wit_S(c)`。

**Support 接法 (重要)**: **不**写 `supp(F_S(c)) = Wit_S(c)` 这种恒等 (太强)。写成 **support certificate / upper bound**:
```
supp_F(S, c) ⊆ Wit̄_S(c) = S ↾ (atoms(c) ∪ W_S(c))
```

### R5-4 — Theorem 5 candidate (在 v2 下打 EmerGNN-as-N 的新定理)

定义 `N_path^L`:
```
F_S(u, v) = ρ(⊕_{p ∈ P_{≤L}(S; u, v)} η(lab(p), λ_S|_p))
```
- `lab(p)` = relation/type label sequence
- `λ_S|_p` = 路径上节点属性的局部取值
- `η` 路径 encoder，`⊕` commutative path aggregator，`ρ` endpoint readout

更贴 EmerGNN: 要求 `η` 由 source-conditioned message passing 沿 path 递推，endpoint readout。

定义 `T_med` (mediator-instance-sensitive targets): `f* ∈ T_med` 若 ∃ queries `(u, v)`, contexts `S, T` 使:
1. `Sig_L(S; u, v) = Sig_L(T; u, v)` (same path signature)
2. `Wit_S(u, v) ≇ Wit_T(u, v)` (different witnesses)
3. `f*(S; u, v) ≠ f*(T; u, v)` (different targets)
4. 差异只来自 mediator-instance semantics

**Theorem 5 (Functor-form separation within Class N)**: Fix `L`. Assume `f* ∈ T_med`.

(i) 对每个 `F ∈ N_path^L` (output factors through `Sig_L`): `F ≠ f*` on the family. `f*` not realizable by `N_path^L`.
(ii) ∃ `F ∈ N_meet` such that `F = f*` on the family, provided witness predicate exposes relevant mediator semantics + witness readout class is universal on finite witness subcontexts.
(iii) 若 fresh extensions 改 `L`-path cone but 不改 witness: ∃ old queries / fresh inclusions 使
  - generic `F ∈ N_path^L`: `δ^old > 0`
  - witness-stable `F ∈ N_meet`: `δ^old = 0` (whenever `Wit_T(j_* c) ≅ j_* Wit_S(c)`)

**Proof idea**:
- Part (i): factorization lemma — `F_S(u, v) = Φ(Sig_L(S; u, v))`。Same `Sig_L` ⇒ same output；but `f*` differs. 矛盾。
- Part (ii): 构造 witness predicate 使 mediator instance ∈ `Wit_S(u, v)`，witness readout 实现 `f*` (universal on finite witness isomorphism classes)。
- Part (iii): fresh extension 加 unrelated mediator-like branch in path cone — path multiset changed ⇒ `N_path` 漂；witness unchanged ⇒ `N_meet` 稳 (Theorem 1A)。

**3 个 break + fallback**:
1. `Sig_L` 太粗 (EmerGNN attention 可能 distinguish more) → 升级为 `FlowSig_L` (architecture-induced complete commutative path-cone summary)
2. Universal strict positivity 太强 → 写"there exists fresh extension witness"
3. mediator-instance semantics 含糊 → "target depends on witness subcontext isomorphism class, not endpoint-conditioned path signature"

**Theorem 5 核心一句**: "Any endpoint-readout path-flow functor that factors through an architecture-induced path-cone invariant cannot realize targets that depend on witness-instance semantics beyond that invariant."

**关键: bite EmerGNN-as-published 的 computation form，不强行说它含 ID**。

### R5-5 — §6 改写结构

- **§6.1** EmerGNN ∈ N (admissible under v2，诚实)
- **§6.2** EmerGNN ∈ N_path (定义 Sup/Prop/Readout + path invariant)
- **§6.2.5** Caveat: N_path 普遍并非 inferior；本文 claim 只限 cold-start DDI mediator-semantic targets (防 reviewer 说普遍贬低 path models)
- **§6.3** Meeting-node + PubMedBERT ∈ N_meet (放 `Wit_S(u,v)` typed definition)
- **§6.4** Theorem 5 instantiation
- **§6.5** Empirical `Δ̂^fresh`: 设计"保持 witness 不变但扰动 path cone"的 fresh-atom 注入实验

### Codex 提的 8 个 fact-check items (写论文前必须核)

1. EmerGNN 初始 drug features 真的只 fingerprint，无 hidden ID table
2. attention / message function 是否真可 factor through commutative path-cone summary (若有 path-order recurrence / bidirectional coupling / non-commutative pooling → 需升 `FlowSig_L`)
3. subgraph extraction 是 "all paths up to L" 还是 sample/pruned (影响 `N_path^L` support 定义)
4. Readout 真 endpoint 还是 pair/subgraph (若后者 → 放宽 "endpoint readout" 到 "path-cone summary readout")
5. Inverse edges / bidirectional propagation 是否改 invariant class
6. Path-level attention normalization 对 irrelevant path mass 反应 (影响 §6.5 empirical 设计)
7. BioPathNet / 相近 models — 做 `N_path` 成一个类 vs 单打 EmerGNN
8. **我们自己 meeting-node implementation 真的 witness-local，不是暗中做大范围 path aggregation**

### Codex 提议下一轮 (Round 6)

1. 把 `N_path^L` 的 **architecture-induced invariant** `FlowSig_L` 定义精确
2. 把 DDI 的 **mediator-instance-sensitive target family** 写成两个最小反例 context `S, T`，让 Theorem 5 从抽象候选变成可验证的分离例子

---

## 当前状态总结

**理论框架已经搭出来 5 轮，结构完备**:
- §1-§9 骨架 (Round 4)
- 在 v2 下打 EmerGNN-as-published 的新角度 + Theorem 5 (Round 5)
- Meeting-node typed status (Round 5)

**核心 thesis (v2 final 表述)**:
> Cold-start 不是 distribution shift，是 *fresh-identity extension*。Nominal semantics 给 cold-start 一个本体而非统计的 characterization：admissibility = Class N (name-free structural)。Class N 内部 functor form 仍有 sub-class 区别；在 cold-start DDI mediator-semantic targets 上，witness-retrieval functors (`N_meet`) 严格 dominate source-conditioned path-flow functors with endpoint readout (`N_path`)。

**下一步选择**:
- **(A)** Round 6: 把 `FlowSig_L` 写死 + 给两个最小反例 context 让 Theorem 5 可验证
- **(B)** Fact-check 8 items 先做 (尤其 #1-#4, #8) — 防止论文写一半被 reviewer 抓
- **(C)** 暂停理论，进 paper-prose drafting
- **(D)** 别的

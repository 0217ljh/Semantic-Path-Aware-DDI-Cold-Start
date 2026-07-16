# KG–Molecule Redundancy Framework — Contribution 3 Spec (2026-05-30)

**Purpose**. 这个 framework 是 paper **Contribution 3（多模态边界融合）** 的精确化版本. 不要跟 C1 (PMP pooling) 和 C2 (LLM-KG 增强) 混在一起.

**三个 contribution 的切分（Option B, codex-reviewed 2026-05-30）**.

经 codex review，原"C1 含 LLM-as-encoder helper"切分有 partially-entangled 问题（i4_typed_sets.json 同一份 output 同时含 taxonomy 和 incidence，无法物理拆分）。改用 **Option B**——C1 纯架构不碰 LLM，所有 LLM-derived 贡献统一在 C2.

| Contribution | 是什么 | 严格不是什么 | 产物 / interface |
|---|---|---|---|
| **C1 PMP** | cold-start invariant scoring primitive。Typed common neighbor pooling + endpoint-stripping invariant + Layer 2 token-semantic attention pool。**Agnostic to mediator graph source** | 不碰 LLM，不构造 mediator graph，不做 routing。**C1 不区分 KG 来源是 raw / enriched** | 一个 score function，consume 任意 (drug → typed token) bipartite graph |
| **C2 Mechanism-aware KG enrichment** | LLM-distilled mechanism mediator graph 构造 pipeline。**包括所有 LLM 贡献**——new role-specific edges、finer relation types (taxonomy)、optional mediator text embeddings (LLM-as-encoder)。Prompt 设计 + 双层 sanitization + leakage audit | 不是 PMP 的内部一部分，是独立的 mediator graph 构造方法学 | 一个 enriched KG (drug → typed mechanism token graph)，任何 KG-consuming 方法 (EmerGNN / SEAL / NCN / PMP) 都可 consume |
| **C3 多模态边界融合（本文档主题）** | KMRR routing + redundancy framework (Indicator 1-4) + case taxonomy + lower bound theorem + mol-bridge 机制 | 不是 mol 信号进 score function（永远不进），是 architectural boundary handling | 一个 routing module + 一组 analysis 工具 |

**关键 framing 调整**（避免 codex review 指出的攻击面）.
- 不再 claim "C1 / C2 完全正交 (orthogonal)"
- 改为 **"modular at the interface, not causally independent"**——C1 consumes mediator graph，C2 constructs one such graph。两者通过 **mediator graph interface** 解耦，但同一份 LLM JSON output 在 C2 内部生产 (taxonomy + incidence + encoding 一体)，不强行拆给 C1 / C2
- LLM 在 paper 里只属于 C2，paper 不写 "C1 用 LLM" 这种表述

**为什么这样切分（独立性 evidence）**.
- C1 没有 C2/C3 也成立——PMP 在 raw KG 上 work（Stage 2a expected ≥ 0.78，待 verify）
- C2 没有 C1/C3 也成立——enriched KG 给任何 GNN baseline (EmerGNN / SEAL / NCN) 都可以用（Ablation 10 必跑，verify standalone utility）
- C3 只在 C1 + C2 之上做 boundary 修补——当 PMP-over-enriched-KG 仍然抓不到信号时（Category D 真分子驱动 KG 缺失），C3 fire 做 mol-bridged 修补

**本文档的覆盖范围（C3 spec）**, **dual role**.
1. **Analysis framework**（paper Section 4）. 量化 KG 和 molecular 信号在 cold-start DDI 上的 redundancy，给 40+ case studies 做 mechanistic attribution，证明 linear-additive multimodal fusion 在 cold-start 上有 architectural lower bound = 0
2. **Architectural module**（paper Section 3.4，PMP 之外的独立模块）. **KG–Mol Redundancy Router (KMRR)** + mol-bridge——在 inference 时检测 (a, b) 的 KG-coverage 和 Mol-complementarity，决定 mol-bridge 何时 fire、fire 多强，确保**真分子驱动且 KG 缺失**的 DDI 能被正确识别 + 处理

**Anchored facts (verified, do not re-question)**.
- HDN-DDI binary S2 on 800-drug = 0.6201（`Code/baseline/hdn_ddi/_results/2026-05-18__binary_cls__seed42.md`）
- EmerGNN binary S2 on 800-drug = 0.7458
- MNAH Stage 1 = 0.7670, Stage 2 PubMedBERT = 0.7722, v2i4 = 0.7804
- BRICS oracle additive ensemble gain = 0.000, b = 0.0, niche_frac = 0.428（`Code/runs/_gate_molecular/gate_analysis.json`）
- **TIGER 和 MKG-FENN 多模态 (KG + mol) 在 cold-start 上表现比 KG-only baseline 差**（user-confirmed verified fact, 2026-05-30）
- merged KG = 178,029 nodes from DrugBank + Hetionet + PrimeKG（`Code/data/KG/_merged_kg/`）

---

## 1. Framework Overview

```
            ┌─── Indicator 1. KG-coverage class ────┐
            │     (M_kg size + type entropy)         │
            │                                        │
(a, b) ─────┼─── Indicator 2. Mol-coverage class ───┼─── KMRR router ──→ routing decision
            │     (BRICS / Murcko / SAR alerts)     │     (α_bridge,
            │                                        │      mol_active,
            │                                        │      attribution log)
            └─── Indicator 3. Per-pair agreement ───┘            │
                  (offline p_kg / p_mol pairwise)                ▼
                                                        Pair-Mediator Pooling
                                                            (PMP forward)
                                                                 │
            Indicator 4. Information-theoretic                   ▼
            redundancy (corpus-level)                  combined score(a, b)
```

四个 indicator 分两组用途.
- **Indicator 1, 2** = 实时计算，**进 KMRR module 作 routing 输入**（model component 角色）
- **Indicator 3, 4** = offline corpus-level 分析，**进 paper Section 4 作 attribution evidence**（analysis 角色）

---

## 1.5 Representation Layer 1 vs Layer 2（核心区分，避免误解 v2i4 为终态）

**为什么这一节要单独写**. 项目 verified 数据上 v2i4 13-d explicit pair feature (0.7804) 是当前主线 best single-seed. 但这个数字是 **Layer 1 的 lossy scalar reduction**, 不是 PMP 的目标终态. 下个 agent 接手时必须明确这一点, 不能把 "scalar count" 当作 work 的 root cause 复制到 PMP.

### Layer 1（已 verified work，但 lossy）. Typed Discrete Token + Set Intersection

**Root cause of work 在这一层**.
- 底层 representation = typed token sets（KG node-by-type 或 LLM-distilled field-by-type）
- Pair operation = typed set intersection / cross
- 是 cold-start work 的 fundamental 原因（drug-identity-invariant, strong inductive bias, pair-level explicit）

**实现到 scalar 是 lossy choice**. v2i4 把 set intersection 的结果 reduce 成 scalar count `|a.cyp_substrate ∩ b.cyp_substrate|`, 把 13 维 hand-engineered 标量喂给 mlp. 丢失了.
- Token identity（CYP3A4 vs CYP2D6 都是 count=1）
- Token semantic（hub enzyme vs niche enzyme 没区分）
- 维度可扩展性（13 维 hand-pick, 其他 cross combination 没考虑, e.g. `cyp_substrate × pd_effect` cross 跳过了）
- LLM 蒸馏出的 rich 语义没用上

### Layer 2（**PMP C1 的完整实现，paper main contribution**）. Typed Token Semantic Embedding + Learned Attention Pool

**完整版（C1 视角，token embedding 来源由 C2 或 KG 提供，C1 不关心来源）**.
```
M(a, b) = typed_set_intersection(a, b)                 # Layer 1 set operation 不变
φ(m) = [ h_m,                                          # mediator embedding (KG-learned or C2-provided, 看 mediator graph 来源)
         type_embed(type(m)),                          # typed embedding
         rel_embed(a → m), rel_embed(m → b) ]          # incident edge type embedding
z_{a,b} = AttnPool({ φ(m) : m ∈ M(a, b) })            # learned attention pool
score = MLP(z_{a,b})
```

**关键 framing 调整**. Layer 2 是 **C1 (PMP architecture) 内部**的 representation 选择, 跟 LLM 解耦. `h_m` 是 mediator 的 embedding——来源由 mediator graph 决定.
- 如果 mediator graph 是 raw KG → `h_m` 是 KG-learned node embedding (PMP-C1 only)
- 如果 mediator graph 是 LLM-enriched (C2 supplied) → `h_m` 可以是 KG-learned 或 C2 提供的 LLM-derived embedding，**这部分属于 C2 的产物**，不是 C1 的设计选择

Layer 2 跟 Layer 1a (scalar count) 的差距.
- Token identity（CYP3A4 跟 CYP2D6 拿不同 embedding）
- 维度自动可扩展（attention 自学跨 type cross, 不需要 hand-pick 13 维）
- 任意 mediator embedding source 可以喂进来——这是 C1 / C2 interface 的体现

### 两层关系图

```
Layer 0 (dense without typed structure)     fail  [v2 dense xattn / v2llm BERT-CLS / v2res shared-residual]
   ↓ add typed discrete structure
Layer 1a (typed discrete + scalar count)   work, lossy  [MNAH 22-d 0.7670, v2i4 13-d 0.7804]
   ↓ replace scalar with token-semantic embedding pool
Layer 2 (typed token embedding + AttnPool)  PMP target [expected ≥ 0.78 with C1, ≥ 0.80 with C1+C2+C3]
```

**Critical**. PMP 的 work 是要在 Layer 2 实现 token-semantic pool, **不是抄 v2i4 的 13 维 hand-engineered scalar**. v2i4 13-d 在 paper 里的角色是 **Layer 1a degenerate baseline**, 验证 typed set intersection 这一脉 work, 但不是终态.

### 对 KMRR module 的影响

KMRR 的 routing 决策（Section 4）跟 representation layer 解耦. KMRR 输出 `α_bridge` 和 `mol_active` 这两个 routing signal, **下游 z_kg / z_bridge 的具体 pooling 实现是 Layer 2**.

具体在 Section 4.4 PMP forward 里, `φ(m)` 必须是 Layer 2 形态（token embedding + type + rel）, 不是 Layer 1a 的 scalar count.

---

## 2. Indicator 1. KG-coverage class（实时，per-pair）

### 2.1 输入

merged KG = `Code/data/KG/_merged_kg/{nodes,edges}__drugbank_hetionet_primekg__mask1.parquet`（KG-only, DDI edges masked，防止 leakage）。

### 2.2 计算

对每对 (a, b)，先算共邻集合.

```
N_kg+llm(a) = 1-hop ∪ 2-hop non-drug neighbors of a (with LLM-distilled edge types from C2)
M_kg(a, b)  = N_kg+llm(a) ∩ N_kg+llm(b)
```

提取两个 scalar 指标.

```
size_score(a, b) = |M_kg(a, b)|
type_entropy(a, b) = − Σ_t p_t · log p_t   where p_t = #mediators of type t / |M_kg|
```

11 个 KG node type 来自 `precompute_meet_features.py:50-62`（protein_gene, pathway, side_effect, disease, anatomy, compound, biological_process, molecular_function, cellular_component, pharmacologic_class, exposure）。C2 LLM 扩展后增加 10 类（cyp_substrate, cyp_inhibitor, cyp_inducer, transporter_substrate, transporter_inhibitor, therapeutic_class, primary_targets, pd_effects, toxicity_mechanisms, clearance）。共 21 类。

### 2.3 分类

| KG-coverage class | 条件 | 解读 |
|---|---|---|
| **KG-rich** | size ≥ 5 AND type_entropy ≥ 1.5 | mediator 充分且多样，KG 信号 dominant |
| **KG-sparse** | size < 3 | mediator 稀疏，需要 mol-bridge fallback |
| **KG-trivial** | size ≥ 5 AND type_entropy < 0.8 | mediator 多但 hub-dominated（e.g. 全是 CYP3A4 这种几乎所有药都连） |
| **KG-moderate** | 其他情况 | 中间状态 |

threshold（5, 1.5, 3, 0.8）应在 train set 上 calibrate，使 KG-rich + KG-sparse 各占合理比例（目标 ~40% / 20% / 10% / 30%）。

### 2.4 实现路径

`Code/my_code/models/screen_s2_v4_pmp/kg_coverage.py`. 输入 batch (a, b) tensor，输出 batch shape `[B, 4]` one-hot class indicator。**deterministic, no learnable parameters**, 可 pre-compute 缓存到 `Code/data/_cache/kg_coverage_class.parquet`。

---

## 3. Indicator 2. Mol-coverage class（实时，per-pair）

### 3.1 输入

SMILES = `Code/data/coldddi_legacy/800drug/drug_smiles__seed42.csv`（1994 行 drug → SMILES）。

### 3.2 三类 fragment extraction

每个药 d 算三组 fragment.

| Fragment family | 工具 | 含义 |
|---|---|---|
| **F_BRICS(d)** | RDKit `BRICS.BRICSDecompose` | retrosynthetic 切片 |
| **F_Murcko(d)** | RDKit `MurckoScaffold` | 骨架 |
| **F_FunGroup(d)** | RDKit `FunctionalGroups.GetFGRules` (Ertl 2017 库) | 官能团（amine, lactone, sulfonamide 等） |
| **F_Alert(d)** | Brenk + PAINS + ToxAlerts SMARTS | 结构警示（reactive groups） |

具体 alert 库.
- Brenk filter（105 SMARTS, 反应性 / 毒理）
- PAINS filter（480 SMARTS, pan-assay interference）
- ToxAlerts (Sushko 2012)（structural alerts for toxicity）
- 自建 cross-reactivity pairs（Michael acceptor ↔ thiol，aldehyde ↔ primary amine，etc.）

### 3.3 Pair-level shared / cross-reactive

```
F_shared(a, b)        = F_BRICS(a) ∩ F_BRICS(b) ∪ F_Murcko(a) ∩ F_Murcko(b)
F_shared_groups(a, b) = F_FunGroup(a) ∩ F_FunGroup(b)
F_cross_react(a, b)   = pairs (g_a ∈ F_Alert(a), g_b ∈ F_Alert(b)) where 
                         g_a + g_b is in cross-reactivity table
```

### 3.4 KG-mol mechanistic mapping table

**关键**. 维护一个 lookup 表，把 fragment 映射到对应的 KG mediator type，用来判断 mol 信号是否冗余于 KG.

```
fragment_to_kg_mediator = {
    "macrolide_ring":           ["cyp_inhibitor"],      # erythromycin, clarithromycin 模板
    "statin_lactone":           ["cyp_substrate"],      # CYP3A4 substrate 标记
    "azole":                    ["cyp_inhibitor"],      # ketoconazole, fluconazole
    "rifamycin_skeleton":       ["cyp_inducer"],
    "SSRI_scaffold":            ["primary_targets"],    # SERT
    "benzodiazepine_core":      ["primary_targets"],    # GABA-A
    ...
}
fragment_kg_explicit = fragment 出现 in fragment_to_kg_mediator AND 
                       对应的 KG mediator 在 M_kg(a, b) 里
```

这个表是 paper Section 4 的 case study 副产物，每个 case 都给一个 row。初版 ~50 row，覆盖 70% common DDI fragment。

### 3.5 分类

| Mol-coverage class | 条件 | 解读 |
|---|---|---|
| **Mol-redundant** | F_shared 主要 fragment 都在 fragment_kg_explicit 里 | mol 信号 KG 已捕获 |
| **Mol-complementary** | F_cross_react 非空 OR F_shared 有 fragment 不在 fragment_kg_explicit | mol 有 KG 没有的信号 |
| **Mol-uninformative** | F_shared 和 F_cross_react 都很弱 | mol 没贡献 |

### 3.6 实现路径

`Code/my_code/models/screen_s2_v4_pmp/mol_coverage.py`. 输入 batch (a, b)，输出 `[B, 3]` one-hot + `[B, K]` fragment attribution vector。**deterministic**, 可 pre-compute 缓存。

---

## 4. KMRR — KG–Mol Redundancy Router（PMP 核心 module）

### 4.1 设计目标

输入 (a, b)，输出.
- `α_bridge(a, b) ∈ [0, 1]`. mol-bridge fusion weight
- `mol_active(a, b) ∈ {0, 1}`. mol-bridge 是否 fire
- `attribution(a, b)`. 解释 routing decision 的 log（paper Section 4 attribution 用）

### 4.2 决策逻辑

按 Indicator 1 + Indicator 2 组合分类，硬决策表.

| KG class | Mol class | α_bridge | mol_active | 解读 |
|---|---|---|---|---|
| KG-rich | Mol-redundant | 0.0 | 0 | KG 覆盖，mol 冗余，**no harm 保证** |
| KG-rich | Mol-complementary | 0.1–0.3 | 1 | KG 主导，mol 微量补充 |
| KG-rich | Mol-uninformative | 0.0 | 0 | 纯 KG |
| KG-moderate | Mol-redundant | 0.0 | 0 | 同 KG-rich + Mol-redundant |
| KG-moderate | Mol-complementary | 0.3–0.5 | 1 | KG-mol 平衡贡献 |
| KG-moderate | Mol-uninformative | 0.0 | 0 | 纯 KG |
| **KG-sparse** | **Mol-redundant** | 0.4 | 1 | **mol fallback**（KG 不够，mol 补 mediator candidate via NN_mol） |
| **KG-sparse** | **Mol-complementary** | **0.7** | **1** | **mol 必须 fire**（真分子驱动且 KG 缺失，**Category D 的 case，C3 核心目标**） |
| KG-sparse | Mol-uninformative | 0.5 | 1 | mol 补 candidate set（fallback） |
| KG-trivial | * | 0.3 | 1 | hub mediator dominated, mol 补强 specificity |

α_bridge 在 architecture 里**最终是 learned**，硬决策表只给 init bias.

```
α_bridge(a, b) = sigmoid( w_class · class_onehot(a, b) + b_class )
其中 w_class 用上表 init，但允许 fine-tune
```

### 4.3 Module 输入输出 contract

```python
class KMRR(nn.Module):
    """KG-Mol Redundancy Router. Pair-conditional routing for PMP mol-bridge."""

    def forward(self, pair_ids: tuple[str, str]) -> dict:
        kg_cov = self.kg_coverage(pair_ids)        # [B, 4] one-hot
        mol_cov = self.mol_coverage(pair_ids)      # [B, 3] one-hot
        class_combined = kg_cov.unsqueeze(2) * mol_cov.unsqueeze(1)  # [B, 4, 3]
        class_flat = class_combined.flatten(1)     # [B, 12]
        alpha_logit = self.alpha_head(class_flat)
        alpha_bridge = torch.sigmoid(alpha_logit)
        mol_active = (alpha_bridge > self.fire_thresh).float()
        return {
            "alpha_bridge": alpha_bridge,
            "mol_active": mol_active,
            "kg_class": kg_cov,
            "mol_class": mol_cov,
            "attribution": self._attribution_log(...),
        }
```

### 4.4 集成进 PMP forward

**关键 framing (Option B)**. C1 PMP forward 中 `φ(m)` 是 mediator embedding，**来源由 mediator graph 决定**——C1 自己不关心。如果 mediator 来自 C2 enriched KG，`h_m` 可包含 C2 提供的 LLM-derived embedding（这是 C2 的产物，不是 C1 内部 LLM 调用）。

```
# C1 PMP Layer 2 forward (architecture-only, agnostic to mediator graph source)
φ(m) = MLP_concat( [
    h_m,                             # mediator embedding (KG-learned or C2-provided)
    type_embed(type(m)),             # typed embedding
    rel_embed(a → m), rel_embed(m → b),
] )

# Pair-Mediator Pooling, primary side
M_primary(a, b) = N(a) ∩ N(b)                              # typed set intersection on whichever mediator graph (raw KG OR C2 enriched)
z_primary(a, b) = AttnPool({ φ(m) : m ∈ M_primary(a, b) })

# Mol-bridge side (Contribution 3, only fires when KMRR says so)
router_out = KMRR(a, b)
if router_out["mol_active"] > 0:
    NN_mol(a) = top-k mol nearest neighbors of a in G1 (by ECFP Tanimoto OR learned m_u)
    NN_mol(b) = top-k mol nearest neighbors of b in G1
    M_bridge(a, b) = (⋃ N(a') for a' in NN_mol(a))
                  ∩ (⋃ N(b') for b' in NN_mol(b))
                  \ M_primary(a, b)             # only add new mediators not already in
    z_bridge(a, b)  = AttnPool({ φ(m) : m ∈ M_bridge(a, b) })   # 同样 Layer 2
else:
    z_bridge(a, b)  = 0  # no-op

# Combined score
α = router_out["alpha_bridge"]
z_final = z_primary + α · z_bridge
score(a, b) = MLP(z_final)
```

**三个 contribution 在 forward 里的体现**.
- **C1 (architecture)**. 整个 forward 框架——typed set intersection + Layer 2 attention pool + endpoint stripping
- **C2 (enrichment)**. `M_primary` 和 `M_bridge` 走的 mediator graph N(·) 是 raw KG (no C2) 还是 LLM-enriched KG (C2 applied)；`h_m` 是 KG-learned 还是 LLM-derived 也由 C2 决定
- **C3 (boundary fusion)**. KMRR 路由 + mol-bridge expansion + α 加权

**Architectural invariants (永远 hold)**.
1. **C1 自己不调 LLM**（C1 invariant，Option B）. C1 forward 代码不出现任何 LLM call. LLM 仅出现在 C2 的 KG 构造 pipeline 里产生 enriched KG，C1 forward 只 consume 这个 KG
2. **mol embedding 永远不进 score function**（C3 invariant）. mol 只用在 KMRR 内部算 mol_cov + NN_mol selection
3. **KG-rich + Mol-redundant pair 上 `α = 0`**（C3 invariant）. PMP 退化到 pure C1 + C2 (no harm 保证)
4. **KG-sparse + Mol-complementary pair 上 `α ≥ 0.7`**（C3 invariant）. **Category D Michael+thiol 类必 fire**
5. **所有 routing 决策 deterministic + interpretable**（C3 invariant）. Attribution log 是 paper-grade evidence
6. **`φ(m)` 必须保留 token identity**（C1 Layer 2 invariant）. 禁止退化为 scalar count over typed set (那是 v2i4 Layer 1a baseline). 单元测试. 对每个 mediator type, attention weight 应在 batch 内体现 token-specific variation
7. **mediator graph interface 是 C1 / C2 唯一耦合点**（Option B invariant）. C1 不假设 mediator graph 来源, C2 提供 (drug -> typed token) bipartite graph + (optional) per-mediator embedding

### 4.5 Training

KMRR `alpha_head` 用以下 loss 训练.

```
L_main = BCE(score(a, b), y(a, b))      # 主 loss
L_router_calibration =                   # 辅助 loss, 防止 α 漂离硬决策表
    || alpha_bridge - alpha_table(class_combined) ||²
total_loss = L_main + λ_router · L_router_calibration
```

`λ_router = 0.1` init。`alpha_table` 用 4.2 节硬决策表的 mid-range 值。

---

## 5. Case Study Taxonomy（paper Section 4.3, 40+ pair）

5 类，每类 6–10 个 verified DDI pair（从 DrugBank description + literature 取，所有 pair 必须存在于我们 800-drug + S2 test split 里）。

### Category A. CYP-mediated PK（~12 pair）

KG-explicit, Mol-redundant。Representative case.

| (a, b) | 机制 | KG attribution | Mol attribution | KMRR prediction |
|---|---|---|---|---|
| clarithromycin, simvastatin | CYP3A4 inhibition → statin exposure ↑ | `a -[inhibits]-> CYP3A4 <-[substrate]- b` | macrolide-lactone + statin lactone | KG-rich, Mol-redundant, α=0 |
| ketoconazole, midazolam | CYP3A4 inhibition | KG explicit | azole + benzodiazepine | KG-rich, Mol-redundant, α=0 |
| fluvoxamine, theophylline | CYP1A2 inhibition | KG explicit | — | KG-rich, Mol-redundant, α=0 |
| ...  (8+ more) | | | | |

### Category B. Transporter-mediated PK（~6 pair）

KG-explicit when P-gp / OATP / BCRP edge present。

| (a, b) | 机制 | KMRR prediction |
|---|---|---|
| verapamil, digoxin | P-gp inhibition | KG-rich, Mol-redundant, α=0 |
| rifampicin, dabigatran | P-gp induction | KG-rich, Mol-redundant, α=0 |
| ... | | |

### Category C. Target / Receptor sharing PD（~8 pair）

KG-explicit via shared target/protein。

| (a, b) | 机制 | KMRR |
|---|---|---|
| fluoxetine, sertraline | shared SERT, serotonin syndrome | KG-rich, Mol-complementary（SSRI scaffold partial overlap）, α=0.1–0.2 |
| amlodipine, nifedipine | shared L-type Ca channel, hypotension | KG-rich, Mol-complementary (DHP scaffold), α=0.1 |
| ... | | |

### Category D. Structural alert / direct reaction（~5 pair）★ Contribution 3 核心 evidence

**KG 没编码、mol 有信号的 pair**。这一类是 framework 必须正确处理的 anchor case。

| (a, b) | 机制 | KG attribution | Mol attribution | KMRR prediction |
|---|---|---|---|---|
| disulfiram, paracetamol metabolite | NAPQI (Michael acceptor) + thiol (disulfiram CSSC) | KG: 无显式 alert edge | F_alert: thiol + Michael acceptor cross-react | **KG-sparse, Mol-complementary, α=0.7** |
| acetaminophen, ethanol | NAPQI formation + GSH depletion | KG: indirect via CYP2E1 | F_alert: para-aminophenol activation | KG-moderate, Mol-complementary, α=0.5 |
| ... (3+ more真实 case, 从 hepatotoxicity / metabolic activation literature 找) | | | | |

**这一类是用户提到的"真正分子驱动且 KG 缺失的 DDI"**，KMRR 必须在这里 α ≥ 0.7 且 mol_active = 1，否则 framework 失败。

### Category E. PD additive via shared phenotype（~5 pair）

KG-explicit via side_effect / phenotype edges。

| (a, b) | 机制 | KMRR |
|---|---|---|
| amiodarone, ondansetron | both → QT prolongation (different ion channels) | KG-rich (shared phenotype), Mol-uninformative, α=0 |
| ... | | |

### Category F. Ambiguous / mixed mechanism（~4 pair）

多机制 overlap。展示 framework 对边缘 case 的处理。

---

## 6. Indicator 3. Per-pair agreement metrics（offline corpus-level）

对所有 S2 test pair 跑.
- `p_kg = PMP-C1+C2 score` (KG-only)
- `p_mol = HDN-DDI binary score` (verified 0.6201 anchor)
- `y = ground truth`

定义.
```
Agreement       = P(p_kg ≥ τ AND p_mol ≥ τ | y = 1)
KG-unique       = P(p_kg ≥ τ AND p_mol < τ | y = 1)
Mol-unique      = P(p_kg < τ AND p_mol ≥ τ | y = 1)
Conflict        = P(disagree)
Redundancy R    = P(both correct) / P(at least one correct)
Complementarity = P(mol-unique correct) / P(at least one correct)
```

**预期结果**（基于已知 verified 数字）.
- p_kg overall AUC = 0.78（PMP target）
- p_mol overall AUC = 0.62（HDN-DDI verified）
- R 预期 0.7+（绝大部分对 KG 都对）
- Complementarity 预期 0.05–0.15（mol unique 贡献是 niche 但非零）
- Mol-unique 那部分必须 majority 落在 Category D + 部分 Category F

---

## 7. Indicator 4. Information-theoretic redundancy（formal）

```
Redundancy = I(p_kg; y) + I(p_mol; y) − I(p_kg, p_mol; y)
```

paper 写 formal lower bound theorem.

### Theorem (cold-start linear additive fusion lower bound)

> 在 cold-start S2 setting 下，任何 linear additive fusion `f(a, b) = α · p_kg(a, b) + β · p_mol(a, b)` 的 expected ranking improvement over `max{p_kg, p_mol}` 上界由 `Complementarity ratio C` 决定. 当 `R → 1`（mol 完全冗余于 KG），任何 linear additive fusion gain = 0.

### Empirical witness

`Code/runs/_gate_molecular/gate_analysis.json` 的 oracle gain = 0.000, optimal b = 0 是这个上界的 empirical witness——即便 test-set tuned optimal blend 也 access 不到 mol 的 marginal value.

### Corollary

**Cold-start multimodal DDI fusion 必须是 conditional / non-additive 形式才有 ranking gain**。KMRR 是这个 corollary 的 architectural instantiation——pair-conditional gate 用 KG-coverage class + Mol-coverage class 联合决定 fusion mode.

### Verified empirical evidence supporting corollary

- TIGER (AAAI 2024) dual-channel concat MG + BKG → cold-start S2 表现差于 KG-only baselines（user-confirmed）
- MKG-FENN 多模态 fusion → cold-start S2 表现差于 KG-only baselines（user-confirmed）
- HDN-DDI molecular-only → cold-start S2 binary 0.6201（verified, 比 EmerGNN 0.7458 低 −12.57 pt）
- BRICS oracle additive gain = 0（verified）
- ✅ 多个独立 evidence 一致支持 "linear additive multimodal fusion 在 cold-start 上 fail"

---

## 8. Paper Section 结构

```
Section 3. Method
  3.1 EmerGNN backbone (anchor)
  3.2 Pair-Mediator Pooling (Contribution 1)
       - PMP architectural principle
       - typed mediator attention pooling
  3.3 LLM-Enriched Mediator Universe (Contribution 2)
       - LLM distillation + sanitization
       - new edge types in KG
  3.4 KG-Mol Redundancy Router and Mol-Bridge (Contribution 3)  ← KMRR 在这
       - Indicator 1, 2 (per-pair real-time)
       - KMRR routing decision
       - Mol-bridge mediator expansion
       - Architectural invariants (no harm + Category D fire)

Section 4. Information Redundancy Analysis
  4.1 Quantitative framework (4 indicators recap, formal definitions)
  4.2 Per-pair agreement metrics on DrugBank S2  (Indicator 3 verified)
       - R, Complementarity, Conflict rates
  4.3 Case study taxonomy (40 pairs, 6 categories) (verified attribution)
       - Category A-F breakdown
       - Mechanistic attribution table
  4.4 Lower bound theorem  (Indicator 4)
       - Statement + proof sketch
       - Oracle gain = 0 empirical witness
       - TIGER / MKG-FENN failure mode confirmation
  4.5 Architectural implications
       - Conditional non-additive fusion necessity
       - KMRR as instance

Section 5. Experiments
  5.1 Main results (PMP-C1, +C2, +C3) vs baselines (EmerGNN, TIGER, MKG-FENN, NCN, etc.)
  5.2 Per-category performance breakdown (Cat A vs B vs ... vs D)
       - Verify KMRR fires correctly on Cat D
  5.3 KMRR routing analysis (gate distribution, attribution)
  5.4 Ablations (see Layer ladder below)
```

### 8.5 Main Ablation Ladder（Layer 1a → Layer 2 progression）

paper Section 5.4 主 ablation table 必须按这条 ladder 走, 让 reviewer 一眼看到 "verified work 的 root cause 在哪一步" + "PMP 真正贡献的是哪一步".

| Stage | Representation | Pair operation | Pool / reduction | S2 AUROC | 状态 |
|---|---|---|---|---|---|
| Stage 0 | dense BERT embedding | element-wise interaction | MLP over Hadamard | 0.57 (mol_only) – 0.77 (combined) | ✗ inert |
| **Stage 1a (verified)** | **typed KG mediator (11 types)** | **typed set intersection** | **scalar count + mlp** | **0.7670 (MNAH)** | ✓ Layer 1a baseline |
| **Stage 1b (verified)** | **typed LLM token (10 fields)** | **typed set intersection + cross** | **hand-engineered 13-d count** | **0.7804 (v2i4)** | ✓ Layer 1a + C2 token universe |
| **Stage 2a (PMP target)** | **typed KG mediator embedding** | **typed set intersection** | **attention pool over `φ(m)`** | **expected ≥ 0.78** | PMP-C1 only (raw KG) |
| **Stage 2b (PMP target)** | **typed KG mediator embedding on C2 enriched KG** | **typed set intersection** | **attention pool over unified mediator graph** | **expected ≥ 0.79** | PMP-C1 + C2 |
| **Stage 2c (PMP target)** | **+ mol-bridged token expansion** | **+ conditional set expansion** | **same Layer 2 pool, KMRR gated** | **expected ≥ 0.80** | PMP-C1 + C2 + C3 |

**关键 narrative for paper Section 5.4**.

> Stage 1a 和 Stage 1b verify 了 typed discrete token + set intersection 是 cold-start work 的 root cause (Layer 1). MNAH (generic KG type) + v2i4 (LLM-distilled fine-grained type) 两个 instance 在不同 token universe 下都 work, 证明 root cause 不依赖具体 token source. 但两者都用 scalar count reduction, **丢失 token identity 和 semantic**. PMP-C1 (Stage 2a) 升级到 typed token embedding + learned attention pool, 解锁了 token-semantic discrimination 和自动 cross-type 组合, 在 raw KG 上 verify C1 architecture. Stage 2b 在 C2 enriched KG 上跑 PMP-C1, 量化 C2 enrichment contribution. Stage 2c 加 C3 KMRR + mol-bridge, 处理 KG-sparse + Mol-complementary boundary case.

### 8.6 完整 10-ablation 矩阵（codex review 2026-05-30 推荐, paper-grade defense）

仅 Main Ladder 不够. AAAI reviewer 会 push back "+1.34 pt 是哪一种因素贡献", "C2 是不是 vocabulary normalization", "v2i4 vs MNAH 是不是 density / overlap statistics 改变". 下面 10 个 ablation 是必跑硬底线. 全跑完才能投稿.

| # | 名称 | 设计 | 测什么 |
|---|---|---|---|
| **A1** | **仅原 KG 已有的机制类型化** (Existing-KG-only mechanism typing) | 仅对原 KG 上已存在的 underlying drug-target / 酶 / 转运体关联应用 LLM 提供的 fine-grained mechanism typing | 分离 "分类粒度细化" vs "新增 edge 又带 typing" |
| **A2** | **仅新增 edge 的机制图** (New-edge-only mechanism graph) | 只保留 LLM 抽出但原 KG 上不存在的 assertion, 丢掉跟 KG 已有 edge 重合的部分 | C2 真正"补全 KG 缺失"价值的隔离 |
| **A3** | **度数匹配对照** (Degree-matched control) | 构造对照 mediator graph, mediator degree 分布跟 v2i4 完全一致, 但用 generic KG token 或随机 type 标签填充 | 排除 gain 是来自 density / overlap statistics 而非 pharmacology semantics |
| **A4** | **类型标签全局打乱** (Type-label permutation) | 保留 drug-token incidence 不变, 但把 mechanism type 名字全局随机重排 | type label 的语义本身有没有作用 |
| **A5** | **类型内 incidence 打乱** (Incidence permutation within type) | 保留 type 频率分布 + 每个 mediator 总频率, 但在同一 type 内 shuffle drug-token 分配 | LLM 抽出的具体 pharmacology 事实重不重要 |
| **A6** | **Schema-only vs schema+instances** | 四阶梯 stage. (i) 纯 generic KG token; (ii) 原 KG 已有 entity + finer role labels only; (iii) LLM 新增 edges 不带 role labels; (iv) LLM 新增 edges + role labels | 分离 "分类粒度" 和 "新事实" 各自贡献 |
| **A7** | **LLM 来源泄漏审计** (LLM source leakage audit) | 验证 LLM extraction prompt + 源文本不含任何 DDI label / "interacts with" 类 DDI phrasing / DrugBank 对测试对 (a, b) 的 interaction description | 防止 LLM 蒸馏间接复制 DDI label 进入特征 |
| **A8** | **冷启动 split 卫生检查** (Cold-start split hygiene) | 验证 G2 unseen drug 没通过别名 / 商品名 / 复方 / 盐型 / 同义词 / DDI-derived annotation 泄漏到 G1 | strict cold-start 协议的基础前提 |
| **A9** | **跨 split 鲁棒性** (Cross-split robustness) | 多个 S2 split (seed42 / 43 / 44 / 独立划分) 上独立 verify v2i4 vs MNAH +1.34 pt gain | gain 不是单 split cherry-picked |
| **A10** | **C2 独立可用性** (Standalone C2 utility) | LLM-enriched KG 喂给非 PMP baseline (EmerGNN / SEAL / NCN / Neo-GNN) 跑一遍 | C2 是否能作为 standalone 可复用 KG enrichment——critical for C2 standalone contribution claim |

**Verdict 触发逻辑**.
- A1 + A2 + A6 联合判 C2 性质. 若 new-edge-only gain 主导且 Schema (iii) 涨点 → C2 真有 completeness 价值; 若仅 (ii) 涨且 (iii) 平 → C2 主要是 vocabulary refinement (弱 contribution)
- A3 判 surface confound. degree-matched 对照若 close 到 v2i4 → 大量 gain 来自 statistics 而非语义
- A4 + A5 判 type label / incidence 是否真有 signal
- A7 + A8 是 leakage 防御硬底线, 失败直接 invalidate 整个 paper
- A9 防 cherry-pick
- A10 判 C2 standalone status. 若仅 PMP 在 enriched KG 涨, C2 不能 standalone claim, 应降级成 PMP 的 sub-component

**reviewer 攻击面预防（更新版）**.
- Q. "v2i4 13-d hand-engineered 维度选得对吗？" → A. Stage 2 通过 learned attention 完全 generalize, 不依赖 hand-pick. v2i4 是 paper 的 Stage 1b baseline, 不是 SOTA claim
- Q. "+1.34 pt 是 C2 还是 statistics confound？" → A. A3 + A4 + A5 联合证据
- Q. "C2 是 just vocabulary engineering done by LLM?" → A. **承认 vocabulary 是 engineered**, 但 (i) A2/A6 量化 new-edge contribution, (ii) A10 验证 standalone reusability across baselines, (iii) paper claim 改成 "mechanism-role mediator graphs provide better inductive interface for S2 cold-start than generic KG neighborhoods, under controlled architecture, endpoint stripping, and leakage audits"
- Q. "C1 / C2 是否真的独立？" → A. **承认两者 modular at interface but not causally independent**. C1 consumes mediator graph (任意来源), C2 constructs one. A10 是 interface 独立性的硬证据 (C2 enriched KG 给非 PMP 方法也涨 → C2 不依赖 PMP)

### 8.7 Paper 概念图 spec（投稿前必画）

paper 投稿前必须有几张概念图, 视觉上清晰传达 contribution boundary. 必须好看且清楚.

#### Figure 1. Three-Contribution Architecture Overview（paper 主图）

**目标**. 一张图让 reviewer 一秒钟看出 C1 / C2 / C3 边界 + interface 关系.

**Layout 要点**.
- 左侧栏 = raw KG (DrugBank + Hetionet + PrimeKG icon) + per-drug SMILES (mol structure icon)
- 中部 = C2 LLM enrichment pipeline (LLM icon + 双层 sanitization filter + arrow into enriched KG)
- 中部 → 右上 = enriched KG (mediator graph 的 drug-token bipartite 视图)
- 右上 → 右中 = C1 PMP forward (mediator graph → typed set intersection → attention pool → score)
- 右下 = C3 KMRR routing + mol-bridge (KG-coverage + mol-coverage indicator → α gate → 控制是否走 mol-bridge)
- 三个 contribution 用三种颜色高亮 (e.g. 蓝色 = C1, 橙色 = C2, 绿色 = C3)
- 用 dashed border 圈出每个 contribution 的 scope, 让"边界"视觉上 explicit

**关键 design choice**. **C1 / C2 / C3 在图上不共享内部组件**, 但通过 arrow 显示数据流和 interface (mediator graph 是 C1 ← C2 的 interface, KMRR routing decision 是 C1 ← C3 的 interface)

#### Figure 2. Representation Layer Hierarchy（Section 1.5 配图）

**目标**. 解释为什么 Layer 2 比 Layer 1a 强, 为什么 v2i4 不是 PMP 终态.

**Layout 要点**.
- 三层堆叠 (Layer 0 / Layer 1a / Layer 2)
- 每层左侧示意图 + 右侧 verified S2 AUROC 数字
- Layer 0. 例子 v2 dense xattn / v2llm BERT-CLS, AUROC 0.57-0.77, ✗ inert (用灰色)
- Layer 1a. 例子 MNAH 22-d / v2i4 13-d, AUROC 0.7670 / 0.7804, ✓ work but lossy (用浅蓝)
- Layer 2. PMP target, expected ≥ 0.78 ~ 0.80, ✓ PMP main contribution (用深蓝高亮)
- 箭头标注从下到上的 "add typed discrete structure" → "replace scalar with token-semantic embedding pool" 两个跃迁
- 右侧示意 token universe (KG 类 vs LLM 类) 用 icon 区分, 强调 representation form 升级 ≠ token source 改变

#### Figure 3. Information Redundancy Lower Bound（Section 4 配图）

**目标**. 视觉化 lower bound theorem. 让 reviewer 看到 "linear additive fusion gain = 0" 不是 hand-wave 而是有 oracle + multiple empirical witness.

**Layout 要点**.
- X 轴 = redundancy ratio R
- Y 轴 = AUROC
- 三条曲线. (i) p_kg 单独 baseline; (ii) p_mol 单独; (iii) oracle additive fusion 上界
- 在 R → 1 处三条曲线 converge, 凸显 oracle gain = 0
- 标 verified data point. BRICS oracle (R 高, gain 0) / HDN-DDI / TIGER / MKG-FENN 在图上的位置
- 加 annotation "PMP + C3 (non-additive fusion) breaks this bound"

#### Figure 4. KMRR Routing Decision Map（Section 3.4 配图）

**目标**. 视觉化 KMRR 决策表 + Category D 的 fire 机制.

**Layout 要点**.
- 2D grid. X 轴 = KG-coverage class (KG-rich / moderate / sparse / trivial), Y 轴 = Mol-coverage class (Mol-redundant / complementary / uninformative)
- 每个 cell 标 α_bridge 值 + mol_active 0/1
- 颜色梯度. 浅色 = α 低 / 不 fire, 深色 = α 高 / 必 fire
- **Category D cell (KG-sparse × Mol-complementary) 单独高亮**, 强调 "真分子驱动且 KG 缺失" 这条 corner case
- 边缘标几个 case study pair 作示例 (clarithromycin+simvastatin → KG-rich+Mol-redundant; disulfiram+paracetamol → KG-sparse+Mol-complementary 等)

#### Figure 5. Case Study Taxonomy（Section 4.3 配图）

**目标**. 视觉化 5 类 case + mechanistic attribution.

**Layout 要点**.
- 5 个 panel 横排 (Category A-E), 每个 panel 显示.
  - 代表 pair 的 mechanism diagram (drug a → mediator → drug b)
  - KG attribution status (✓ explicit / ⚠ partial / ✗ missing)
  - Mol attribution status (同上)
  - 该 category 在 verified test pair 中的占比
- 颜色按 mechanism type (PK = 蓝, PD = 橙, structural alert = 红)

**Style 要求（所有 figure 共通）**.
- 用 TikZ / Inkscape 矢量图, 不要 PNG bitmap
- 字体 ≥ 8pt, 保证 single-column 缩放后可读
- 颜色 colorblind-friendly (e.g. ColorBrewer Set2 系列)
- 每张图带 clear caption + standalone (不依赖正文也能理解大致 message)

---

## 9. Implementation TODO List（按 priority）

### Priority 1（地基，必须先做）

| Task | 位置 | 依赖 |
|---|---|---|
| `kg_coverage.py` — Indicator 1 (deterministic) | `Code/my_code/models/screen_s2_v4_pmp/` | KG cache 已有 |
| `mol_coverage.py` — Indicator 2 (BRICS + Murcko + FunGroup + Alert) | 同上 | RDKit, Brenk/PAINS SMARTS lib（pip 可装） |
| `fragment_to_kg_mediator.json` — 初版 50 row 映射表 | `Code/data/_cache/` | 手工 + 从 case studies derive |
| `kmrr.py` — KMRR router module | 同上 | Indicator 1 + 2 |

### Priority 2（PMP 主架构）

| Task | 位置 |
|---|---|
| `pmp_trainer.py` — Contribution 1 主架构 (typed mediator attention pool) | `Code/my_code/models/screen_s2_v4_pmp/` |
| `pmp_c2_llm_enriched.py` — Contribution 2 enrich KG with LLM edges | 同上 |
| `pmp_c3_mol_bridge.py` — Contribution 3 integrate KMRR + mol bridge | 同上 |

### Priority 3（Analysis 框架）

| Task | 位置 |
|---|---|
| `compute_agreement_metrics.py` — Indicator 3 | `Code/scripts/` |
| `compute_information_redundancy.py` — Indicator 4 | `Code/scripts/` |
| `annotate_case_studies.py` — 40 pair manual annotation 工具 | `Code/scripts/` |
| `case_studies/` 目录 — 40 pair markdown attribution | `Notes/Case-Studies/` |

### Priority 4（Baseline 对照）

| Task | priority |
|---|---|
| TIGER baseline on our 800-drug S2 | already verified by user, 拿数字落盘到 `_results/` |
| MKG-FENN baseline on our 800-drug S2 | already verified by user, 同上 |
| NCN / MPLP / Neo-GNN / BUDDY S2 | task #11–#14 pending |

---

## 10. Architectural Invariants（必须保证, 测试时验证）

| Invariant | 测试 |
|---|---|
| KG-rich + Mol-redundant pair 上 α = 0 | random sample 100 such pair, assert all α < 0.05 |
| KG-sparse + Mol-complementary pair 上 α ≥ 0.6 | random sample 50 such pair, assert all α ≥ 0.6 |
| Category D 5 个 anchor pair 上 mol_active = 1 | unit test |
| mol embedding 从未出现在 score function 的 input dependency 里 | gradient trace test |
| KMRR routing 决策 100% interpretable (有 attribution log) | log inspection |

---

## 11. Lower bound 在 paper 里怎么写（draft）

> **Theorem (Cold-start additive multimodal fusion bound)**. Let $p_{\text{kg}}, p_{\text{mol}}$ be calibrated single-modality predictors with AUROCs $A_{\text{kg}}, A_{\text{mol}}$ on cold-start S2 pairs. Define redundancy $R = P(\text{both correct}) / P(\text{at least one correct})$ where "correct" means the model's score ranks the positive above the negative. Any linear additive fusion $f_{\alpha,\beta} = \alpha p_{\text{kg}} + \beta p_{\text{mol}}$ satisfies
>
> $$\text{AUROC}(f_{\alpha,\beta}) \leq \max\{A_{\text{kg}}, A_{\text{mol}}\} + (1 - R) \cdot (1 - \max\{A_{\text{kg}}, A_{\text{mol}}\})$$
>
> When $R \to 1$ (mol completely redundant on KG), the marginal gain from any linear additive fusion is zero.
>
> **Empirical witness**. On DrugBank S2 cold-start, oracle additive ensemble (test-set-tuned optimal blend) achieves gain = 0.000 with optimal blend weight $b^\star = 0$ (verified `gate_analysis.json`). TIGER and MKG-FENN, both linear additive multimodal architectures, underperform KG-only baselines under strict S2 cold-start, consistent with the theorem.
>
> **Corollary**. Cold-start multimodal DDI fusion requires conditional non-additive architecture to extract molecular signal beyond the KG-redundant baseline. KMRR (Section 3.4) is one such instantiation, fusing molecular information only via pair-conditional mediator-set expansion, never via additive score combination.

---

## 12. 一句话总结

> 这份 framework 是 paper Contribution 3 的 dual-role 实例化, 但只有在 **Option B 三 contribution 切分 (codex-reviewed 2026-05-30)** 下才有 paper-grade defensibility. C1 = pure PMP architecture (不碰 LLM, agnostic to mediator graph source). C2 = mechanism-aware KG enrichment (所有 LLM 贡献统一在此, 包括 new edges + taxonomy + optional encoding, standalone deliverable). C3 = **KMRR routing + redundancy framework + mol-bridge**——本文档主题, dual-role. Analysis 角色给 paper Section 4 提供 redundancy quantification + 40 pair case studies + lower bound theorem (TIGER / MKG-FENN failure 是 empirical witness). Module 角色是 PMP 之外的独立 routing module, 在 KG-rich + Mol-redundant pair 上自动关闭 (no-harm), 在 KG-sparse + Mol-complementary pair (Category D, 真分子驱动且 KG 缺失) 上必须 fire α ≥ 0.7. **C1 不假设 mediator graph 来源, C2 不依赖 PMP, 两者 modular at interface but not causally independent (不强 claim orthogonality, codex 推荐 framing)**. PMP representation 必须在 Layer 2 (typed token embedding + learned attention pool) 实现, MNAH 22-d / v2i4 13-d 是 Layer 1a baseline 不是终态. 10 个 codex-推荐 ablation 是 paper 投稿前的硬底线 (Section 8.6). 5 张 paper figure spec 见 Section 8.7.

# Method Design — Perception Components

> 把 i1-i4 insight 落到具体的 perception-capable model components 上. 这里只列 layout / blueprint, 不写代码.

## 1. Insight → Component 映射

| Insight | Core thesis | Component Name | 负责的"感知"机制 |
|---|---|---|---|
| **i1** PK/PD two paradigms | DDI 信号 split into molecular-mediated (PK) vs effect-mediated (PD) | **C4. Type-Channel Splitter** | 把 mediator 分流到合适的 channel 让模型"知道在看哪种机制" |
| **i2** Meeting-node anchor | 表征应锚在 N(u)∩N(v) 而非 drug 端点 | **C2. Meeting-Node Extractor** | 让模型"看到" shared mediator 集合而不是孤立的 endpoint |
| **i3** Attention 不能注入外部先验 | Internal attention 是数据内部加权; perception 需要外部 prior | **C3. Pair-Conditional Perception** | "感知"哪些 mediator 对该 (u, v) pair 真正 relevant |
| **i4** Node-name text 语义 | PubMedBERT(name) 是 cold-start 唯一稳定外部 prior 来源 | **C1. Stable-Feature Init** | 给所有节点提供 cold-start 稳定的"语义起点" |

## 2. 整体 Pipeline 布局

```
drug pair (u, v) + merged KG
    │
    ▼
[C1] Stable-Feature Init                            ← i4
    │  X_init = PubMedBERT(node.name) → proj_d
    │
    ▼
[C2.a] Shallow GCN K=2 encoder over full KG          ← i2 (浅层避免 over-smoothing)
    │  H = GCN_2(X_init, edge_index)
    │
    ▼
[C2.b] Meeting-Node Extractor                        ← i2
    │  M = (N₁(u) ∪ N₂(u)) ∩ (N₁(v) ∪ N₂(v))
    │
    ▼
[C4] Type-Channel Splitter                           ← i1
    │  M_PK = {m ∈ M : kind(m) ∈ molecular kinds}
    │  M_PD = {m ∈ M : kind(m) ∈ effect kinds}
    │
    ▼
[C3] Pair-Conditional Perception (per channel)       ← i3 + i4
    │  for each m ∈ M_PK:
    │    r_ext(m) = external prior from (i4 BERT cos, kind score, KG centrality)
    │    w_m^PK = perception_PK(h_u, h_v, h_m, r_ext(m))
    │  emb_PK = Σ_m w_m^PK · h_m
    │  (analogously for M_PD)
    │
    ▼
[C5] Pair Decoder                                    ← (utility, not a new insight)
    │  z_pair = combine(h_u, h_v, emb_PK, emb_PD)
    │  logit = MLP(z_pair)
    │
    ▼
sigmoid → P(DDI | u, v)
```

## 3. 单组件 spec (interface only)

### C1. Stable-Feature Init (i4)
- **Input**: 节点集合, node.name (字符串)
- **Output**: `X_init ∈ R^{N × d}`
- **变体**:
  - C1-a: PubMedBERT [CLS] → random projection d
  - C1-b: PubMedBERT [CLS] → PCA-d
  - C1-c (ablation): random Gaussian d (作为 "perception 没有外部 prior" 的对照)
  - C1-d (ablation): shuffled-within-kind name → 测 specific-name 贡献
- **何时 freeze**: init 阶段 frozen, 后续 GCN 输出可学
- **跨 baseline 适用**: 任何接受 node-feature init 的模型都可换 C1

### C2. Meeting-Node Extractor + Encoder (i2)
- **C2.a Encoder**:
  - Input: `X_init`, `edge_index`
  - Output: `H ∈ R^{N × d}`
  - Architecture: 2-layer GCN (depth=2 per E2 sweet spot)
  - 可换 GAT / SAGE — 但 attention 受 i3 perception block 限制, 单独 GAT 不够
- **C2.b Extractor**:
  - Input: `(u, v)`, `adjacency`
  - Output: `M ⊆ V` (meeting node set), `|M|` variable per pair
  - 提取规则: 1-hop intersection ∪ 2-hop intersection
  - 可选筛选: only keep `M` whose kind is mediator-relevant (排除 Drug-Drug)
- **关键设计点**: extractor 是 *graph 结构上的硬性 retrieval*, 不是 learned. 这是 i2 的核心——anchor location 不靠模型 learn

### C3. Pair-Conditional Perception (i3 + i4)
- **核心 i3 commitment**: attention weights 必须 *pair-conditional + external-prior-injected*, 不能仅 data-internal
- **Input**: `h_u, h_v ∈ R^d`, `m ∈ M`, `h_m ∈ R^d`, `r_ext(u, v, m) ∈ R^k` 外部先验向量
- **Output**: `w_m ∈ R_+` (scalar weight)
- **变体**:
  - C3-a (baseline / failed per E8.2): `w_m = softmax(MLP(h_m))` — internal only, **expected to fail**
  - C3-b (proposed): `w_m = softmax(MLP([h_u; h_v; h_m; r_ext(u, v, m)]))` — pair-conditional with external prior
  - C3-c (stronger): two-stage — first selector outputs top-k mediator candidates, then within candidates re-weight with finer features
- **External prior r_ext 候选** (4 类, 全部可同时):
  - PubMedBERT 语义距离: `cos(BERT(u), BERT(m))`, `cos(BERT(v), BERT(m))`
  - Kind 类型 one-hot of m (供 channel routing 用)
  - KG-structural prior: `degree(m)`, `1/degree(m)` (rare mediator weighting), `shortest-path-length(u, m)`, `shortest-path-length(v, m)`
  - Pretrained relevance score (optional): 从 E7 meeting-node LR 学到的 kind coefficient 作 weighted prior
- **每个 channel 独立 perception**: `perception_PK` 和 `perception_PD` 是独立参数, 因为 PK 信号空间 ≠ PD 信号空间 (per i1)

### C4. Type-Channel Splitter (i1)
- **Input**: `M`, node kind map
- **Output**: `M_PK` (molecular), `M_PD` (effect)
- **Partition rule** (declarative, 不 learn):
  - `M_PK kinds`: `Gene, Protein, gene/protein, Pathway, pathway, Biological_Process, Molecular_Function`
  - `M_PD kinds`: `Side Effect, effect/phenotype, Disease, disease, Symptom, Anatomy`
  - `M_other`: 其它 (PharmaClass, CellComp, etc.) — 可入 PK / PD / 单独第三通道
- **是否要在 M_other 上独立 perception?**: 留 ablation 决定; 默认放进 PK or PD by case
- **Channel embedding**:
  - `emb_PK ∈ R^d` = pair-conditional-aggregated M_PK
  - `emb_PD ∈ R^d` = pair-conditional-aggregated M_PD

### C5. Pair Decoder (utility, no new insight)
- **Input**: `h_u, h_v, emb_PK, emb_PD`
- **Output**: scalar logit
- **变体**:
  - C5-a (simple): `logit = MLP([h_u; h_v; emb_PK; emb_PD])`
  - C5-b (symmetric): `logit = MLP([h_u + h_v; |h_u - h_v|; emb_PK; emb_PD])` — 保证 pair 对称性
  - C5-c (factorized): separate PK/PD logits, sum or learn-weight 后再 sigmoid

## 4. Ablation matrix

| Variant | C1 init | C2 encoder | C2 extractor | C3 perception | C4 PK/PD split | Note |
|---|---|---|---|---|---|---|
| **A0 baseline GCN** | random | GCN K=2 | — | — | — | E2 baseline, AUC 0.68 |
| **A1 +PubMedBERT** | C1-b | GCN K=2 | — | — | — | E3 baseline, AUC 0.71 |
| **A2 +meeting-node retrieval** | C1-b | GCN K=2 | C2.b | uniform pool | — | M-readout no perception |
| **A3 +internal attention** | C1-b | GCN K=2 | C2.b | C3-a | — | i3 prediction: fails |
| **A4 +pair-conditional perception** | C1-b | GCN K=2 | C2.b | C3-b | — | i3 unlocked |
| **A5 +PK/PD split** | C1-b | GCN K=2 | C2.b | C3-b | C4 | **full method** |
| **A6 −PubMedBERT** | random | GCN K=2 | C2.b | C3-b | C4 | i4 ablation |
| **A7 −meeting-node** | C1-b | GCN K=2 | — | C3-b on all neighbors | C4 | i2 ablation |
| **A8 −perception** | C1-b | GCN K=2 | C2.b | uniform pool | C4 | i3 ablation |
| **A9 −PK/PD** | C1-b | GCN K=2 | C2.b | C3-b | — | i1 ablation |

主结果: **A5 (full)** 期望比 **A0 baseline** 高 ~10pt AUC, 比 **A1 PubMedBERT-only** 高 ~5pt, 比每个 −X ablation 高 ~2-5pt.

EmerGNN / KnowDDI 等 baseline 单独跑 (按 published 实现), 在同 fair-protocol 下出现在主 table.

## 5. 文件结构 layout

```
Code/my_code/models/
├── perception_ddi/          # 新主 method 目录
│   ├── __init__.py
│   ├── components/
│   │   ├── c1_stable_init.py        # C1 PubMedBERT init
│   │   ├── c2_kg_encoder.py         # C2.a shallow GCN encoder
│   │   ├── c2_meeting_extractor.py  # C2.b meeting node extraction
│   │   ├── c3_perception.py         # C3 pair-conditional perception
│   │   ├── c4_pkpd_split.py         # C4 type channel splitter
│   │   └── c5_decoder.py            # C5 pair decoder
│   ├── model.py                     # 组装 5 个 component 成完整 PerceptionDDI 模型
│   ├── ablations.py                 # A0–A9 配置
│   └── README.md                    # 模块说明
└── GCN/                              # 现有 GCN, 不动
```

```
Code/my_code/eval/
└── ddi_eval.py                       # 共享 eval: AUC + paired bootstrap CI on S0/S1/S2
```

```
Code/configs/
└── perception_ddi/
    ├── A0.yaml ... A9.yaml          # 每个 ablation 一个 config
    └── full.yaml                    # A5 主方法
```

## 6. Training/eval plan layout

- **数据**: ColdDDI seed42/43/44 splits (warm S0, semi-cold S1, both-cold S2)
- **超参锁定**:
  - GCN K=2, hidden=128, lr=5e-3, weight_decay=1e-5, epochs=25 (per E3)
  - Perception MLP: 2-layer hidden=64, dropout=0.3
  - r_ext dimension: 8-12 (BERT cos × 2 + kind one-hot × 7 + structural × 3)
- **Training**: 标准 BCE on positive + drug-replacement negatives (same protocol as ColdDDI)
- **Eval**: AUC + AUPR + F1 on test_S2 (primary); test_S0 (warm reference); paired bootstrap CI (1000) for variant comparisons
- **Multi-seed**: 3 seeds each (42/43/44)

## 7. 不在本期范围

- EmerGNN/KnowDDI 等 baseline 实现 (单独 baseline branch)
- SMILES/molecular-graph channel (cross-modality, future work, see HDN-DDI 讨论)
- Multi-class DDI type prediction (现暂限 binary pair classification)
- Theory framework (v1 / v2 / nominal) 跟 code 的对应关系 — 留下一轮

## 8. Bring up 顺序 (建议)

1. **C1 Stable-Feature Init**: 现有 E3 已实现, 抽出为独立 module
2. **C2 KG Encoder + Meeting-Node Extractor**: GCN 已有, extract module 是新, ~1 day
3. **C5 Pair Decoder**: simple MLP, ~半 day
4. → 第一个可跑版本: **A2 (meeting-node + uniform pool)**, 测一下 sanity
5. **C3 Pair-Conditional Perception**: 核心新 module, ~2 day
6. → **A4 (perception 单 channel)**, 测 perception 是否真 unblock
7. **C4 PK/PD Split**: ~半 day
8. → **A5 (full method)**, 主结果
9. **Ablation suite**: A0–A9 全跑, 出主 table

## 9. 与现有 baseline 关系

- 现有 `Code/my_code/models/GCN/` (node classification on Cora): 不动, 保留作 reference
- E2 / E3 / E7 的脚本: 在 exp2-0513-test-insight/ 作 historical record, 不替换
- 新 PerceptionDDI 跟它们对接通过 *共享 KG / 共享 splits / 共享 PubMedBERT cache*

## 10. 跟 v1 / v2 理论的接口 (后续 round)

每个 component 对应理论:
- C1 ⇄ stable σ-field (v1 C1) / labeled context attributes (v2)
- C2.b ⇄ witness functor `Wit_S(u,v)` (v2 §6.3)
- C3 ⇄ external-prior selector (v1 C3) / Class N_meet readout (v2)
- C4 ⇄ i1 task structuring
- 整体 ⇄ N_meet ∈ Class N (v2 §6.3 typed definition)

写论文时 method section 对接 theory section: 每个 component 给一个 "this realizes Theorem X" 的 cross-reference.

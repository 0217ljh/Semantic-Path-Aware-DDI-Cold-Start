# Codebase Architecture — Stable Infrastructure + Swappable Components

> **核心原则**: 把 "固定不动的" (data / training loop / eval / logging) 跟 "经常换的" (model components / 不同 idea) 物理隔离. 不同 idea = 不同 component class + 不同 YAML config, 不改 infrastructure.

## 1. 三层划分

```
┌──────────────────────────────────────────────┐
│ Experiment layer  (configs/, run.py)         │  ← per idea: 写一个 YAML, 跑
├──────────────────────────────────────────────┤
│ Composition layer (models/)                  │  ← 把 components 串起来
├──────────────────────────────────────────────┤
│ Component layer   (components/)              │  ← 每个 idea 一个新 file
├──────────────────────────────────────────────┤
│ Infrastructure    (core/)                    │  ← 一次写好, 永不动
└──────────────────────────────────────────────┘
```

**Infrastructure**: data loading, training loop, eval, logging, config system, registry. **写一次, 之后所有 idea 共享**.

**Component**: 每个 idea 是这一层加 1-2 个新 file. 实现一个固定 interface 就好.

**Composition + Experiment**: idea 之间切换 = 改 YAML config, **不改 code**.

## 2. 目录布局

```
Code/my_code/
├── core/                              # ★ 写一次永不动
│   ├── __init__.py
│   ├── config.py                      # YAML config loader + schema validation
│   ├── registry.py                    # 装饰器式 component registry
│   ├── trainer.py                     # 通用 training loop
│   ├── evaluator.py                   # AUC / NLL / bootstrap CI
│   ├── data_module.py                 # 共享 data loading (KG, splits, pairs)
│   ├── feature_cache.py               # PubMedBERT / Morgan FP / 邻居 cache
│   ├── checkpoint.py                  # save / load model state
│   ├── logger.py                      # 统一 logging (console + run dir)
│   └── seed_utils.py                  # determinism
│
├── components/                        # ★ 每个 idea 加 1-2 file
│   ├── __init__.py                    # auto-import 让 registry 看到所有 component
│   ├── init/                          # C1 stable-feature init
│   │   ├── random.py                  # RandomGaussianInit
│   │   ├── pubmedbert.py              # PubMedBERTInit
│   │   ├── node2vec.py
│   │   ├── shuffled_pubmedbert.py     # ablation
│   │   └── typename_pubmedbert.py     # ablation
│   ├── encoder/                       # C2.a KG encoder
│   │   ├── identity.py                # 不做 encoding (X_init 直接用)
│   │   ├── gcn2.py                    # GCN K=2
│   │   ├── gcn_k.py                   # 任意 depth, 用于 over-smoothing sweep
│   │   ├── gat.py                     # 单独 attention encoder
│   │   ├── pairnorm_gcn.py            # PairNorm rescue
│   │   └── jk_gcn.py                  # Jumping Knowledge
│   ├── extractor/                     # C2.b meeting-node extraction
│   │   ├── none.py                    # 不做 extraction (drug-anchored)
│   │   ├── intersect_1hop.py
│   │   ├── intersect_1_2hop.py        # 主默认
│   │   └── relation_typed.py          # 关系类型 aware
│   ├── perception/                    # C3 pair-conditional perception
│   │   ├── uniform_pool.py            # mean over M
│   │   ├── internal_attention.py      # MLP(h_m) → softmax (i3 failure mode)
│   │   ├── pair_conditional.py        # 加 h_u, h_v 条件 (i3 unlock)
│   │   ├── pair_cond_external_prior.py# 加 r_ext (full perception, recommended)
│   │   └── two_stage_selector.py      # selector + scorer 分阶段
│   ├── splitter/                      # C4 PK/PD channel splitter
│   │   ├── none.py
│   │   ├── pkpd_kind.py               # declarative kind mapping
│   │   └── pkpd_learned.py            # learned soft assignment (ablation)
│   └── decoder/                       # C5 pair decoder
│       ├── mlp.py                     # simple MLP
│       ├── symmetric.py               # 保证 swap(u, v) 不变
│       └── factorized.py              # PK head + PD head, late fusion
│
├── models/                            # ★ 组装层 (轻薄)
│   ├── __init__.py
│   ├── perception_ddi.py              # 主模型: 把 6 个 slot 按 config 串起来
│   └── baselines/                     # 第三方 baseline (独立目录)
│       ├── emergnn.py
│       ├── knowddi.py
│       ├── sumgnn.py
│       └── ...
│
└── experiments/                       # ★ 入口 + configs
    ├── run.py                         # 入口: python run.py --config <path>
    ├── configs/
    │   ├── _base.yaml                 # 共享 default (data path, eval protocol)
    │   ├── ablations/
    │   │   ├── A0_baseline.yaml       # 随便填 random init + GCN + 无 perception
    │   │   ├── A1_pubmedbert.yaml
    │   │   ├── A2_meeting_pool.yaml
    │   │   ├── A3_internal_attn.yaml
    │   │   ├── A4_pair_cond.yaml
    │   │   ├── A5_full.yaml           # ★ 主方法
    │   │   ├── A6_no_pubmedbert.yaml
    │   │   ├── A7_no_meeting.yaml
    │   │   ├── A8_no_perception.yaml
    │   │   └── A9_no_pkpd.yaml
    │   ├── baselines/
    │   │   ├── emergnn.yaml
    │   │   └── knowddi.yaml
    │   └── exploratory/               # 未来 idea (跨 modality, hetero edges, ...)
    │       └── ...
    └── analysis/                      # 用 run.py 出的 outputs 做 analysis
        └── compare_ablations.py
```

## 3. Stable infrastructure 详细 spec

### core/config.py
- 接口: `cfg = Config.load(yaml_path)` → 校验 schema, 支持 `_base` 继承
- 字段类型 lock 好: `data.kg_nodes_path: str`, `train.epochs: int >= 1`, etc.
- 不同 idea 之间 **YAML 是唯一变化**

### core/registry.py
- 装饰器 `@register("init", "pubmedbert")` 把 component class 登记到 registry
- 工厂: `build_component("init", "pubmedbert", cfg)` 返回实例
- 加新 idea = 写新 file + 装饰器, registry 自动看到

### core/data_module.py
- 函数 / class 提供: `load_kg() / load_splits(seed, split_name) / load_pairs() / id2idx`
- **所有 model 共享 KG / splits / pair-id-mapping**
- 不在 model code 里重新读 parquet (避免每次实验重 IO)

### core/feature_cache.py
- 单例 cache: PubMedBERT[CLS] / Morgan FP / 1-hop 邻居 / 2-hop 邻居 / kind map
- 首次访问加载到 disk; 再访问 mmap
- Components 只需要 `cache.get("pubmedbert")` 就拿到 X_full

### core/trainer.py
- 通用 BCE / multi-label / multi-class loop
- 接受 model + optimizer + dataloader + callbacks
- 支持 train_negatives 周期更新 (per ColdDDI protocol)

### core/evaluator.py
- AUC / AUPR / F1 / NLL on test_S0/S1/S2
- Paired bootstrap CI (1000 reps) 跨 variant
- Output: 每 experiment 生成 `results.json` + `per_pair_predictions.parquet`

### core/checkpoint.py
- 每 run 一个目录 `runs/<config_name>_<timestamp>/`
- 自动保存: model state, config, results.json, predictions, logs
- 支持 reload 续训

### core/logger.py
- 统一 logging 到 console + `runs/<...>/train.log`
- 关键 metric 写 jsonl (per epoch)

## 4. Swappable component 统一 interface

每个 component class 实现一个 *固定的 forward signature*. 不同 idea 改 forward 内部, signature 不动:

| Slot | Class interface | Forward signature |
|---|---|---|
| **C1 init** | `Init(cfg, n_nodes)` | `forward() → X ∈ R^{N × d}` |
| **C2.a encoder** | `Encoder(cfg, in_dim, out_dim)` | `forward(X, edge_index) → H ∈ R^{N × d}` |
| **C2.b extractor** | `Extractor(cfg, adjacency)` | `forward(u, v) → M ⊆ [N]` |
| **C3 perception** | `Perception(cfg, dim)` | `forward(H, u, v, M, r_ext) → w ∈ R^{|M|}` |
| **C4 splitter** | `Splitter(cfg, kind_map)` | `forward(M) → {channel_name: M_subset}` |
| **C5 decoder** | `Decoder(cfg, dim)` | `forward(h_u, h_v, channel_embs) → logit ∈ R` |

新 idea = 新 component class with `forward` 实现 + `@register("slot", "idea_name")`.

## 5. Composition layer (`models/perception_ddi.py`)

总共 ~50 行, **不变**. 内容大致:

```python
class PerceptionDDI:
    def __init__(self, cfg):
        self.init = build_component("init", cfg.init.type, cfg.init.params)
        self.encoder = build_component("encoder", cfg.encoder.type, cfg.encoder.params)
        self.extractor = build_component("extractor", cfg.extractor.type, cfg.extractor.params)
        self.perception = build_component("perception", cfg.perception.type, cfg.perception.params)
        self.splitter = build_component("splitter", cfg.splitter.type, cfg.splitter.params)
        self.decoder = build_component("decoder", cfg.decoder.type, cfg.decoder.params)

    def forward(self, pair, edge_index):
        # 6 个 slot 串起来; 每个 slot 是哪个具体 idea 由 cfg 决定
        ...
```

新 idea **永远不用改这个文件**.

## 6. Experiment YAML 例子 (idea = 一个 config)

**A5_full.yaml** (主方法):
```yaml
_base: ../_base.yaml
init:
  type: pubmedbert
  params: {projection: pca, dim: 128}
encoder:
  type: gcn2
  params: {hidden: 128, dropout: 0.1}
extractor:
  type: intersect_1_2hop
  params: {restrict_kinds: ["Protein", "Gene", "SideEffect", "Phenotype", "Disease", "Pathway"]}
perception:
  type: pair_cond_external_prior
  params: {hidden: 64, prior_dims: [bert_cos, kind_onehot, log_degree]}
splitter:
  type: pkpd_kind
  params: {pk_kinds: [Protein, Gene, Pathway], pd_kinds: [SideEffect, Phenotype, Disease]}
decoder:
  type: symmetric
  params: {hidden: 64}
train: {epochs: 25, lr: 0.005, weight_decay: 1e-5, batch_size: 16384}
eval: {splits: [s0, s2], metrics: [auc, aupr, f1, nll], bootstrap_ci: 1000}
data: {seed: 42}
```

**A4_pair_cond.yaml** (perception 但不 PK/PD split):
```yaml
_base: A5_full.yaml
splitter: {type: none}
```

**A0_random_baseline.yaml**:
```yaml
_base: A5_full.yaml
init: {type: random, params: {dim: 128}}
extractor: {type: none}
perception: {type: uniform_pool}
splitter: {type: none}
decoder: {type: mlp}
```

→ idea 之间切换 = 改几行 YAML.

## 7. 跑流程

```bash
# 跑 A5 full method on seed 42
python -m my_code.experiments.run --config experiments/configs/ablations/A5_full.yaml --seed 42

# 跑 A4 on seed 43
python -m my_code.experiments.run --config experiments/configs/ablations/A4_pair_cond.yaml --seed 43

# 批量 ablation
python -m my_code.experiments.run --config-dir experiments/configs/ablations --seeds 42,43,44

# Output 在 runs/A5_full_2025-05-14_..._seed42/
#   model.pt  config.yaml  results.json  predictions.parquet  train.log
```

## 8. 新 idea 加入流程 (举例 — 想加一个 "graph transformer" encoder)

1. 写 `components/encoder/graph_transformer.py`:
   - class `GraphTransformerEncoder` with same `forward(X, edge_index) → H` interface
   - 顶部 `@register("encoder", "graph_transformer")`
2. 写 `experiments/configs/exploratory/A_gt.yaml`:
   ```yaml
   _base: ../ablations/A5_full.yaml
   encoder: {type: graph_transformer, params: {layers: 4, heads: 8}}
   ```
3. 跑: `python -m my_code.experiments.run --config experiments/configs/exploratory/A_gt.yaml`

**两个新 file, 一个 config, 不改 任何 infrastructure / 任何 other component**.

## 9. 跟现有 code 的关系

- `Code/my_code/models/GCN/` (现有 Cora node-classification GCN): 保留不动
- `Code/my_code/train/`, `Code/my_code/eval/` (现有): 跟新 `core/` 同时存在, 旧 script 仍可跑 (E2/E3/E7 历史脚本不受影响)
- 新的 `core/` 是 **DDI-specific 的统一 infrastructure**, 跟旧的 general utilities 并存

## 10. 实施顺序建议

**P0 (一次性 ~3 day)** — 固定 infrastructure:
1. core/config.py + core/registry.py (~半 day)
2. core/data_module.py + core/feature_cache.py (~1 day, 抽 E3 现有 loading)
3. core/trainer.py + core/evaluator.py (~1 day, 抽 E3 现有 train loop)
4. core/checkpoint.py + core/logger.py (~半 day)

**P1 — 第一个 runnable idea**:
1. 写 6 个最基本 component variants (init=pubmedbert, encoder=gcn2, extractor=none, perception=uniform_pool, splitter=none, decoder=mlp)
2. 写 `models/perception_ddi.py` composition
3. 写 `A0` config
4. 跑通, 验证 AUC 跟 E3 baseline 一致

**P2 — 增量加 component**:
- 每加一个 component = 一个 file + 一个 config + 一次 run
- 增量 ablation 自然完成 A0 → A1 → ... → A5

**P3 — Baseline 接入**:
- `models/baselines/` 接入 EmerGNN/KnowDDI (单独 model, 不走 component composition)
- 共享 `core/` 的 data / eval

## 11. 这套架构的好处

1. **加 idea 几乎零摩擦**: 1-2 个新 file + 1 个 YAML
2. **复现性**: 每 run 完整保存 config + results, ML reviewer 可以 reproduce
3. **Ablation 自然**: A0-A9 不是 9 个不同脚本, 是 9 个 YAML, infrastructure 一致
4. **理论对应**: paper 里 method section 可以一一对应 component, 每个 slot 解释 "this realizes Theorem X"
5. **跟 baseline 隔离干净**: 我们方法走 component composition, 第三方 baseline 在 `models/baselines/` 单独路径

## 12. 短板 + 接受

- Initial 投入: ~3 day 写 infrastructure 之前不出 paper figure. 接受是因为之后所有 idea 都几乎 0 摩擦
- Over-engineering 风险: 如果只跑 1-2 idea, 就不必这么搭. 但我们规划要跑 ≥10 个 ablation × 3 seeds × 3 splits, infrastructure 投入回报高
- 不是 mmcv / Hydra 的 full version, 只取了 registry + config-composition pattern 这两个最核心

## 13. 下一步建议

如果你 ok 这个布局:
- 让我先写 `core/` 4 个最关键 file (config, registry, data_module, trainer) 作 P0 skeleton
- 跑通一个最 trivial `A0` ablation 确认 pipeline 工作
- 然后逐个 component 加进来

如果你想先看具体几个 file 的 detailed 设计 (例如 registry decorator 怎么写 / config 继承怎么 implement), 告诉我重点 zoom in 哪里.

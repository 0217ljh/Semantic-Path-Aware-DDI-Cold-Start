# Codebase Architecture v2 — Idea-Folder Centric, Minimal

> 简化版. 每个 idea = 一个文件夹. 内部含: 该 idea 的所有新 component + config + README. 不按 component-type 拆 init/encoder/perception/... 这种 subdirectory.

## 1. 顶层布局 (3 部分 — 完)

```
Code/my_code/
├── core/                  # ★ 写一次, 永不动 (~7 files)
├── ideas/                 # ★ 每 idea 一个 folder
└── experiments/           # ★ 跨 idea 组合的 YAML 配置 + 入口
```

就 3 个顶层目录, 不再多.

## 2. core/ — Infrastructure (一次性)

```
core/
├── config.py              # YAML loader + 继承 (_base) + schema 校验
├── registry.py            # @register("slot", "name") 装饰器
├── data.py                # 共享 data: KG / splits / pairs / neighbors
├── cache.py               # PubMedBERT / Morgan FP / kind map 单例 cache
├── trainer.py             # BCE training loop + checkpoint
├── evaluator.py           # AUC / AUPR / NLL / bootstrap CI
├── run.py                 # 入口: python -m my_code.core.run --config <path>
└── defaults/              # 最简单 trivial component (跑 baseline 用)
    ├── init_random.py
    ├── encoder_gcn2.py
    ├── extractor_none.py
    ├── perception_uniform.py
    ├── splitter_none.py
    └── decoder_mlp.py
```

**这 13 个 file 是 infrastructure + 默认 trivial components, 一次写好之后不动**. 任何新 idea 都通过 `ideas/<idea_name>/` 加, 不进 core/.

## 3. ideas/ — 每 idea 一个 folder

每 idea folder 固定 3 件东西:
1. `component.py` — 该 idea 引入的新 component(s)
2. `config.yaml` — 单独跑这个 idea 的最简 config (只激活这一个 idea)
3. `README.md` — 这个 idea 在干什么 + 对应 insight + 实验结果链接

```
ideas/
├── idea1_meeting_node/             # ← 对应 i2 (meeting-node anchor)
│   ├── component.py                # 新 component: MeetingNodeExtractor
│   ├── config.yaml                 # extractor: meeting_1_2hop, 其余 default
│   └── README.md
│
├── idea2_perception/               # ← 对应 i3 (pair-conditional perception)
│   ├── component.py                # 新 component: PairConditionalPerception + external prior wiring
│   ├── config.yaml
│   └── README.md
│
├── idea3_pkpd_split/               # ← 对应 i1 (PK/PD two paradigms)
│   ├── component.py                # 新 component: PKPDSplitter + FactorizedDecoder
│   ├── config.yaml
│   └── README.md
│
├── idea4_pubmedbert_init/          # ← 对应 i4 (node-name text)
│   ├── component.py                # 新 component: PubMedBERTInit
│   ├── config.yaml
│   └── README.md
│
└── (idea5_*, idea6_*, ... 未来 idea)
```

**每 idea folder 最多 3 个 file**. 不在 idea folder 内部再分子目录.

如果一个 idea 引入多个 component (e.g., idea3 同时引入 splitter 和 factorized decoder), 都放在该 idea 的 `component.py` 里, 不拆 subfolder.

## 4. experiments/ — 组合 YAML

```
experiments/
├── configs/
│   ├── _base.yaml                      # 全局 default
│   ├── A0_baseline.yaml                # 0 个 idea 激活 (全 core/defaults)
│   ├── A1_i4_only.yaml                 # idea4
│   ├── A2_i2_i4.yaml                   # idea1 + idea4
│   ├── A4_i2_i3_i4.yaml                # idea1 + idea2 + idea4
│   ├── A5_full.yaml                    # idea1+2+3+4 (主方法)
│   ├── A6_no_i4.yaml                   # full − idea4
│   ├── ...                             # 其余 ablation
│   └── baselines/
│       └── emergnn.yaml
└── analysis/                            # 跨 run 比较脚本 (可选)
```

每个 ablation = 一个 YAML, 通过 `_base: A5_full.yaml` 然后 override 一个 slot 实现.

## 5. Component slot — 6 个固定

无论加多少 idea, model 总是 6 个 slot:

| Slot | 默认 (core/defaults/) | idea folder 里的 swap |
|---|---|---|
| `init` | `init_random` | `idea4/init_pubmedbert` |
| `encoder` | `encoder_gcn2` | (暂时无 idea swap) |
| `extractor` | `extractor_none` | `idea1/extractor_meeting` |
| `perception` | `perception_uniform` | `idea2/perception_pair_cond` |
| `splitter` | `splitter_none` | `idea3/splitter_pkpd` |
| `decoder` | `decoder_mlp` | `idea3/decoder_factorized` |

未来如果发现需要 7th slot (e.g., 新加一个 "negative sampler" 或 "loss reweighting"), 加进 core, 但谨慎. 当前 6 个够.

## 6. Component interface — 极简

每个 component 一个 class, 一个 register 装饰器, 一个 forward. 全文件 < 100 行.

```
ideas/idea1_meeting_node/component.py:

  from core.registry import register

  @register("extractor", "meeting_1_2hop")
  class MeetingNodeExtractor:
      def __init__(self, cfg, adjacency_cache):
          # 读 cfg.restrict_kinds 等
          ...
      def forward(self, u, v):
          # 返回 M ⊆ [N]
          ...
```

新 idea = **一个 file + 一个装饰器 + 一个 class + 一个 config**. 

## 7. YAML 例子

**A0_baseline.yaml** (无任何 idea 激活):
```yaml
_base: _base.yaml
init:       {type: random}
encoder:    {type: gcn2}
extractor:  {type: none}
perception: {type: uniform}
splitter:   {type: none}
decoder:    {type: mlp}
```

**A5_full.yaml** (4 个 idea 全激活):
```yaml
_base: _base.yaml
init:       {type: pubmedbert,        params: {projection: pca, dim: 128}}    # idea4
encoder:    {type: gcn2,              params: {hidden: 128}}
extractor:  {type: meeting_1_2hop,    params: {restrict_kinds: [Protein, Gene, SideEffect, Phenotype, Disease, Pathway]}}  # idea1
perception: {type: pair_cond_external, params: {hidden: 64, priors: [bert_cos, kind, log_degree]}}  # idea2
splitter:   {type: pkpd_kind,         params: {pk: [Protein, Gene, Pathway], pd: [SideEffect, Phenotype, Disease]}}  # idea3
decoder:    {type: factorized,        params: {hidden: 64}}                   # idea3
```

**A6_no_i4.yaml** (full minus idea4):
```yaml
_base: A5_full.yaml
init: {type: random}     # 只改这一行
```

**ideas/idea1_meeting_node/config.yaml** (只激活 idea1 单独跑, 用来 isolate 该 idea 的贡献):
```yaml
_base: ../../experiments/configs/A0_baseline.yaml
extractor: {type: meeting_1_2hop, params: {restrict_kinds: [...]}}
```

## 8. 运行

```bash
# 跑 A5 (full method)
python -m my_code.core.run --config experiments/configs/A5_full.yaml --seed 42

# 跑 idea1 单独
python -m my_code.core.run --config ideas/idea1_meeting_node/config.yaml --seed 42

# 批量
python -m my_code.core.run --config-dir experiments/configs --seeds 42,43,44

# 输出到 runs/A5_full__seed42__2025-05-14_...
#   model.pt, config.yaml, results.json, predictions.parquet, train.log
```

## 9. 新 idea 加入完整流程

**例**: 想加 idea5 = "graph-transformer encoder"

1. `mkdir ideas/idea5_graph_transformer`
2. 写 `ideas/idea5_graph_transformer/component.py`:
   - class `GraphTransformerEncoder`
   - `@register("encoder", "graph_transformer")`
3. 写 `ideas/idea5_graph_transformer/config.yaml`:
   - `_base: ../../experiments/configs/A0_baseline.yaml`
   - `encoder: {type: graph_transformer, params: ...}`
4. 写 `README.md` 解释这个 idea
5. 可选: 写 `experiments/configs/A_gt.yaml` 把 idea5 跟 i2/i3/i4 组合

**改动只在 ideas/idea5/ 内**, 不动 core, 不动其他 idea, 不动 composition.

## 10. 跟 idea 解耦的好处

- **加 idea 零摩擦**: 新建 folder, 写 component class, 加 register 装饰器, 写 config — 完事
- **删 idea 零摩擦**: 删除 idea 文件夹即可 — 其他 idea 不受影响
- **idea 之间组合**: experiments/configs/ 里的 YAML 选哪些 idea 用
- **paper reviewer 友好**: 每个 idea folder 自带 README, 解释 "this realizes i2 / i3 / etc."
- **历史可追溯**: idea 不被改动, 总是看 idea folder 就知道当初设计

## 11. 文件总数 (minimal target)

- core/: 7 file (config, registry, data, cache, trainer, evaluator, run) + 6 default components = 13
- ideas/idea1-4/: 每个 3 file = 12
- experiments/configs/: 10-12 个 YAML (ablations + baselines)

总共 **~40 file**. 之后每加一个 idea, +3 file. 这是极简版本.

## 12. P0 实施顺序

1. `core/config.py` + `core/registry.py` (~半 day)
2. `core/data.py` + `core/cache.py` (~1 day, 抽 E3 已有 loading)
3. `core/trainer.py` + `core/evaluator.py` (~1 day)
4. `core/run.py` (~半 day)
5. `core/defaults/` 6 个 trivial component (~半 day, 都 < 50 行)
6. 跑 A0_baseline 验证 pipeline 跟 E3 baseline AUC 一致
7. 然后 idea1, idea2, ... 逐个加 (每 idea ~半 day)

P0 总计 ~3-4 day. 之后每 idea 半 day.

## 13. 接受 / 调整

如果你 ok 这个 v2 简洁布局, 我可以:
- 选项 A: 开始 P0 写 core/ infrastructure 的 4 个最关键 file
- 选项 B: 先写一个 idea folder 的 detailed sample (e.g., idea1_meeting_node/ 的 component / config / README 三个 file 的 详细 spec, 让你看 idea-level 的具体样子)
- 选项 C: 还有想调整的地方

不再写 v3 除非有明确需求.

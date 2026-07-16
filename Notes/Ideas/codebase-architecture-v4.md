# Codebase Architecture v4 — Idea-by-Number, Slot-Mirror Internal Structure

> v3 调整: 
> 1. idea 文件夹**只用序号** (`idea1/`, `idea2/` ...), 描述放 README.md
> 2. idea 内部**按 pipeline slot 分子文件夹** — 该 idea 给 pipeline 哪个 slot 贡献组件, 就有那个 slot 的子文件夹
> 3. 同一 idea 多个 sub-module 都放进对应 slot 子文件夹里

## 1. 顶层布局 (跟 v3 相同, 3 部分)

```
Code/my_code/
├── perception_ddi/                # baseline 包 (注册到 coldddi, 一次性)
├── ideas/                         # 每 idea 一个序号文件夹
└── experiments/                   # 跨 idea 组合 YAML
```

## 2. perception_ddi/ — baseline 包 (一次写好)

```
perception_ddi/
├── __init__.py                    # @register("perception_ddi") + auto-import ideas
├── baseline.py                    # PerceptionDDIBaseline(BaselineModel) — fit/predict_proba/save/load
├── model.py                       # nn.Module — 6 slot 串起来 (50-80 行, 永不动)
├── slot_registry.py               # @register_slot("slot_name", "variant_name") + build_slot
├── config.py                      # YAML loader (含 _base 继承) + build_model_from_config
└── defaults/                      # 6 个 trivial 默认 (跑 A0 baseline 用)
    ├── init_random.py
    ├── encoder_gcn2.py
    ├── extractor_none.py
    ├── perception_uniform.py
    ├── splitter_none.py
    └── decoder_mlp.py
```

**6 个固定 slot** (主 pipeline 结构):
`init` → `encoder` → `extractor` → `perception` → `splitter` → `decoder`

整个 pipeline 由 `model.py` 串起来:
```
X = init.forward()
H = encoder.forward(X, edge_index)
M = extractor.forward(u, v)
w = perception.forward(H, u, v, M, r_ext)
channels = splitter.forward(M, w)
logit = decoder.forward(h_u, h_v, channels)
```

## 3. ideas/ — 每 idea 一个序号文件夹

### 总体规则
- 文件夹名: `idea1/`, `idea2/`, `idea3/`, ...
- 描述什么: 进 README.md
- 内部子文件夹: **只为该 idea 实际贡献的 slot 创建**, 不需要的 slot 不建空文件夹
- 每个 slot 子文件夹里可以有多个 file (variant / sub-module / helpers)

### 具体布局

```
ideas/
├── idea1/                           # README 内会注明: i2 meeting-node anchor
│   ├── README.md
│   ├── extractor/                   # 该 idea 给 extractor slot 的贡献
│   │   ├── meeting_1_2hop.py        # 主 variant
│   │   ├── relation_aware.py        # 额外 variant
│   │   └── _neighbor_cache.py       # 共享 helper (下划线开头, 不注册)
│   └── config.yaml                  # 单独激活 idea1 的 config
│
├── idea2/                           # README: i3 pair-conditional perception
│   ├── README.md
│   ├── perception/                  # 给 perception slot 的贡献
│   │   ├── pair_conditional.py
│   │   ├── external_prior.py        # 加 r_ext 的 variant
│   │   ├── two_stage.py             # selector + scorer 分阶段 variant
│   │   └── _prior_featurizer.py     # 内部 helper: 拼 BERT cos + kind + degree
│   └── config.yaml
│
├── idea3/                           # README: i1 PK/PD 双 paradigm
│   ├── README.md
│   ├── splitter/
│   │   └── pkpd_kind.py
│   ├── decoder/                     # idea3 同时 contribute 到 decoder slot
│   │   └── factorized.py            # PK head + PD head + late fusion
│   └── config.yaml
│
├── idea4/                           # README: i4 PubMedBERT name semantics
│   ├── README.md
│   ├── init/
│   │   ├── pubmedbert.py            # 主 variant
│   │   ├── shuffled.py              # ablation: shuffled-within-kind name
│   │   └── typename.py              # ablation: kind name only
│   └── config.yaml
│
└── (idea5/, idea6/, ... 未来)
```

### 每个 idea 必备 3 类东西

| 类型 | 命名规则 | 说明 |
|---|---|---|
| `README.md` | 固定名字 | idea 描述 + 对应 insight + 引入的 slot/variant 列表 + 实验结果链接 |
| `<slot>/<variant>.py` | slot 子文件夹 | 一个 file 一个 component class + `@register_slot("<slot>", "<variant>")` 装饰器 |
| `config.yaml` | 固定名字 | 单独激活该 idea 的 config (其余 slot 走 default) |

**`_*.py` 是该 idea 内部 helper, 不注册 — 仅被该 idea 的 components 使用**.

### README.md 模板

```markdown
# Idea N

**Corresponding insight**: i2 (meeting-node anchor) — see `Notes/Settings/Insights/i2.md`

## What this idea contributes

- New `extractor` variant: `meeting_1_2hop`
- New `extractor` variant: `relation_aware`
- Replaces default `extractor_none` (uniform read all of `H`)

## Components in this folder

- `extractor/meeting_1_2hop.py` — class `MeetingNodeExtractor`, registers as `("extractor", "meeting_1_2hop")`
- `extractor/relation_aware.py` — class `RelationAwareExtractor`, registers as `("extractor", "relation_aware")`
- `extractor/_neighbor_cache.py` — internal helper, not registered

## Pipeline integration

In a YAML config:
```yaml
extractor: {type: meeting_1_2hop, params: {restrict_kinds: [Protein, Gene, ...]}}
```

## Single-idea config

`config.yaml` activates this idea alone (other slots = defaults). 跑:
```
bash experiments/run.sh ideas/idea1/config.yaml 42 S2
```

## Theoretical correspondence

This realizes Theorem 5 of v2 nominal framework (witness functor `Wit_S`). See `Notes/Settings/Insights/theory-framework-v2-deeper.md` §6.3.

## Experimental results

See `Notes/Log/<run-dir>/results_S2.json` (after first run).
```

### config.yaml 模板 (idea1 单独跑)

```yaml
_base: ../../experiments/configs/A0_baseline.yaml
extractor:
  type: meeting_1_2hop
  params:
    restrict_kinds: [Protein, Gene, SideEffect, Phenotype, Disease, Pathway]
```

## 4. 跟主 pipeline 怎么集成

集成路径有 3 个 entry point:

### A. Component 注册 (build-time)
`perception_ddi/__init__.py` 在 import 时:
```python
# 1. 注册我们的 baseline 到 coldddi 全局 registry
from perception_ddi.baseline import PerceptionDDIBaseline  # decorator fires

# 2. 让 slot registry 看到所有 ideas/ 下的 component
from perception_ddi.slot_registry import _auto_import_ideas
_auto_import_ideas()
```

`_auto_import_ideas()` 遍历 `ideas/*/<slot>/*.py` 并 import 它们 — 每个 file 顶部的 `@register_slot(...)` 装饰器就自动触发, slot_registry 完整填充.

### B. YAML config → model 实例 (instantiation)
`perception_ddi.config.build_model_from_config(cfg_dict)`:
```python
for slot in ["init", "encoder", "extractor", "perception", "splitter", "decoder"]:
    slot_cfg = cfg_dict[slot]
    slot_components[slot] = build_slot(slot, slot_cfg["type"], slot_cfg.get("params", {}))
model = PerceptionDDIModel(slot_components)
```

### C. ColdDDI evaluate dispatch (run-time)
`coldddi.evaluate --method perception_ddi --config-path ...` 走到 `PerceptionDDIBaseline.__init__(config=...)`, 内部调 (B), 拿到 model 后走标准 `fit / predict_proba`.

**3 个 entry point 都不需要每加 idea 修改**. 只要 idea 文件夹按规约写, 自动 wire up.

## 5. experiments/ — 跨 idea 组合

```
experiments/
├── configs/
│   ├── _base.yaml                   # model + train default
│   ├── A0_baseline.yaml             # 0 idea (全 defaults)
│   ├── A1.yaml                      # idea4 only
│   ├── A2.yaml                      # idea1 + idea4
│   ├── A4.yaml                      # idea1 + idea2 + idea4
│   ├── A5.yaml                      # idea1 + idea2 + idea3 + idea4 (full)
│   ├── A6.yaml                      # A5 - idea4
│   ├── ...
│   └── baselines/
│       └── emergnn.yaml             # 对照: coldddi 现成 EmerGNN, 直接跑
└── run.sh                           # thin wrapper for coldddi.evaluate
```

### A5.yaml 例子 (跨 4 idea 组合)

```yaml
_base: _base.yaml
init:       {type: pubmedbert,         params: {projection: pca, dim: 128}}   # idea4
encoder:    {type: gcn2,               params: {hidden: 128, depth: 2}}       # defaults
extractor:  {type: meeting_1_2hop,     params: {restrict_kinds: [...]}}       # idea1
perception: {type: pair_cond_external, params: {hidden: 64, priors: [...]}}   # idea2
splitter:   {type: pkpd_kind,          params: {pk: [...], pd: [...]}}        # idea3
decoder:    {type: factorized,         params: {hidden: 64}}                  # idea3
```

YAML 不关心"哪个 idea", 只关心"哪个 variant 填进哪个 slot". idea 是组织 code 的概念, 不是 runtime 概念.

## 6. 加新 idea 完整流程 (举例 idea5 = "molecular-fingerprint side channel")

```bash
mkdir my_code/ideas/idea5
```

写 4 件东西:
- `idea5/README.md` (描述 + slot 列表 + 用法)
- `idea5/init/morgan_fp.py` (新 init variant: `class MorganFPInit + @register_slot("init", "morgan_fp")`)
- `idea5/init/morgan_fp_combined.py` (variant: morgan ⊕ pubmedbert 拼接)
- `idea5/config.yaml` (`_base: A0`; `init: {type: morgan_fp}`)

如果 idea5 还想给 decoder 加一个 cross-modal aggregator:
- `idea5/decoder/cross_modal.py` (`@register_slot("decoder", "cross_modal")`)

跑 idea5 单独:
```bash
bash experiments/run.sh ideas/idea5/config.yaml 42 S2
```

跑跟 idea1-4 组合:
- 新建 `experiments/configs/A6_with_mol.yaml`:
```yaml
_base: A5.yaml
init: {type: morgan_fp_combined, params: {...}}
```

**任何步骤不动 perception_ddi/, 不动其他 idea, 不动 experiments 已有 configs**.

## 7. P0 实施顺序 (跟 v3 相同)

1. **(½ day)** 安装 coldddi 并验证 `from coldddi.data import PairDataset` 可用
2. **(1 day)** 写 `perception_ddi/` 7 个 file (baseline / model / slot_registry / config / 6 defaults)
3. **(½ day)** 跑通 A0_baseline via `coldddi.evaluate`, AUC 跟 E3 random-init 对齐
4. **(½ day)** idea4 (PubMedBERT init) — 验证 AUC 跟 E3 PubMedBERT-init 对齐
5. **逐 idea 加 (每 ~半 day)**: idea1 → idea2 → idea3 → A5 full
6. **(½ day)** 出 main table: A0 / A1 / A2 / A4 / A5 / A6-A9 ablations + EmerGNN baseline

## 8. 关键 design 原则 (永不动)

1. **idea 序号化**: 不在文件夹名里 encode 任何语义. 描述全在 README. 删/重排 idea 随意.
2. **Slot-mirror internal**: idea 内部子文件夹 1:1 对应 pipeline slot. 看子文件夹就知道 idea 改哪些 slot.
3. **多 variant 同 slot 共存**: 同一 idea 可在同 slot 提供多个 variant (main + ablation + alternative), 都放进同个 slot 子文件夹.
4. **Helper 用下划线前缀**: 不注册的 internal helper file 用 `_*.py`, 跟注册的 component file 视觉区分.
5. **跨 idea 组合靠 YAML**: idea 之间不互相 import. 组合只在 experiments/configs/AN.yaml 通过 type/params 完成.

## 9. 你选 / 调整

**(A)** 接受 v4. 开始 P0 step 1 (装 coldddi as dependency, 试 import)
**(B)** 看具体 file 的 detailed spec (比如 `perception_ddi/baseline.py` fit 怎么写 / slot_registry 装饰器实现)
**(C)** 还想调整 layout

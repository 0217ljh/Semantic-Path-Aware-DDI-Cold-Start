# Codebase Architecture v3 — Plug into ColdDDI as a Baseline

> v2 简化版的进一步收紧。**不再写 data loading / train loop / evaluate / negatives — 全部复用 ColdDDI `Code-Released-Formal` 已有 pipeline**. 我们只贡献一个新 baseline 注册进去, 加上 idea folder 管理 swap-able components.

## 1. 复用什么 (ColdDDI 已有, 不重写)

`D:\My-Research\03-Projects\ColdDDI\Code-Released-Formal\` 已经写好且 release-grade:

| ColdDDI 模块 | 提供什么 |
|---|---|
| `coldddi.data.PairDataset` | `from_release_dir(root, seed)` 自动加载 KG + splits + drugs.csv + negatives |
| `coldddi.data.SplitFolds` | train / val_s0/s1/s2 / test_s0/s1/s2 共 7 个 split 的统一接口 |
| `coldddi.data.negatives` | `get_train_negatives(epoch)` 1:1 drug-replacement negative, 跨 epoch 不同 |
| `coldddi.data.KnowledgeGraph` | KG (enzymes/targets/transporters/carriers/pathways) 加载 |
| `coldddi.baselines.BaselineModel` | ABC: `fit(train, val, kg) / predict_proba(pairs, kg) / save(path) / load(path)` |
| `coldddi.baselines.register("name")` | 装饰器把 baseline 注册到 registry |
| `coldddi.evaluate` | CLI: `--method <name> --data <root> --seed --setting {S0,S1,S2,all}` 自动跑 fit + eval |
| 8 个 baselines (`deepddi/ssi-ddi/dsn-ddi/hdn-ddi/emergnn/tiger/mkg-fenn/textddi`) | 已经实现好, 直接当 baseline 对比 |

**整个 train/val/test 流水线 + 负样本 + KG + drugs CSV + 指标计算全部由 coldddi 负责**. 我们只贡献 model.

## 2. 我们要做什么

**核心**: 一个新 baseline `PerceptionDDIBaseline`, 内部 6-slot 可组合, 不同组合 = 不同 idea / ablation.

```
Code/my_code/
├── perception_ddi/                  # ★ 我们的 baseline 包 (注册到 coldddi)
│   ├── __init__.py                  # @register("perception_ddi")
│   ├── baseline.py                  # PerceptionDDIBaseline: fit/predict_proba/save/load
│   ├── model.py                     # nn.Module: 6-slot 装配 (50 行不变)
│   ├── slot_registry.py             # 我们项目内的 component slot registry
│   ├── config.py                    # 把 yaml/dict → 6 slot 实例
│   └── defaults/                    # 6 个 trivial 默认 component
│       ├── init_random.py
│       ├── encoder_gcn2.py
│       ├── extractor_none.py
│       ├── perception_uniform.py
│       ├── splitter_none.py
│       └── decoder_mlp.py
│
├── ideas/                           # ★ 每 idea 一个 folder (3 file)
│   ├── idea1_meeting_node/          # i2
│   │   ├── component.py             # 新 extractor 类 + @register_slot
│   │   ├── config.yaml              # 单独激活该 idea 的 model config
│   │   └── README.md
│   ├── idea2_perception/            # i3
│   ├── idea3_pkpd_split/            # i1
│   ├── idea4_pubmedbert_init/       # i4
│   └── (future)
│
└── experiments/                     # ★ 跨 idea 组合 + 跑 ColdDDI evaluate
    ├── configs/
    │   ├── _base.yaml               # model-level default
    │   ├── A0_baseline.yaml
    │   ├── A5_full.yaml
    │   └── ...
    └── run.sh                       # 调 coldddi evaluate 的 wrapper
```

**总文件数**: perception_ddi/ ~12 file (写一次), ideas/ 4 个 folder × 3 file = 12, configs ~10. 约 35 file. 最小化.

## 3. perception_ddi/ — Our baseline package

### perception_ddi/__init__.py
- 触发 register: `from .baseline import PerceptionDDIBaseline` (装饰器自动注册到 coldddi 的 `_REGISTRY`)
- 自动 import 所有 ideas 下的 component (让 slot registry 看到)

### perception_ddi/baseline.py
- `class PerceptionDDIBaseline(BaselineModel)` 继承 coldddi 的 ABC
- `__init__(self, *, config: dict | str | Path)` — 接受 YAML path 或 dict, 加载 model config
- `fit(self, train: PairDataset, val=None, kg=None)`:
  - 从 `train.get_train_negatives(epoch)` 拿 negatives (每 epoch 重 sample)
  - 在 `train.kg` 上跑 model forward
  - 标准 BCE training loop
  - 可选 `val` 做 early stopping
- `predict_proba(self, pairs: DataFrame, kg=None) → np.ndarray`:
  - 把 (drug_a_id, drug_b_id) DF 转 indices, 走 model.forward
  - 输出 1-D sigmoid prob 数组
- `save(path)` / `load(path)`: 标准 manifest + state_dict
- 装饰器: `@register("perception_ddi")` (coldddi 的)

### perception_ddi/model.py
- `class PerceptionDDIModel(nn.Module)`
- `__init__(self, slot_components)`: 接受 6 个 slot 实例
- `forward(self, pair_idx_batch, kg_data)`: 6 slot 串起来 (init → encoder → extractor → perception → splitter → decoder)
- ~80 行, 永不动

### perception_ddi/slot_registry.py
- 项目内部的 slot registry (跟 coldddi baseline registry 是两个不同 layer)
- 装饰器 `@register_slot("perception", "pair_cond_external")`
- 工厂 `build_slot("perception", "pair_cond_external", params)` → 实例

### perception_ddi/config.py
- `load_perception_config(yaml_path) → dict` (含 `_base` 继承)
- `build_model_from_config(config) → PerceptionDDIModel`
  - 调 slot_registry 拿 6 个 slot 实例
  - 装配

### perception_ddi/defaults/
6 个 trivial 默认 component, 每个 < 50 行 + 装饰器 `@register_slot("init", "random")` 等. 让 A0 baseline 跑得起来.

## 4. ideas/ideaN_xxx/ — 每个 idea 3 个 file

### ideas/idea1_meeting_node/component.py
```python
from perception_ddi.slot_registry import register_slot

@register_slot("extractor", "meeting_1_2hop")
class MeetingNodeExtractor:
    def __init__(self, cfg, adjacency_cache):
        ...
    def forward(self, u_idx, v_idx):
        # returns M_indices: LongTensor
        ...
```

### ideas/idea1_meeting_node/config.yaml
单独跑这个 idea (其余 default):
```yaml
_base: ../../experiments/configs/A0_baseline.yaml
extractor:
  type: meeting_1_2hop
  params: {restrict_kinds: [Protein, Gene, SideEffect, ...]}
```

### ideas/idea1_meeting_node/README.md
- Idea 在干什么 (i2 的 meeting-node 锚定)
- 引入的 component (MeetingNodeExtractor)
- 对应 insight: `Notes/Settings/Insights/i2.md`
- 跑这个 idea: `bash experiments/run.sh ideas/idea1_meeting_node/config.yaml`
- 跑结果链接 (放到对应 `Notes/Log/...` 目录)

## 5. experiments/ — Cross-idea YAMLs

### experiments/configs/_base.yaml
```yaml
# Model-level defaults (slot 默认填随机/简单的)
init:       {type: random,    params: {dim: 128}}
encoder:    {type: gcn2,      params: {hidden: 128, depth: 2}}
extractor:  {type: none}
perception: {type: uniform}
splitter:   {type: none}
decoder:    {type: mlp,       params: {hidden: 64}}

# Training-level defaults
train:
  epochs: 25
  lr: 5e-3
  weight_decay: 1e-5
  batch_size: 16384
  negative_epochs: 5     # 每 N epoch 重采样 train negatives
```

### experiments/configs/A5_full.yaml
```yaml
_base: _base.yaml
init:       {type: pubmedbert,         params: {projection: pca, dim: 128}}      # idea4
extractor:  {type: meeting_1_2hop,     params: {restrict_kinds: [...]}}          # idea1
perception: {type: pair_cond_external, params: {hidden: 64, priors: [...]}}      # idea2
splitter:   {type: pkpd_kind,          params: {pk: [...], pd: [...]}}           # idea3
decoder:    {type: factorized,         params: {hidden: 64}}                     # idea3
```

### experiments/run.sh
```bash
#!/bin/bash
CONFIG=$1
SEED=${2:-42}
SETTING=${3:-S2}
DATA_ROOT=${DATA_ROOT:-/path/to/coldddi/release}

# 把我们的 model config 路径传给 perception_ddi baseline (通过 env 或 CLI arg)
PERCEPTION_CONFIG=$CONFIG python -m coldddi.evaluate \
  --method perception_ddi \
  --data $DATA_ROOT \
  --seed $SEED \
  --setting $SETTING \
  --out runs/$(basename $CONFIG .yaml)_seed${SEED}/
```

**整个 train / eval / negative resampling / multi-split (S0/S1/S2) 全部 coldddi 框架负责**.

## 6. 如何把 perception_ddi 注册到 coldddi

两种实现路径:

### Option A: 直接 import (推荐)
我们的 `Code/my_code/perception_ddi/__init__.py` 顶上:
```python
from coldddi.baselines.base import register, BaselineModel
```
然后我们的 `PerceptionDDIBaseline` 用 `@register("perception_ddi")`. ColdDDI 的 `evaluate.py` 通过 `--method perception_ddi` 就能 dispatch 到我们.

要求: **coldddi 包必须在 PYTHONPATH 里**. 通过 `pip install -e D:/My-Research/03-Projects/ColdDDI/Code-Released-Formal` 安装即可.

### Option B: 把我们的 baseline 提交进 ColdDDI 仓库
作为 `coldddi.baselines.perception_ddi.*`. 这需要改 ColdDDI 仓库, 不推荐 (跨项目耦合).

**走 Option A**, 跨仓库零耦合, ColdDDI 当 dependency.

## 7. 跑流程 (端到端)

```bash
# 1. install coldddi as editable
pip install -e D:/My-Research/03-Projects/ColdDDI/Code-Released-Formal

# 2. 在 Semantic-Path-Aware-DDI-Cold-Start 项目里跑 A5
cd D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Code
bash my_code/experiments/run.sh my_code/experiments/configs/A5_full.yaml 42 S2

# 3. 输出 (coldddi evaluate 标准):
# runs/A5_full_seed42/
#   ├── manifest.json
#   ├── model state files
#   ├── results_S2.json       # AUC/AUPR/F1/NLL on test_S2
#   └── train.log
```

## 8. 新 idea 加入流程 (举例 idea5 = "molecular fingerprint side-channel")

1. `mkdir my_code/ideas/idea5_mol_fp`
2. 写 `component.py`: 新 slot 或新 component 类 + `@register_slot(...)`
3. 写 `config.yaml`: `_base: A5_full.yaml`, override 相关 slot
4. 写 `README.md`: 描述 + insight link
5. `bash run.sh my_code/ideas/idea5_mol_fp/config.yaml 42 S2`

**改动只在新 idea folder 内**.

## 9. 跟 v2 的具体 simplification

| v2 | v3 (vs v2 进一步简化) |
|---|---|
| 自写 `core/data.py` `core/trainer.py` `core/evaluator.py` `core/run.py` | **全部不写**, 用 `coldddi.data` `coldddi.evaluate` |
| 自管 cache (PubMedBERT / Morgan FP) | 各 component 自己 lazy load + 在 component 内 cache; 不再 core-level |
| 入口 `core/run.py` | 直接 `coldddi.evaluate --method perception_ddi --config ...` |
| 自己实现 negative sampling | 用 `train.get_train_negatives(epoch)` |
| 自己实现 7-split eval | coldddi evaluate 已支持 S0/S1/S2/all |

净文件减少: v2 是 ~40 file, v3 是 ~30 file. infrastructure 几乎全靠 coldddi.

## 10. 跟 v1 / v2 理论的接口 (不变)

每个 idea 的 component 仍可在 `README.md` 里写:
> This realizes Theorem X (v2 nominal sigma-field sufficiency) by ...

理论文档跟 code 通过 README 交叉引用, 不放在 code 里.

## 11. P0 实施顺序 (修订, 总~3 day)

1. **(½ day)** 验证能否在 our project 内 import `coldddi.*` (装好, 试 `from coldddi.data import PairDataset` 跑通 toy example)
2. **(½ day)** 写 `perception_ddi/baseline.py` 骨架 (fit/predict_proba/save/load 框架, 内部先用空 model — 验证能被 coldddi evaluate dispatch 到)
3. **(½ day)** 写 `perception_ddi/slot_registry.py` + `model.py` 6-slot composition
4. **(½ day)** 写 6 个 defaults/*.py — 全 trivial, A0 baseline 跑得通
5. **(½ day)** 跑通 `A0_baseline` end-to-end via `coldddi evaluate`, AUC 跟 E3 random-init baseline 对得上 (sanity)
6. **(½ day)** 加 idea4 (PubMedBERT init) 验证组件机制 — 跟 E3 PubMedBERT-init baseline AUC 对得上
7. **逐 idea 加入** — idea1 / idea2 / idea3 各 ~半 day

P0 总~3 day, 之后每 idea ~半 day, 全 9 个 ablation 跑下来约 1 周.

## 12. 不引入 v3 vs v2 没必要的改动

- 仍保留 6 个 slot (init / encoder / extractor / perception / splitter / decoder)
- 仍每 idea 一个 folder, 3 file
- 仍 cross-idea 组合通过 experiments/configs/*.yaml
- 唯一变化: data/eval/train pipeline 全部 outsource 给 coldddi, 我们的 baseline 是 coldddi 注册的一个 plugin

## 13. 你选 / 调整

(选项 A) 接受 v3, 我开始 P0 第 1 步 — 测试 `import coldddi` 在我们 project 工作 (装 dependency / 跑 1-pair toy fit-predict)

(选项 B) 想看具体某个 file 的 detailed spec (e.g., `perception_ddi/baseline.py` 的 fit 函数怎么写, slot_registry 装饰器怎么实现)

(选项 C) 想调整 layout (例如 觉得 ideas/ 应该改成别的命名, 或 slot 数量要变, 或不想跑 coldddi 当 dependency)

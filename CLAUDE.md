# Project Rules

科研 ML 项目模板。代码文档见 `Code/docs/`（1-Overview 到 13-Log）。
研究笔记见 `Notes/`（项目内，同时被 Obsidian vault 索引）。

本项目应 clone 到 Obsidian vault 的 `03-Projects/` 下使用，
使 Obsidian 能搜索和链接项目笔记。

## 禁止编造事实（最最重要，违反一次都不能容忍）

**核心规则**：任何**具体数字 / 引文 / paper 报告值 / 表格数据 / 文件路径 /
代码行号 / 函数签名 / API 行为 / 库版本 / 数据集统计量**，引用之前**必须
亲自打开源头 verify**——Read 文件、Grep 代码、读 PDF、跑命令。**不许凭
记忆或印象说**。

**为什么"最最重要"**：凭印象说出来的"paper 报告 X 左右"、"通常是 Y"、
"这个 API 大概是 Z 用法"，会让用户基于**错误前提**做关键决策（停训练、
改代码、用错库），是直接的伤害行为。比"代码改错"更恶劣，因为它伪装成
事实在引导决策。

**已发生过的反模式（必须看一眼引以为戒）**：
- ❌ "EmerGNN 在 cold-start S2 上 F1 ~0.85+" → 真实是 25.0±2.8，差 60 个百
  分点；基于这数字误导用户停训练，浪费了用户的注意力 + 信任
- 这是**真实发生在本项目的事件**。被用户骂"你扯淡吧"应当视为最高优先
  级警告，立刻 audit 自己刚才所有陈述里哪些是凭印象说的

**强制流程**：

1. 任何要引用具体值之前，先用 Read / Grep / WebFetch / Bash 拿到原始证据
2. 引用时**附 file:line 或 paper page/Table/行号**，让用户能复核
3. 没查的时候**必须用"我没查 / 我不确定 / 让我去翻一下"开头**，不许伪装
   成肯定语气
4. 如果发现自己刚说错了，**立刻明确承认错误 + 给出 verify 后的正确值**，
   不许装没看见 / 找借口 / 顺着错的方向继续说
5. 给推荐 / 让用户做决定（停训练、改超参、删文件）之前，**必须先 verify
   推荐所基于的事实**——基于错误事实的推荐是双重伤害

**典型场景对照**：

| 不许说 | 必须说 |
|---|---|
| "paper 报告 AUC 大概 0.85" | "让我翻 Table 1"（然后 Read PDF / Grep 提取文本） |
| "torch.cat 已被 deprecated" | "我去查一下 PyTorch 文档" |
| "这个函数在 line 142 定义" | Grep 拿到精确行号再说 |
| "通常 batch_size=128 是默认" | Read argparse 定义看默认值 |
| "我的复现已 PASS"（没跑过真数据） | "代码层 PASS、真数据未 verify" |

**这条规则覆盖所有其他规则**——即使在执行别的任务（修复 bug、写代码、
review），只要要引用具体事实，都必须先 verify。

## 代码修改规则（最重要）

- **禁止删除或重构**已有函数/类/文件，除非我明确说「请重构 X」或「请删除 X」
- 修改任何文件前，**先用一句话描述你要改什么**，等我确认后再动手
- 不得自动合并/简化两个功能相似的函数
- 新增功能**优先新建文件**，不改动已有逻辑
- 不得更改已有函数的签名（参数顺序、参数名、返回值类型）
- 不得移动文件或重命名模块，除非我明确要求

## 命名规范

- 变量 / 函数：`snake_case`（如 `compute_loss`, `load_dataset`）
- 类名：`PascalCase`（如 `GCNMethod`, `BaseDatasetBuilder`）
- 常量：`UPPER_SNAKE_CASE`（如 `HAS_PYG`, `DEFAULT_SEED`）
- 配置文件：与 `Code/configs/` 四层结构对应（`data/`, `model/`, `training/`, `runs/`）
- 实验标识：`Code/runs/<method_name>/<exp_name>/<timestamp>/`

## 目录结构

详细目录结构见 `Code/docs/2-Repo_Layout.md`。关键约定：

```
project_root/                    ← 所有命令从这里运行（cwd = 项目根目录）
├─ Code/                         # 所有代码
│  ├─ my_code/                   # 所有源码
│  │  ├─ pipeline/               # 编排逻辑（run.py, registry.py）
│  │  ├─ models/                 # 模型结构（每个模型一个子文件夹，实现 Method 接口）
│  │  ├─ data/                   # 数据加载、预处理、特征工程
│  │  ├─ train/                  # 训练循环（trainer.py, callbacks.py）
│  │  ├─ predict/                # 推理（CSV 流式写入）
│  │  ├─ eval/                   # 评测（evaluator.py, metrics.py）
│  │  ├─ io/                     # 路径管理、产物写盘、日志
│  │  └─ utils/                  # 可复用工具（config, seeding, types）
│  ├─ configs/                   # YAML 配置（data/ model/ training/ runs/）
│  ├─ data/                      # 数据集（只读）
│  ├─ runs/                      # 实验输出（自动生成，不手动修改）
│  ├─ scripts/                   # CLI 入口（所有可运行脚本必须在此目录，train.py 等）
│  ├─ docs/                      # 代码文档
│  └─ requirements.txt
├─ Notes/                        # Obsidian vault（研究笔记，不放脚本，被 vault 索引）
│  ├─ Literature/                # 文献笔记
│  ├─ Ideas/                     # 改进思路、实验想法（仅 .md 文件）
│  └─ Log/                       # 实验日志、会议记录
├─ Paper/                        # 参考文献 PDF（.gitignore 忽略 PDF）
├─ Notebooks/                    # Jupyter notebooks
├─ CLAUDE.md
└─ README.md
```

## 运行规范

- **cwd 永远是项目根目录**，所有命令都从这里执行
- 运行脚本的标准格式：`python Code/scripts/<script_name>.py [args]`
- **禁止在 Notes/、Notebooks/ 等非 Code/ 目录放置 .py 脚本**
- 新脚本必须放在 `Code/scripts/`，命名规范：`<action>_<target>.py`
  - 例：`train_gnn.py`, `run_ablation.py`, `analyze_results.py`, `prepare_data.py`
- WSL 环境运行：`wsl bash -ic "conda activate <env> && cd <project_root_wsl_path> && python Code/scripts/<script>.py"`

## 脚本命名规范

| 前缀 | 用途 | 示例 |
|------|------|------|
| `train_` | 训练模型 | `train_stgnn.py`, `train_sklearn.py` |
| `run_` | 运行完整实验流程 | `run_ablation.py`, `run_baselines.py` |
| `prepare_` | 数据预处理/生成 | `prepare_data.py`, `prepare_splits.py` |
| `analyze_` | 分析结果 | `analyze_failure.py`, `analyze_ceiling.py` |
| `eval_` | 评估模型 | `eval_predictions.py` |
| `export_` | 导出结果/图表 | `export_figures.py` |

## Notes/ 使用说明

- `Notes/` 是 Obsidian vault，**仅放 .md 文件**，不放 .py 脚本
- `Notes/` 存放本项目的研究笔记，随项目 git 同步
- 项目应 clone 到 Obsidian vault 的 `03-Projects/<project_name>/` 下，这样 Obsidian 能直接搜索和链接笔记
- Claude Code 可以读取 `Notes/` 下所有 markdown 笔记，并参与讨论
- 可用 `[[双向链接]]` 关联 vault 中其他项目或论文笔记

### 5 阶段研究工作流（用 vault Templater 模板生成）

新项目第一步：在 Obsidian 里跑 `TPL-Proj-Init` → 覆盖顶层 `README.md` 为 dashboard（同时注册到 vault 的 `HOME.md` 和 `03-Projects/index.md` 进度表）。

之后每个阶段都用对应模板，弹一次"One-line TL;DR"，模板会自动落到正确子文件夹：

| 阶段 | 模板 | 落位 |
|---|---|---|
| 1. Insight / Purpose | `TPL-Proj-Insight` | `Notes/Settings/Insights/` |
| 2. Setting（锁定） | `TPL-Proj-Setting` | `Notes/Settings/Setting.md` |
| 3. Paper Review MOC | `TPL-Proj-PaperReview` | `Notes/Settings/Paper-Review/Review-MOC.md` |
| 3. 单篇 paper 简评 | `TPL-Proj-PaperNote` | `Notes/Settings/Paper-Review/` |
| 4. Dataset / Baselines | 无模板（按 context 自定，可写进 `Setting.md` 末尾） | — |
| 5. Idea + 高层迭代日志 | `TPL-Proj-Idea` | `Notes/Ideas/` |
| 5.1. 单次实验详细日志 | `TPL-Proj-ExpLog` | `Notes/Experiments/` |

每个模板写的 TL;DR 一句话会自动出现在项目 dashboard 的 Dataview 表里，外面一眼能看到当前进度。

## 架构约束

- 新模型必须实现 **Method Protocol**：`setup()`, `train_step()`, `eval_step()`, `state_dict()`, `load_state_dict()`
- 新模型在 `Code/my_code/models/<name>/` 下创建子文件夹，然后在 `Code/my_code/models/__init__.py` 用 `register("method_name")(MethodClass)` 注册
- Batch 遵循 dict-like 合约：必须有 `x` key，可选 `y`, `train_mask`/`val_mask`/`test_mask`
- 配置合并逻辑在 `Code/my_code/utils/config.py`，不要在其他地方重复实现

## 实验管理

- 每次实验必须有独立的 config（或通过 `--set` 覆盖参数）
- 结果写入 `Code/runs/<run_id>/`，**不覆盖旧结果**
- 新模型变体应创建新的 Method 类，不修改已有 Method

### 日志收集规范（重要）

所有 baseline / experiment 跑出来的 log 必须按统一规则归档，方便之后批量整理与对比。

**1. Run ID 统一标识符**

每次 run 用一个 `run_id` 标识，格式：
```
<timestamp>__<script>__<tag>__seed<N>
e.g.  2026-05-14_18-30-21__run_baseline__emergnn_merged_e5__seed42
```

**2. 每 run 的产物目录**

```
Code/runs/<run_id>/
├─ results.json              # 主结果 (AUC/NLL/F1/config/data stats)
├─ train.log                 # 该次 run 的完整 stdout+stderr
├─ predictions_s0.parquet    # 可选: per-pair eval predictions
├─ predictions_s1.parquet
├─ predictions_s2.parquet
└─ model/                    # 可选: 模型 state
```

**3. 全局 log 收集目录**

所有 run 的 log + 一行 summary 同时写到 `Code/runs/_logs/`:
```
Code/runs/_logs/
├─ index.csv                 # 所有 run 的 metadata + 主指标一行一行追加
├─ <run_id>.log              # 每个 run 的 stdout+stderr 镜像 (mirror of train.log)
└─ ...
```

`index.csv` 字段（追加，不覆盖）:
```
run_id, timestamp, script, tag, baseline, kg_source, seed, epochs,
fit_time_s, auc_s0, auc_s1, auc_s2, nll_s2, status, run_dir
```

**4. 实现责任**

- 任何 `Code/scripts/*.py` 跑实验的脚本必须:
  - 用 `make_run_id(script_name, tag, seed)` 生成 run_id
  - 创建 `Code/runs/<run_id>/`, 把 results.json / predictions / model 写进去
  - 把 stdout+stderr 同时镜像写到 `Code/runs/<run_id>/train.log` **和** `Code/runs/_logs/<run_id>.log`
  - 跑完后追加一行到 `Code/runs/_logs/index.csv`
- 共享工具放 `Code/my_code/utils/run_logger.py` (或类似), 各脚本调用统一接口
- 即使脚本崩了, 也要把已收集的部分 log 写到 `_logs/<run_id>.log` (try/finally 包住)

**5. 检索约定**

- 整理结果一律从 `Code/runs/_logs/index.csv` 开始 (一次 grep 拿全)
- 单独看某 run 的 log: `cat Code/runs/_logs/<run_id>.log`
- 重训某 run 但不污染历史: 新 run_id (新 timestamp), 旧的不动

### 训练进度日志规范（重要）

**任何训练循环必须打 loss 到 log**, 静默循环禁止. 两种粒度都要:

**Per-step (高频, 看曲线/早期发散)**:
- 间隔 `log_step_every` 个 step 打一次, **`log_step_every` 是可调超参数** (默认 50, 通过 CLI 传入)
- 格式: `[ep <epoch>/<total_epochs> step <step>/<total_steps>] loss=<mean_last_N>`
- loss 用最近 N 步的滚动均值, 不打 raw 单步 (太抖)

**Per-epoch (低频, 看趋势)**:
- 每个 epoch 末必打一次, 不可调
- 格式: `[ep <epoch>/<total_epochs>] mean_loss=<X> time=<Ys>` + 可选 val metric

**实现责任**:
- 共享 helper 在 `Code/my_code/utils/train_progress.py`, 提供 `TrainProgress` 类
- 任何 baseline / 新 model 的 `fit()` 内循环必须用 `TrainProgress.step(loss_value)` + `TrainProgress.epoch_end(extra=...)`
- 模型类 `__init__` 必须收 `log_step_every: int = 50` 参数, 透传给 `TrainProgress`
- `Code/scripts/run_baseline.py` 等入口脚本必须提供 `--log-step-every N` CLI flag, 默认 50

**反面例子** (禁止):
```python
for epoch in range(self.n_epochs):  # silent, 没法监控
    for batch in dataloader:
        ...  # no print
```

**正面例子**:
```python
from my_code.utils.train_progress import TrainProgress

prog = TrainProgress(self.n_epochs, log_step_every=self.log_step_every)
for epoch in range(self.n_epochs):
    prog.epoch_start(epoch)
    for batch in dataloader:
        loss = ...
        prog.step(loss.item())
    prog.epoch_end(extra={"val_auc": val_auc} if val_auc else None)
```

### 训练中间 eval / 中间 save 规范

参照 HuggingFace `Trainer` 的接口风格. 任何训练循环要支持以下 6 个超参 (CLI 透传):

| 超参 | 取值 | 默认 | 说明 |
|---|---|---|---|
| `eval_strategy` | `"no"` / `"epoch"` / `"steps"` | `"epoch"` | 训练中间何时跑 val eval |
| `eval_steps` | int | `500` | 当 `eval_strategy="steps"` 时, 每 N step 跑一次 |
| `save_strategy` | `"no"` / `"epoch"` / `"steps"` | `"no"` | 训练中间何时存 checkpoint (跟最终 save 不冲突) |
| `save_steps` | int | `500` | 当 `save_strategy="steps"` 时, 每 N step 存一次 |
| `save_total_limit` | int | `3` | 中间 checkpoint 上限, 超过则删最旧 |
| `load_best_model_at_end` | bool | `True` | 训练完用 val 最佳 epoch 的 state 加载回来 |

**实现**: `TrainProgress` 提供 `should_eval_step()` / `should_eval_epoch()` / `should_save_step()` / `should_save_epoch()` 决策方法; baseline `fit()` 自己拥有 eval 函数和 save 函数, 但 *是否触发* 由 progress 控制. **Eval 结果必须 print 到 log** (用 `progress.log_eval(metrics, scope)`):

```
[<baseline>] [ep 5/50 step 250/2160] loss=0.4123
[<baseline>] [eval @ ep 5 step 250] val_auc=0.7214 val_nll=0.5832
[<baseline>] [save @ ep 5 step 500] checkpoint-0500/
[<baseline>] [ep 5/50] mean_loss=0.4012 time=58.3s val_auc=0.7214
```

**中间 checkpoint 位置**: `<run_dir>/checkpoints/checkpoint-<step>/` (baseline 自己的 `.save()` schema). 超过 `save_total_limit` 自动删最旧 (按数字大小排序).

**`load_best_model_at_end`** 仅当 eval 跑过 ≥ 1 次时生效. 跟踪 val_auc 最佳 epoch/step 的 state, 训练结束加载回来.

**反面例子** (禁止):
```python
for epoch in range(self.n_epochs):
    ...
    if val is not None:
        auc = self._validate(val)  # 静默计算, 不 print
        if auc > best:
            best_state = ...
```

**正面例子**:
```python
for epoch in range(self.n_epochs):
    prog.epoch_start(epoch)
    for batch in dataloader:
        loss = ...
        prog.step(loss.item())
        if prog.should_eval_step():
            metrics = self._eval(val)  # returns {"val_auc": ..., "val_nll": ...}
            prog.log_eval(metrics, scope="step")
            best_state = update_best(metrics, best_state)
        if prog.should_save_step():
            self.save_checkpoint(prog.step_count)
    if prog.should_eval_epoch():
        metrics = self._eval(val)
        prog.log_eval(metrics, scope="epoch")
        best_state = update_best(metrics, best_state)
    if prog.should_save_epoch():
        self.save_checkpoint(epoch=epoch+1)
    prog.epoch_end(extra=metrics if metrics else None)
```

### 复现代码 (Reproduction) 规范（重要，必须遵循）

本节是 `Code/reproductions/<MethodName>/` 下所有复现工作的**唯一权威规范**——
文件夹布局、命名约定、工作流程都以此为准。其他章节出现冲突时以本节为准。
baseline 派生有独立规范 (见 §"从 reproduction 派生 baseline" 系列章节)。

#### 1. 文件夹布局（强制）

```
Code/reproductions/<MethodName>/
├── _results/                       ← 复现结果汇总: 单/多 fold/seed 跑出的数字,
│                                     是否符合原论文 Table N 报告值
├── _reviews/                       ← agent 写代码 / review 产出的 review 报告
├── _Original-Dataset/              ← 跟上游 paper / github 一一对应的数据集
│   ├── <raw dataset files>         ← 例: ddis.csv / drug_smiles.csv /
│   │                                       cold_start/ / warm_start/
│   └── necessary/                  ← 模型必需的辅助产物 (KG / 分子图 / cache)
│       ├── <name>__official.<ext>      官方 ship 的版本 (github 直接下载)
│       ├── <name>__mine.<ext>          我们 builder 重建的版本
│       └── build_<name>.py             我们自己实现的构造代码 (基于 paper +
│                                       github 反推, 用于日后迁移到本项目数据)
├── _paper-and-GitHub/              ← 原文 + 上游仓库归档
│   ├── *.pdf                       ← 原始论文全文 PDF
│   ├── _paper_text.txt             ← (可选) PDF utf-8 提取, 便于 grep
│   ├── github.txt                  ← 上游 github URL, 单行 plain URL
│   └── <full-git-clone>/           ← 完整 git clone 快照, 仅供阅读
└── <reproduction-code>.py          ← 我们写的复现代码 (从 git clone 迁移到本项目环境;
                                     严禁 import _paper-and-GitHub/<git-clone>/ 下任何文件)
```

**关键命名约定**:
- 所有 "项目元数据 / agent 产物 / 上游一一对应资料" 的子文件夹都用下划线前缀:
  `_results/`, `_reviews/`, `_Original-Dataset/`, `_paper-and-GitHub/`
  - 下划线 → 不被当作 Python package, 不会被 `__init__.py` import
  - 视觉上一眼区分 "工具/元资料" vs "我们写的复现代码"
- `_Original-Dataset/` 默认入 `.gitignore` (大文件 + 版权)
- 我们写的复现代码 `.py` 直接平铺在 reproduction root

#### 2. 必需文件命名 (`__official` / `__mine`)

`_Original-Dataset/necessary/` 下的辅助产物 (pkl/npz/json/...) **必须**用后缀
显式区分来源:

```
<base>__official.<ext>   ← 上游 ship 的版本 (github 下载 / paper supplement)
<base>__mine.<ext>       ← 我们 builder 重建的版本
```

**复现端读取优先级** (当两者都存在):
1. **优先 `__official`** → load + log `[<name>] USING OFFICIAL: <path>`
2. `__official` 缺 + `__mine` 存在 → load + log `USING MINE (official missing): <path>`
3. 两者都缺 → **报错并提示用户**, **复现端不要 auto-build** (见 §3 step 6)

#### 3. 工作流程 (6 步, 严格按顺序)

| Step | 动作 | 失败时 |
|---|---|---|
| 1 | 检查 `_paper-and-GitHub/` 下是否有 `*.pdf` 和完整 git clone | 没有则**停止并 warn**, 不继续 |
| 1' | PDF 转 txt → `_paper_text.txt`, 读 paper 全文 + git clone 全文 | — |
| 2 | 根据阅读结果确定该 baseline 需要哪些数据 + 必须辅助文件, 全部下载到 `_Original-Dataset/` (和 `_Original-Dataset/necessary/<name>__official.*`) | 数据缺无法继续 |
| 3 | 实现复现代码 (放 reproduction root). **Codex 多轮 review**, 至少**无 critical 错误** | review 不过关回到 step 3 |
| 4 | 给用户**单 seed CLI 命令**, 跑一次, 检查复现质量 (理想 ±2pt 内符合 paper 报告值) | 数字偏大需解释 / 调试 |
| 5 | 实现 `_Original-Dataset/necessary/build_<name>.py` (反推 paper + github 算法), 跑一次跟 `__official` 对比是否一致, 写 report 到 `_reviews/`; 如果算法实在反推不出, 在 `_reviews/` 里明确说明原因 | — |
| 6 | **复现端 pipeline 不需要 detect-and-generate 必须文件的逻辑** (只需 detect → load → 报错); detect-and-auto-build 是**baseline 端**的事 | — |

**step 3 review 操作**:
- 读 paper PDF **全文** (不只 abstract / methods 段落)
- **第一优先级**: 从 paper title + abstract + intro contributions 段显式提取
  **3-5 个核心 contribution / novelty claim**, 列成清单, 排在 review 报告
  最前面。每条:
  - 给出 paper 原文引用 (page/section)
  - 列对应的 code 位置 (file:line)
  - 做**端到端实测验证** (trace 看核心机制真的在数据上生效, 不只是看调用链)
  - 标 ✅/⚠/❌
- 通过 `github.txt` 拿到 repo URL, **逐文件** WebFetch 仓库所有 `.py` / `.ipynb` /
  `.sh`, 训练 / 测试 / 评估 / 数据处理全部要看
- 仔细对比每个复现文件 vs 原始仓库对应文件
- paper 的关键数学定义 / 评估协议 / 超参 sweep 范围必须跟实现交叉验证
- 任何"我没看 X"或"我没跑 X"的部分必须在 review 报告里 explicit 标出来
- review 报告分三类列不一致: paper-vs-impl / repo-vs-impl / paper-vs-repo

**禁止反模式** (review 时):
- ❌ 用通用 GNN/Transformer/RL review 模板拼装 checklist
  → 把 paper-specific 核心 contribution 稀释成普通一项
- ❌ 只看代码调用链对得上 (`func_X` 被调用了) 就声明 "verified"
  → 必须实际 trace 数据流, 看核心模块在真实输入上产生预期输出
- ❌ 拍脑袋按"架构/超参/loss/训练流程"通用维度组织 checklist
  → 应该先回答 "这篇 paper 的核心 contribution 是什么", 从那里反推
- ❌ 只看 abstract / README 就声明 "verified faithful"
- ❌ 在 review 前未访问 `_paper-and-GitHub/` 目录
- ❌ 把 "看起来对" 当 "verified"

#### 4. HDN-DDI 实例 (canonical example)

```
Code/reproductions/HDN-DDI/
├── _results/2026-05-17__fold0_faithful.md       ← fold 0 实测 vs paper Table 3
├── _reviews/2026-05-17__paper_faithful.md       ← contribution-first review
├── _Original-Dataset/
│   ├── ddis.csv, drug_smiles.csv                ← raw dataset
│   ├── cold_start/, warm_start/                 ← splits
│   └── necessary/
│       ├── id_data_dict__official.pkl              ← 上游 ship (40.5 MB)
│       ├── id_data_dict__mine.pkl                  ← 我们 builder 产物 (sanity)
│       └── build_hierarchical_pkl.py               ← 我们的 builder 代码
├── _paper-and-GitHub/
│   ├── Sun and Zheng 2025 ...pdf                ← 全文 PDF
│   ├── _paper_text.txt
│   ├── github.txt                               ← github URL (单行)
│   └── jcsun-00__HDN-DDI/                       ← git clone 快照
├── _pkl_loader.py                               ← detect+load (优先 official)
├── data_preprocessing.py                        ← 复现代码 (byte-exact port)
├── models.py / layers.py / custom_loss.py
├── run_faithful.py                              ← CLI 入口
├── run_3fold.py                                 ← fold 0/1/2 wrapper
└── README.md
```

#### 5. 严禁

- ❌ 不读 paper 全文就声明 "verified faithful" (只看 abstract / README)
- ❌ paper 数据放到 `Code/data/` 顶层 (那是项目自有数据空间, 跨 baseline 共享)
- ❌ paper 数据放到 `Code/reproductions/<Name>/data/` 或 `dataset/` (路径不统一)
- ❌ 复现代码 import `_paper-and-GitHub/<git-clone>/` 下任何文件 (失去独立性)
- ❌ `_Original-Dataset/` 入 git (大文件 + 版权)
- ❌ 复现 pipeline 里 auto-build 必须文件 (复现端只 detect → load → 缺则报错;
   auto-build 只在 baseline 侧)
- ❌ 两个版本必需文件都存在时默认用 mine (必须默认 official)
- ❌ 走 pipeline 不 log 用的是 official 还是 mine
- ❌ 跑 baseline 时 subprocess 调复现端 builder (跨目录依赖, 见 baseline 规范)

### Baseline 规范（重要，必须遵循）

本节是 `Code/baseline/<method_name>/` 下所有 baseline 代码的**唯一权威规范**——
文件夹布局、命名约定、工作流程都以此为准。其他章节出现冲突时以本节为准。
跟 §"复现代码 (Reproduction) 规范" 配套: reproduction 是 paper-faithful
参考, baseline 把同样算法搬到本项目数据集上, 两侧通过文件系统解耦, 严禁 import 互调。

#### 1. 文件夹布局（强制）

```
Code/baseline/<method_name>/
├── _results/                       ← 跑我们数据集的结果汇总 (单/多 seed 数字)
├── _reviews/                       ← agent code review 报告
├── _data/                          ← 本 baseline 专属的辅助文件 (KG / 分子图 cache /
│   │                                 ...). **项目自有 dataset (drug 池 / DDI triple
│   │                                 / KG raw) 不放这里, 统一放 `Code/data/`,
│   │                                 baseline 从那调用即可**.
│   └── necessary/                  ← 模型必需的辅助产物
│       ├── <name>__mine.<ext>          我们 builder 重建的版本 (跑 baseline 必需输入)
│       │                               一般 baseline 侧**只有 __mine**, 因为我们的数据集
│       │                               没有上游 official 对应版本; 如果存在两者, 命名 +
│       │                               读取优先级跟 reproduction 侧一致 (见 §复现规范 §2)
│       └── build_<name>.py             我们的构造代码 (从 reproduction 侧
│                                       `_Original-Dataset/necessary/build_*.py`
│                                       **复制** 过来, 独立维护; 见 §2 step 2)
├── <task1>/                        ← 任务子文件夹 (e.g. binary_cls/, multi_cls/,
│   ├── __init__.py                   multi_label_cls/, ...)
│   └── baseline.py                   每个任务一个独立 BaselineModel 实现
├── <task2>/
│   └── ...
├── __init__.py                     ← 顶层 re-export 所有 task 变体
├── models.py / layers.py / ...     ← 跨 task 共享的 model / layer / featurizer
└── _shared.py                      ← (可选) 共享工具 (e.g. detect-and-build pipeline)
```

**关键约定**:
- 项目元数据子文件夹用下划线前缀: `_results/`, `_reviews/`, `_data/`
  - 下划线 → 不被当 Python package
  - 视觉上一眼区分 "工具/元资料" vs "task 代码 / 共享代码"
- baseline 必需文件 (pkl/npz/cache) 命名一律用 `__mine` 后缀, 即使
  没有 `__official` 对应物——这是显式标记 "这是 OUR builder 产物,
  不要去找 official 版本"

#### 2. 工作流程 (4 步, 严格按顺序)

| Step | 动作 | 失败时 |
|---|---|---|
| 1 | 参考 reproduction 实现 task-specific + 通用代码. **Codex 多轮 review**, 至少**无 critical 错误** | review 不过关回到 step 1 |
| 2 | 把 reproduction 的 `_Original-Dataset/necessary/build_*.py` **复制** (非剪切) 到 `baseline/<method>/_data/necessary/`. 两份独立维护 (§文件级独立性) | — |
| 3 | 在 baseline 通用代码 (一般是 `_shared.py` 或类似) 实现 **detect-and-generate** 逻辑: `检测文件名 → 不存在 → 进入生成流程 → 落盘`; `检测文件名 → 存在 → 加载 → go on` | — |
| 4 | 给用户**单 seed CLI 命令**, 跑一次, 检查在我们数据集上的效果 | 数字异常需调试 |

**step 2 "复制不剪切" 的原因**:
- reproduction 端 builder 是 paper-faithful, 不允许动 (§文件级独立性)
- baseline 端 builder 可能因为我们数据集的 schema 不同需要小改 (e.g. 不同的
  id column 名称); 但初始**必须 byte-identical**, drift 时写 test 防漂移
- 物理上是两份文件, 不允许 baseline subprocess 跨目录调 reproduction 的脚本

**step 3 baseline detect-and-generate 区别于 reproduction**:
- reproduction 端 detection: missing 则**报错** (不 auto-build)
- baseline 端 detection: missing / stale / corrupt 则**subprocess auto-build**
- 原因: reproduction 必需文件是上游 ship 的, 缺失说明用户没下载; baseline
  必需文件本来就是我们自己 build 的, 缺失是正常状态 (首次跑 / 换数据集),
  auto-build 是合理 contract

#### 3. 任务子文件夹约定

每个 task (binary_cls / multi_cls / multi_label_cls / ...) 一个**独立子文件夹**,
不允许平铺多个 task 的 baseline.py 在同一层。

**强制约束**:
1. 任务区分以**子文件夹**为单位, 不允许 `baseline_multiclass.py` 这种文件名后缀平铺
2. 跨 task 共享的代码 (model / layers / featurizer / KG builder) 留**顶层**,
   子文件夹只放 task-specific 训练 / 推理逻辑
3. 顶层 `__init__.py` 必须 re-export 所有 task 变体, 外部 `from
   baseline.<method> import <ClassName>` 不受子文件夹结构变化影响
4. 子文件夹 `__init__.py` 只 re-export 本 task 下的类
5. 多个 paper-faithful 程度变体在同一 task 下 (如 flat vs BRICS-aware) 仍可
   平铺到同一 task 子文件夹下 (e.g. `binary_cls/baseline.py` +
   `binary_cls/baseline_brics.py`)

**反例**:
```
baseline/<method>/
├── baseline.py             ← ❌ binary 在顶层
├── baseline_multiclass.py  ← ❌ multi-class 在顶层
└── model_multiclass.py     ← ❌ multi-class 专属 model 在顶层
```

→ 应改为:
```
baseline/<method>/
├── models.py               ← 共享
├── layers.py               ← 共享
├── binary_cls/baseline.py
└── multi_cls/
    ├── baseline.py
    └── model.py
```

#### 4. 派生时允许 vs 禁止的偏差 (跑我们数据集时的"paper-faithful 边界")

baseline 跑我们数据集, 在 paper-faithful 之外有些必要改动 (数据 schema /
任务头 / 评测协议), 但**算法核心**必须忠实复现。

**case A: baseline task 跟 paper 原始 task 一致** (e.g. paper 做 multi-class,
我们也做):

| 维度 | reproduction (paper) | baseline (我们数据集) | 允许偏差 |
|---|---|---|---|
| 数据源 | paper 数据 schema | 项目数据 schema | ✅ |
| 评测 split | paper 设置 | 项目 cold-start (S0/S1/S2) | ✅ |
| 数据集 vocab 大小 (K, \|V\|) | paper 数 | 跟我们数据走 | ✅ |
| model 架构 | paper code | 同 paper, 跟 reproduction 同源算法 | ❌ |
| Loss 公式 + reduction (sum/mean) | paper 公式 | 必须等价 | ❌ |
| Optimizer + lr + weight_decay | paper hyperopt 最佳 | 必须一致 | ❌ |
| LR scheduler | paper 设置 | 必须一致 | ❌ |
| 训练协议 (per-epoch 数据采样 / KG 重采样等) | paper 核心 | 必须一致 | ❌ |
| Batch size | paper 调优值 | 必须一致 (否则等价 lr 漂移) | ❌ |
| Best ckpt 选择标准 | paper 主指标 | 必须用 paper 主指标 | ❌ |
| Eval 频率 | paper 设置 | 可不同 (计算开销考虑) | ✅ |

**case B: baseline task 是 paper 没做的任务** (e.g. paper 做 multi-class, 我们补 binary):

允许且应当改造: 输出头维度 / Loss 函数族 (BCE vs CE) / 负样本采样 / 主指标 (AUC vs F1)

**不允许借口 "新任务" 省略 paper 算法核心**:
- 不能说 "我们 binary 不需要 shuffle_train" → shuffle_train 是 paper 的
  inductive 训练范式, 新任务同样受益
- 不能说 "我们 binary 不需要 LR scheduler" → 这是优化器设置, 任务无关

**典型反例 (EmerGNN multi_cls 第一版犯过)**:
- ❌ 缺 `shuffle_train` → 从 paper 训练范式退化成 transductive, cold-start 数字明显低于 paper 实现
- ❌ Loss `reduction='mean'` (应 `sum`) → 等价于隐式 lr 缩小 batch_size 倍
- ❌ 缺 `ReduceLROnPlateau` → 后期收敛差异
- ❌ best ckpt 选 `top1_acc` (应 `macro_f1`) → 与 paper 主指标不一致

**典型反例 (HDN-DDI binary_cls 第一版犯过)**:
- ❌ 没用 builder, 在 `baseline/hdn_ddi/mol_features.py` 写 "independent
  re-implementation", 每次 fit() on-the-fly BRICS 分解
- ❌ 项目数据 val_auc 0.64, paper Ds2 AUC 0.96 → 差 32 pt; **根因**: 应该用
  step 2 复制过来的 builder, 不该自己再写一份

#### 5. HDN-DDI / EmerGNN 实例

```
Code/baseline/hdn_ddi/                       Code/baseline/emergnn/
├── _results/                                ├── _results/
├── _reviews/                                ├── _reviews/
├── _data/                                   ├── _data/  (待 align)
│   └── necessary/                           │   └── necessary/
│       ├── hdn_ddi_mol_graphs__mine.pkl     │       └── <kg cache>__mine.pkl
│       └── build_hierarchical_pkl.py        ├── binary_cls/baseline.py
├── binary_cls/baseline.py                   └── multi_cls/baseline.py
├── multi_cls/baseline.py + model.py
├── _shared.py  (detect + autobuild)
└── models.py / layers.py / mol_features.py
```

> ⚠ HDN-DDI baseline 已对齐本 spec (2026-05-17 完成). 其他 baseline
> (`emergnn` / `tiger` / `dsn_ddi` / `ssi_ddi`) 还没创建 `_data/necessary/`
> 子文件夹; 用到必须文件 (KG cache 等) 的需要按本节 spec 迁移。

#### 6. 严禁

- ❌ baseline 跨目录 import / subprocess 调用 `reproductions/` 下的代码 / builder
  (复制独立维护, 不是借用; 否则 silent drift)
- ❌ baseline `_data/` 散在项目共享 `Code/data/` 顶层 (`Code/data/` 是项目
  自有 dataset 空间, baseline-specific 产物不该混进去)
- ❌ baseline 必需 pkl 没用 `__mine` 后缀 (即使没 `__official` 对应物也要挂)
- ❌ baseline 不带 detect-and-generate, 要求用户手动跑 builder 才能 train
  (说明 baseline 入口 contract 不完整)
- ❌ 用 "为了 baseline 自给自足, 我重写了一份 X" 当理由——重写 ≠ build via builder
- ❌ 多个 task 的 baseline.py 平铺同一层 (必须建子文件夹)
- ❌ 拍脑袋 "为了简化省略 paper 的 X 步骤" (paper 算法细节不允许私自简化)
- ❌ 任务无关优化器细节 (scheduler / weight_decay / loss reduction) 当
  "项目自由发挥" 去改

#### 必需文件 detection pipeline（**非对称**: reproduction vs baseline）

复现端和 baseline 端 detection pipeline 的语义**不同** (这是 §"复现代码规范"
step 6 定下的)。此处只做对照表, 详细行为见各端规范:

| 端 | detection 行为 | 落位 |
|---|---|---|
| reproduction | detect → 优先 `__official` → fallback `__mine` → 都缺则**报错** (不 auto-build) | `<repro>/_pkl_loader.py` 或同名 loader |
| baseline | detect → 缺 / stale / corrupt → **subprocess build 本目录 builder** → load | 见即将到来的 baseline 规范 |

**两端共同要求**:
- 路径检测用 `Path.is_file()` 显式判断, 不要靠 `open()` 抛 FileNotFoundError
- 走 pipeline 必须 print 一行 stderr 标识用哪个文件 / 走哪个分支:
  - reproduction: `USING OFFICIAL: <path>` 或 `USING MINE (official missing): <path>`
  - baseline: `cache HIT / STALE rebuild / BUILT`
- Builder 跨目录 subprocess **永远禁止** (§文件级独立性: 同名文件可并存,
  sha 不同; 各侧维护独立副本; subprocess 跨目录借用脚本会造成 silent drift)
- 多候选源命名必须用 `__official` / `__mine` 后缀 (规则见 §"复现代码规范" §2)

**HDN-DDI 实例**:
```
reproductions/HDN-DDI/                       baseline/hdn_ddi/
├── _pkl_loader.py                           ├── _shared.py
│  (detect+load, 缺则报错)                   │   (ensure_mol_graphs_pkl:
└── _Original-Dataset/necessary/             │    detect+autobuild via subprocess)
    ├── id_data_dict__official.pkl           ├── build_hierarchical_pkl.py
    ├── id_data_dict__mine.pkl               │   (独立 builder copy)
    └── build_hierarchical_pkl.py            └── data/
        (独立 builder copy)                       └── hdn_ddi_mol_graphs__mine.pkl
```

**反例 (本项目真实发生过)**:
- ❌ `baseline/hdn_ddi/_shared.py` 用 `subprocess.run([..., "Code/reproductions/.../build_hierarchical_pkl.py", ...])`
  跨目录调用 reproduction 侧 builder 生成项目 pkl —— silent drift 风险, 已修
- ❌ reproduction 端 `data_preprocessing.py` 在 import 时直接 `pkl.load(open(...))`,
  缺 pkl 时连 detection log 都没有就 FileNotFoundError —— 已修, 改走 `_pkl_loader`

#### Code review 报告归档规范（重要）

任何按上面规范做完一次完整 code review 后, **必须把 review 报告落盘**,
不能只留在聊天 / 对话窗口里。

**归档位置**: co-located with the code being reviewed.

```
<reviewed-path>/_reviews/<YYYY-MM-DD>__<scope>.md
```

例:
```
Code/baseline/hdn_ddi/_reviews/2026-05-17__baseline.md
Code/reproductions/HDN-DDI/_reviews/2026-05-17__paper_faithful.md
Code/baseline/emergnn/_reviews/2026-05-21__multi_cls__rerun.md
```

**为什么 co-located 不放 Notes/**:
- 跟着代码走, 移动代码时 review 历史一起移动
- 阅读代码时一眼能看到历史 review 结论, 不用跨目录跳转
- review 引用的 file:line 跟仓库实际行号同步演化时, 至少在同一个版本快照里

**`_reviews/` 目录约定**:
- 下划线前缀 → 不是 Python package, 不会被 `__init__.py` import
- 每次 review 一个 .md 文件, **不覆盖历史** (类似 `Code/runs/<run_id>/`
  的不覆盖语义), 文件名带日期可以累积演化
- 同一天多次 review 加后缀 (`...__round2.md`)

**报告必含字段** (对应 review 规范的检查项):
1. **元信息**:
   - 日期
   - **Primary reviewer**: 谁起草这份报告 (e.g. `Claude (sonnet-4.5)`)
   - **Independent reviewer**: 第二意见来源 (e.g. `codex (gpt-5-codex)`)
     —— **必填**, review 规范要求 codex 独立 review 作为最后 gate
   - 触发方式 ("用户要求 review X")
2. **已读 vs 未读清单**: paper 章节 + github 文件 + 项目内文件
3. **paper 核心 contribution 抽取**: 3-5 条, 带 paper 原文引用
4. **逐 contribution 实测验证**: 每条带 file:line + 实测 trace 数据
5. **CLAUDE.md 各规范条款对照**: file 独立性 / layout / 训练规范等
6. **Codex 独立 verdict** (必填章节): 复制 codex 原话, 不要总结改写
7. **发现的 issue + 已修内容**: 每条 file:line + 修复 commit (如果已 commit)
8. **未决 issue / future work**

**Reviewer 字段填法 (避免误导)**:
- ❌ `Reviewer: Claude (sonnet)` 单字段 → 看不出第二意见来自哪里, 也看不出
  到底有没有走过 codex 独立 gate
- ✅ 拆成两个字段写 (primary = 起草报告的 agent, independent = 独立 verdict 的 agent)
- 如果某次 review 跳过了 codex (e.g. 用户指定"快速 review"), 必须显式写
  `Independent reviewer: 未调用 (原因: ...)`, 不能空着假装走过

**禁止**:
- ❌ review 只在对话里说一遍, 不落盘 (信息会丢)
- ❌ review 报告放在 git 不追踪的临时位置 (`/tmp/`, 桌面)
- ❌ 把多次 review 累加到同一文件覆盖历史 (违反"不覆盖历史"原则)
- ❌ Primary / Independent reviewer 混在一个字段, 让读者分不清第二意见来源

#### Reproduction / experiment 结果归档规范（重要）

review 和 result 是**两种不同 artifact**, 必须分目录存放, 不能混在一起,
否则后续 grep / 时间线整理时分不开"这是审稿意见"还是"这是实验数字"。

**归档位置**: co-located with the code being reproduced, 但放在独立的
`_results/` 目录下。

```
<reproduced-path>/_results/<YYYY-MM-DD>__<scope>.md
```

例:
```
Code/reproductions/HDN-DDI/_results/2026-05-17__fold0_faithful.md
Code/reproductions/HDN-DDI/_results/2026-05-20__fold1_faithful.md
Code/reproductions/HDN-DDI/_results/2026-05-21__3fold_mean.md
Code/baseline/emergnn/_results/2026-05-25__binary_cls__seed42.md
```

**目录约定** (与 `_reviews/` 完全平行):
- 下划线前缀 → 不是 Python package
- 每次跑出一组结果一个 .md, **不覆盖历史** (跟 `Code/runs/<run_id>/` 同语义)
- 单 fold / 单 seed 一个文件, 多 fold / 多 seed 聚合时再写一个汇总文件
  (filename 用 `__3fold_mean` / `__multiseed` 后缀区分)

**报告必含字段**:
1. **Run identity**: run_id, run_dir 路径, artifacts 清单 (results.json /
   train.log / 权重), wall time
2. **Config**: 关键超参表 (从 train.log 复制, 不要事后回忆)
3. **Data stats**: split 大小, 数据集 / fold 标识
4. **Final metrics**: 自己跑的数字 (单 fold / 单 seed 标清楚)
5. **vs paper 对比**: paper 原始数字 + Δ + 是否在容忍区间; paper 数字必须给
   出处 (table N / 文件 line 号)
6. **Verdict + justification**: PASS / FAIL / PASS-with-caveats; 如果是
   PASS-with-caveats 必须列出 caveat (e.g. 只跑了 1 fold)
7. **Caveats**: 已知的限制 (单 fold 没有 std / early-stop 触发点 / 命名歧义等)
8. **Reproduce 命令**: 一行能复跑的 CLI 命令

**`_reviews/` vs `_results/` 该往哪放**:
| 内容 | 归档目录 |
|---|---|
| 代码 review 报告 (paper 对齐 / contribution 验证 / codex verdict) | `_reviews/` |
| 实验结果汇报 (跑了什么 / 跑出多少 / 跟 paper 差多少) | `_results/` |
| review + result 混合 (e.g. "review 完顺便跑了一遍") | 拆成两个文件, 各自归档 |

**结果该挂在 `reproductions/` 还是 `baseline/` 下** (按**数据集**判, 不按代码目录判):

| 用的数据集 | 归档位置 | 语义 |
|---|---|---|
| **上游原始数据集** (paper 自带 pkl / csv, e.g. HDN-DDI 的 1706-drug + 86-rel pkl) | `reproductions/<Method>/_results/` | 验证我们能 reproduce paper Table N 的数字 |
| **本项目自己构造的数据集** (e.g. 我们的 1900-drug DrugBank) | `baseline/<method>/_results/` | 测 baseline 在我们 setting 下的真实水平, 用于跟我们方法对比 |
| 本项目数据集 + reproductions/ 的 byte-exact 代码 (临时 sanity check) | `baseline/<method>/_results/` (按"数据集"归类) | 罕见; 文件标题必须标"用 reproductions/ 代码跑我们数据集" |

**判断规则一句话**: 看跑出来的数字**用来证明什么**。

- 想证明"我们的复现忠实于上游" → 数据集必须用上游原始的, 结果归 `reproductions/`
- 想证明"baseline 在我们 setting 下打多少分" → 数据集必须用我们自己的, 结果归 `baseline/`

**典型反例**:
- ❌ 在 `reproductions/HDN-DDI/_results/` 放"用 paper 代码跑我们 1900-drug 数据集"
  的结果 → 应去 `baseline/hdn_ddi/_results/`, 因为这不是 reproduce paper, 是测
  baseline 在我们 setting 下的表现
- ❌ 在 `baseline/hdn_ddi/_results/` 放"用 baseline 代码跑上游 86-rel 原始数据集"
  的结果 → 应去 `reproductions/HDN-DDI/_results/`, 因为这是 reproduction 验证

**禁止**:
- ❌ 结果文件放在代码目录顶层 (e.g. `REPRODUCTION_RESULTS.md`),
  跟 `README.md` / `*.py` 混在一起
- ❌ 把单 fold 结果跟多 fold 聚合结果写进同一文件覆盖更新
- ❌ 引用 paper 数字但不给章节 / line 出处
- ❌ 凭代码所在目录决定结果归档位置 (必须看**数据集**是上游的还是我们自己的)

#### 文件级独立性（重要）

派生时还必须保证两个目录的**文件物理独立**（语义可以一致, 文件不可共享）：

1. **reproductions/ 是 source of truth, 派生过程严禁修改**
   - 任何"修复 baseline 实现"的工作都不能 touch reproductions/ 下的文件
   - 不能把 reproductions/ 的核心模块改成 baseline 共用 (e.g. 提取公共模块
     然后双方 import) —— 这会让 reproductions/ 不再是上游 byte-exact 的 mirror

2. **跨目录引用全面禁止**
   - ❌ `from reproductions.<Method> import ...` 在 baseline/ 下
   - ❌ `from baseline.<method> import ...` 在 reproductions/ 下
   - ❌ symlink / hardlink 跨目录共享文件
   - 同名文件 (layers.py, models.py 等) 可以并存, 但 sha 必须不同
     (各自独立维护副本, 走各自项目 namespace import)

3. **派生 = 在 baseline/ 里独立 copy + 改造**
   - 直接 copy reproductions/ 的代码到 baseline/, 然后按"派生 review 规范"
     的"允许偏差/禁止偏差"列表做必要改造
   - 不要 import 复用, 不要 symlink 借用

4. **遵守"禁止删除/重构已有文件"**
   - baseline/ 旧版即便有 bug (e.g. 没用 paper 的核心 contribution), 也不直接改
   - 新增 `<file>_<variant>.py` (如 `mol_features_brics.py`, `baseline_brics.py`),
     通过 `__init__.py` 用 `register("<method>_<variant>")` 注册新变体跟旧的并存
   - 旧 baseline 保留, docstring 加 deprecation note 指向新版

5. **drift 监控**
   - 周期性 `sha1sum baseline/<method>/*.py reproductions/<Method>/*.py` 检查
   - 名义上 "functionally 等价" 的核心模块 (e.g. `models.py`, `layers.py`),
     如果两边语义出现不一致, 写一个 test (e.g. `tests/test_<method>_drift.py`)
     断言关键 forward pass 结果一致, 防止悄悄 drift

**典型反例**:
- ❌ baseline/hdn_ddi/ 直接 `from reproductions.HDN-DDI.data_preprocessing import DrugDataset`
  → 一旦 reproductions/ 因为对齐 upstream 改动, baseline 跟着挂掉
- ❌ 把 reproductions/HDN-DDI/models.py 改成两边共用版本
  → 破坏 reproductions/ 跟 paper byte-exact 的地位
- ❌ 在 baseline/hdn_ddi/baseline.py 直接改成 BRICS-aware
  → 违反"禁止重构已有文件", 应该新建 baseline_brics.py 并存

## Git 版本控制

- 只用 `main` 分支，保持 main 始终可运行
- 本地开发 → push → 服务器 pull → 跑实验；服务器上尽量不改代码
- 服务器上的 `Code/runs/` 结果不入 git，需要时手动拷回本地分析
- Commit message 使用前缀：`feat:` 新功能 / `fix:` 修复 / `data:` 数据处理 / `exp:` 实验配置 / `docs:` 文档 / `refactor:` 重构
- 不得 `git push --force` 到 main
- 提交前检查：不提交 `.env`、密钥、大文件（PDF/数据集/权重）
- `Code/data/`、`Code/runs/`、`Paper/*.pdf` 已在 `.gitignore` 中

## 禁止行为

- 不得删除任何 `.py` 文件
- 不得修改 `Code/data/` 目录下的任何文件（只读）
- 不得修改 `Code/runs/` 下已有的实验结果
- 不得修改 `Code/docs/` 下的文档，除非我明确要求
- 不得修改 `Notes/` 下已有笔记的内容，除非我明确要求
- 不得在 `Code/my_code/utils/` 中放入依赖实验参数的代码
- 不得引入新的全局可变状态
- **不得在 Notes/ 或项目根目录放置 .py 脚本**

## 代码风格

- 保持现有代码风格一致：`from __future__ import annotations` 在文件头部
- 模块级 docstring 简洁描述用途
- type hints 用于函数签名
- 导入顺序：stdlib → third-party → local

## Local Environment

- **OS**: Windows 11 + **WSL2**
- **GPU**: NVIDIA (CUDA-enabled), 通过 WSL2 暴露
- **Conda env**: `project_1` (Python 3.x, **PyTorch with CUDA**)
- **WSL project root**: `/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start`
- Code dir: 项目根目录（所有命令从这里执行）
- wandb: false
- code_sync: git

### 🚨 强制规则 (违反一次都不能容忍): 所有 Python 命令都用 WSL conda env

**任何**执行 Python 的命令 (训练 / 评测 / 诊断脚本 / 临时 one-liner /
import smoke test / 任何 `python -c "..."` / 任何 `python xxx.py`) **必须**
通过 WSL conda env 跑, 绝对不能用 agent 默认的 Windows Python。

**正确格式** (从项目根目录):
```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/<script>.py [args]"
```

**为什么这条规则不能省**:
- Agent 默认 Windows Python 是 `torch 2.8.0+cpu` (**没 CUDA**)
- WSL conda env `project_1` 才是 **GPU 版本**
- 用错环境的后果:
  - 训练 / 评测脚本 silently 跑在 CPU 上, 100 epoch 从 ~50 min 变成 ~10 小时
  - 部分库 (e.g. emergnn 的 propagation kernel) 在 Windows CPU 下 segfault
  - 数字可能没问题但训练永远跑不完, agent 还以为是在 GPU 上

**已发生的事故** (2026-05-20):
- Agent 用 Windows Python 启了 emergnn-kgonly 和 emergnn-multimode 两个任务
- multimode 立即 segfault (exit 139)
- kgonly silently 在 CPU 上 step 50/3359 of ep 1/100 龟速跑 (估算总时长 ~10 小时, 不是 50 min)
- 浪费用户时间 + 计算资源

**违反这条规则的检测方式**:
- 任何 `python -c "import torch; print(torch.cuda.is_available())"` 返回 `False`
  → 当前不在 GPU env, 立刻 stop 重新用 wsl bash 调
- 任何训练 task 跑了 5+ min 还没出 epoch 1 完整 log → 99% 是 CPU silent fallback
- 任何 Python segfault (exit 139) → 多半是 Windows Python 跑了 cuda-required 库

**Agent 习惯检查清单** (每次要跑 Python 之前 ask 自己):
1. 这个命令需要 GPU 吗? (训练 / 评测 / 重型 forward 都需要)
2. 我现在的 Bash tool 默认是 Windows native? 还是 WSL?
3. 如果是 Windows native → 必须 wrap 成 `wsl bash -ic "conda activate project_1 && cd <wsl_path> && ..."`
4. 路径要从 `D:\...` 转成 `/mnt/d/...`

**小 utility 不需要 GPU 的也强烈建议走 WSL conda env**, 保持 import 版本一致, 避免 numpy/pandas/torch 跨 env 行为不一致。

## Remote Server

（待配置 — 有远程服务器时在此添加 SSH、conda env、code dir 等信息）

- SSH: (例: `ssh my-gpu-server`)
- GPU: (例: 4x A100 80GB)
- Conda env: (例: `research`)
- Code dir: (例: `/home/user/project/`)
- code_sync: rsync | git

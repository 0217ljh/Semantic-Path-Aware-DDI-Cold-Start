# Project Rules

科研 ML 项目模板。代码文档见 `Code/docs/`（1-Overview 到 13-Log）。
研究笔记见 `Notes/`（项目内，同时被 Obsidian vault 索引）。

本项目应 clone 到 Obsidian vault 的 `03-Projects/` 下使用，
使 Obsidian 能搜索和链接项目笔记。

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
- 结果写入 `Code/runs/<method>/<exp_name>/<timestamp>/`，**不覆盖旧结果**
- 新模型变体应创建新的 Method 类，不修改已有 Method

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

（在此填写项目的本地环境信息）

- GPU: (例: NVIDIA RTX 5080 (16GB), CUDA 12.8)
- OS: (例: Windows 11 + WSL2)
- Conda env: (例: `my_env` (Python 3.11, PyTorch))
- Code dir: 项目根目录（所有命令从这里执行）
- 运行命令: `wsl bash -ic "conda activate <env> && cd <project_root_wsl_path> && python Code/scripts/<script>.py"`
- wandb: false
- code_sync: git

## Remote Server

（待配置 — 有远程服务器时在此添加 SSH、conda env、code dir 等信息）

- SSH: (例: `ssh my-gpu-server`)
- GPU: (例: 4x A100 80GB)
- Conda env: (例: `research`)
- Code dir: (例: `/home/user/project/`)
- code_sync: rsync | git

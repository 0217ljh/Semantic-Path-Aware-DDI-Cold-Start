# Project Template

可复用的科研 ML 项目框架：数据处理 → 训练 → 预测 → 评测，支持 main 与 baselines 统一入口。

## 项目结构

```
project_root/
├─ Paper/          # 参考文献 PDF
├─ Code/           # 所有代码、配置、数据、实验输出
├─ Notebooks/      # Jupyter notebooks
├─ Notes/          # 研究笔记（Obsidian vault）
└─ CLAUDE.md       # Claude Code 项目指令
```

- **Paper/** — 存放参考文献 PDF（git 忽略 PDF 文件）
- **Code/** — 源码、配置、数据集、实验结果（运行代码时 cwd 应为 `Code/`）
- **Notebooks/** — Jupyter notebooks，用于数据探索和实验
- **Notes/** — 用 Obsidian 打开此文件夹，管理文献笔记、改进思路、实验日志

## 快速开始：Cora + GCN 节点分类

### 1. 安装依赖

```bash
pip install -r Code/requirements.txt
# 若使用 PyTorch Geometric，需与当前 PyTorch 版本匹配，见 https://pytorch-geometric.readthedocs.io/
```

### 2. 一键训练 + 评测

在 `Code/` 目录下执行：

```bash
cd Code
python scripts/train.py --run cora_gcn --mode train
```

### 3. 输出位置

- 运行目录：`Code/runs/main/cora_gcn/<timestamp>/`
- 产物：`meta/resolved_config.yaml`、`ckpt/`、`metrics/`、`preds/`、`logs/`

详见 [Code/docs/2-Repo_Layout](Code/docs/2-Repo_Layout.md)、[Code/docs/6-Init_Path](Code/docs/6-Init_Path.md)。

## 代码与流程分布

| 步骤 | 位置 | 说明 |
|-----|------|------|
| CLI | `Code/scripts/train.py` | 解析 `--run` / `--data`+`--model`+`--training`、`--mode`、`--set` |
| 入口 | `Code/my_code/pipeline/run.py` | 编排：load_cfg → init_run_paths → 数据 → 模型 → train/predict → evaluate |
| 配置 | `Code/my_code/utils/config.py` | `load_cfg(args)` 合并配置与 `--set` 覆盖 |
| 路径 | `Code/my_code/io/run_paths.py` | `init_run_paths(cfg)` → `runs/<method>/<exp_name>/<timestamp>/` |
| 模型 | `Code/my_code/models/` | 每个模型一个子文件夹，实现 Method Protocol |
| 训练 | `Code/my_code/train/trainer.py` | `train(cfg, loaders, method, paths)` |
| 预测 | `Code/my_code/predict/predictor.py` | `predict(...)` → 写 preds CSV |
| 评测 | `Code/my_code/eval/evaluator.py` | `evaluate(cfg, predict_output, paths)` → metrics JSON |

## 文档

见 [Code/docs/1-Overview](Code/docs/1-Overview.md) 与文档索引。

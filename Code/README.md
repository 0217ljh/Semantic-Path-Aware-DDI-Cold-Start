# Code — Quickstart & Layout

可复用的科研 ML 项目代码框架：数据处理 → 训练 → 预测 → 评测，支持 main 与 baselines 统一入口。

## 快速开始：Cora + GCN 节点分类

### 1. 安装依赖

```bash
pip install -r requirements.txt
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

详见 [docs/2-Repo_Layout](docs/2-Repo_Layout.md)、[docs/6-Init_Path](docs/6-Init_Path.md)。

## 代码与流程分布

| 步骤 | 位置 | 说明 |
|-----|------|------|
| CLI | `scripts/train.py` | 解析 `--run` / `--data`+`--model`+`--training`、`--mode`、`--set` |
| 入口 | `my_code/pipeline/run.py` | 编排：load_cfg → init_run_paths → 数据 → 模型 → train/predict → evaluate |
| 配置 | `my_code/utils/config.py` | `load_cfg(args)` 合并配置与 `--set` 覆盖 |
| 路径 | `my_code/io/run_paths.py` | `init_run_paths(cfg)` → `runs/<method>/<exp_name>/<timestamp>/` |
| 模型 | `my_code/models/` | 每个模型一个子文件夹，实现 Method Protocol |
| 训练 | `my_code/train/trainer.py` | `train(cfg, loaders, method, paths)` |
| 预测 | `my_code/predict/predictor.py` | `predict(...)` → 写 preds CSV |
| 评测 | `my_code/eval/evaluator.py` | `evaluate(cfg, predict_output, paths)` → metrics JSON |

## 文档

见 [docs/1-Overview](docs/1-Overview.md) 与文档索引。

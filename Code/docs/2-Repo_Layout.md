# Repo Layout

```text
project_root/
├─ README.md
├─ requirements.txt                  # 或 environment.yml（二选一）
├─ .gitignore
│
├─ docs/                             # 不展开
│
├─ configs/
│  ├─ data/
│  │  ├─ main_branch_1.yaml
│  │  ├─ main_branch_2.yaml
│  │  ├─ baseline_x.yaml
│  │  └─ baseline_y.yaml
│  │
│  ├─ model/
│  │  ├─ main_branch_1.yaml
│  │  ├─ main_branch_2.yaml
│  │  ├─ baseline_x.yaml
│  │  └─ baseline_y.yaml
│  │
│  ├─ training/
│  │  ├─ main_branch_1.yaml
│  │  ├─ main_branch_2.yaml
│  │  ├─ baseline_x.yaml
│  │  └─ baseline_y.yaml
│  │
│  └─ runs/                          # 可选：记录三块配置的组合（复现实验更方便）
│     ├─ exp_main_branch1.yaml
│     ├─ exp_main_branch2.yaml
│     ├─ exp_baseline_x.yaml
│     └─ exp_baseline_y.yaml
│
├─ data/                             # 方案 B：每个数据集自包含（raw->processed 统一一次）
│  ├─ dataset_a/
│  │  ├─ README.md
│  │  ├─ raw/
│  │  └─ processed/
│  │     ├─ manifest.json
│  │     └─ ... processed artifacts ...
│  │
│  ├─ dataset_b/
│  │  ├─ README.md
│  │  ├─ raw/
│  │  └─ processed/
│  │     ├─ manifest.json
│  │     └─ ...
│  │
│  └─ dataset_c/
│     ├─ README.md
│     ├─ raw/
│     └─ processed/
│        ├─ manifest.json
│        └─ ...
│
├─ runs/                             # 统一实验输出（见 6-Init_Path：method 分支 + exp_name + timestamp）
│  ├─ main/
│  │  └─ <exp_name>/<timestamp>/    # 例：exp_lr001/20250119-143022/
│  │     ├─ meta/
│  │     │  ├─ resolved_config.yaml  # 合并后的 cfg 快照
│  │     │  ├─ cmd.txt               # 完整 CLI 命令
│  │     │  └─ meta.json             # 运行时信息（mode, ranks, source_ckpt 等）
│  │     ├─ ckpt/                    # 仅 train 模式写入
│  │     │  ├─ last.pt
│  │     │  └─ best.pt
│  │     ├─ metrics/
│  │     │  ├─ train.jsonl           # 可选：训练步/epoch 日志
│  │     │  ├─ val.json
│  │     │  └─ test.json
│  │     ├─ preds/                   # 推理记录（11-Prediction：CSV 流式写入）
│  │     │  ├─ val.csv
│  │     │  └─ test.csv
│  │     └─ logs/
│  │        └─ stdout_rank0.log
│  │
│  └─ baselines/
│     ├─ baseline_x/
│     │  └─ <exp_name>/<timestamp>/...
│     └─ baseline_y/
│        └─ <exp_name>/<timestamp>/...
│
│
├─ code/
│  ├─ pipeline/
│  │  ├─ run.py                      # 统一入口：load_dataset->dataset/loader->train->eval
│  │  ├─ load_dataset.py             # 通用第一步：ensure processed（create or load）
│  │  └─ registry.py                 # main/baseline 的注册与加载
│  │
│  ├─ data/
│  │  ├─ preprocess/
│  │  │  └─ build.py                 # raw -> processed（项目级统一，只做一次）
│  │  ├─ dataset.py                  # processed -> Torch Dataset / IterableDataset
│  │  ├─ collate.py                  # collate_fn（可选）
│  │  └─ dataloaders.py              # build_loaders(...)
│  │
│  ├─ methods/
│  │  ├─ base.py                     # fit + load_from_checkpoint + predict(_batch)
│  │  ├─ main/
│  │  │  ├─ model.py
│  │  │  ├─ features.py              # 可选：读取 processed 后的额外组合/缓存
│  │  │  └─ __init__.py
│  │  └─ baselines/
│  │     ├─ baseline_x/
│  │     │  ├─ model.py
│  │     │  ├─ features.py           # 可选：读取 processed 后的额外组合/缓存
│  │     │  └─ __init__.py
│  │     └─ baseline_y/...
│  │
│  ├─ train/
│  │  ├─ trainer.py                  # 统一训练入口（通用）
│  │  └─ callbacks.py                # 可选：ckpt/log/earlystop 等 hooks
│  │
│  ├─ eval/
│  │  ├─ evaluator.py                # 统一评测入口（通用）
│  │  └─ metrics.py
│  │
│  ├─ io/
│  │  ├─ run_paths.py                # 生成 runs/main/... 或 runs/baselines/... 路径
│  │  ├─ artifacts.py                # 约定输出文件：ckpt/metrics/log/config
│  │  └─ logging.py
│  │
│  └─ utils/
│     ├─ config.py                   # 合并 data/model/training cfg + overrides
│     ├─ seeding.py                  # 复现相关（seed/determinism）
│     ├─ types.py                    # Sample/Batch 等通用类型
│     ├─ paths.py                    # 可选：文件/路径小工具
│     └─ misc.py                     # 可选：少量通用函数（避免变垃圾桶）
│
├─ tests/                            # 可选但建议：保障框架不漂移
│  ├─ test_dataset_contract.py
│  ├─ test_method_contract.py
│  └─ test_artifacts.py
│
└─ scripts/
   ├─ train.py                       # CLI：--run 或 --data/--model/--training，--mode train|predict（4-scripts）
   ├─ eval.py
   └─ report.py                      # 可选：汇总 runs
```

---

## 与各文档的对应

| 路径/模块 | 文档 |
|----------|------|
| configs/ | [5-Config](5-Config.md)（data/model/training/runs 合并与 --set） |
| data/（raw + processed/<data_key>） | [7-GetOrLoadData](7-get_or_load_data.md)；resources/ 见 [8-Feature_Process](8-Feature_Process.md) |
| runs/ 布局（method 分支 + exp_name + timestamp） | [6-Init_Path](6-Init_Path.md)（meta/ckpt/metrics/preds/logs） |
| code/pipeline/run.py | [3-Pipeline](3-Pipeline.md) |
| code/data/（dataset, collate, dataloaders） | [9-DatasetAndCollate](9-Torch_dataset_collate.md)（R4/R5） |
| code/methods/ | [10-Training](10-Training.md) method 接口；registry 见 [2-Repo_Layout](2-Repo_Layout.md) |
| code/train/ | [10-Training](10-Training.md) |
| code/predict/ | [11-Prediction](11-Prediction.md)（preds 为 **CSV** 流式写入） |
| code/eval/ | [12-Evaluation](12-Evaluation.md) |
| code/io/run_paths.py | [6-Init_Path](6-Init_Path.md) |
| code/utils/config.py | [5-Config](5-Config.md) |

**说明：** 产物写盘由 pipeline 调用 `io/*`、`train/*`、`predict/*`、`eval/*` 完成，scripts 只解析参数并调用 `pipeline/run.py`。preds 格式以 [11-Prediction](11-Prediction.md) 为准（CSV）；metrics 为 JSON（见 6-Init_Path §6.5.4）。

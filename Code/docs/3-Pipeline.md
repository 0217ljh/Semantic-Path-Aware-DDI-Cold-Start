```mermaid
flowchart TD
  %% 外部入口
  A["scripts/train.py / eval.py<br/>(Parse CLI args)"]
  B["pipeline/run.py<br/>(Entry Point)"]

  %% 子图：run.py 内部编排
  subgraph RUN["Inside pipeline/run.py (Orchestration)"]
    direction TB

    R1["load_cfg(args)<br/>(utils/config.py)"]
    R2["init_run_paths(cfg)<br/>(io/run_paths.py)"]

    R2_5["compute_data_key(cfg)<br/>data_key := dataset + split_spec + split_seed + preprocess_version"]

    R3{"Base processed exists?<br/>check: data/&lt;dataset&gt;/processed/&lt;data_key&gt;/"}
    R3Y["Load base.pkl<br/>(data, splits, extra)"]
    R3N["Build base processed<br/>raw/* → processed/&lt;data_key&gt;/<br/>save base.pkl + DONE"]
    R3M["Base processed ready"]

    R3_6["optional: ensure resources caches<br/>processed/&lt;data_key&gt;/resources/&lt;method_key&gt;/<br/>if missing: build + save resource.pkl + DONE"]


    R4["build_dataset(cfg, processed + resources)<br/>(data/dataset.py)<br/><b>→ torch.utils.data.Dataset / IterableDataset</b>"]
    R5["build_loaders(cfg, dataset)<br/>(data/dataloaders.py)<br/><b>→ DataLoader + collate_fn (batching)</b>"]

    R7{"Feature cache exists?<br/>(method_key bound to data_key)"}
    R7Y["Load feature cache"]
    R7N["Build feature cache<br/>(methods/*/features.py)<br/>save (example)"]

    R6["load_method(cfg.model.name)<br/>(pipeline/registry.py)"]

    %% 模式分支：train 则先训练得最佳模型路径，再与 predict-only 共用「预测」步骤
    M{"mode?"}
    T["train(...)<br/>(train/trainer.py)<br/><b>→ TrainOutput (best_ckpt_path, ...)</b>"]
    PRED["predict(cfg, loaders, method, paths)<br/>(predict/ predictor)<br/>load ckpt；在选定数据集 val/test 上 run → <b>PredictOutput</b>"]
    P["evaluate(...)<br/>(eval/evaluator.py)"]

    R10["save artifacts<br/>(io/artifacts.py + io/logging.py)"]

    %% 连线：train 路径返回最佳模型路径后，pipeline 用该路径做 predict；predict-only 直接用 cfg.ckpt
    R1 --> R2 --> R2_5 --> R3
    R3 -- "Yes" --> R3Y --> R3M
    R3 -- "No"  --> R3N --> R3M
    R3M --> R3_6 --> R4 --> R5 --> R7
    R7 -- "Yes" --> R7Y --> R6
    R7 -- "No"  --> R7N --> R6

    R6 --> M
    M -- "train" --> T --> PRED
    M -- "predict (no-train)" --> PRED
    PRED --> P
    P --> R10
  end

  %% 外部到子图
  A -->|"args: --run OR --data/--model..."| B
  B --> R1

  ```

**Train 路径：** `train(...)` 返回 **TrainOutput**（含 `best_ckpt_path`）；pipeline 用该路径设置 `cfg["ckpt"]`，再调用 **predict** 在选定的 val/test 上跑预测，得到 **PredictOutput**，最后 **evaluate**。

**Predict-only 路径：** 不经过 training；直接用 `cfg["ckpt"]` 调用 **predict** 在选定的 val/test 上跑预测 → **PredictOutput** → **evaluate**。

---

## 与文档对应（核对用）

| 步骤 | 文档 | 说明 |
|------|------|------|
| A | [4-scripts](4-scripts.md) | scripts/train.py、eval.py 解析 CLI，调用 pipeline/run.py |
| R1 | [5-Config](5-Config.md) | load_cfg(args) @ utils/config.py |
| R2 | [6-Init_Path](6-Init_Path.md) | init_run_paths(cfg) @ io/run_paths.py |
| R2_5 | [7-GetOrLoadData](7-get_or_load_data.md)、required_config_keys | data_key = f(dataset, split_spec, split_seed, preprocess_version) |
| R3 | [7-GetOrLoadData](7-get_or_load_data.md) | 检查/加载/构建 base.pkl + DONE @ data/&lt;dataset&gt;/processed/&lt;data_key&gt;/ |
| R3_6 | [8-Feature_Process](8-Feature_Process.md) | 可选资源缓存 @ processed/&lt;data_key&gt;/resources/&lt;method_key&gt;/ |
| R4 | [9-DatasetAndCollate](9-Torch_dataset_collate.md) | build_dataset @ data/dataset.py |
| R5 | [9-DatasetAndCollate](9-Torch_dataset_collate.md) | build_loaders @ data/dataloaders.py |
| R7 | [8-Feature_Process](8-Feature_Process.md)（可选） | 特征缓存（method_key 绑定 data_key），methods/*/features.py |
| R6 | [2-Repo_Layout](2-Repo_Layout.md)、[10-Training](10-Training.md) | load_method(cfg.model.name) @ pipeline/registry.py |
| M | [5-Config](5-Config.md)、[4-scripts](4-scripts.md) | mode: train \| predict |
| T | [10-Training](10-Training.md) | train(...) @ train/trainer.py → TrainOutput |
| PRED | [11-Prediction](11-Prediction.md) | predict(...) @ predict/predictor.py → PredictOutput |
| P | [12-Evaluation](12-Evaluation.md) | evaluate(...) @ eval/evaluator.py |
| R10 | [6-Init_Path](6-Init_Path.md)、[2-Repo_Layout](2-Repo_Layout.md) | 写盘 ckpt/metrics/preds/logs/meta @ io/artifacts.py、logging.py |
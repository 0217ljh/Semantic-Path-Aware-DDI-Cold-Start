### 1. 背景

我希望维护一套可复用的项目基础框架，用于未来任何类型的 research/engineering project。该框架提供完整且稳定的端到端工作流（从数据处理到训练、预测与评测），并支持在同一工作流下运行主模型（main model）与一系列 baselines，从而保证实验组织一致、输出统一、对比方便、复现可靠。

### 2. 核心原则

* **工作流稳定：** 主流程（pipeline orchestrator）尽量不随项目/方法变化而改动。
* **数据自由：** 每个 project 的数据处理逻辑可以完全自定义，由我自己实现。
* **统一边界：** 所有 project 最终只需要满足一个统一接口——能够构建 PyTorch Dataset / DataLoader（含可选 collate_fn）并被训练/预测/评测流程消费。
* **任务无关：** 监督信号是可选的；框架支持 classification / regression / ranking / generation / unsupervised 等不同任务形态。
* **Dataset 形式无关：** 支持 map-style Dataset 与 IterableDataset；`__len__` 可选（当数据流式或不可知时）。
* **batch 最小契约：** batch 为 dict-like，至少包含 x（输入）；y（监督信号）与 meta（元信息）为可选字段。
* **方法可插拔：** 主模型与 baselines 都以插件方式接入（registry + 配置切换），同一入口运行与评测。
* **推理与评测分离：** 推理（predict）与指标计算（evaluate）为独立步骤；evaluate 只读预测结果（如 CSV），不依赖具体方法实现，便于统一对比与扩展指标。
* **产物统一：** 训练、预测与评测的输出目录结构与文件命名保持一致（ckpt/, metrics/, preds/, logs/, meta/），便于批量对比与自动化分析。

### 3. 目标

* **定义通用标准工作流：** prepare → dataset/loader → [train →] **predict** → **evaluate**（+ 可选 figures/report）。  
  - **两种运行模式：** `train`（先训练得最佳模型，再在 val/test 上预测并评测）与 `predict`（仅加载已有 checkpoint 在选定 split 上预测并评测）。
* 在同一工作流中支持：main model + baselines，并做到「一键切换方法」（改 config / `cfg["model"]["name"]`）。
* 通过稳定接口约束，确保不同方法不会各自发明一套输出/评测方式。
* 为长期复用提供规范：目录结构、运行入口、产物（ckpt / metrics / preds / log）组织方式。

### 4. 范围

本规范覆盖：

* 项目级标准工作流（阶段划分与职责边界，见 [3-Pipeline](3-Pipeline.md)）
* 最小接口契约（以 Torch Dataset/DataLoader 为核心边界；train/predict/evaluate 接口在对应文档定义）
* 实验产物与 run 目录的统一约定（[6-Init_Path](6-Init_Path.md)，用于复现/对比/归档）
* main model 与 baselines 的接入方式（[2-Repo_Layout](2-Repo_Layout.md) registry + 配置）

### 5. 非目标

* 不规定任何特定领域的数据 schema
* 不规定具体预处理实现细节（由每个 project 自己实现）
* 不限制特征存储格式（只要求最终能构建成可迭代的 Dataset）

### 6. 文档索引

| 文档 | 内容概要 |
|------|----------|
| [2-Repo_Layout](2-Repo_Layout.md) | 仓库结构、configs/data/runs/code、methods 与 registry |
| [3-Pipeline](3-Pipeline.md) | 端到端流程图：load_cfg → paths → data → dataset/loaders → method → train/predict → evaluate |
| [4-scripts](4-Scripts.md) | CLI 入口：train.py / eval.py，参数与 mode |
| [5-Config](5-Config.md) | load_cfg、合并 data/model/training、--set 覆盖 |
| [6-Init_Path](6-Init_Path.md) | init_run_paths、run_dir 布局（ckpt/, metrics/, preds/, logs/, meta/） |
| [7-GetOrLoadData](7-Get_or_Load_Data.md) | 基础 processed 缓存、data_key、global_seed |
| [8-Feature_Process](8-Feature_Process.md) | 资源缓存 R3_6、特征缓存 R7（可选） |
| [9-DatasetAndCollate](9-Torch_Dataset_And_Collate.md) | build_dataset / build_loaders、batch 契约 |
| [10-Training](10-Training.md) | train(...) 接口、TrainOutput、ckpt/metrics/log |
| [11-Prediction](11-Prediction.md) | predict(...) 接口、PredictOutput、流式 CSV、多卡聚合 |
| [12-Evaluation](12-Evaluation.md) | evaluate(...) 接口、EvalOutput、从 cfg 读指标列表、读 CSV 算指标 |
| [required_config_keys](required_config_keys.md) | 各模块所需 config key 汇总 |

### 7. 扩展性与研究多样性

本框架通过以下设计支持**泛用实现**与**长期研究的多样性**：

* **方法无关：** 训练与预测只依赖 method 的约定接口（setup, train_step, eval_step, load_state_dict 等），不关心具体模型结构或任务形态；新增方法只需实现接口并注册，无需改 pipeline。
* **指标可配置：** 评测阶段从 cfg 读取要计算的指标列表（如 `cfg["evaluation"]["metrics"]`），不同任务/项目可配置不同指标，evaluator 通用。
* **数据与资源可扩展：** 数据处理、可选资源缓存（R3_6）、可选特征缓存（R7）均由 project 实现；pipeline 只做「存在则加载、否则构建」的编排，便于接入新数据集或新特征。
* **双模式：** train 与 predict-only 共用同一 predict → evaluate 路径，便于「只跑评测」或「多 checkpoint 对比」而不重复训练。
* **产物统一：** 所有 run 的目录结构与文件名一致，便于跨方法、跨实验的批量分析与复现。

在遵守上述接口与目录约定的前提下，你可以自由替换数据管线、模型实现和指标定义，而无需改动 pipeline 核心；适合长期迭代与多方向研究。
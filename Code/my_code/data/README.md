# Data 模块布局

按数据集划分：每个数据集一个目录，下含 `preprocess/`、`feature/`、`dataset/`。注册在各自 `__init__.py` 中完成，入口通过 `data/load.py` 按名查表。

## 目录结构

```
data/
├── base.py          # BaseDatasetBuilder 接口
├── registry.py      # 注册表：preprocess / feature / dataset / collate
├── load.py          # 薄调度：get_processed, get_features, build_dataset, build_collate_fn, build_loaders
├── __init__.py      # import cora, mnist 触发注册；再导出 load 中 API
├── cora/
│   ├── __init__.py      # 注册 cora 的 preprocess, feature, dataset, collate
│   ├── preprocess/      # raw → processed
│   │   └── build.py     # CoraBuilder
│   ├── feature/         # 特征增强
│   │   ├── add_degree.py
│   │   └── main.py
│   └── dataset/         # processed+features → Dataset + collate
│       ├── dataset.py   # MyPyGDataset, build_datasets
│       └── collate.py   # collate_cora
└── mnist/
    ├── __init__.py
    ├── preprocess/build.py
    ├── feature/         # 可为空
    └── dataset/
        ├── dataset.py   # TorchSplitDataset, build_datasets
        └── collate.py   # collate_torch_batch
```

## 新增数据集

1. 新建 `data/<name>/`，下设 `preprocess/`、`feature/`、`dataset/`。
2. 在 `<name>/__init__.py` 中：`register_preprocess(name, Builder())`、`register_dataset_builder(name, build_datasets)`、`register_collate(name, collate_fn)`；若有 feature 则 import feature 子包并在 feature 内 `register_feature(dataset_name, method_key, build_fn, load_fn)`。
3. 在 `data/__init__.py` 中增加 `import my_code.data.<name>`。

## Config

- `cfg["data"]["dataset"]`：数据集名，用于查 registry。
- `cfg["data"]["feature_method"]`：特征方法名，与 dataset 一起查 `get_feature_fns(dataset, method_key)`。

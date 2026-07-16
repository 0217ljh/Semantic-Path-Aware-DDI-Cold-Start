"""SumGNN baseline (KG subgraph-summarization DDI) for the unified benchmark.

Per-task unified wrappers live under multi_cls/ and multi_label_cls/ (imported by the
runner by module path). The ported SumGNN core is in _core/ (independent DGL-2.4 copy of
the reproduction); the on-disk data builder is _data/necessary/build_sumgnn_data.py.
"""

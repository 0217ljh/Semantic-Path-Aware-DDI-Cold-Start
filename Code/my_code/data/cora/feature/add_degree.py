"""Feature cache add_degree: node degree from edge_index. For Cora."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict


def build_feature_cache(processed: dict, cfg: dict, feature_dir: str) -> Dict[str, Any]:
    data = processed["data"]
    edge_index = data["edge_index"]
    try:
        import torch
        edge_index = edge_index.numpy() if hasattr(edge_index, "numpy") else getattr(edge_index, "__array__", lambda: edge_index)()
    except Exception:
        import numpy as np
        edge_index = np.asarray(edge_index)
    n = data.get("num_nodes")
    if n is None and "x" in data:
        n = data["x"].shape[0]
    if n is None:
        n = int(edge_index.max()) + 1 if edge_index.size else 0

    data_cfg = cfg.get("data") or {}
    method_key = (cfg.get("model") or {}).get("name", "main")
    feature_cfg = data_cfg.get(method_key, {}).get("feature", {}) or data_cfg.get("feature", {})
    feature_type = feature_cfg.get("feature_type", "degree")
    if feature_type == "degree":
        import numpy as np
        deg = np.zeros(n, dtype=np.int64)
        for j in range(edge_index.shape[1]):
            u = int(edge_index[0, j])
            if 0 <= u < n:
                deg[u] += 1
        by_index = {"node_degree": deg}
    else:
        by_index = {}

    data_key = data_cfg.get("dataset", "")
    features = {
        "by_index": by_index,
        "collate": {},
        "meta": {"method_key": method_key, "data_key": data_key, "n": n},
    }
    path = Path(feature_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "feature.pkl", "wb") as f:
        pickle.dump(features, f)
    (path / "DONE").write_text("")
    return features


def load_feature_cache(feature_dir: str) -> Dict[str, Any]:
    with open(Path(feature_dir) / "feature.pkl", "rb") as f:
        return pickle.load(f)

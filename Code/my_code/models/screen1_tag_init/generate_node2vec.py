"""Build variant C (Node2Vec / DeepWalk-style) embedding cache.

CPU-only (uses gensim Word2Vec). Generates random walks on the merged KG
and trains skip-gram. Saves to `screen1_tag_init/deepwalk_d64__seed42.pt`.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen1_tag_init import init_features as ift  # noqa: E402


def main():
    print(f"[node2vec] starting at {time.strftime('%H:%M:%S')}")
    import pandas as pd
    nodes_df = pd.read_parquet(ift.ntb.MERGED_NODES)
    canonical_ids = [str(x) for x in nodes_df["id"].tolist()]
    print(f"[node2vec] {len(canonical_ids):,} nodes")

    t0 = time.time()
    _, emb = ift.build_init("C", n_dim=64, seed=42)
    print(f"[node2vec] DONE in {time.time()-t0:.0f}s, shape={emb.shape}")


if __name__ == "__main__":
    main()

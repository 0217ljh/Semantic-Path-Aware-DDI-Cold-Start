"""Quick path-verification smoke: uses default REPO_DATA_DIR (no overrides)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from data_loader import REPO_DATA_DIR, load_data

print(f"REPO_DATA_DIR = {REPO_DATA_DIR}")
print(f"  exists? {REPO_DATA_DIR.exists()}")
print(f"  has drugbank? {(REPO_DATA_DIR / 'drugbank' / 'ddi.txt').exists()}")

t0 = time.time()
bundle = load_data(dataset="drugbank", extractor="khop-subtree", khop=2, fixed_num=4)
print(f"load_data via default path: {time.time() - t0:.1f}s")
print(f"  interactions: {bundle.interactions.shape}")
print(f"  num_drugs_DDI: {bundle.stats['num_drugs_DDI']}")
print(f"  num_nodes: {bundle.stats['num_nodes']}")
print("PATH OK")

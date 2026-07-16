# code-adapter

Self-contained experiment pipeline for the Semantic-Path-Aware DDI cold-start
paper (adapter `H = H_base + M_A M_B`). Built by **copying** reusable modules
from `Code/my_code/rank_analysis/` (which still powers the intro figure and is
left untouched) and adding new modules stage by stage.

## Import convention (locked)

The folder name `code-adapter` contains a hyphen, so it is **not** an importable
Python package name (matches the project's `code-idea-*` convention). Never do
`import code-adapter...`. Instead every entrypoint / test adds this folder to
`sys.path` once, then imports its top-level modules directly (module names have
no hyphen):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # .../Code/code-adapter
from specs import TaskSpec
from data.loader import load_rank_data
```

All Python runs go through the WSL conda env:

```bash
wsl bash -ic "conda activate project_1 && cd <project_root> && \
    python Code/code-adapter/tests/test_data_load.py"
```

## Stage layout (maps 1:1 to the pipeline module flow)

| stage | dir / file | status |
|---|---|---|
| 1 dataset load -> `RankData` | `data/loader.py`, `specs.py` | done, tested |
| 2 KG + semantic prep (M_uv, z_m/z_r) | `kg/` | todo |
| 3 protocol (协议1/协议2) epoch | `protocol/` | todo |
| 4 model = backbone + adapter (M_A.M_B) | `models/` | todo |
| 5-6 training loop + extraction | `train/` | todo |
| 7 metrics | `metrics/` | todo |
| 8 analysis (RQ1/RQ2/RQ3) | `analysis/` | todo |
| entry | `run.py` | todo |

Tests live in `tests/`; each stage gets an acceptance test with explicit
success criteria.

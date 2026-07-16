"""CLI entry for PMP v1 per-drug mediator dict cache builder.

Thin wrapper. Delegates to
my_code.models.screen_s2_v4_pmp.v1.precompute_pmp_cache.main.

Usage:
  python Code/scripts/precompute_pmp_v1_cache.py            # first build
  python Code/scripts/precompute_pmp_v1_cache.py --force    # rebuild
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))

from my_code.models.pmp_v1.precompute_pmp_cache import main  # noqa: E402

if __name__ == "__main__":
    main()

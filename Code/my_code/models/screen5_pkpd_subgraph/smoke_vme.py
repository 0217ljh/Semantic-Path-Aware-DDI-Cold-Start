"""Smoke test for VME generation pipeline (small spend ~$0.005)."""
from __future__ import annotations

import sys
from pathlib import Path

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "Code"))

from my_code.models.screen5_pkpd_subgraph.vme_generator import generate_vme_via_gpt4o


def main():
    # Three test gaps — minimal cost
    gaps = [
        ("Aspirin", "Ibuprofen", "PK"),
        ("Carbamazepine", "Diazepam", "PD"),
        ("Warfarin", "Phenytoin", "PK"),
    ]
    out = generate_vme_via_gpt4o(gaps, budget_usd=0.10, model="gpt-4o", skip_if_cached=True)
    print()
    print("=" * 60)
    print("VME generation smoke results:")
    for gap, text in out.items():
        print(f"  {gap[0]} <-> {gap[1]} ({gap[2]}): {text[:120]}")


if __name__ == "__main__":
    main()

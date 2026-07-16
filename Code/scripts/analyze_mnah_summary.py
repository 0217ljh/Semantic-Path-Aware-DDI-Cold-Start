"""Verified MNAH (Stage 1, i2) results — for paper narrative.
Reads results.json across the MNAH canonical reps + controls + generalization.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def show(label, patterns):
    print(f"=== {label} ===")
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(str(ROOT / p))))
    for f in files:
        try:
            d = json.load(open(f))
            m = d.get("metrics",{}).get("test_s2",{})
            tag = os.path.basename(os.path.dirname(f))
            auc = m.get("auc")
            nll = m.get("nll")
            auc_emer = m.get("auc_emergnn_only")
            auc_aux = m.get("auc_aux_only")
            print(f"  {tag[:88]:88s}  auc={auc}  emer_only={auc_emer}  aux_only={auc_aux}")
        except Exception as e:
            print(f"  ERR {f}: {e}")
    print()

show("Anchor (EmerGNN, no MNAH)",
     ["Code/runs/*s2anchor*drugbank*/results.json"])
show("MNAH Stage 1 canonical (3 reps, seed42)",
     ["Code/runs/*mnah_v2s1_drugbank_seed42__seed42*/results.json",
      "Code/runs/*mnah_v2s1_drugbank_seed42_rep*/results.json"])
show("MNAH shuffle control (codex r13)",
     ["Code/runs/*mnah_v2s1_SHUFFLECTRL*/results.json"])
show("MNAH degree-only control",
     ["Code/runs/*mnah_v2s1_DEGREECTRL*/results.json"])
show("MNAH seed43 generalization (release split)",
     ["Code/runs/*mnah_v2s1_GEN_release_seed43*/results.json"])

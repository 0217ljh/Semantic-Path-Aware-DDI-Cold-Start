"""Quick summary of CACR + S2-anchor results — for narrative use only.
Reads results.json from Code/runs/ and prints headline AUROC per run.
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
            print(f"  {tag[:88]:88s}  auc={auc}  nll={nll}")
        except Exception as e:
            print(f"  ERR {f}: {e}")
    print()
show("ANCHOR (no CACR)",
     ["Code/runs/*s2anchor*drugbank*/results.json"])
show("CACR canonical lam0.5 warm5 ddi100 (reps)", [
    "Code/runs/*cacr_ddi100_lam0.5_warm5_drugbank_seed42*/results.json",
    "Code/runs/*cacr_ddi100_lam0.5_warm5_drugbank_rep*/results.json",
])
show("CACR ddi-rate variant (ddi075)", ["Code/runs/*cacr_ddi075*lam0.5*/results.json"])
show("CACR lambda sweep", [
    "Code/runs/*cacr_ddi100_lam0.25*/results.json",
    "Code/runs/*cacr_ddi100_lam1.0*/results.json",
])
show("CACR non-DDI light dropout", ["Code/runs/*cacr_ddi075_nond02*/results.json"])
show("Nodedup (lowdeg lam0.1)", ["Code/runs/*nodedup*/results.json"])

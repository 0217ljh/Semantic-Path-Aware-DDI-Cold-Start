"""Find CONCRETE modality blind-spot cases for the Gap-slide motivation.

Molecular blind spot = a POSITIVE (interacting) test_s2 pair that is
structurally DISSIMILAR (low Morgan/Tanimoto) yet shares KG mechanism
mediators (high s_tau) — i.e. molecular similarity cannot see the interaction
but the KG bridge can. KG blind spot = drugs with ~zero KG neighbourhood
(handled separately; printed for completeness).

Verified data only (no fabrication). Run:
  python Code/scripts/analyze_spmn_v1_modality_blindspots.py --seed 42 --l-max 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for cand in [cur, *cur.parents]:
        if (cand / "Code" / "data" / "KG").is_dir():
            return cand
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

from data_utils.dataset import PairDataset  # noqa: E402
from my_code.models.spmn_v1.retrieval import N_TYPES  # noqa: E402

PKL_DIR = ROOT / "Code/data/coldddi_legacy/800drug"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--l-max", type=int, default=3)
    ap.add_argument("--tanimoto-max", type=float, default=0.20)
    args = ap.parse_args()

    ds = PairDataset.from_pkl(str(PKL_DIR / f"seed{args.seed}.pkl"))
    pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].reset_index(drop=True)

    # s_tau (shared KG mediators) from the v3and cache: rows 0..len(pos)-1 of
    # test_s2 are the positives (stacked pos-then-neg by _binary_frame).
    cache = ROOT / "Code/data/_cache" / \
        f"spmn_v1_phase2_supports_v3and_seed{args.seed}_lmax{args.l_max}.npz"
    z = np.load(cache)
    struct = z["test_s2__struct"]
    s_tau_sum = struct[:len(pos), :N_TYPES].sum(1)            # shared mediators
    n_active = (struct[:len(pos), :N_TYPES] > 0).sum(1)

    # SMILES + Morgan FP + names.
    smi = pd.read_csv(PKL_DIR / f"drug_smiles__seed{args.seed}.csv")
    col_id = [c for c in smi.columns if "id" in c.lower()][0]
    col_smi = [c for c in smi.columns if "smile" in c.lower()][0]
    id2smi = dict(zip(smi[col_id].astype(str), smi[col_smi].astype(str)))
    nodes = pd.read_parquet(
        ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet",
        columns=["id", "name"])
    id2name = dict(zip(nodes["id"].astype(str), nodes["name"].astype(str)))

    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    fp_cache: dict[str, object] = {}

    def fp(did: str):
        if did in fp_cache:
            return fp_cache[did]
        m = Chem.MolFromSmiles(id2smi.get(did, "")) if did in id2smi else None
        f = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m else None
        fp_cache[did] = f
        return f

    rows = []
    for i in range(len(pos)):
        a, b = str(pos.iloc[i, 0]), str(pos.iloc[i, 1])
        fa, fb = fp(a), fp(b)
        if fa is None or fb is None:
            continue
        tani = DataStructs.TanimotoSimilarity(fa, fb)
        rows.append((a, b, tani, float(s_tau_sum[i]), int(n_active[i])))

    df = pd.DataFrame(rows, columns=["a", "b", "tanimoto", "s_tau_sum", "n_active_types"])
    print(f"=== molecular blind-spot: low Tanimoto + high shared KG mediators "
          f"(interacting test_s2 pairs, n={len(df)}) ===")
    blind = df[(df["tanimoto"] <= args.tanimoto_max) & (df["s_tau_sum"] > 0)]
    blind = blind.sort_values(["tanimoto", "s_tau_sum"], ascending=[True, False])
    print(f"pairs with Tanimoto<={args.tanimoto_max} AND shared mediators>0: "
          f"{len(blind)} / {len(df)}")
    print("--- 10 most structurally-dissimilar interacting pairs WITH KG bridge ---")
    for _, r in blind.head(10).iterrows():
        na, nb = id2name.get(r["a"], "?")[:24], id2name.get(r["b"], "?")[:24]
        print(f"  {r['a']}({na}) x {r['b']}({nb})  "
              f"Tanimoto={r['tanimoto']:.3f}  shared_mediators={r['s_tau_sum']:.0f}  "
              f"types={r['n_active_types']}")
    # quick stats: how common is the molecular blind spot?
    interacting = df["s_tau_sum"] > 0
    lowsim = df["tanimoto"] <= args.tanimoto_max
    print(f"\n[stat] {100*lowsim.mean():.1f}% of interacting test_s2 pairs have "
          f"Tanimoto<={args.tanimoto_max}; of those, "
          f"{100*(blind.shape[0]/max(1,lowsim.sum())):.1f}% still share KG mediators "
          f"(molecular blind but KG sees them)")


if __name__ == "__main__":
    main()

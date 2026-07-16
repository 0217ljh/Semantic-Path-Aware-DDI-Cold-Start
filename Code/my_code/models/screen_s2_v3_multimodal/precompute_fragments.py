"""E-frag prerequisite — per-drug BRICS fragment features (richer molecular source).

codex 019e6747: fragments are the most promising richer-molecular source for PK lift (PK is
substructure/metabolism-motif driven). Per drug: BRICS-decompose SMILES into fragments, build
a global top-K fragment vocabulary, emit a per-drug count vector. Deterministic, no GPU.

Output: Code/data/_cache/molecular_fragments_brics.npz
  drug_ids (N,) str | frag (N, V) float32 count vector | vocab (V,) str (fragment SMILES)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_FILE = Path(__file__).resolve()
PROJECT_ROOT = _FILE.parents[4]
SMILES_CSV = PROJECT_ROOT / "Code/data/coldddi_legacy/800drug/drug_smiles__seed42.csv"
OUT = PROJECT_ROOT / "Code/data/_cache/molecular_fragments_brics.npz"
VOCAB_TOPK = 768


def main() -> None:
    df = pd.read_csv(SMILES_CSV)
    id_col = "drugbank_id" if "drugbank_id" in df.columns else df.columns[0]
    ids = df[id_col].astype(str).tolist()
    smis = df["smiles"].astype(str).tolist()

    per_drug_frags: list[list[str]] = []
    global_counts: Counter = Counter()
    n_fail = 0
    for smi in smis:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            per_drug_frags.append([]); n_fail += 1; continue
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            frags = []
        per_drug_frags.append(frags)
        global_counts.update(set(frags))  # document frequency

    vocab = [f for f, _ in global_counts.most_common(VOCAB_TOPK)]
    vidx = {f: i for i, f in enumerate(vocab)}
    V = len(vocab)
    frag = np.zeros((len(ids), V), dtype=np.float32)
    for i, frags in enumerate(per_drug_frags):
        c = Counter(frags)
        for f, n in c.items():
            j = vidx.get(f)
            if j is not None:
                frag[i, j] = float(n)
    cov = int((frag.sum(axis=1) > 0).sum())
    print(f"[frag] drugs={len(ids)} parse_fail={n_fail} vocab={V} "
          f"drugs_with_frag={cov} ({100*cov/len(ids):.1f}%) "
          f"mean_frags/drug={frag.sum(axis=1).mean():.1f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, drug_ids=np.array(ids), frag=frag, vocab=np.array(vocab))
    print(f"[frag] saved -> {OUT} (frag {frag.shape})", flush=True)


if __name__ == "__main__":
    main()

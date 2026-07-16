"""Complementarity gate — MOLECULAR side (codex r19).

Builds a motif-COUNT-only molecular model (NO Morgan FP, to isolate NEW molecular
signal from EmerGNN's existing Morgan input) on seed42 800-drug, evaluates S2, and
DUMPS per-example test_s2 logits so we can later compute:
  - motif-only AUC vs MNAH ~0.77
  - corr(MNAH logits, motif logits)
  - val-tuned 2-logit ensemble a*MNAH + b*motif
  - shuffled-motif control (scramble motif↔drug assignment)

Molecular features: BRICS fragment decomposition (RDKit) -> per-drug motif-count
vector over a top-K motif vocab. Pair feature = log1p(count_u + count_v) per motif
(symmetric, presence-in-either). Cold-start-stable: any SMILES -> BRICS motifs.

Outputs to Code/runs/_gate_molecular/:
  motif_only_logits.npz  (y_true, motif_logit, order=pos-then-neg for test_s2)
  results.json           (motif_only_auc, shuffled_auc, vocab_size, n_*)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_root() -> Path:
    cur = Path(__file__).resolve()
    for c in [cur, *cur.parents]:
        if (c / "Code" / "data" / "KG").is_dir():
            return c
    raise FileNotFoundError


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "Code"))

PKL = ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"
OUT = ROOT / "Code/runs/_gate_molecular"
TOPK_MOTIFS = 300


def brics_motifs(smiles: str) -> list[str]:
    from rdkit import Chem
    from rdkit.Chem import BRICS
    if not smiles or pd.isna(smiles):
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        frags = BRICS.BRICSDecompose(mol)  # set of fragment SMILES (with dummy atoms)
    except Exception:
        return []
    # strip dummy-atom labels [n*] -> canonical-ish motif token
    out = []
    for f in frags:
        m = Chem.MolFromSmiles(f)
        if m is None:
            out.append(f)
        else:
            out.append(Chem.MolToSmiles(m))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from data_utils import PairDataset
    ds = PairDataset.from_pkl(str(PKL))

    # drug -> SMILES
    drugs = ds.drugs
    if drugs is None or "smiles" not in drugs.columns:
        raise RuntimeError("PairDataset.drugs lacks SMILES")
    did2smi = {str(r["drugbank_id"]): r["smiles"] for _, r in drugs.iterrows()}
    print(f"[gate-mol] drugs with SMILES: {sum(1 for v in did2smi.values() if isinstance(v,str) and v)}")

    # per-drug BRICS motif counts
    print("[gate-mol] BRICS decomposition ...", flush=True)
    drug_motifs: dict[str, Counter] = {}
    vocab_counter: Counter = Counter()
    for did, smi in did2smi.items():
        ms = brics_motifs(smi if isinstance(smi, str) else "")
        c = Counter(ms)
        drug_motifs[did] = c
        vocab_counter.update(c.keys())
    vocab = [m for m, _ in vocab_counter.most_common(TOPK_MOTIFS)]
    vidx = {m: i for i, m in enumerate(vocab)}
    print(f"[gate-mol] motif vocab (top {TOPK_MOTIFS}): {len(vocab)} motifs", flush=True)

    def drug_vec(did: str) -> np.ndarray:
        v = np.zeros(len(vocab), dtype=np.float32)
        for m, n in drug_motifs.get(did, {}).items():
            j = vidx.get(m)
            if j is not None:
                v[j] = n
        return v

    drug_vecs = {did: drug_vec(did) for did in did2smi}

    def pair_feats(df: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, (a, b) in enumerate(zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str))):
            X[i] = np.log1p(drug_vecs.get(a, 0) + drug_vecs.get(b, 0))
        return X

    # train (pos + epoch_0 neg), test_s2 (pos + neg)
    train_pos = ds.splits.train[["drug_a_id", "drug_b_id"]].copy()
    train_neg = ds.get_train_negatives(0, regenerate=True)[["drug_a_id", "drug_b_id"]]
    te_pos = ds.splits.test_s2[["drug_a_id", "drug_b_id"]].copy()
    te_neg = ds.get_negatives("test_s2")[["drug_a_id", "drug_b_id"]]
    print(f"[gate-mol] train pos/neg = {len(train_pos)}/{len(train_neg)}; "
          f"test_s2 pos/neg = {len(te_pos)}/{len(te_neg)}", flush=True)

    Xtr = np.vstack([pair_feats(train_pos), pair_feats(train_neg)])
    ytr = np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))])
    Xte = np.vstack([pair_feats(te_pos), pair_feats(te_neg)])
    yte = np.concatenate([np.ones(len(te_pos)), np.zeros(len(te_neg))])

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, ytr)
    logit = lr.decision_function(Xte)
    auc = roc_auc_score(yte, logit)
    print(f"[gate-mol] motif-only test_s2 AUC = {auc:.4f}", flush=True)

    # shuffled-motif control: scramble drug->vec assignment, retrain
    rng = np.random.default_rng(123)
    dids = list(drug_vecs.keys())
    perm = rng.permutation(len(dids))
    shuf_vecs = {dids[i]: drug_vecs[dids[perm[i]]] for i in range(len(dids))}

    def pair_feats_shuf(df):
        X = np.zeros((len(df), len(vocab)), dtype=np.float32)
        for i, (a, b) in enumerate(zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str))):
            X[i] = np.log1p(shuf_vecs.get(a, 0) + shuf_vecs.get(b, 0))
        return X

    Xtr_s = np.vstack([pair_feats_shuf(train_pos), pair_feats_shuf(train_neg)])
    Xte_s = np.vstack([pair_feats_shuf(te_pos), pair_feats_shuf(te_neg)])
    lr_s = LogisticRegression(max_iter=2000, C=1.0)
    lr_s.fit(Xtr_s, ytr)
    auc_s = roc_auc_score(yte, lr_s.decision_function(Xte_s))
    print(f"[gate-mol] SHUFFLED-motif test_s2 AUC = {auc_s:.4f} (should be near chance)", flush=True)

    # dump test_s2 logits (order: pos then neg) + the pair ids for MNAH alignment
    np.savez(OUT / "motif_only_logits.npz",
             y_true=yte, motif_logit=logit,
             pair_a=np.concatenate([te_pos["drug_a_id"].astype(str).values, te_neg["drug_a_id"].astype(str).values]),
             pair_b=np.concatenate([te_pos["drug_b_id"].astype(str).values, te_neg["drug_b_id"].astype(str).values]),
             label=yte)
    (OUT / "results.json").write_text(json.dumps({
        "motif_only_auc": float(auc), "shuffled_motif_auc": float(auc_s),
        "vocab_size": len(vocab), "topk": TOPK_MOTIFS,
        "n_train_pos": int(len(train_pos)), "n_train_neg": int(len(train_neg)),
        "n_test_pos": int(len(te_pos)), "n_test_neg": int(len(te_neg)),
        "note": "motif-only excludes Morgan; isolates NEW molecular signal. "
                "test_s2 logits order = pos then neg.",
    }, indent=2))
    print(f"[gate-mol] saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

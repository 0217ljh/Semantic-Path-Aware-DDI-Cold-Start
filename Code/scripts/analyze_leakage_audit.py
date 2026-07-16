"""Comprehensive leakage audit for the 800-drug seed42 cold-start pipeline.

Verifies (1) SEEN/UNSEEN drug disjointness, (2) train/test pair disjointness,
(3) merged KG has no DDI edges, (4) meet_feat sanity, (5) z_m InfoNCE training-set
excludes UNSEEN, (6) LLM sanitized text post-redact partner/DDI-phrase residue,
(8) train negative sampler restricted to SEEN drugs.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Code"))
from data_utils import PairDataset  # noqa

print("=" * 72)
print("LEAKAGE AUDIT  --  800-drug seed42 cold-start S2")
print("=" * 72)

ds = PairDataset.from_pkl(str(ROOT / "Code/data/coldddi_legacy/800drug/seed42.pkl"))
tr = ds.splits.train
te = ds.splits.test_s2
val = ds.splits.val_s2
seen = set(tr["drug_a_id"].astype(str)) | set(tr["drug_b_id"].astype(str))
test_unseen = set(te["drug_a_id"].astype(str)) | set(te["drug_b_id"].astype(str))
val_unseen = set(val["drug_a_id"].astype(str)) | set(val["drug_b_id"].astype(str))

# ---- 1: drug disjointness ----
print("\n[1] DRUG SET DISJOINTNESS")
print(f"  SEEN (in train): {len(seen)}")
print(f"  UNSEEN test_s2: {len(test_unseen)}")
print(f"  UNSEEN val_s2:  {len(val_unseen)}")
print(f"  SEEN n test_s2: {len(seen & test_unseen)}  (expect 0)")
print(f"  SEEN n val_s2 : {len(seen & val_unseen)}  (expect 0)")
print(f"  test_s2 n val_s2 UNSEEN: {len(test_unseen & val_unseen)}")
print(f"  total distinct drugs touched: {len(seen | test_unseen | val_unseen)}")


def canon(df):
    return set((a, b) if a <= b else (b, a)
               for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)))


tr_p = canon(tr); te_p = canon(te); val_p = canon(val)
print("\n[2] PAIR DISJOINTNESS (canonical)")
print(f"  train+: {len(tr_p)}  test_s2+: {len(te_p)}  val_s2+: {len(val_p)}")
print(f"  train n test_s2: {len(tr_p & te_p)}  (expect 0)")
print(f"  train n val_s2 : {len(tr_p & val_p)}  (expect 0)")
print(f"  test  n val_s2 : {len(te_p & val_p)}  (expect 0)")

# ---- 3: merged KG DDI edges? ----
print("\n[3] MERGED KG (mask1) DDI-EDGE AUDIT")
edges = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet")
nodes = pd.read_parquet(ROOT / "Code/data/KG/_merged_kg/nodes__drugbank_hetionet_primekg.parquet")
drug_set = set(nodes.loc[nodes["kind"].isin(["Drug", "drug"]), "id"])
sd = edges["src"].isin(drug_set); dd = edges["dst"].isin(drug_set)
ddedges = int((sd & dd).sum())
print(f"  total edges: {len(edges):,}  drug nodes: {len(drug_set):,}")
print(f"  drug->drug edges (any rel): {ddedges}  (expect 0 for DDI-masked)")
if ddedges > 0:
    print(edges[sd & dd].head(5).to_string())

# ---- 4: meet_feat sanity on test_s2 pairs ----
print("\n[4] meet_feat (MNAH count cache) sanity on test_s2 pairs")
mf = pd.read_parquet(ROOT / "Code/data/_cache/meet_feat_drugbank_seed42_kgonly_v1.parquet")
mf_canon = pd.Series(
    [(a, b) if a <= b else (b, a)
     for a, b in zip(mf["drug_a_id"].astype(str), mf["drug_b_id"].astype(str))],
    index=mf.index)
mask = mf_canon.isin(te_p)
te_rows = mf.loc[mask]
fcols = [c for c in mf.columns if c.startswith("f_")]
nonzero = int((te_rows[fcols].sum(axis=1) > 0).sum())
print(f"  test_s2 pairs covered: {len(te_rows)}/{len(te_p)}")
print(f"  with >=1 mediator count: {nonzero}")
print(f"  mean total mediators per test pair: {te_rows[fcols].sum(axis=1).mean():.2f}")
print(f"  NOTE non-zero is NOT leakage; counts are over biomedical (DDI-masked) KG")

# ---- 5: z_m InfoNCE training set ----
print("\n[5] z_m InfoNCE training-set leakage check")
mu = np.load(ROOT / "Code/data/_cache/molecular_aligned_infonce.npz", allow_pickle=True)
zm_ids = np.array(mu["drug_ids"]).astype(str)
zm_seen = set(zm_ids[mu["is_seen"].astype(bool)])
print(f"  z_m records flagged is_seen=True: {len(zm_seen)}")
print(f"  zm_seen n test_s2 UNSEEN: {len(zm_seen & test_unseen)}  (expect 0)")
print(f"  zm_seen n val_s2 UNSEEN:  {len(zm_seen & val_unseen)}  (expect 0)")
print(f"  zm_seen subset of SEEN train graph: {zm_seen <= seen}  (expect True)")

# ---- 6: LLM sanitized text post-redact residue ----
print("\n[6] LLM sanitized text — post-redact partner-drug / DDI-phrase residue")
recs = []
for ln in open(ROOT / "Code/data/_cache/llm_pharma/llm_pharma.jsonl", encoding="utf-8"):
    try:
        recs.append(json.loads(ln))
    except Exception:
        pass
id2name = json.loads((ROOT / "Code/data/KG/drugbank/filtered/id2name.json").read_text(encoding="utf-8"))
name_to_id = {v.lower(): k for k, v in id2name.items() if len(v) >= 5}
DDI_PHRASES = ("interact", "coadminist", "co-administ", "combined with", "combination with",
               "concomitant", "avoid with", "contraindicated", "with inhibitor", "with inducer",
               "increase levels", "decrease levels", "co-medic", "co-prescri")
n_with = n_part = n_phr = 0
flagged = 0
for r in recs:
    if r.get("leakage_flag"):
        flagged += 1
    san = (r.get("sanitized_text") or "").lower()
    if not san:
        continue
    n_with += 1
    own = (r.get("name") or "").lower()
    hit = False
    for nm in name_to_id:
        if nm == own:
            continue
        if re.search(r"\b" + re.escape(nm) + r"\b", san):
            hit = True; break
    if hit:
        n_part += 1
    if any(p in san for p in DDI_PHRASES):
        n_phr += 1
print(f"  total jsonl records: {len(recs)}")
print(f"  flagged at distill time (raw text had partner/DDI): {flagged}")
print(f"  records with non-empty sanitized_text: {n_with}")
print(f"  POST-REDACT still contains other drug name: {n_part}  (expect ~0)")
print(f"  POST-REDACT still contains DDI phrase:     {n_phr}   (expect ~0)")

# ---- 8: train negative sampler restricted to SEEN ----
print("\n[8] TRAIN NEGATIVE SAMPLING (epoch 0)")
tn = ds.get_train_negatives(0, regenerate=True)
tn_p = canon(tn)
tn_drugs = set(tn["drug_a_id"].astype(str)) | set(tn["drug_b_id"].astype(str))
print(f"  train neg pairs: {len(tn_p)}")
print(f"  train_neg n test_s2 pairs: {len(tn_p & te_p)}  (expect 0)")
print(f"  train_neg n val_s2 pairs:  {len(tn_p & val_p)}  (expect 0)")
print(f"  train_neg drugs subset of SEEN: {tn_drugs <= seen}  (expect True)")
print(f"  train_neg drugs: {len(tn_drugs)}/{len(seen)}")

# ---- 9: test negative pool sanity ----
print("\n[9] TEST_s2 negative pool")
test_neg = ds.get_negatives("test_s2")
tn_pairs = canon(test_neg)
tn_drugs = set(test_neg["drug_a_id"].astype(str)) | set(test_neg["drug_b_id"].astype(str))
print(f"  test_s2 neg pairs: {len(tn_pairs)}")
print(f"  test_neg n train+: {len(tn_pairs & tr_p)}  (expect 0)")
print(f"  test_neg n test+:  {len(tn_pairs & te_p)}  (expect 0)")
print(f"  test_neg drugs subset of UNSEEN test set: {tn_drugs <= test_unseen}  (expect True)")
print(f"  test_neg drugs: {len(tn_drugs)}")

print("\n" + "=" * 72)
print("Audit complete. Any (expect 0) line showing >0 is a leakage finding.")
print("=" * 72)

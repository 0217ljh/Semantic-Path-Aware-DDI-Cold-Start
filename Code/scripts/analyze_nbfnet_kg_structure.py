"""Structural analysis of the KG/DDI that bounds NBFNet on cold-start S2.

Reports KG relation distribution, drug KG-degree stats, G1/G2 pool sizes, and
the AUC of a trivial 'number of shared KG neighbors' predictor on S2/S0 splits
(the 2-hop-overlap signal floor). Supports BOTH KG sources:
  --kg drugbank : the 5-bucket DrugBank KG (build_kg_from_kb)
  --kg merged   : the merged DrugBank+Hetionet+PrimeKG drug-incident KG
                  (build_kg_from_merged_parquet) — what PMP v1.1-v1.6 use.

Backs `my_code/models/nbfnet_v1_7/_reviews/2026-06-05__structural_limits_analysis.md`.

Run:
  python Code/scripts/analyze_nbfnet_kg_structure.py --kg merged
  python Code/scripts/analyze_nbfnet_kg_structure.py --kg drugbank
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.emergnn.kg_builder import build_kg_from_kb
from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet
from data_utils import PairDataset

MERGED_EDGES = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"

ap = argparse.ArgumentParser()
ap.add_argument("--kg", choices=["drugbank", "merged"], default="merged")
args = ap.parse_args()

ds = PairDataset.from_pkl(str(ROOT / "data/coldddi_legacy/800drug/seed42.pkl"))
kg = ds.kg

drug_ids = set()
for _n, df in ds.splits.items():
    drug_ids.update(df["drug_a_id"].astype(str)); drug_ids.update(df["drug_b_id"].astype(str))
if hasattr(kg, "drug_ids"):
    drug_ids.update(kg.drug_ids)
drug_id_list = sorted(drug_ids)

if args.kg == "drugbank":
    kb = {"my_enzyme_list": kg.enzymes, "my_target_list": kg.targets,
          "my_transporter_list": kg.transporters, "my_carrier_list": kg.carriers,
          "my_pathway_list": kg.pathways}
    art = build_kg_from_kb(kb, drug_id_list, keep_only_known_drugs=True)
    id2rel = {0: "target", 1: "enzyme", 2: "transporter", 3: "carrier", 4: "pathway"}
else:
    art = build_kg_from_merged_parquet(MERGED_EDGES, drug_id_list, verbose=False)
    id2rel = {v: k for k, v in art["rel2id"].items()}

e2i = art["entity2id"]; n_ent = art["n_ent"]; trip = np.asarray(art["triplets"], dtype=np.int64)
n_rel = art.get("n_rel", int(trip[:, 2].max()) + 1)

print(f"==== KG source = {args.kg} ====")
print(f"== KG ==  n_ent={n_ent}  n_edges={len(trip)}  n_rel={n_rel}  n_drugs(vocab)={len(drug_id_list)}")
rels, cnts = np.unique(trip[:, 2], return_counts=True)
order = np.argsort(-cnts)
print("  top relations by edge count:")
for i in order[:12]:
    print(f"    rel {int(rels[i]):3d} ({id2rel.get(int(rels[i]), '?'):28s}): {int(cnts[i])} edges")

# drug -> set(neighbor) adjacency (heads are drugs by construction)
nbr = defaultdict(set)
for h, t, r in trip:
    nbr[int(h)].add(int(t))
drug_nodes = set(int(h) for h in np.unique(trip[:, 0]))
deg = np.array([len(nbr[d]) for d in drug_nodes])
print(f"== drug KG-degree ==  drugs_with_edges={len(drug_nodes)}  "
      f"mean={deg.mean():.1f} median={np.median(deg):.0f} min={deg.min()} max={deg.max()}")

g1 = set(map(str, ds.splits.g1_drugs)); g2 = set(map(str, ds.splits.g2_drugs))
g2_with_kg = sum(1 for d in g2 if d in e2i and len(nbr.get(e2i[d], ())) > 0)
print(f"== pools ==  |G1|={len(g1)} |G2|={len(g2)}  G2_with_KG_edges={g2_with_kg}/{len(g2)}")

def shared_counts(df):
    out = []
    for a, b in zip(df["drug_a_id"].astype(str), df["drug_b_id"].astype(str)):
        if a in e2i and b in e2i:
            out.append(len(nbr.get(e2i[a], set()) & nbr.get(e2i[b], set())))
        else:
            out.append(-1)
    return np.array(out)

for split in ["test_s2", "val_s2", "test_s0"]:
    pos = getattr(ds.splits, split)[["drug_a_id", "drug_b_id"]]
    neg = ds.get_negatives(split)[["drug_a_id", "drug_b_id"]]
    sp = shared_counts(pos); sn = shared_counts(neg)
    spv = sp[sp >= 0]; snv = sn[sn >= 0]
    y = np.concatenate([np.ones(len(spv)), np.zeros(len(snv))])
    score = np.concatenate([spv, snv]).astype(float)
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    print(f"== {split} ==  n_pos={len(spv)} n_neg={len(snv)}")
    print(f"   shared-neighbor count  POS mean={spv.mean():.2f} median={np.median(spv):.0f} "
          f"%>=1={(spv>=1).mean()*100:.1f}  |  NEG mean={snv.mean():.2f} "
          f"median={np.median(snv):.0f} %>=1={(snv>=1).mean()*100:.1f}")
    print(f"   AUC of trivial 'shared-neighbor-count' predictor = {auc:.4f}")

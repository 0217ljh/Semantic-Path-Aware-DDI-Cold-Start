"""B-class (PK-B ∪ PD-B) vs Negative path-structure comparison."""
import pandas as pd, networkx as nx, sys, numpy as np
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

KG = r"D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Code/data/KG"
OUT = r"D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Log/insight-discovery"

ds = pd.read_parquet(f"{OUT}/eval_600_PKB_PDB_NEG.parquet")
print(f"Loaded dataset: {len(ds)} pairs")
print(ds['pkpd_4way'].value_counts())

# === Load Hetionet, build graph ===
hetio_nodes = pd.read_csv(f"{KG}/hetionet/hetionet-v1.0-nodes.tsv", sep='\t')
hetio_edges = pd.read_csv(f"{KG}/hetionet/hetionet-v1.0-edges.sif", sep='\t',
                          header=None, names=['head','relation','tail'])
G = nx.MultiGraph()
for _, r in hetio_nodes.iterrows():
    G.add_node(r['id'], kind=r['kind'])
for _, r in hetio_edges.iterrows():
    G.add_edge(r['head'], r['tail'], relation=r['relation'])
print(f"Hetionet graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")

# Pre-cache CcSE neighbor sets for shared-SE count
hetio_cse = hetio_edges[hetio_edges['relation'] == 'CcSE']
cse_by_drug = hetio_cse.groupby('head')['tail'].apply(set).to_dict()

def db_to_hetio(d): return f"Compound::{d}"

def analyze_pair(a, b):
    """Returns dict of structural features for pair (a, b) in Hetionet."""
    src, dst = db_to_hetio(a), db_to_hetio(b)
    feat = {
        'in_kg': (src in G) and (dst in G),
        'src_in_kg': src in G,
        'dst_in_kg': dst in G,
        'shared_se_count': 0,
        'total_2hop_paths': 0,
        'paths_via_SideEffect': 0,
        'paths_via_Gene': 0,
        'paths_via_Disease': 0,
        'paths_via_Other': 0,
    }
    if not feat['in_kg']:
        return feat
    se_a = cse_by_drug.get(src, set())
    se_b = cse_by_drug.get(dst, set())
    feat['shared_se_count'] = len(se_a & se_b)
    feat['src_se_count'] = len(se_a)
    feat['dst_se_count'] = len(se_b)

    # 2-hop path enumeration via shared neighbors
    nbrs_a = set(G.neighbors(src)) - {dst}
    nbrs_b = set(G.neighbors(dst)) - {src}
    shared = nbrs_a & nbrs_b
    total = 0
    by_kind = Counter()
    for nbr in shared:
        kind = G.nodes[nbr].get('kind', 'Unknown')
        # multi-edge: count each combination
        n_edges_a = len(G[src][nbr])
        n_edges_b = len(G[nbr][dst])
        paths_here = n_edges_a * n_edges_b
        total += paths_here
        by_kind[kind] += paths_here
    feat['total_2hop_paths'] = total
    feat['paths_via_SideEffect'] = by_kind.get('Side Effect', 0)
    feat['paths_via_Gene'] = by_kind.get('Gene', 0)
    feat['paths_via_Disease'] = by_kind.get('Disease', 0)
    feat['paths_via_Other'] = total - feat['paths_via_SideEffect'] - feat['paths_via_Gene'] - feat['paths_via_Disease']
    return feat

# === Apply to all 600 pairs ===
print("\nEnumerating paths for 600 pairs...")
results = []
for _, row in ds.iterrows():
    f = analyze_pair(row['drug_a_id'], row['drug_b_id'])
    f['pair_id'] = row['pair_id']
    f['class'] = row['pkpd_4way']
    results.append(f)
res = pd.DataFrame(results)
print(f"Done. Pairs in KG by class:")
print(res.groupby('class')['in_kg'].agg(['sum', 'count']).rename(columns={'sum':'in_kg', 'count':'total'}))

# === Compare: B-class (PKB + PDB) vs NEG ===
res['group'] = res['class'].map({'PK-B':'B', 'PD-B':'B', 'NEG':'NEG'})
in_kg_only = res[res['in_kg']]
print(f"\nIn-KG pairs by group: {in_kg_only['group'].value_counts().to_dict()}")

# Distribution comparison
print(f"\n{'='*80}")
print(f"B-class (PKB + PDB) vs NEG path-structure distribution (Hetionet, in-KG pairs only)")
print(f"{'='*80}")

def describe(s):
    if len(s) == 0: return "(empty)"
    return f"n={len(s):>3}  mean={s.mean():>6.1f}  median={s.median():>5.1f}  p25={s.quantile(.25):>5.1f}  p75={s.quantile(.75):>5.1f}  max={s.max():>5}"

for feat in ['shared_se_count', 'total_2hop_paths', 'paths_via_SideEffect', 'paths_via_Gene', 'paths_via_Disease']:
    print(f"\n[{feat}]")
    for grp in ['B', 'NEG']:
        sub = in_kg_only[in_kg_only['group'] == grp]
        print(f"  {grp:>4}: {describe(sub[feat])}")

# === Discrimination AUC: use each feature alone to classify B vs NEG ===
from collections import defaultdict
try:
    from sklearn.metrics import roc_auc_score
    print(f"\n{'='*80}")
    print(f"AUC if used as B-vs-NEG classifier (higher feature → predict B)")
    print(f"{'='*80}")
    sub_kg = in_kg_only.copy()
    sub_kg['y'] = (sub_kg['group'] == 'B').astype(int)
    for feat in ['shared_se_count', 'total_2hop_paths', 'paths_via_SideEffect', 'paths_via_Gene', 'paths_via_Disease',
                 'src_se_count', 'dst_se_count']:
        if feat not in sub_kg.columns: continue
        if sub_kg[feat].isna().any():
            sub_kg[feat] = sub_kg[feat].fillna(0)
        auc = roc_auc_score(sub_kg['y'], sub_kg[feat])
        print(f"  {feat:<25}: AUC = {auc:.3f}")

    # Combine simple features (sum normalized) — proxy for what GNN/path-method could extract
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    X = sub_kg[['shared_se_count', 'paths_via_SideEffect', 'paths_via_Gene', 'paths_via_Disease',
                'src_se_count', 'dst_se_count']].fillna(0)
    y = sub_kg['y']
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    auc_cv = cross_val_score(clf, Xs, y, scoring='roc_auc', cv=5)
    print(f"\n  Combined (LogReg, 5-fold CV, balanced class weight):")
    print(f"    AUC = {auc_cv.mean():.3f} ± {auc_cv.std():.3f}")

except ImportError:
    print("\n(sklearn not available, skipping AUC)")

# === Save augmented results ===
res.to_parquet(f"{OUT}/eval_600_path_features.parquet", index=False)
print(f"\nSaved per-pair features: eval_600_path_features.parquet")

# === Also separately compare PK-B vs NEG and PD-B vs NEG ===
print(f"\n{'='*80}")
print(f"PK-B vs NEG  and  PD-B vs NEG  (in-KG pairs only)")
print(f"{'='*80}")
for cls in ['PK-B', 'PD-B']:
    sub = in_kg_only[in_kg_only['class'].isin([cls, 'NEG'])].copy()
    sub['y'] = (sub['class'] == cls).astype(int)
    print(f"\n  {cls} vs NEG (n_{cls}={(sub['class']==cls).sum()}, n_NEG={(sub['class']=='NEG').sum()}):")
    try:
        for feat in ['shared_se_count', 'paths_via_SideEffect', 'paths_via_Gene', 'total_2hop_paths']:
            auc = roc_auc_score(sub['y'], sub[feat].fillna(0))
            print(f"    {feat:<25}: AUC = {auc:.3f}")
    except:
        pass

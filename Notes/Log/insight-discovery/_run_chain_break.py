"""Sub-finding A: chain break analysis on 200-pair evaluation set."""
import pandas as pd, json, re, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

KG = r"D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Code/data/KG"
OUT = r"D:/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start/Notes/Log/insight-discovery"
ds = pd.read_parquet(f"{OUT}/eval_200_PKB_PDB.parquet")

# === Build L3 keyword dictionary (data-driven, from freq scan + clinical knowledge) ===
L3_DICT = {
    'receptor_channel_enzyme': [
        'cyp450', 'cyp 3a4', 'cyp3a4', 'cyp 2d6', 'cyp2d6', 'cyp 2c9', 'cyp2c9',
        'cyp 2c19', 'cyp2c19', 'cyp 1a2', 'cyp1a2', 'cyp 3a5', 'cyp3a5',
        'isoenzyme', 'isoform',
        'alpha-1 adrenergic', 'alpha-2 adrenergic', 'alpha adrenergic',
        'beta-1 adrenergic', 'beta-2 adrenergic', 'beta adrenergic',
        'dopamine receptor', 'dopaminergic',
        'serotonin receptor', '5-ht receptor',
        'muscarinic receptor', 'nicotinic receptor',
        'histamine receptor', 'h1 receptor', 'h2 receptor',
        'opioid receptor', 'mu receptor', 'kappa receptor',
        'gaba receptor', 'nmda receptor',
        'p-glycoprotein', 'p-gp',
    ],
    'drug_class': [
        'anticholinergic', 'antihistamine', 'antihistaminic',
        'sedative', 'sedating', 'hypnotic',
        'opioid', 'opiate',
        'phenothiazine', 'phenothiazines',
        'tricyclic antidepressant', 'tcas',
        'ssri', 'maoi',
        'beta-blocker', 'beta blocker', 'ace inhibitor',
        'statin', 'hmg-coa', 'nsaid', 'corticosteroid',
        'thiazide', 'loop diuretic',
        'cyp inhibitor', 'cyp inducer', 'cyp substrate',
        'anticoagulant', 'antiplatelet',
        'cns depressant', 'cns stimulant',
        'qt-prolonging', 'qt prolonging',
        'serotonergic', 'neuroleptic', 'antipsychotic',
        'anxiolytic', 'muscle relaxant', 'skeletal muscle relaxant',
    ],
    'system_organ': [
        'central nervous system', 'cns',
        'cardiovascular system', 'cardiac',
        'hepatic', 'liver',
        'renal', 'kidney',
        'gastrointestinal', 'pulmonary', 'respiratory',
        'hematologic', 'autonomic',
    ],
    'mechanism_phenomenon': [
        'qt prolongation', 'qtc prolongation', 'qt interval',
        'torsade de pointes', 'torsades de pointes', 'torsades',
        'ventricular arrhythmia', 'cardiac arrhythmia',
        'sudden death', 'bleeding', 'hemorrhage',
        'orthostatic hypotension', 'hypotension',
        'sedation', 'somnolence', 'drowsiness',
        'cns depression', 'respiratory depression', 'respiratory-depressant',
        'hepatotoxicity', 'liver injury',
        'nephrotoxicity', 'renal failure',
        'cardiotoxicity', 'serotonin syndrome',
        'hyperkalemia', 'hypokalemia',
        'rhabdomyolysis', 'myopathy',
        'electrolyte imbalance',
        'cholinergic crisis', 'anticholinergic syndrome',
        'extrapyramidal', 'neuroleptic malignant syndrome',
        'tachycardia', 'bradycardia',
    ],
}

L3_ALL = []
for layer, terms in L3_DICT.items():
    for t in terms:
        L3_ALL.append((t, layer))
unique_layers = {t: layer for t, layer in L3_ALL}

def extract_L3(text):
    if not isinstance(text, str):
        return []
    t = text.lower()
    found = []
    for term, layer in L3_ALL:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, t):
            found.append((term, layer))
    return found

ds['L3_terms_found'] = ds['ddinter_mechanism'].map(extract_L3)
ds['L3_count'] = ds['L3_terms_found'].map(len)

print(f"L3 term extraction summary:")
print(f"  Pairs with at least 1 L3 term:  {(ds['L3_count']>=1).sum()}/200")
print(f"  Pairs with at least 3 L3 terms: {(ds['L3_count']>=3).sum()}/200")
print(f"  Avg L3 terms per mech text:  PK-B={ds[ds['pkpd_4way']=='PK-B']['L3_count'].mean():.2f}  "
      f"PD-B={ds[ds['pkpd_4way']=='PD-B']['L3_count'].mean():.2f}")

layer_counts = Counter()
for terms in ds['L3_terms_found']:
    for _, layer in terms:
        layer_counts[layer] += 1
print(f"\nTotal L3 term occurrences across 200 texts, by layer:")
for layer, c in layer_counts.most_common():
    print(f"  {layer:<25}: {c}")

all_found = []
for terms in ds['L3_terms_found']:
    all_found.extend([t[0] for t in terms])
print(f"\nTop 20 L3 terms found:")
for term, c in Counter(all_found).most_common(20):
    print(f"  {c:>3}  {term}")

# === Step C.2: KG node existence for each L3 term ===
print(f"\n{'='*70}")
print(f"Step C.2: KG node existence for each L3 term")
print(f"{'='*70}")

hetio_nodes = pd.read_csv(f"{KG}/hetionet/hetionet-v1.0-nodes.tsv", sep='\t')
hetio_node_names = set(hetio_nodes['name'].astype(str).str.lower())

prime = pd.read_csv(f"{KG}/primekg/kg.csv", low_memory=False,
                    usecols=['x_id','x_type','x_name','y_id','y_type','y_name'])
prime_node_names = set()
prime_node_names.update(prime['x_name'].dropna().astype(str).str.lower())
prime_node_names.update(prime['y_name'].dropna().astype(str).str.lower())
print(f"Hetionet unique node names: {len(hetio_node_names):,}")
print(f"PrimeKG unique node names:  {len(prime_node_names):,}")

def check_in_kg(term, name_set):
    if term in name_set:
        return ('exact', term)
    for n in name_set:
        if term in n:
            return ('substring', n)
    return (None, None)

used_terms = sorted(set(all_found))
print(f"\n--- L3 terms USED in mechanism texts vs KG findability ---")
print(f"{'Term':<35} {'Layer':<24} {'Hetio':<10} {'Prime':<10}")
print("-" * 90)
for term in used_terms:
    layer = unique_layers[term]
    ht, _ = check_in_kg(term, hetio_node_names)
    pt, _ = check_in_kg(term, prime_node_names)
    ht_str = ht if ht else "MISSING"
    pt_str = pt if pt else "MISSING"
    print(f"{term:<35} {layer:<24} {ht_str:<10} {pt_str:<10}")

print(f"\n--- Layer-level findability (weighted by occurrence in 200 texts) ---")
print(f"{'Layer':<25} {'Total occur':>12} {'Hetio match':>14} {'Prime match':>14} {'Either':>12}")
for layer in L3_DICT.keys():
    occ = sum(1 for terms_in_text in ds['L3_terms_found']
              for t, lyr in terms_in_text if lyr == layer)
    h_occ = sum(1 for terms_in_text in ds['L3_terms_found']
                for t, lyr in terms_in_text
                if lyr == layer and check_in_kg(t, hetio_node_names)[0])
    p_occ = sum(1 for terms_in_text in ds['L3_terms_found']
                for t, lyr in terms_in_text
                if lyr == layer and check_in_kg(t, prime_node_names)[0])
    either = sum(1 for terms_in_text in ds['L3_terms_found']
                 for t, lyr in terms_in_text
                 if lyr == layer and
                 (check_in_kg(t, hetio_node_names)[0] or check_in_kg(t, prime_node_names)[0]))
    h_pct = 100*h_occ/max(occ,1)
    p_pct = 100*p_occ/max(occ,1)
    e_pct = 100*either/max(occ,1)
    print(f"{layer:<25} {occ:>12} {h_occ:>5} ({h_pct:>4.0f}%)  {p_occ:>5} ({p_pct:>4.0f}%)  {either:>5} ({e_pct:>4.0f}%)")

# Save per-pair findability for later use
ds['L3_in_kg_count'] = ds['L3_terms_found'].apply(
    lambda terms: sum(1 for t, _ in terms
                       if check_in_kg(t, hetio_node_names)[0] or check_in_kg(t, prime_node_names)[0]))
ds['L3_findable_ratio'] = ds.apply(
    lambda r: r['L3_in_kg_count'] / r['L3_count'] if r['L3_count'] > 0 else None, axis=1)

print(f"\n--- Per-pair L3 findability ratio ---")
for cls in ['PK-B', 'PD-B']:
    sub = ds[ds['pkpd_4way'] == cls]
    valid = sub['L3_findable_ratio'].dropna()
    print(f"  {cls}: n={len(valid)}, mean={valid.mean():.2f}, median={valid.median():.2f}, "
          f"min={valid.min():.2f}, max={valid.max():.2f}")

# Save augmented dataset for downstream use
ds.to_parquet(f"{OUT}/eval_200_with_L3_analysis.parquet", index=False)
print(f"\nSaved augmented dataset: eval_200_with_L3_analysis.parquet")

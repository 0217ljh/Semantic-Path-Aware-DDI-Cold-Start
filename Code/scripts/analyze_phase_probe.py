"""GO/NO-GO 'phase probe' for the dual-source rotational-interference DDI idea.

Question (codex kill test): on same-property up/down DDI class pairs, does a
CONTINUOUS rotational compatibility on shared-mediator relation structure
separate DIRECTION better than a BINARY-SIGN version? And are the learned up/down
class orientations ~antipodal?

This is the cheap pre-build diagnostic. 1-hop shared mediators only (dominant
signal); L=2 left as a follow-up. Read-only over KG + 215 typed labels.

GO  if continuous beats sign by a material macro-F1 margin on the paired-direction
    slice AND up/down orientations are ~antipodal (cos < -0.7).
NO-GO if margin ~0 or orientations not antipodal -> rotational story is decoration.

Run:
  python Code/scripts/analyze_phase_probe.py
"""
import json, re, sys, collections
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.emergnn.kg_builder_merged import build_kg_from_merged_parquet
from data_utils import PairDataset

MERGED = ROOT / "data/KG/_merged_kg/edges__drugbank_hetionet_primekg__mask1.parquet"
TYPEMAP = ROOT / "data/_cache/ddi_type_map.json"
SEED = 42
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

# ---------------- 1. KG: drug -> {mediator: 18-dim relation count} -------------
ds = PairDataset.from_pkl(str(ROOT / "data/coldddi_legacy/800drug/seed42.pkl"))
drug_ids = set()
for _n, df in ds.splits.items():
    drug_ids.update(df["drug_a_id"].astype(str)); drug_ids.update(df["drug_b_id"].astype(str))
if hasattr(ds.kg, "drug_ids"):
    drug_ids.update(ds.kg.drug_ids)
art = build_kg_from_merged_parquet(MERGED, sorted(drug_ids), verbose=False)
e2i = art["entity2id"]; trip = np.asarray(art["triplets"], np.int64)
n_rel = art["n_rel"]; n_drugs = len(art["drug_ids"])
print(f"KG: n_ent={art['n_ent']} n_rel={n_rel} n_drugs={n_drugs} n_edges={len(trip)}")

nbr = collections.defaultdict(lambda: collections.defaultdict(lambda: np.zeros(n_rel, np.float32)))
for h, t, r in trip:
    nbr[int(h)][int(t)][int(r)] += 1.0
nbr = {d: dict(m) for d, m in nbr.items()}

# ---------------- 2. parse 215 types -> (property, direction) ------------------
d = json.load(open(TYPEMAP, encoding="utf-8"))
idx_to_type = d["idx_to_type"]; pti = d["pair_to_idx"]
UP = re.compile(r"\b(increase|increased|potentiat)\b", re.I)
DN = re.compile(r"\b(decrease|decreased|reduce|reduced)\b", re.I)

def parse(t):
    up, dn = bool(UP.search(t)), bool(DN.search(t))
    if up == dn:
        return None  # skip neither / both (ambiguous)
    direction = "up" if up else "dn"
    # property extraction (templated DrugBank strings)
    m = re.search(r"may (?:increase|decrease) the (.+?) of\b", t, re.I)
    if not m:
        m = re.search(r"The (.+?) of .*can be (?:increased|decreased)", t, re.I)
    if not m:
        m = re.search(r"The (.+?) can be (?:increased|decreased)", t, re.I)
    prop = m.group(1).strip().lower() if m else None
    return (prop, direction)

cls_pd = {}
for i, t in enumerate(idx_to_type):
    pd_ = parse(t)
    if pd_ and pd_[0]:
        cls_pd[i] = pd_
# paired properties = have both up and dn classes
prop_dirs = collections.defaultdict(set)
for i, (p, dr) in cls_pd.items():
    prop_dirs[p].add(dr)
paired_props = {p for p, s in prop_dirs.items() if {"up", "dn"} <= s}
print(f"parsed classes={len(cls_pd)}  paired properties (have up&dn)={len(paired_props)}")
print("  paired props:", sorted(paired_props)[:25])

# ---------------- 3. build paired-slice dataset of pair -> (prop, dir) ---------
# pair_to_idx keys are canonical 'A|B'. class must be in a paired property.
cls_to_pd = {i: cls_pd[i] for i in cls_pd if cls_pd[i][0] in paired_props}
samples = []  # (a_idx, b_idx, prop, y)  y=1 up, 0 dn
PER = 4000    # cap pairs per (prop,dir) to keep it afternoon-scale
cnt = collections.Counter()
items = list(pti.items()); rng.shuffle(items)
for key, ci in items:
    if ci not in cls_to_pd:
        continue
    prop, dr = cls_to_pd[ci]
    if cnt[(prop, dr)] >= PER:
        continue
    a, b = key.split("|")
    if a not in e2i or b not in e2i:
        continue
    ai, bi = e2i[a], e2i[b]
    if ai not in nbr or bi not in nbr:
        continue
    cnt[(prop, dr)] += 1
    samples.append((ai, bi, prop, 1 if dr == "up" else 0))
print(f"slice samples={len(samples)}  (cap {PER}/dir)")

# ---------------- 4. featurize: per pair, shared-mediator delta/rho ------------
MAXM = 64
feats = []  # per sample: (delta (M,18), rho (M,))
keep = []
for ai, bi, prop, y in samples:
    na, nb = nbr[ai], nbr[bi]
    shared = [m for m in na.keys() if m in nb and m != ai and m != bi]
    if not shared:
        continue
    if len(shared) > MAXM:
        # keep top-MAXM by rho
        shared.sort(key=lambda m: float(na[m].sum() * nb[m].sum()), reverse=True)
        shared = shared[:MAXM]
    xa = np.stack([na[m] for m in shared]); xb = np.stack([nb[m] for m in shared])
    delta = xa - xb                                  # (M,18)
    rho = xa.sum(1) * xb.sum(1)                       # (M,)
    feats.append((torch.tensor(delta), torch.tensor(rho)))
    keep.append((prop, y))
print(f"pairs with >=1 shared mediator: {len(keep)} / {len(samples)} "
      f"({100*len(keep)/max(len(samples),1):.1f}%)")
props = sorted(set(p for p, _ in keep))
yall = np.array([y for _, y in keep])
print(f"label balance up={int(yall.sum())} dn={int((1-yall).sum())}")

# train/val split
idx = np.arange(len(keep)); rng.shuffle(idx)
ntr = int(0.7 * len(idx)); tr, va = idx[:ntr], idx[ntr:]

# ---------------- 5. continuous (rotational) vs sign probe ---------------------
def pad(batch_idx):
    M = max(feats[i][0].shape[0] for i in batch_idx)
    D = torch.zeros(len(batch_idx), M, n_rel); R = torch.zeros(len(batch_idx), M)
    for j, i in enumerate(batch_idx):
        de, rh = feats[i]; m = de.shape[0]
        D[j, :m] = de; R[j, :m] = rh
    return D, R

class Cont(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.theta = torch.nn.Parameter(0.01*torch.randn(n_rel)); s.head = torch.nn.Linear(2,1)
    def forward(s, D, R):
        ang = D @ s.theta                       # (B,M)
        feat = torch.stack([(R*torch.cos(ang)).sum(1), (R*torch.sin(ang)).sum(1)], -1)
        return s.head(feat).squeeze(-1)

class Sign(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.w = torch.nn.Parameter(0.01*torch.randn(n_rel)); s.head = torch.nn.Linear(1,1)
    def forward(s, D, R):
        proj = D @ s.w                          # (B,M)
        feat = (R*torch.tanh(8.0*proj)).sum(1, keepdim=True)   # soft sign in {-,+}
        return s.head(feat).squeeze(-1)

ytr = torch.tensor(yall[tr], dtype=torch.float32); yva = torch.tensor(yall[va], dtype=torch.float32)
Dtr, Rtr = pad(tr); Dva, Rva = pad(va)

def train(model):
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(400):
        opt.zero_grad(); out = model(Dtr, Rtr); loss = lossf(out, ytr); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (model(Dva, Rva) > 0).float().numpy()
    return f1_score(yva.numpy(), pred, average="macro")

f1_cont = train(Cont()); f1_sign = train(Sign())
print(f"\n=== paired-direction (up vs dn) macro-F1 on held-out ===")
print(f"  continuous (rotational 2D): {f1_cont:.4f}")
print(f"  sign-restricted (1D +/-)  : {f1_sign:.4f}")
print(f"  Delta_paired = {f1_cont - f1_sign:+.4f}")

# antipodal check: per-property up vs dn centroid angle in (cos,sin) plane
cont = Cont(); train(cont)
with torch.no_grad():
    Dall, Rall = pad(np.arange(len(keep)))
    ang = Dall @ cont.theta
    z = torch.stack([(Rall*torch.cos(ang)).sum(1), (Rall*torch.sin(ang)).sum(1)], -1).numpy()
cosvals = []
for p in props:
    mu = np.array([p2 == p for p2, _ in keep])
    zu = z[mu & (yall == 1)]; zd = z[mu & (yall == 0)]
    if len(zu) < 5 or len(zd) < 5:
        continue
    cu = zu.mean(0); cd = zd.mean(0)
    c = float(cu @ cd / (np.linalg.norm(cu)*np.linalg.norm(cd) + 1e-9))
    cosvals.append(c)
A = float(np.median(cosvals)) if cosvals else float("nan")
print(f"  antipodal A = median cos(up_centroid, dn_centroid) over {len(cosvals)} props = {A:.3f} (want < -0.7)")

go = (f1_cont - f1_sign) > 0.05 and A < -0.7
print(f"\n=== VERDICT: {'GO' if go else 'NO-GO / borderline'} ===")
print("  (caveat: 1-hop shared mediators only, sampled; L=2 not yet tested)")

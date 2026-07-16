"""Read-only: promote/antagonize breakdown of DDI types. TEMP."""
import json, re, collections, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "data/_cache/ddi_type_map.json"
print("exists:", p.is_file(), "|", p)
d = json.load(open(p, encoding="utf-8"))
print("top-level keys:", list(d.keys()))
idx_to_type = d.get("idx_to_type") or []
pti = d.get("pair_to_idx", {})
print("n_types:", len(idx_to_type), "| n_pairs:", len(pti))
svals = list(itertools.islice(pti.values(), 8))
print("sample pair_to_idx values:", svals, "| value type:", type(svals[0]).__name__ if svals else None)

inc = re.compile(r"increase|increased|higher|potentiat|risk or severity", re.I)
dec = re.compile(r"decrease|decreased|reduce|reduced|lower|antagon", re.I)

def bucket(t):
    hi, ho = bool(inc.search(t)), bool(dec.search(t))
    return "both" if hi and ho else "inc" if hi else "dec" if ho else "neither"

tb = [bucket(t) for t in idx_to_type]
print("type-level:", dict(collections.Counter(tb)))

paircnt = collections.Counter()
for v in pti.values():
    if isinstance(v, int) and 0 <= v < len(tb):
        paircnt[tb[v]] += 1
    else:
        paircnt["nonint_or_oob"] += 1
print("pair-level:", dict(paircnt))

print("--- all 215 types (idx | bucket | string) ---")
for i, t in enumerate(idx_to_type):
    print(f"{i:3d} | {tb[i]:7s} | {t}")

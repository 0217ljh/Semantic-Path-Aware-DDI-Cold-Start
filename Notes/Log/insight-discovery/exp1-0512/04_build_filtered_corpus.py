"""V2 filter on the 400-pair eval dataset (200 PK-B + 200 PD-B, all with DDInter
mechanism text). Scope-limited so user can hand-review.

DEFINITION OF LEAK (per user, 2026-05-13):
  A sentence leaks the DDI label iff it gives CLINICAL DECISION / PRESCRIPTIVE
  guidance — i.e., advises on how to HANDLE the pair, rather than describing a
  pharmacological phenomenon.

KEEP:
  - Drug names (drug_a, drug_b, others) — never mask
  - Individual drug pharmacology, side effects, mechanism of action
  - Drug class descriptions
  - Phenomenon descriptions even framed as "combining causes additive Z"
  - "Increased risk of Z when combined" / "may result in additive Z" — phenomenon

REMOVE (the leak signals):
  - Prescriptive verbs: "should be (avoided|used with caution|monitored|adjusted)",
      "is/are recommended", "not recommended", "contraindicated", "must be",
      "consider (avoiding|alternative|reducing)", "advised"
  - Caution words: caution / careful / monitoring / vigilant / vigilance
  - Treatment decisions: alternative therapy, dose adjustment may be needed,
      discontinue, interrupt therapy, withhold
  - Direct interaction claims: "X interacts with Y", "drug-drug interaction"
"""
from __future__ import annotations
import sys, re, json, time
from pathlib import Path
import pandas as pd

_HERE = Path(__file__).resolve()
_ROOT = next(p for p in [_HERE, *_HERE.parents] if (p / "Code" / "data" / "KG").is_dir())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KG = _ROOT / "Code" / "data" / "KG"
OUT = _HERE.parent

# ----------------------------------------------------------------------------
# 1. Leak vocabulary (v2 — prescriptive guidance only)
# ----------------------------------------------------------------------------
LEAK_PATTERNS = [
    # Direct interaction claim (the word itself reveals DDI)
    r"\b(?:may|can|might)\s+interact\b",
    r"\binteract(?:s|ion|ions)\s+(?:with|may|can|might|between)",
    r"\bdrug[- ]drug\s+interaction",
    r"\bdrug\s+interaction",

    # Prescriptive guidance
    r"\bshould\s+(?:be|not|generally|always|never|preferably|ideally|carefully)\b",
    r"\b(?:are|is)\s+(?:recommended|not\s+recommended|contraindicated|advised|warranted)\b",
    r"\b(?:are|is)\s+(?:to\s+be\s+)?avoided\b",
    r"\bmust\s+be\b",
    r"\bought\s+to\b",
    r"\bnot\s+recommended\b",
    r"^\s*RECOMMENDED\s*[:.\-]",         # DDInter prescriptive header line
    r"\brecommended\s+(?:to|for|that|when|dose|dosage|approach|alternative|monitoring)\b",
    r"\bcontraindicat",
    r"\bavoid(?:ed|ing)?\s+(?:concurrent|concomitant|coadministration|combination|use\s+with|simultaneous)",
    r"\bavoid(?:ance)?\b",

    # Caution / monitoring
    r"\bcaution(?:\s+(?:is|should|must|are))?\b",
    r"\bcareful(?:ly)?\s+monitor",
    r"\bclose(?:ly)?\s+monitor",
    r"\b(?:should|must|may)\s+be\s+monitor",
    r"\bmonitor(?:ed|ing)?\s+(?:for|closely|carefully|frequently|regularly)\b",
    r"\bvigilan(?:t|ce)\b",

    # Treatment decisions
    r"\balternative\s+(?:therap|treatment|agent|medication|drug|approach)",
    r"\bdose\s+(?:adjustment|reduction|modification|titration)\s+(?:may|might|should|is)",
    r"\b(?:reduce|lower|decrease|adjust|modify)\s+(?:the\s+)?dose\b",
    # discontinu*: only when it's a prescriptive verb action, not a phenomenon noun
    r"\bdiscontinue(?:d|s)?\s+(?:therap|treatment|use|administration|the\s+drug|both|coadministration|concomitant|if|when|immediately)",
    r"\bshould\s+be\s+discontinued\b",
    r"\binterrupt(?:ed|ion)?\s+(?:therap|treatment|use)",
    r"\bwithhold",
    r"\bwithdraw\s+(?:therap|treatment)",
    r"\bif\s+(?:concurrent|concomitant|coadministration|combination)\s+(?:is|use|therap)\s+(?:is\s+)?(?:necessary|required|deemed)",

    # Clinical warning
    r"\bwarning\b", r"\bwarned\b",
    r"\b(?:patients?\s+(?:should|must))\b",
    r"\bclinically\s+(?:significant|relevant|important)\s+interaction",
]
LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), flags=re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(<])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def filter_text(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return dict(filtered_text="", removed_sentences=[], n_orig=0,
                    n_kept=0, n_removed=0, char_orig=0, char_filtered=0)
    sentences = split_sentences(text)
    kept, removed = [], []
    for s in sentences:
        if LEAK_REGEX.search(s):
            removed.append(s)
        else:
            kept.append(s)
    return dict(
        filtered_text=" ".join(kept),
        removed_sentences=removed,
        n_orig=len(sentences), n_kept=len(kept), n_removed=len(removed),
        char_orig=len(text), char_filtered=len(" ".join(kept)),
    )


# ----------------------------------------------------------------------------
# 2. Build / load the 400-pair eval set (200 PK-B + 200 PD-B with mechanism)
# ----------------------------------------------------------------------------
EVAL_FILE = OUT.parent / "eval_400_PKB_PDB.parquet"
if EVAL_FILE.exists():
    print(f"Loading existing eval set: {EVAL_FILE.name}")
    eval_df = pd.read_parquet(EVAL_FILE)
else:
    print(f"Building eval set (200 PK-B + 200 PD-B with DDInter mechanism)...")
    pkpd_csv = pd.read_csv(KG / "drugbank" / "enriched" / "ddi_pk_pd_labels.csv")
    def _4w(row):
        label = row["pk_pd_label"]
        if pd.isna(label) or label == "Mixed": return label
        if label == "PK":
            kws = str(row.get("matched_pk_keywords", "") or "").lower()
            return "PK-B" if ("metabolism" in kws or "serum concentration" in kws) else "PK-A"
        pkws = str(row.get("matched_pd_keywords", "") or "").lower()
        return "PD-B" if ("risk" in pkws or "adverse" in pkws or "severity" in pkws) else "PD-A"
    pkpd_csv["pkpd_4way"] = pkpd_csv.apply(_4w, axis=1)
    labels_map = dict(zip(pkpd_csv["ddi_type"], pkpd_csv["pkpd_4way"]))

    mech = pd.read_csv(KG / "drugbank" / "filtered" / "drugbank_with_mechanisms.csv")
    mech = mech[mech["ddinter_mechanism"].notna() & (mech["ddinter_mechanism"].str.len() > 0)].copy()
    mech["pkpd_4way"] = mech["ddi_type"].map(labels_map)
    id2name = json.load(open(KG / "drugbank" / "filtered" / "id2name.json"))
    mech["drug_a_name"] = mech["drug_a_id"].map(id2name)
    mech["drug_b_name"] = mech["drug_b_id"].map(id2name)

    pool_pkb = mech[mech["pkpd_4way"] == "PK-B"]
    pool_pdb = mech[mech["pkpd_4way"] == "PD-B"]
    SEED = 42
    samp_pkb = pool_pkb.sample(n=200, random_state=SEED).reset_index(drop=True)
    samp_pdb = pool_pdb.sample(n=200, random_state=SEED).reset_index(drop=True)
    samp_pkb["pair_id"] = ["PKB-" + str(i+1).zfill(3) for i in range(200)]
    samp_pdb["pair_id"] = ["PDB-" + str(i+1).zfill(3) for i in range(200)]
    eval_df = pd.concat([samp_pkb, samp_pdb], ignore_index=True)[
        ["pair_id", "pkpd_4way", "drug_a_id", "drug_a_name",
         "drug_b_id", "drug_b_name", "ddi_type", "description", "ddinter_mechanism"]
    ]
    eval_df.to_parquet(EVAL_FILE, index=False)
    print(f"  saved: {EVAL_FILE.name}  ({len(eval_df)} rows)")

print(f"Eval set: {len(eval_df)} rows  ({eval_df['pkpd_4way'].value_counts().to_dict()})")

# ----------------------------------------------------------------------------
# 3. Apply v2 filter
# ----------------------------------------------------------------------------
print(f"\nApplying v2 filter (drug names preserved)...")
t0 = time.time()
results = []
for r in eval_df.itertuples(index=False):
    out = filter_text(r.ddinter_mechanism)
    results.append(dict(
        pair_id=r.pair_id,
        pkpd_4way=r.pkpd_4way,
        drug_a_id=r.drug_a_id, drug_a_name=r.drug_a_name,
        drug_b_id=r.drug_b_id, drug_b_name=r.drug_b_name,
        ddi_type=r.ddi_type,
        original_text=r.ddinter_mechanism,
        filtered_text=out["filtered_text"],
        removed_sentences=" || ".join(out["removed_sentences"]),
        n_sent_orig=out["n_orig"],   n_sent_kept=out["n_kept"],   n_sent_removed=out["n_removed"],
        char_orig=out["char_orig"], char_filtered=out["char_filtered"],
    ))
corpus = pd.DataFrame(results)
print(f"Done in {time.time()-t0:.1f}s")

# ----------------------------------------------------------------------------
# 4. Save outputs
# ----------------------------------------------------------------------------
out_parquet = OUT / "filtered_corpus_400.parquet"
out_csv     = OUT / "filtered_corpus_400.csv"
out_xlsx    = OUT / "filtered_corpus_400.xlsx"
out_stats   = OUT / "filter_stats_400.json"
out_samples = OUT / "filter_samples_400.txt"

corpus.to_parquet(out_parquet, index=False)
corpus.to_csv(out_csv, index=False, encoding="utf-8-sig")
# Excel: drop the long removed_sentences column to avoid huge cells
corpus.drop(columns=["removed_sentences"]).to_excel(out_xlsx, index=False)

stats = {
    "filter_version": "v2.1 — prescriptive-only, drug names preserved (RECOMMENDED-header catch + discontinu noun-form preserved)",
    "n_rows": int(len(corpus)),
    "n_PKB": int((corpus["pkpd_4way"] == "PK-B").sum()),
    "n_PDB": int((corpus["pkpd_4way"] == "PD-B").sum()),
    "char_reduction": {
        "mean_original":   float(corpus["char_orig"].mean()),
        "mean_filtered":   float(corpus["char_filtered"].mean()),
        "mean_pct_kept":   float((corpus["char_filtered"] / corpus["char_orig"]).mean() * 100),
    },
    "sentence_reduction": {
        "mean_orig_sent":     float(corpus["n_sent_orig"].mean()),
        "mean_kept_sent":     float(corpus["n_sent_kept"].mean()),
        "mean_pct_sent_kept": float((corpus["n_sent_kept"] / corpus["n_sent_orig"]).mean() * 100),
    },
    "rows_with_empty_filtered_text":   int((corpus["char_filtered"] == 0).sum()),
    "rows_unchanged_by_filter":        int((corpus["char_orig"] == corpus["char_filtered"]).sum()),
    "rows_filtered_text_<30char":      int((corpus["char_filtered"] < 30).sum()),
}
out_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

# Write 30 readable samples (10 PKB / 10 PDB / 10 mixed-by-length)
sampled_idx = (
    corpus[corpus["pkpd_4way"]=="PK-B"].head(10).index.tolist()
    + corpus[corpus["pkpd_4way"]=="PD-B"].head(10).index.tolist()
    + corpus.sort_values("char_orig", ascending=False).head(10).index.tolist()
)
sampled_idx = list(dict.fromkeys(sampled_idx))  # dedup preserve order
with open(out_samples, "w", encoding="utf-8") as f:
    for i, idx in enumerate(sampled_idx):
        r = corpus.iloc[idx]
        f.write(f"\n{'='*100}\n")
        f.write(f"[{i+1}] {r['pair_id']}  {r['drug_a_id']} ({r['drug_a_name']}) "
                f"↔ {r['drug_b_id']} ({r['drug_b_name']})  class={r['pkpd_4way']}\n")
        f.write(f"ddi_type: {r['ddi_type']}\n")
        f.write(f"{'-'*100}\n[ORIGINAL ({r['char_orig']} chars / {r['n_sent_orig']} sentences)]:\n")
        f.write(r["original_text"] + "\n")
        f.write(f"{'-'*100}\n[FILTERED ({r['char_filtered']} chars / {r['n_sent_kept']} sent)]:\n")
        f.write((r["filtered_text"] or "(empty)") + "\n")
        f.write(f"{'-'*100}\n[REMOVED ({r['n_sent_removed']} sentences — prescriptive language):\n")
        f.write(r["removed_sentences"].replace(" || ", "\n  • ") + "\n")

print(f"\nSaved:")
print(f"  {out_parquet}")
print(f"  {out_csv}")
print(f"  {out_xlsx}")
print(f"  {out_stats}")
print(f"  {out_samples}  (30 samples for spot-checking)")

print(f"\n=== v2 stats on 400 pairs ===")
print(json.dumps(stats, indent=2))

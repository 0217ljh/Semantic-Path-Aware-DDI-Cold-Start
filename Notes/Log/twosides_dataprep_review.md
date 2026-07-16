# Codex review — TWOSIDES unified data-prep (prepare_twosides.py)

- Date: 2026-06-29
- Primary reviewer: Claude (opus-4.8) — drafted plan + script
- Independent reviewer: codex (gpt-5-codex), thread 019f153e, 2 rounds (plan + final code)
- Trigger: standing project rule "每个数据集处理步都要 codex 讨论 + review"; user
  asked to query the TWOSIDES source and handle it.
- Scope: `Code/scripts/prepare_twosides.py` → `Code/data/ddi_unified/`
  {twosides_warm_ml, twosides_s1_ml, twosides_s2_ml}. Format facts in
  [[twosides_source_format]].

## Round 1 — plan review (before writing)
codex verdicts:
- (a) pos/neg pairing: do NOT rely on row order, do NOT extend the fixed row schema;
  persist pairing in a SEPARATE sidecar file. → adopted `<split>_pair_links.parquet`.
- (b) S0 3-fold: re-splitting = benchmark drift; codex preferred official single
  split. → user explicitly chose to re-split into 3 seeded folds; documented as a
  protocol choice, not an EmerGNN-official reproduction.
- (c) pair_format="ordered_source", no canonicalization → confirmed correct
  (canonicalizing would destroy the head-corruption negative structure).
- (d) **benchmark-integrity**: cold validators must NOT run on negatives — eval
  negatives are endpoint-corruptions and would falsely trip S1/S2 disjointness. →
  adopted eval-side = positives only.

## Round 2 — final code review
Deviation I raised: codex round-1 said derive train-seen from positives only; I set
train-seen = ALL train-row drugs (pos+neg) after verifying EmerGNN
`TWOSIDES/load_data.py:75-77` populates `train_ent` from every train line (pos AND
neg). codex verdict: **"Yes. train_seen = all train rows is the correct faithful
choice"** — a drug seen only as a corrupted-negative endpoint is still "seen" in the
exact sense EmerGNN uses; positives-only would understate exposure and could falsely
certify a row as cold. Correct asymmetry = train side all rows, eval side positives.

codex found 3 issues (all fixed):
1. (real) `drugbank_id` backfilled the entity id when source `db` is null → fabricates
   a DrugBank-like id. FIXED: null db → `pd.NA` ("<NA>" after str-coercion); 269/604
   drugs have no DrugBank map, 335 keep real DB ids.
2. (doc) header claimed cold validators run positives-only, but code uses the
   asymmetric rule. FIXED: docstring rewritten to state train=all-rows, eval=positives.
3. (weak) KG pointer used a `<setting>` glob placeholder. FIXED: concrete dir
   `Code/reproductions/EmerGNN/_Original-Dataset/TWOSIDES/data`.

codex final: "Other than the drugbank_id backfill, I do not see another
benchmark-corrupting issue. S1/S2 split semantics are coherent under the
EmerGNN-compatible seen-set definition."

## Verified output
- All 3: task=multilabel, 200 labels (label_vocab = relation2id[0..199] UMLS CUIs),
  y_label_ids ∈ 0..199, no empty-label rows, 1:1 pos:neg, morgan dim 1024,
  pair_format ordered_source, kg scope=dataset.
- twosides_s2_ml fold0: drug_split 514 train_seen + 83 eval_unseen; test-positive
  drugs leaking into train-seen = 0 (true cold S2).
- twosides_s1_ml fold0: 514 train_seen + 90 eval_unseen; check_cold_s1
  (exactly-one-seen on eval positives) passed.
- pair_links sidecar integrity: pos_pair_id all map to positive rows, neg_pair_id all
  to negative rows.

Verdict: PASS (data-prep). Caveat: S0 warm 3-fold is a project protocol choice, not an
EmerGNN-official split (user-approved).

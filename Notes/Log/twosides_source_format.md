# TWOSIDES dataset source + format (verified from authoritative loader)

Verified 2026-06-29 by reading EmerGNN github `TWOSIDES/load_data.py` (authoritative)
and the shipped files under
`Code/reproductions/EmerGNN/_Original-Dataset/TWOSIDES/data/`. All numbers below were
counted from the real files, not recalled.

## Provenance
- Original source = **TWOSIDES** (Tatonetti et al. 2012, ref [23] in EmerGNN paper),
  the polypharmacy side-effect database. PubChem-CID drugs, UMLS-CUI side effects.
- The split + KG packaging we use is **EmerGNN's** preprocessing (LARS-research/EmerGNN),
  shipped as `S0 / S1_* / S2_*` directories.

## On-disk layout
```
TWOSIDES/data/
  S0/            <- warm / double-known  (single split, NO cv folds)
  S1_1 .. S1_12345   <- one-drug-unseen, 5 folds
  S2_1 .. S2_12345   <- both-drugs-unseen (cold-start), 5 folds
  necessary/
    id2drug__official.json        604 drugs: {entity_id: {cid, db, smiles}}
    id2drug_feat__official.pkl    604 drugs: {entity_id: {cid,db,smiles,Morgan,rdkit2d}}
    relation2id__official.json    523 relations (id->UMLS CUI); ids 0..199 = the
                                  200 DDI side-effect labels (all C-prefixed),
                                  200..522 = KG relations
    entity2id__official.json      33444 KG entities (drugs are ids 0..603)
  S*/{train,valid,test}_ddi.txt   the DDI pairs (format below)
  S*/{train,valid,test}_KG.txt    per-split KG triplets (h t r), cumulative
```

## `*_ddi.txt` row format (TAB-separated, 4 fields)
`x \t y \t z \t w`
- `x`, `y` = head/tail drug entity id (int, 0..603)
- `z` = comma-joined **200-wide multihot** over side-effect labels (`z1=map(int,z.split(','))`)
- `w` = flag: **1 = positive, 0 = negative**

Files already ship **1:1 positive:negative**, paired by consecutive rows.

### Negative construction (verified from raw rows)
A negative row **corrupts ONE endpoint** of its paired positive and keeps the SAME
relation multihot. e.g. S2_1/train:
```
h=4   t=5  w=1  labels=[5..18]      <- positive
h=197 t=5  w=0  labels=[5..18]      <- negative: head 4->197, tail+labels unchanged
```
So a negative = "this corrupted pair does NOT have these side-effects". NOT a
relation-corruption. Paper line 639: "follow [6] to randomly sample one negative
sample for each (u,i,v)".

## Per-split stats (counted from files)
| setting | split | pos (=neg) | drugs |
|---|---|---|---|
| S0 (warm)      | train/valid/test | 28889 / 4128 / 8253 | 604 (shared pool) |
| S1_1 (one-unseen) | train/valid/test | 30134 / 3570 / 6699 | 514 / 544 / 574 |
| S2_1 (both-unseen)| train/valid/test | 30134 / 106 / 355  | 514 / 30 / 60 (eval disjoint from train) |

- Label width W = **200** for every split.
- S2 valid/test drugs are disjoint from the 514 train drugs (true cold-start S2).

## Loader semantics (`load_data.py`)
- `process_files_ddi`: stores `pos_triplets[split] = [x, y, *z1]`, same for neg.
- `shuffle_train`: the **S0/S1/S2 setting is chosen by the dir name prefix**
  (`params.dataset.startswith('S1'/'S2'/'S0')`); inside train it carves fact-graph
  (both endpoints in a subsampled train_ent) vs supervised pos/neg (S1=exactly-one in
  train_ent, S2=neither in train_ent).
- KG: `train_KG` then `valid_KG = train+valid`, `test_KG = +test` (cumulative,
  split-scoped) -> dataset-scoped KG reference in our unified layout (like ddi800).

## Mapping to our unified layout (planned, pending codex review)
3 dataset_ids: `twosides_warm_ml` (S0) / `twosides_s1_ml` (S1) / `twosides_s2_ml` (S2).
- task = multilabel; row schema `pair_id, drug_a_id, drug_b_id, is_positive, y_label_ids`.
- drug_id = entity id (str); source_drug_id = CID; drugbank_id = db; smiles from id2drug;
  morgan from id2drug_feat (1024? to verify).
- label_vocab = {0..199 -> relation2id[str(i)] UMLS CUI}.
- S1/S2 -> take folds `_1/_12/_123` as fold0/1/2 (3-fold). S0 -> single split (open Q).
- S2 = cold_s2 (+drug_split), S1 = cold_s1 (exactly-one-seen), S0 = transductive.
- OPEN design Qs for codex: (a) how to keep pos/neg pairing (group_id?); (b) S0 to
  3-fold or keep single; (c) drug_id = entity-id vs CID.

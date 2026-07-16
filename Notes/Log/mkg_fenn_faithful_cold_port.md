# MKG-FENN unified port — FAITHFUL cold-start design (verified from official source)

Date: 2026-07-02. User chose the FAITHFUL path (not the simplified 4ch-for-all).
All facts below verified from the official upstream at
`Paper/Reference/Original-Code/MKG-FENN/Code and Datasets/code/` (paths relative to that dir).

## Paper task = MULTICLASS 65-event (softmax CE), NOT multilabel
- `--event_num default=65` (MKG-FENN-task1.py:39, task2.py:38, task3.py:38)
- `loss_function = nn.CrossEntropyLoss()` (task1.py:175, task2.py:172, task3.py:172)
- softmax only at eval (task1.py:206, task2.py:212, task3.py:215)
- task1 = warm 5-fold pair CV; task2 = ONE-unseen cold; task3 = BOTH-unseen cold.

## CRITICAL: the three official models DIFFER (md5 all distinct). Cold ≠ warm.
diff modeltask1.py (warm) vs modeltask3.py (both-unseen cold):
1. **Channel count**: warm fuses 4 channels (`embedding_num*4*2 -> ... -> 65`, concat
   gnn1/gnn2/gnn3/**gnn4**). COLD fuses **3 channels** (`embedding_num*3*2`, concat
   gnn1/gnn2/gnn3 only — GNN4 molecular-property channel dropped from FUSION).
2. **Cold imputation**: cold model forward takes `datas=(idx, train_or_test, test_adj)`;
   when `train_or_test==1`, for each unseen test drug it REPLACES that drug's per-channel
   embedding with the MEAN of its nearest-seen-neighbour drugs' embeddings:
   `drug_f[i] = sum(drug_f[pos])/len(pos)` (modeltask3.py, each GNN forward). warm model has none.

## Nearest-seen-neighbour structure (test_adj), built in task3.py:344-395
- Per test(unseen) drug `j`, per channel `k in {0,1,2,3}`: scan `drug_sim{k}[j]`, take the
  argmax-similarity drug(s) `p` s.t. `p not in test_drug` and `p != j` (i.e. most similar SEEN
  drug(s); ties kept as a list). `test_adj[i][k][j] = [most-similar-seen-drug-ids]`.
- Per-channel drug-drug similarity (task3.py:313-319):
  - `drug_sim1 = Jaccard(feature_matrix1)`  (drug->entity / KG1)
  - `drug_sim2 = Jaccard(feature_matrix2)`  (drug->Morgan substructure / KG2)
  - `drug_sim4 = Jaccard(feature_matrix4)`  (drug->property / KG4)
  - `drug_sim3 = find_dif(feature_matrix3)` (drug->DDI / KG3; custom, task3.py ~:233-245)
  - feature_matrix1..4 = per-channel per-drug feature vectors (from the 4 KGs). Impl agent
    must read the exact feature_matrix construction + Jaccard/find_dif defs in task3.py and port.

## Regime -> model mapping for our unified benchmark (FAITHFUL)
- S0 (transductive, both seen)  -> 4-channel model (modeltask1-style; repo model.py IS this).
- S1 (inductive, one unseen)    -> 3-channel + impute (modeltask2-style).
- S2 (inductive, both unseen)   -> 3-channel + impute (modeltask3-style).
Repo `Code/baseline/mkg_fenn/model.py` is ported from modeltask1 (4ch, no impute) per its
docstring:3-11 -> faithful for S0 ONLY. Cold model variant must be NEW (do not edit model.py).

## KG1 source (native, not merged KG)
- MKG-FENN's 4 "KGs" are intrinsic channels (kg_builder.build_all_kgs). Only KG1 needs an
  external KG object: drug->{enzyme,target,transporter,carrier,pathway}.
- Our repo has these 5 native tables at `Code/data/KG/drugbank/filtered/drug_*.csv`, loadable
  via `data_utils.kg.KnowledgeGraph.from_filtered_dir(...)`. Use THIS (native drug side-info),
  NOT resources.kg.source (that's the merged triple KG). KG2/KG3/KG4 need only SMILES + train DDI.
- Cold-start OK: enzyme/target/etc. are drug-intrinsic -> unseen drugs still have KG1 edges;
  only KG3 (DDI) is empty for unseen -> self-loop (kg_builder.py:135-137). build_kg1 keys by drug
  and only checks dict1 membership (kg_builder.py:66-68), so unseen drugs are included.

## File structure (codex 019f23fe confirmed option a)
- LEAVE top `baseline.py` (binary, coldddi imports, dead) + `model.py` UNTOUCHED.
  `__init__.py` does not import baseline.py (exports []), so package import is fine.
- NEW: `multi_cls/baseline.py` (multiclass core, regime-aware), `multi_cls/model_cold.py`
  (3ch+impute cold model, ported from modeltask2/3), `multi_cls/baseline_unified.py`,
  `multi_cls/__init__.py`. Register ("mkg_fenn","multiclass") in run_baseline_unified.py _MODULES.
- `multi_cls/driver_task3*.py` are research drivers (not registry baselines) — reference only.

## Multiclass core design (case A, same task as paper)
- Reuse kg_builder.build_all_kgs UNCHANGED. Reuse `_build_dict1` idea (all drugs from splits +
  drugs table -> unseen reachable, baseline.py:113-123).
- event_num = # train-observed ddi_type classes (re-vocab from train, EmerGNN template
  emergnn/multi_cls/baseline_unified.py); CLAUDE.md allows vocab-size change in case A.
- Loss = CrossEntropyLoss (paper). Best-ckpt = **val macro-F1** (paper primary; val not test).
- predict returns (n, n_labels_GLOBAL) by scattering (n, K_train) to global class ids
  (EmerGNN-style); OOV-in-train gold -> ~0 mass -> counted wrong (runner oov_target_rate).
- Must-keep training protocol (task3.py): deterministic seeding (:48-56), symmetric pair
  augmentation (a,b)+(b,a) (:176-185), Adam+CE no scheduler no early-stop, per-epoch shuffle,
  per-epoch eval + best-by-macro-F1 (:215-223).
- PAPER_HYPERPARAMS (baseline.py:48-56): embedding 128, neighbor_sample 6, dropout 0.3, lr 1e-2,
  wd 1e-8, batch 256, 50 epochs. (Impl agent: confirm vs official argparse defaults.)

## Then (later, separate stop-and-report): binary (adapt existing binary logic, fresh file) +
## multilabel (Case-B, TWOSIDES 200-label; note twoside drugs mostly lack enzyme/target -> KG1 ghost).

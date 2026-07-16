# Screen 1 — TAG Node Init

**Hypothesis (i4)**: biomedical-LM textual embeddings of node names/descriptions outperform topology-only init (random, type-onehot, node2vec) for cold-start DDI prediction, because they carry pre-trained pharmacological semantics that survive in the absence of KG neighbors.

**Backbone**: EmerGNN multimode (frozen design). Only `node_init` tensor varies.

**Comparison anchor**: EmerGNN multimode @ seed 42 800-drug
- test_s0 AUC = 0.9895
- test_s1 AUC = 0.8328
- test_s2 AUC = **0.7462**  ← screen 1 must beat this

## Variants (see first_step_plan.md §4.5)

| ID | Init | Role |
|---|---|---|
| A | Random Gaussian 64d | lower bound |
| B | Node-type one-hot → 64d | topology lower bound |
| C | Node2Vec 64d | topology strong baseline |
| D | PubMedBERT [CLS] on full drug profile + node name | primary claim |
| D-name | PubMedBERT [CLS] on drug name only ("Aspirin") | profile-richness control |
| E | PubMedBERT [CLS] on within-kind shuffled names | semantic-vs-capacity control |
| F | PubMedBERT [CLS] on kind string only ("Drug") | type-name-only control |
| H | Qwen-72B input-embedding mean-pool | non-contextual frontier-LM control (deferred until weights confirmed) |

## Gates
1. D > E by ≥ 2 pt test_s2 AUC, paired bootstrap CI excluding 0
2. D > D-name by ≥ 1 pt (profile richness matters)
3. D > F by ≥ 1 pt (text content beats type label)

## Files
- `node_text_builder.py` — extracts `{node_id: text}` from merged KG nodes + drug profiles
- `encoder.py` — PubMedBERT [CLS] encoder with disk cache
- `init_features.py` — produces the 8 init tensors at d=64
- `emergnn_with_init.py` — EmerGNN subclass that accepts external init tensor
- `run_screen1.py` — entry script

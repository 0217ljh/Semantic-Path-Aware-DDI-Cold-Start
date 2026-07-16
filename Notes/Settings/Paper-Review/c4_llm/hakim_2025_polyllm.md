# PolyLLM: Polypharmacy Side Effect Prediction via LLM-based SMILES Encodings

- **Authors**: Sadra Hakim, Alioune Ngom
- **Year / Venue**: 2025 / Frontiers in Pharmacology
- **Link**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12351685/
- **Read depth**: full-text summary
- **Cluster**: c4

## TL;DR
Use chemical LLMs (ChemBERTa, sentence-BERT, GPT text-embedding, BERT) to encode SMILES strings into dense drug embeddings, then aggregate to drug pairs (simple sum) and predict 964 polypharmacy side effects with either MLP (multi-label) or GNN (link prediction). Best combo (Deepchem ChemBERTa + GNN) reaches AUC 0.9228, beating Decagon (0.872).

## Problem & Setting
Decagon benchmark (TWOSIDES-derived): 645 drugs, 964 side-effect types, 4.58M drug-pair-effect associations. Multi-label task. 80/10/10 split with 10-fold CV.

## Method (core)
- **SMILES encoding**: each drug's SMILES is passed through a language model (ChemBERTa/BERT/GPT/etc).
- **Pair aggregation**: element-wise sum of two drug embeddings - notably uniform pooling.
- **Classifier**: MLP (multi-label sigmoid over 964 effects) OR GNN (bipartite drug-pair / side-effect graph, GraphConv + dot-product decoder).

LLM role: SMILES encoder (text-mode LLM treats SMILES as language). No structural reasoning, no KG paths.

## Cold-start handling
Transductive 80/10/10 over (drug-pair, side-effect) edges. Same drugs appear in train and test. Authors do not claim cold-start; standard supervised split. The GNN explicitly uses supervised edges separate from message-passing edges to avoid target leakage at edge level, but drug-overlap leakage persists.

## Key contributions
- Systematic comparison of multiple LLM/text encoders for SMILES on the same downstream task.
- Clean ablation: Deepchem ChemBERTa > fine-tuned ChemBERTa > Mol2vec > GPT > BERT on AUC.
- GNN > MLP for the same encoder, suggesting relational signal matters even with strong text embeddings.

## Limitations / gaps (as relevant to our insights)
- No drug-level cold-start eval - critical missing piece.
- Sum-pooling of two drug embeddings is the simplest possible aggregation - directly the noise-pooling failure mode i3 warns about.
- Mechanism-blind side effects (immune-mediated, e.g. Adenopathy 0.78 AUC) confirm the limits of pure-chemical inputs - we should cite this as evidence that PD-side effects need non-chemical signal.
- No drug-name / description channel tested, only SMILES through LMs.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Indirect support - chemical-LLM encodings work well on chemistry-driven effects but degrade on immune/PD-driven effects, consistent with the PK-vs-PD split.
- **i2 (meeting node + over-smoothing)**: Not applicable; bipartite graph is shallow.
- **i3 (pooling noise + attention limit)**: Direct evidence - the simple sum-aggregation works but caps performance; their best AUC is bounded by exactly this pooling design.
- **i4 (node-name semantic prior)**: Partial counter-example - they use SMILES not name/description; reasonable to cite as showing that *which text* the LLM sees matters. Our i4 argues node-name is the cold-start-stable signal, not SMILES.

## Notes
- LLMs tested: ChemBERTa (Deepchem + fine-tuned), BERT, Sentence-BERT, GPT text-embedding-3-small, Mol2vec, Doc2vec.
- Best: Deepchem ChemBERTa + GNN, AUC 0.9228 / AUPRC 0.8944.
- Single SMILES per drug - no canonicalization sweep.
- Useful as: "SMILES-LM" baseline contrast to our text-name-LM proposal.

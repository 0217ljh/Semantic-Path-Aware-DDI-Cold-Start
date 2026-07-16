# RISE-DDI: Informative Subgraph Extraction with Deep Reinforcement Learning for Drug-Drug Interaction Prediction

- **Authors**: Jiancong Xie, Wentao Wei, Chi Zhang, Jiahua Rao, Yuedong Yang (Sun Yat-sen University; Pengcheng Laboratory)
- **Year / Venue**: 2026 (AAAI-26), Proceedings of the AAAI Conference on Artificial Intelligence
- **Link**: https://ojs.aaai.org/index.php/AAAI/article/view/37105
- **Read depth**: abstract-only
- **Cluster**: c2

## TL;DR
RISE-DDI casts informative-subgraph extraction as a Markov Decision Process and trains a deep RL agent to select, edge-by-edge, the subgraph most useful for predicting the DDI of a given pair. A structure-aware reward couples KG topology with molecular features. Up to 20% gains over baselines in both transductive and inductive settings.

## Problem & Setting
DDI prediction with both transductive (existing-existing) and **inductive** (unseen drugs) settings. Subgraph extraction is the central object, not embedding learning.

## Method (core)
1. **MDP formulation** — state = current partial subgraph; action = pick next edge to add or stop; transition = subgraph update.
2. **RL policy** trained to maximise a reward that combines (i) structural informativeness, (ii) molecular-feature alignment.
3. **Predictor** runs on the extracted subgraph.

## Cold-start handling
Inductive evaluation is reported. Because the policy reasons over a *pair-specific* subgraph rather than per-drug embeddings, it can in principle handle unseen drugs — though the abstract doesn't break out S1 vs S2.

## Key contributions
- First explicit DDI subgraph extractor trained as an RL agent — moves beyond fixed-K-hop or attention pruning.
- Combines KG topology with chemical structure inside the reward, addressing one of our i3 concerns (need to combine multiple priors).
- Reports large gains in inductive split.

## Limitations / gaps (as relevant to our insights)
- Abstract gives only headline numbers; method details (reward shape, state encoding, stop criterion) need full paper to verify.
- No text semantics on path/subgraph nodes.
- No PK/PD typology.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: not addressed.
- **i2 (meeting node + over-smoothing)**: subgraph is pair-anchored — reduces drug-drug message-flow distance issues. Whether the final readout is meeting-node-style or drug-pair-style needs full text to confirm.
- **i3 (pooling noise + attention limit)**: RL policy *selects* edges (binary keep/drop) rather than only *reweighting* them — partially answers the "attention can only reweight" critique. But the prior still comes from the reward, which is internal, not from external semantics.
- **i4 (node-name semantic prior)**: not used.

## Notes
Recent AAAI-26 work; the RL-pruning angle is the most aggressive answer yet to i3. Need to read the full proceedings paper to capture exact method.

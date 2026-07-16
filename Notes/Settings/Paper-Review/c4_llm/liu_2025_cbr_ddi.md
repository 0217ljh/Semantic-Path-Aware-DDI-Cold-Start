# Case-Based Reasoning Enhances the Predictive Power of LLMs in Drug-Drug Interaction

- **Authors**: Guangyi Liu, Yongqi Zhang, Xunyuan Liu, Quanming Yao (Tsinghua, HKUST)
- **Year / Venue**: 2025 / arXiv:2505.23034
- **Link**: https://arxiv.org/abs/2505.23034
- **Read depth**: full-text (abstract + extended summary)
- **Cluster**: c4

## TL;DR
CBR-DDI augments off-the-shelf LLMs with a case-based reasoning module: retrieve similar historical drug-pair cases (via hybrid LLM-text + GNN-graph similarity), inject both factual KG knowledge and mechanistic patterns from cases into the prompt. Achieves a +28.7% absolute accuracy gain over CBR baseline and a ~5x improvement over the bare LLM on S1, no fine-tuning required.

## Problem & Setting
Multi-class (DrugBank, 86 types) and multi-label (TWOSIDES, 200 effects) DDI prediction under three settings:
- S0: both drugs seen,
- S1: one emerging + one existing,
- S2: both drugs emerging (true cold-start, matches our setting).

## Method (core)
Three-stage pipeline:
1. **Retrieval (LLM-GNN collaboration)**: Hybrid similarity over a case base. Semantic similarity from LLM-generated drug descriptions; structural similarity from GNN encoding of KG subgraphs. λ≈0.5 best.
2. **Reuse (dual-layer prompting)**: External factual layer (KG paths from EmerGNN) + internal regularity layer (interaction mechanisms distilled from retrieved cases). Both layers ablated and shown necessary.
3. **Refinement (K-Medoids sampling)**: Trims case repository >90% with no perf loss.

LLM role: reasoning engine over retrieved evidence (RAG-style with structured prompts), NOT fine-tuned. KG (HetioNet, 34K nodes / 1.69M edges) provides paths.

## Cold-start handling
Explicit S2 protocol where neither drug appears in training set. CBR-DDI is reported to be strongest exactly on S1/S2 - meaningful for our cold-start claim. Follows splits from Zhang 2023 / Abdullahi 2025.

## Key contributions
- First framework to combine CBR with LLM specifically for DDI cold-start.
- Empirical separation of "factual KG knowledge" vs "mechanistic case knowledge" - both needed.
- Strong S1/S2 numbers without parameter updates → cheap to deploy.

## Limitations / gaps (as relevant to our insights)
- Relies on textual + KG-path inputs; explicitly skips molecular structure - authors flag this.
- Retrieved cases still come from training distribution; novel mechanisms unseen in any case may fail.
- LLMs used (Llama3.1-8B/70B, DeepSeek-V3) may have memorized public DDI facts; no leakage audit.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Implicit - the "interaction mechanism" channel could carry PK or PD reasoning, but the paper doesn't separate them. Worth probing: does case retrieval naturally cluster by PK vs PD?
- **i2 (meeting node + over-smoothing)**: GNN component is for retrieval only (encoding subgraphs around each drug), not for direct drug-drug message passing across many hops. They effectively side-step deep GNN over-smoothing by handing reasoning back to the LLM.
- **i3 (pooling noise + attention limit)**: Strong support - they explicitly bypass uniform pooling by injecting selected high-similarity cases as a sparse, curated prior. Two layers (factual + regularity) act as an external prior that attention alone wouldn't extract.
- **i4 (node-name semantic prior)**: Direct support - LLM-generated drug descriptions used for semantic retrieval. Node-name + text description is the bridge between cold-start drugs and the case base. This is structurally close to our claim.

## Notes
- LLMs tested: Llama3.1-8B-Instruct, Llama3.1-70B-Instruct, DeepSeek-V3 (no proprietary needed).
- KG: HetioNet via EmerGNN for path extraction.
- Most architecturally aligned paper with our framing in this cluster; useful as direct baseline.
- Code likely released (check arXiv v1).

# EmerGNN: Emerging Drug Interaction Prediction Enabled by Flow-based Graph Neural Network with Biomedical Network

- **Authors**: Yongqi Zhang (4Paradigm), Quanming Yao (Tsinghua), Ling Yue, Xian Wu, Ziheng Zhang, Zhenxi Lin, Yefeng Zheng (Tencent Jarvis Lab)
- **Year / Venue**: 2023, *Nature Computational Science*; arXiv 2311.09261
- **Link**: https://arxiv.org/abs/2311.09261 ; https://www.nature.com/articles/s43588-023-00558-4 ; code: https://github.com/LARS-research/EmerGNN
- **Read depth**: **full-text PDF** (16-page main + Method section)
- **Cluster**: c2 (path-based / multi-hop reasoning)

## TL;DR
EmerGNN predicts DDI for emerging (cold-start) drugs by (i) constructing a path-based subgraph between the query drug pair on an augmented biomedical network with inverse edges added, (ii) running a **Bellman-Ford-style flow GNN** that propagates drug u's fingerprint along the directed subgraph until it accumulates at drug v, and (iii) using relation-type attention conditioned on the (fu, fv) fingerprints to weight messages along the way. The final pair representation is the bidirectional concatenation (u→v and v→u). **Defines and benchmarks the canonical S1 (emerging × existing) and S2 (emerging × emerging) cold-start splits we should adopt.**

## Problem & Setting
- Two tasks: multi-class DDI type prediction (DrugBank: 86 types) and multi-label binary side-effect prediction (TWOSIDES).
- Drug splits: random 7:1:2 split of the **drug node set** VD into VD-train / VD-valid / VD-test. The validation and test drugs are "emerging" — never appear in the DDI network during training.
- **S0**: existing × existing (standard transductive).
- **S1**: emerging × existing.
- **S2**: emerging × emerging (both unseen).
- Biomedical network = HetioNet (24+ relation types over genes, diseases, side effects, anatomies, pathways, pharmacologic classes).
- Crucially, the biomedical network is also split: training-time NB-train excludes emerging drugs' adjacent edges, so the model cannot peek at test drugs through KG during training.

## Method (core, four-step)

### Step 1 — Augmented network construction
Integrate the DDI network ND with the biomedical network NB into N' = ND ∪ NB. Add **inverse edges** for every relation (e, r, e') becomes also (e', r_inv, e). The full augmented network:

> N = N' ∪ {(e, r_inv, e') : (e', r, e) ∈ N'}

Adding inverse edges means a path like `a -r1→ b ←r2- c` can be rewritten as `a -r1→ b -r2_inv→ c`, making the path strictly forward-flowing for the GNN.

### Step 2 — Path-based subgraph extraction
For query pair (u, v), collect ALL paths of length ≤ L (typical L=3) between u and v in N. Union into subgraph G_{u,v}^L. Edges are directed from u toward v after the inverse-augmentation trick.

### Step 3 — Flow-based GNN
Let V_{u,v}^ℓ = entities exactly ℓ-steps from u AND (L−ℓ)-steps from v. So V_{u,v}^0 = {u} and V_{u,v}^L = {v}. The GNN iterates ℓ = 1..L:

> h_{u,e}^{(ℓ)} = σ ( W^{(ℓ)} · Σ_{e' ∈ V_{u,v}^{ℓ−1}} φ( h_{u,e'}^{(ℓ−1)}, h_r^{(ℓ)} ) )    for e ∈ V_{u,v}^ℓ

Initial: h_{u,u}^{(0)} = f_u (Morgan fingerprint of u, dimension d).

Message function with attention:
> φ(h, h_r) = α_r^{(ℓ)} · (h ⊙ h_r)

Relation-type attention:
> α_r^{(ℓ)} = σ( w_r^{(ℓ)} · [f_u ; f_v] )

— so the attention weight is **drug-pair-conditional** (depends on both query drug fingerprints) but applied **per relation type per layer** (the same r at layer ℓ gets the same weight regardless of which specific edge it is on).

After L steps, h_{u,v}^{(L)} is the pair representation **accumulated at v** starting from u's fingerprint.

### Step 4 — Bidirectional readout
Run flow in both directions (u→v and v→u with the same parameters), get h_{u,v}^{(L)} and h_{v,u}^{(L)}. Concatenate:

> l(u, v) = W_rel · [h_{u,v}^{(L)} ; h_{v,u}^{(L)}]

This gives logits over interaction types.

## Key empirical findings (from results section)

- **L=3 is optimal**, L=1 too shallow (signal can't reach), **L>3 degrades** (irrelevant info accumulates, performance drops). This is precisely the over-smoothing onset in our framing.
- **Top attended relations** (Fig. 3c): `CrC` (Compound resembles Compound) is dominant — i.e., paths through **structurally similar drugs** are the most useful signal. Other prominent: `CrD` (cause disease), `CbG` (binds gene), `PCiC` (pharma class includes compound), `CcSE` (causes side effect).
- The model "implicitly finds similar drugs" — Group 1 (paths through u1 where u1 has interaction i1 = i_pred with v) have higher fingerprint similarity to u than Group 2 (random drugs).
- Top-3 or Top-5 attention relations are **sufficient**: pruning to those relations gives near-full performance. Random 10/30/50% edge sampling badly hurts.
- Ablations:
  - Without inverse edges → worse (loses directional info)
  - SumGNN-style subgraph encoding → worse than flow
  - Uni-directional only → worse than bidirectional
- Computational footprint: relatively small parameter count + memory; subgraphs (~tens of thousands of edges per pair on DrugBank) much smaller than the full KG (~3.6M edges).

## Cold-start handling
EmerGNN's primary mechanism is **path-conditioned similarity inference**:
- An emerging drug u is connected to KG entities (proteins, diseases, SE) even without DDI history.
- The flow finds paths from u through KG to some existing drug u_1 (most often via `CrC`, i.e., u resembles u_1), and uses u_1's known DDIs to inform the predicted DDI for u.
- "If u resembles u_1 and u_1 interacts with v in way i, then u likely interacts with v in way i."

## Key contributions
- First flow-based GNN architecture for DDI explicitly designed for emerging drugs.
- Defines and benchmarks the canonical S1/S2 cold-start splits with paired biomedical-network splits (no emerging drug leakage at training time).
- Provides interpretable path-level attention (which paths matter for a prediction).
- Open-source code + Zenodo data release.

## Limitations (from authors)
- Emerging drug must be in the biomedical network (otherwise no path).
- Computational cost higher than per-drug embedding methods.
- L>3 degrades — implicit over-smoothing acknowledgment.

## Relevance to our insights

- **i1 (PK / PD two paradigms)**: NOT addressed. Single flat readout over all interaction types. No mechanism-type aware reasoning.
- **i2 (meeting-node anchor + over-smoothing)**: CLOSEST of all c2 papers — flow accumulates a pair representation at v (rather than reading drug embeddings independently). But anchor is still drug node v, not a shared mediator M. Over-smoothing implicitly visible at L>3 ablation.
- **i3 (pooling noise + attention insufficient)**: relation-type-level attention is a fixed-granularity selector; the **same r gets the same α regardless of which specific edge instance** at that hop. This is coarser than per-instance perception. No external semantic prior.
- **i4 (node-name semantic prior)**: NOT used. Intermediate biomedical entities are ID-embedded; node text/name semantics are unused.

## Notes
- This is the **#1 must-compare baseline** for our paper.
- Their S1/S2 split + biomedical-network split protocol is the cleanest cold-start benchmark currently published — we should adopt it directly.
- Their `L=3 optimal, L>3 degrades` ablation is independent empirical support for our over-smoothing argument (i2) in the DDI context.
- Their "paths through CrC are most attended" is a hint that EmerGNN's success on S2 mostly leans on drug-similarity inference, not on full causal-chain reasoning — relevant for our boundary discussion.

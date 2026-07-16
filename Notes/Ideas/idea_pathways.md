Research Plan: LLM-Enhanced Flow GNN for Cold-Start DDI Prediction
Baseline: EmerGNN (Zhang et al., Nat Comp Sci 2023) — flow-based GNN with bidirectional path propagation on biomedical KG. Remaining the only deep-learning method explicitly designed for emerging-drug DDI, but with no method-level follow-up since 2023. K-Paths (KDD 2025) uses EmerGNN as backbone but explicitly discards the flow mechanism.

Process
Step 1. Enhance the KG-side flow mechanism via LLM integration.

Stay within the biomedical KG; do not touch molecular graph yet. Explore which of the following LLM-integration angles yields stable gain over EmerGNN on cold-start (S1/S2) without hurting S0:

TAG-based node enrichment (Text-Attributed Graph). Biomedical entities (drugs, enzymes, targets, pathways) carry textual descriptions. Use LLM to embed these descriptions as node features, replacing EmerGNN's Morgan-FP-only / random init. Direct cold-start payoff: emerging drugs lack KG neighbors but have textual MoA/indication, which becomes the substitute signal.
PK/PD path recognition. DDI mechanisms split into PK-mediated (metabolic-enzyme inhibition, transporter competition, ...) vs PD-mediated (receptor synergy/antagonism, ...). EmerGNN's flow treats all paths equally; LLM can label each path with its mechanism class, used as a soft prior on path attention.
Automated pooling nodes (meet-in-middle). Instead of EmerGNN's "propagate L hops then read terminal-node embedding", propagate from both ends simultaneously toward an intermediate junction node; read the junction embedding directly. Hypothesis to test: 1-hop aggregation at the junction encodes bidirectional causal signal that EmerGNN needs L hops to reach. Independent of LLM choice, but compatible with both above.
LLM-decoder verification of KG path causality. Use LLM as a constrained discriminator (discrete-token decoder over a closed effect-type vocabulary) to verify whether an extracted path semantically supports the predicted DDI effect. Avoids hallucination by design (LLM doesn't generate new edges, only scores). Output probability feeds back as path-level attention prior. Core value: when KG path is structurally present but semantically incoherent, or when KG is missing causal edges, LLM's pre-trained pharmacological knowledge fills the gap — and stays auditable because output is in a fixed vocabulary.
These are exploratory directions; one or two of them is enough to support the paper.

Step 2. Introduce molecular graph as a second representational source.

Only triggered once Step 1 has produced stable gain. Replace EmerGNN's single Morgan FP (1024-bit, low-dim, lossy) with a richer molecular representation (e.g., learned atom-level GNN embedding, or substructure-aware encoding).

Key constraint identified empirically (TIGER on our data, S0 0.93 → S2 0.56, -37pt drop): hybrid methods that late-fuse mol-graph and KG channels with equal weight collapse on cold-start. The reason: mol signal is invariant to cold-start (drug structure is always known), but KG signal degrades sharply for cold drugs (few or no DDI edges). Equal-weight fusion lets the degraded KG channel contaminate the prediction.

Therefore Step 2 must avoid equal-weight late fusion. Candidate strategies (one chosen later based on Step 1's chosen architecture):

Adaptive fusion conditioned on drug's KG-connectivity / cold-vs-warm status
Mol-graph as KG node init refinement (single-channel, no fusion problem)
Mol substructures injected as new node types into KG so flow naturally traverses both
Other gating designs that down-weight the weaker channel per drug
# Cluster c8 — GNN Over-Smoothing, Expressivity, and Depth: Summary

## Scope
Theory and applied papers establishing that deep GNNs suffer two coupled pathologies — over-smoothing (representations collapse onto a low-dimensional subspace) and over-squashing (exponentially many distant-node messages compressed through narrow bottlenecks) — and that the standard mitigations (normalization, edge dropping, residual injection, rewiring) only delay rather than eliminate the collapse.

## Papers reviewed (7)
1. Li, Han, Wu 2018 — "Deeper Insights into GCN" (AAAI). Identifies Laplacian smoothing as the mechanism behind GCN's shallowness.
2. Xu, Li, Tian et al. 2018 — JKNet (ICML). Adaptive per-node depth via jumping skip connections.
3. Oono and Suzuki 2020 — "Exponentially Lose Expressive Power" (ICLR). Rigorous non-asymptotic exponential-rate proof for ReLU GCN over-smoothing.
4. Rong, Huang, Xu, Huang 2020 — DropEdge (ICLR). Random edge dropping delays but does not remove collapse.
5. Zhao and Akoglu 2020 — PairNorm (ICLR). Pairwise-distance normalization that preserves geometric separation.
6. Chen, Wei, Huang, Ding, Li 2020 — GCNII (ICML). Initial residual plus identity mapping enables 64-layer GCN.
7. Alon and Yahav 2021 — Bottleneck of GNNs (ICLR). Identifies over-squashing as a dual problem to over-smoothing.
8. Topping, Di Giovanni, Chamberlain, Dong, Bronstein 2022 — Curvature and over-squashing (ICLR). Geometric (Forman-Ricci) characterization of where messages are squashed.

(Eight papers logged — one over target — because over-smoothing and over-squashing are co-foundational for our argument.)

## Synthesis for our paper (i2 emphasis)

**The argument we can confidently make, with citations:**

- Over-smoothing is exponential in depth (Oono and Suzuki 2020), not merely asymptotic. For typical GCN settings, representations collapse within 4-8 layers (Li et al. 2018, Zhao and Akoglu 2020).
- Over-squashing is the dual problem: even when collapse is delayed, exponentially many messages must funnel through the narrow embedding dimension (Alon and Yahav 2021), and the bottleneck localizes precisely at negatively-curved edges in the graph topology (Topping et al. 2022).
- Standard mitigations are all patches on the symptom: edge dropping (Rong et al. 2020), normalization (Zhao and Akoglu 2020), initial residuals (Chen et al. 2020), or adaptive depth (Xu et al. 2018). None eliminate the underlying tension between needing depth for receptive field and paying for it with collapse.
- Implication for drug-pair distance reasoning: pushing message-passing along the full drug-to-drug shortest path (typically 3-5 hops in biomedical KGs) lies in the regime where collapse is already severe and over-squashing is acute. The principled architectural fix is to anchor reasoning at an intermediate biological "meeting" entity (shared target, shared pathway, shared transporter), so the model never needs deep stacking.

## Has the over-smoothing × DDI specific angle been published?

**Short answer: NO, not as a primary contribution. The angle is open.**

Detailed findings:
- All canonical over-smoothing theory papers (c8 set) evaluate on citation graphs (Cora, Citeseer, Pubmed), OGB, and synthetic Erdős-Rényi. **None** evaluate on DDI or polypharmacy benchmarks.
- GNN-based DDI literature (KGNN, SumGNN, Decagon, KnowDDI) acknowledges over-smoothing informally and uses standard mitigations (residual, skip, shallow architectures by default), but no paper isolates an over-smoothing analysis for DDI prediction or shows depth-vs-DDI-AUC curves as a central claim.
- DDI surveys (Frontiers Chemistry 2026, Quantitative Biology 2024, Nature Sci Reports 2025) mention over-smoothing only in passing as a generic GNN concern. None point to a paper that has specifically tied over-smoothing to drug-pair shortest-path length in a biomedical KG.
- The closest neighbor work is CurvDrop (WWW 2023) which applies Ricci-curvature-guided edge dropping for over-smoothing+over-squashing relief, but on standard benchmarks, not DDI.

**Implication:** Our paper has a clean opening to claim a specific contribution of the form: "for cold-start DDI on biomedical KGs, the typical drug-to-drug shortest path lies in the depth regime where existing over-smoothing theory (Oono and Suzuki 2020) predicts exponential representation collapse, and we are the first to (a) demonstrate this empirically on DDI benchmarks and (b) sidestep it by anchoring on the meeting biological entity rather than enlarging the GNN receptive field."

This is a publishable angle. The c8 cluster gives us the theory; our experiments would give the DDI-specific empirical evidence.

## Recommended citation block for the paper
- Foundational (must cite): Li et al. 2018, Oono and Suzuki 2020.
- Dual problem: Alon and Yahav 2021, Topping et al. 2022.
- Mitigation toolkit (contrast against): Rong et al. 2020 (DropEdge), Zhao and Akoglu 2020 (PairNorm), Chen et al. 2020 (GCNII), Xu et al. 2018 (JKNet).

## Cross-cluster handoffs
- KG+GNN DDI methods that do not engage with depth → c1.
- Depth-tolerant methods that bypass message passing entirely (e.g., transformer-based, path-encoder-based) → may belong in c2 (path / multi-hop reasoning).
- Cold-start-specific architectural fixes that incidentally help with over-smoothing → c3.

# MedKGQA: Medical Knowledge Graph QA for Drug-Drug Interaction Prediction based on Multi-hop Machine Reading Comprehension

- **Authors**: Peng Gao, Feng Gao, Jian-Cheng Ni, Yu Wang, Fei Wang
- **Year / Venue**: 2022 (v1), revised 2024; arXiv 2212.09400
- **Link**: https://arxiv.org/abs/2212.09400 ; https://arxiv.org/html/2212.09400v3
- **Read depth**: full-text (abstract + HTML)
- **Cluster**: c2

## TL;DR
MedKGQA reframes DDI prediction as a **multi-hop machine reading comprehension** problem over a constructed drug-protein KG. It extracts drug-protein triplets from open-domain documents, builds a directed graph following metabolic pathway logic, and uses a GNN to traverse drug → protein → ... → drug chains. Reports +4.5% accuracy on the Qangaroo MedHop benchmark.

## Problem & Setting
DDI prediction as multi-hop QA. Document-based: questions are answered by reading closed-domain literature plus the KG built from open-domain docs.

## Method (core)
1. **KG construction** — extract drug-protein triplets from literature; build a directed graph respecting metabolic-pathway direction.
2. **Entity embeddings** for drug-protein attributes.
3. **GNN over the directed KG** to learn multi-hop drug → protein → protein → drug patterns.
4. **MRC layer** ties graph reasoning to textual evidence from the source document.

## Cold-start handling
Not explicitly evaluated as S2. But because the system reads documents at inference time, a drug with no historical DDI label can still be reasoned about as long as it appears in any document with protein annotations — closer to a retrieval-based cold-start.

## Key contributions
- Uses **drug-protein** as the meeting-node by construction — paths are drug → protein (→ protein) → drug.
- Couples textual evidence (MRC) with KG paths — one of the few c2 works to use document semantics.
- Directional path respects metabolic logic, not just topology.

## Limitations / gaps (as relevant to our insights)
- Only drug-protein triplets — does not include phenotype/SE layer, so PD-side mechanisms are missed.
- MRC scoring is text-extraction; not the same as learned semantic priors over node names.
- Cold-start performance not separately reported.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: PK-leaning — drug-protein-protein-drug chains are firmly in the PK / molecular-mechanism layer. Lacks the PD / phenotype path. Useful evidence that path **typology matters**.
- **i2 (meeting node + over-smoothing)**: best PK example of meeting-node reasoning — proteins literally sit at the mediator position. Strong support for our i2 claim that paths through a shared mediator are the right anchor.
- **i3 (pooling noise + attention limit)**: directional pathway structure acts as a strong external prior on which edges to traverse — implicit support that priors beat pure attention.
- **i4 (node-name semantic prior)**: uses textual evidence at the document level but not node-name-level semantics on the KG nodes themselves. Partial credit.

## Notes
The cleanest existing example of "meeting node = protein" reasoning. Confirms one half (PK) of our i1 split; the missing PD half is exactly our gap.

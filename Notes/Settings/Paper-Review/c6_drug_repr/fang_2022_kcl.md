# KCL: Molecular Contrastive Learning with Chemical Element Knowledge Graph

- **Authors**: Yin Fang, et al.
- **Year / Venue**: 2022 / AAAI
- **Link**: https://arxiv.org/abs/2112.00544 ; code https://github.com/Fangyinfff/KCL
- **Read depth**: abstract
- **Cluster**: c6 (precursor to KANO)

## TL;DR
The original chemical-element-KG contrastive scheme that KANO extends. Builds a Chemical Element KG (element-element relations, attributes), augments each molecular graph with that KG, and contrasts the original vs. knowledge-augmented graph view. Three encoders + knowledge-aware message passing.

## Method (core)
- Chemical Element KG describes relations among elements and their chemical attributes.
- Knowledge-guided graph augmentation creates a positive view per molecule.
- Contrastive loss aligns the two views; self-supervised, no downstream label.

## Relevance to our project
- Earliest clean example of "align molecular representation to a KG via contrastive learning without any task label." Cite alongside KANO as the molecule↔KG self-supervised alignment lineage.
- Same caveat: KG is element-level (generic), not drug-specific neighborhood.

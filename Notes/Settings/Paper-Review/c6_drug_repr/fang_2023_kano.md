# KANO: Knowledge Graph-Enhanced Molecular Contrastive Learning with Functional Prompt

- **Authors**: Yin Fang, Qiang Zhang, et al. (HICAI-ZJU)
- **Year / Venue**: 2023 / Nature Machine Intelligence (5, 542-553)
- **Link**: https://www.nature.com/articles/s42256-023-00654-0 ; code https://github.com/HICAI-ZJU/KANO
- **Read depth**: abstract + method
- **Cluster**: c6 (drug representation) — also relevant to molecule↔KG alignment

## TL;DR
Pretrains a molecular GNN by contrastively aligning each molecule against an ElementKG-augmented view (atom-level chemistry knowledge from the Periodic Table + functional-group facts from Wikipedia). At fine-tuning, injects a functional-group "prompt" derived from the same KG. Strong on 14 molecular-property datasets.

## Method (core)
- ElementKG: small curated KG of chemical elements (periodicity, metallicity) and functional groups.
- Contrastive pretraining: augment each molecular graph with element/functional-group subgraph knowledge; pull the molecule embedding toward its knowledge-augmented view, push apart others. NO downstream label needed — pretraining is self-supervised over intrinsic chemistry + KG.
- Functional prompt at fine-tuning re-injects the KG knowledge.

## Cold-start handling
Not a DDI/cold-start paper, but the alignment is structurally exactly what we need: the molecule is aligned to KG-derived knowledge via a contrastive objective that uses ONLY intrinsic chemistry (the molecule's own atoms/groups) and a generic chemistry KG — never a supervised interaction edge. The aligned encoder then transfers to any (including unseen) molecule.

## Relevance to our project (DDI-edge-independent alignment)
- **Template for our alignment objective**: align molecular-graph embedding to KG knowledge using intrinsic structure only, with no DDI label. KANO's ElementKG is chemistry-level; we would instead align molecular embeddings to the biomedical KG NEIGHBORHOOD (targets/enzymes) of the drug.
- The functional-prompt idea is a clean way to inject KG priors at inference for an unseen drug without retraining.
- Limitation for us: ElementKG is shared across all molecules (element-level), so alignment is generic; our drugs need drug-SPECIFIC KG neighborhoods. The contrastive recipe transfers; the KG content must be swapped.

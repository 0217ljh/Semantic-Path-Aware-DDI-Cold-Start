# MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding

- **Authors**: Chopra S, Sanchez-Rodriguez G, Mao L, Feola AJ, Li J, Kira Z
- **Year / Venue**: 2025 / arXiv:2506.08356 (cs.CV)
- **Link**: https://arxiv.org/abs/2506.08356
- **Read depth**: abstract + method summary
- **Cluster**: c7

## TL;DR
Medical vision-language model that routes multi-scale image features through specialized experts via a Mixture-of-Experts module **conditioned on the report/diagnosis type**. This is the closest published instance of "route a modality conditioned on the semantic type of the task" — but it lives in medical imaging, not DDI, and the routing condition is diagnostic context, not a PK/PD interaction paradigm. It establishes that conditional modality routing is a sound, working idea while leaving the DDI mechanism-type application open.

## Problem & Setting
Medical vision-language alignment (CT, X-ray, etc.) against clinical reports. Zero-shot and linear-probe benchmarks.

## Method (core)
- Swin Transformer feature pyramid → MoE module conditioned on report type → specialized expert branches capture modality-specific visual semantics.
- Routing is dynamic and does not require modality-specific supervision at inference.

## Cold-start handling
N/A (imaging task, not DDI).

## Key contributions
- Demonstrates condition-aware (report-type-conditioned) routing of a modality through specialized experts improves multimodal alignment.
- Establishes the general MoME/modality-routing template in a biomedical setting.

## Limitations / gaps (as relevant to our insights)
- Not DDI. Routing condition is diagnostic/report context, not a pharmacological interaction mechanism.
- Routes within a single (visual) modality across scales; does not route a molecular modality on/off by interaction semantics.

## Relevance to our insights
- **Routing-novelty gap (core)**: This is the strongest evidence that "conditional modality routing" is an established, validated pattern in biomedical multimodal learning (also MoME, M4oE, MedMoE family) — but ALWAYS conditioned on data/task modality, never on a PK/PD interaction paradigm, and never in DDI. Our contribution — routing the molecular modality ONLY into the PK channel based on interaction mechanism type — is therefore a novel transfer of a known pattern to an unoccupied problem.
- **i1 (PK/PD split)**: Supports the architectural feasibility of mechanism-conditioned routing; we instantiate the router on the PK/PD axis rather than on imaging modality.

## Notes
- Cite as "conditional modality gating / MoME is established in biomedical multimodal ML (MedMoE, M4oE) but has never been keyed to DDI interaction mechanism (PK vs PD)." This is the cleanest available framing for the routing-novelty claim — the mechanism is borrowed, the routing key (PK/PD) and the DDI cold-start application are new.
- Companion general-ML references for the related-work paragraph: MoME (vision-language), M4oE (medical image segmentation).

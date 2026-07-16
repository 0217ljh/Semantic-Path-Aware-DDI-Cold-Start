# Lit backing — i1-as-routing v2 (learned PK/PD gate) — 2026-05-25

Search for [[README_i1_routing_v2_learned]]. Four buckets + novelty/gap check.

> **Verification status**: papers below were retrieved by a WebSearch sub-agent, NOT yet
> personally verified against the source PDFs. Per project no-fabrication rule, treat
> titles/venues/claims as agent-reported until confirmed. The arXiv 2511.06662 competitor
> details (its gate/attention mechanism) MUST be read from the actual PDF before writing
> any differentiation paragraph.

## Verdict (agent)
Core contribution — a **learned PK/PD-emergent soft gate** routing between an
aggregate-molecular (PK) channel and a sign-aware cross-attention *select-compose* over
phenotype neighbors (PD), for S2 cold-start — appears **novel**. Each building block is
well-supported. Strongest reviewer risk = the 2511.06662 competitor.

## Bucket 1 — pair-conditional / cross-drug attention selection
All do cross-drug attention over MOLECULAR SUBSTRUCTURES; none over side-effect/phenotype
neighbors → our effect-layer cross-attention is unoccupied territory.
- **SSI-DDI** (Nyamabo et al., 2021, Briefings in Bioinformatics) — canonical cross-drug
  substructure co-attention baseline. STRONG.
- **DSN-DDI** (Li et al., 2023, Briefings in Bioinformatics) — dual-view intra+inter-view;
  inter-view = one drug's substructures attend across the other. STRONG.
- **SA-DDI** (Yang et al., 2022, Chem. Sci. / PMC9337739) — size-adaptive substructures +
  substructure-substructure interaction module. STRONG.
- **Multi-Scale Graph Neural Process w/ Cross-Drug Co-Attention** (2025, arXiv 2509.15256)
  — most recent; framed as beyond late-fusion co-attention. Good "go-beyond" cite. STRONG.
- **Taco-DDI** (2025, Neural Networks) — graph-transformer + dynamic co-attention. MEDIUM.

## Bucket 2 — learned modality gating / adaptive fusion
Gate structure-vs-structure modalities, NOT a structure-channel vs effect/phenotype
channel → our gate's semantic role (PK vs PD) is novel.
- **AdaptMol** (2025, arXiv 2505.11878) — adaptive multi-level attention fusion of SMILES+
  graph; shows naive concat worsens sparsity/hurts cross-modal interaction. Directly
  supports "uniform fusion hurts, learned gate helps." STRONG.
- **UKGE** (2024, Neural Networks) — multimodal geometric+semantic+KG fusion. MEDIUM.
- **UniMAP** (2023, arXiv 2310.14216) — deep SMILES-graph cross-modality fusion. MEDIUM.

## Bucket 3 — why structure predicts PK but not PD
- **Hou et al., 2017, J. Cheminformatics (PMC5340788)** — DDI via structural similarity,
  explicitly separates PK vs PD; structurally similar drugs share enzymes/transporters (PK)
  vs same receptor/site (PD). Best single mechanistic cite for the two-channel split. STRONG.
- **PK & PD DDI: Research Methods and Applications** (2023, PMC10456269) — PD changes
  response via agonism/antagonism WITHOUT affecting kinetics; PK = ADME/metabolism. Anchors
  "PD is emergent." STRONG.
- **CYP450/transporter PK-DDI review** (Chemis Group) — PK DDIs dominated by CYP/P-gp/OATP,
  PBPK-predictable. MEDIUM.
- **Caveat cite** (PMC6685287) — DrugBank DDIs mostly inferred PK → structure-only models
  are PK-biased → motivates a dedicated PD channel. MEDIUM.

## Bucket 4 — novelty / closest prior work
- **Dual-Pathway Fusion of EHRs and KG for Unseen DDIs** (2025, arXiv 2511.06662) —
  **CLOSEST / direct competitor.** Learned fusion between a KG/structure pathway (PK-ish)
  and an EHR pathway (PD/empirical), teacher-student for cold-start, attends over
  side-effect neighbors. BUT its second channel is EHR observational co-occurrence, not a
  KG cross-attention select-compose over phenotype neighbors; its split is *data-source*
  (mechanistic vs observational), NOT an explicit learned PK/PD-emergent gate conditioned
  on mediator composition. Close in spirit, different mechanism. MUST cite + differentiate
  (after reading the real PDF).
- **Systematic review: molecular structures, KGs, cold-start in DDI** (2025, Comp. Biol.
  Med.) — frames exactly our problem space; cite for positioning. STRONG.
- **CSMDDI** (2022, PMC8851772) — cold-start multi-type DDI, single-channel, no PK/PD
  routing. MEDIUM.

No found method uses an explicit mechanism-typed (PK/PD) learned gate between an
aggregate-molecular channel and a select-compose-effect channel.

## Action items into the design
1. Differentiate from 2511.06662 explicitly (KG select-compose + mediator-conditioned gate
   vs EHR observational dual-pathway). Read the PDF first.
2. Cite Hou 2017 + PMC10456269 as the mechanistic justification for the two-channel split.
3. Cite AdaptMol for "uniform fusion hurts" (the TIGER/MKG-FENN collapse framing).
4. Position effect-layer cross-attention as novel vs Bucket-1 (those are substructure-only).

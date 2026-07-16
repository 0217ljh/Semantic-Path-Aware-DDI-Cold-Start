# Learning to Describe for Predicting Zero-shot Drug-Drug Interactions (TextDDI)

- **Authors**: Fangqi Zhu, Yongqi Zhang, Lei Chen, Bing Qin, Ruifeng Xu
- **Year / Venue**: 2023, EMNLP
- **Link**: https://arxiv.org/abs/2403.08377 (also aclanthology.org/2023.emnlp-main.918)
- **Read depth**: full-text
- **Cluster**: c3

## TL;DR
First clean demonstration that **textual drug descriptions** (Background, Indication, Mechanism of Action sections from DrugBank/PubChem) are sufficient for zero-shot DDI prediction on truly unseen drugs. Pairs an LM-based classifier with an RL-based sentence selector that picks the most informative spans. Directly validates our insight i4.

## Problem & Setting
**Zero-shot DDI prediction**: chronological split — test interactions involve at least one new drug never seen during training. Defined as the mapping `(D_new × D) ∪ (D × D_new) → I`. Drug Interaction sections explicitly excluded from descriptions to prevent leakage. Evaluated on DrugBank and TWOSIDES.

## Method (core)
1. **LM predictor** (RoBERTa-base): input is a concatenated prompt "[drug-A description] [SEP] [drug-B description]" up to 256 tokens; CLS token → linear classifier over interaction types.
2. **RL information selector**: framed as MDP where each step samples one sentence from the drug descriptions. Policy is an MLP comparing sentence embedding to current prompt state. Reward = improvement in DDI predictor quality score.
3. Joint training alternates between LM and selector.

## Cold-start handling
- Drug is represented **purely by its biomedical text** — no SMILES, no KG, no topology.
- Therefore a new drug is fully usable from day-one as long as a textual description exists.
- The RL selector learns which textual aspects (mechanism, target enzyme, indication) are diagnostic for DDI — implicitly distilling PK vs PD signals from text.

## Key contributions
- Reframes cold-start DDI as a text-to-label task.
- DrugBank zero-shot F1: 52.5% vs 32.9% for best symbolic baseline (Decagon).
- TWOSIDES zero-shot accuracy: 80.3% vs 75.9%.
- Shows that "drug ID only" and "drug name only" prompts also work, but full descriptions are much better — strong evidence that semantic text matters.

## Limitations / gaps (as relevant to our insights)
- Pure text — no structural/topological reasoning over biomedical KG. Cannot exploit shared mediators (enzymes, transporters) explicitly.
- Black-box LM — no interpretable path or mechanism.
- Requires high-quality, leakage-free descriptions; smaller for drugs in early development.
- Does not distinguish PK vs PD reasoning explicitly, though the selector likely does so implicitly.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Implicit at best — the RL selector may pick mechanism-of-action sentences (PK) or side-effect sentences (PD) depending on the pair, but architecturally undifferentiated.
- **i2 (meeting node + over-smoothing)**: Sidesteps GNN entirely. Doesn't demonstrate that meeting-node anchoring helps; merely that text is sufficient.
- **i3 (pooling noise + attention limit)**: RL selector is exactly an "external semantic prior" mechanism — RL reward acts as a hard supervisor over which sentences are kept. Very aligned with our i3 critique of pure attention.
- **i4 (node-name semantic prior)**: **Strong validation** — this paper is the clearest published evidence that node-name / textual semantics is the dominant stable signal under cold-start. Sets the bar we need to match while also exploiting KG structure.

## Notes
- Should be cited as the canonical text-only cold-start baseline alongside ZeroDDI.
- Performance numbers (52.5% F1 vs 32.9% Decagon) give us a quantitative anchor for the "structure-only methods collapse under cold-start" claim.
- Public code available via the OpenReview / authors' repo.

# ZeroDDI: A Zero-Shot Drug-Drug Interaction Event Prediction Method with Semantic Enhanced Learning and Dual-modal Uniform Alignment

- **Authors**: Ziyan Geng et al.
- **Year / Venue**: 2024, IJCAI
- **Link**: https://www.ijcai.org/proceedings/2024/671 (arXiv:2407.0891)
- **Read depth**: full-text
- **Cluster**: c3 (zero-shot DDI events, where "zero-shot" is over event class)

## TL;DR
Zero-shot DDI-event prediction: classifies drug pairs into **unseen event classes** by aligning drug-pair representations with BioBERT-encoded event-class descriptions. Decomposes events into (Effect, Sign, Pattern) attributes and uses cross-modal substructure-text fusion plus uniform-alignment contrastive loss.

## Problem & Setting
**Zero-shot DDI-Event (ZS-DDIE)**: unseen DDI event classes at test time. Note: "zero-shot" here is over **event classes**, not over drugs — drugs themselves may appear in training. Each event class decomposed via three attributes:
- Effect (e.g., "bradycardia") — from MeSH
- Sign (increased / decreased)
- Pattern (directional)

Evaluated on DrugBank-derived event set (175 train + 68 unseen classes).

## Method (core)
1. **Biological semantic enhanced representation (BRL)**:
   - Class-level semantics from full DDIE description (BioBERT).
   - Attribute-level semantics from Effect terms via MeSH lookup (BioBERT).
2. **Substructure-guided fusion (SSF)**: cross-modal attention between molecular substructure (GIN) embeddings and BioBERT text tokens.
3. **Dual-modal uniform alignment (DUA)** loss:
   - Alignment loss (contrastive)
   - Class uniformity (distribute class reps on unit sphere)
   - Instance uniformity (mirror for drug-pair reps)

## Cold-start handling
- Cold-start is over **event classes**, not drugs. Drugs are encoded via GIN on molecular graph (i.e., structural).
- Unseen event classes are handled via BioBERT-encoded textual class descriptions — semantics are the bridge.
- No drug-level cold-start protocol reported.

## Key contributions
- Decomposes DDI events into reusable attribute primitives (Effect/Sign/Pattern) — enables compositional zero-shot.
- Cross-modal alignment between SMILES-derived structure and BioBERT-derived event text.
- 17.57% accuracy on conventional ZSL vs 8.25% for 3DGT-DDI baseline; 26.48% harmonic mean on generalized ZSL.

## Limitations / gaps (as relevant to our insights)
- Class-level cold-start, not drug-level. Cannot be used as direct baseline for our S2 drug-cold-start setting.
- Drugs are still represented by GIN on molecular graph — vulnerable when chemotype is novel.
- Requires manual attribute annotation for new event classes.
- Performance degrades severely under heavy class imbalance.

## Relevance to our insights
- **i1 (PK/PD two paradigms)**: Partial — the (Effect, Sign, Pattern) decomposition implicitly captures some PD vocabulary, but does not separate PK and PD reasoning architecturally.
- **i2 (meeting node + over-smoothing)**: Not applicable (no KG GNN).
- **i3 (pooling noise + attention limit)**: Uses cross-modal attention with **explicit textual prior** (BioBERT class embeddings). This is closer to the spirit of i3's "external semantic prior" demand — but applied to event side rather than drug side.
- **i4 (node-name semantic prior)**: **Strong evidence in the same direction**, but applied to event-class names rather than drug nodes. Demonstrates that BioBERT-encoded biomedical terminology is a viable bridge for unseen classes; we extend the same principle to unseen drugs.

## Notes
- Pairs well with TextDDI (zero-shot over drugs) — together they motivate that semantics on both sides matters.
- BioBERT-on-class-names trick is portable to drug-node names in our setting.

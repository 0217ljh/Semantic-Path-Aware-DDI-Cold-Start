# Cluster c6 Summary — Drug Representation (SMILES / Molecular Structure / Fingerprint) for DDI

## Scope
Methods that encode drugs primarily from their **molecular structure** (SMILES, 2D molecular graphs, 3D conformations, learned fingerprints) for DDI prediction. KG-heavy methods sit in c1; multi-event focus sits in c5; cold-start-only methodology sits in c3. Some papers here also use KG / text as a secondary signal but the molecular branch is central.

## Papers (7)

| File | Method | Year | Encoder | Cold-start? | Role for our story |
|---|---|---|---|---|---|
| `ryu_2018_deepddi.md` | DeepDDI | 2018 | ECFP4 → Structural Similarity Profile → DNN | implicit only | Canonical PK-only baseline, no substructure resolution |
| `nyamabo_2021_ssi_ddi.md` | SSI-DDI | 2021 | GAT on mol graph + co-attention substructure pairs | implicit (inductive-capable) | Strawman for i3: flat attention over substructure bag |
| `nyamabo_2022_gmpnn_cs.md` | GMPNN-CS | 2022 | Gated MPNN with edge gates, size-adaptive substructures | **explicit inductive eval** | Authors concede substructure-redundancy pooling noise; key i3 citation |
| `yang_2022_sa_ddi.md` | SA-DDI | 2022 | D-MPNN + substructure attention + partner-conditioned SSIM | **explicit S1 + S2 cold eval** | Shows partner-conditioned attention partially fixes i3 but still degrades cold |
| `chen_2021_muffin.md` | MUFFIN | 2021 | MPNN (mol) + TransE (KG) with bi-level fusion | transductive only | Closest precedent for "two-signal fusion"; KG side non-inductive |
| `he_2022_3dgt_ddi.md` | 3DGT-DDI | 2022 | SchNet on 3D conformation + SciBERT on sentence | inductive-capable, not S2-benched | Closest c6 precedent for using text alongside structure |
| `chithrananda_2020_chemberta.md` | ChemBERTa (+ ChemBERTaDDI 2025) | 2020 / 2025 | RoBERTa MLM on 77M SMILES | inductive by construction | Strongest pure-PK encoder; scaffold for i4 analogy |

## Common architectural pattern in cluster c6

1. SMILES → atom/bond features (RDKit) or token sequence.
2. Encoder: MPNN / GAT / D-MPNN / 3D-GNN / SMILES transformer.
3. Drug-pair interaction: substructure-substructure attention (SSI-DDI, GMPNN-CS, SA-DDI) **or** pooled drug-vector pair-MLP (DeepDDI, ChemBERTa, MUFFIN).
4. Classification head over DDI types.

## Trajectory 2018 → 2025

- **2018 DeepDDI** — flat similarity profile, no substructure resolution.
- **2021 SSI-DDI** — substructure-pair co-attention, symmetric.
- **2022 GMPNN-CS** — gated edges produce learnable variable-size substructures; explicit inductive eval.
- **2022 SA-DDI** — partner-conditioned attention (SSIM); explicit cold-start eval shows clear degradation.
- **2022 3DGT-DDI** — geometric (3D) + literature text fusion.
- **2021 MUFFIN** — structure + KG fusion (transductive only).
- **2020 ChemBERTa → 2025 ChemBERTaDDI** — transformer SMILES pretraining transferred to DDI.

The frontier moved from global descriptors → substructure attention → partner-conditioned attention → external-signal fusion (3D, KG, language pretraining). Two pain points persist: (a) pooling over substructures remains data-driven without external priors, and (b) cold-start S2 numbers are consistently weaker than warm-start.

## How cluster c6 relates to our four insights

### i1 — PK vs PD two reasoning paradigms
**All c6 methods are PK-side.** They reason about molecular structure / chemical interaction. None encode downstream physiological effect overlap. MUFFIN adds KG (some PD signal via gene/disease relations) and 3DGT-DDI adds clinical text, but neither separates the two paradigms architecturally. c6 is the canonical "PK paradigm" cluster.

### i2 — Anchor at meeting node, not drug
c6 methods anchor at either (a) the drug-level vector pair (DeepDDI, ChemBERTa, MUFFIN) or (b) the substructure-substructure pair (SSI-DDI, GMPNN-CS, SA-DDI). The substructure pair is the closest c6 analogue to a "meeting node" but it lives in chemistry space, not in biomedical / phenotypic space. SA-DDI's SSIM is the most explicit attempt to anchor on the interaction rather than the drug.

### i3 — Uniform pooling is noisy; attention can't inject external priors
This is the **most directly contested point** in c6. The whole substructure-attention line (SSI-DDI → SA-DDI) is the cluster's attempt to fix uniform pooling. **GMPNN-CS itself acknowledges substructure redundancy** as an open problem. **SA-DDI's cold-start degradation** is empirical evidence that data-driven attention without external priors hits a ceiling. None of the seven papers inject external pharmacological priors (e.g. known reactive sites, known DDI mechanism annotations) into the pooling — all attention weights are learned from data alone.

### i4 — Node-name biomedical text as cold-stable signal
**SPECIAL contrast for c6.** Molecular features (SMILES, fingerprints, 3D conformations) are cold-stable in the **chemistry** sense — they can be computed for any new drug. Our i4 argues for cold-stability in the **biomedical semantic** sense — node names carry pharmacological meaning even when the drug is unseen. The two stability properties are **complementary**:
- Chemistry-stable signals (c6): say what the molecule looks like, but say nothing about pharmacology.
- Name-stable signals (i4): say what the drug *means* clinically/biologically, but say nothing about chemistry.

ChemBERTa is the cleanest analogue: it shows that *language model pretraining on chemistry text* gives cold-stable embeddings. We do the same recipe on *biomedical node-name text*, an orthogonal corpus. 3DGT-DDI is the closest c6 precedent for adding text, but it uses **sentence-level** literature text (which may not exist for new drugs), not entity-level node names (which always exist).

## Takeaway for our paper

c6 is the PK-side baseline cluster. The frontier (SA-DDI, GMPNN-CS) admits two open problems we should foreground:
1. **Substructure pooling is data-driven without external priors** — direct support for i3.
2. **Cold-start S2 numbers degrade significantly even with state-of-the-art molecular encoders** — direct support for "molecular signal alone hits a ceiling, need a complementary cold-stable signal."

Our work should be positioned as **adding an orthogonal, biomedical, node-name-text-based cold-stable signal** to the c6 molecular encoder line. Cite SA-DDI for the cold-start ceiling, GMPNN-CS for the explicit acknowledgment of pooling noise, and ChemBERTa as the analogy template (LM pretraining → cold-stable embedding) applied to a different corpus.

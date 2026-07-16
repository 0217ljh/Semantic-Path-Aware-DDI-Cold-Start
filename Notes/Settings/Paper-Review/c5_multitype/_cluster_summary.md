# Cluster c5 Summary: Multi-Type / Event-Type / Multi-Label DDI

## Papers reviewed (8)

| # | Paper | Year | Venue | Classes | Cold-start? |
|---|---|---|---|---|---|
| 1 | Ryu et al. DeepDDI | 2018 | PNAS | 86 (multi-label sentence templates) | partial (SMILES inductive, random split) |
| 2 | Zitnik et al. Decagon | 2018 | Bioinformatics (ISMB) | 964 polypharmacy effects (TWOSIDES) | no (transductive R-GCN) |
| 3 | Deng et al. DDIMDL | 2020 | Bioinformatics | 65 events (DS1) | Tasks 1/2/3 explicit |
| 4 | Chen et al. MUFFIN | 2021 | Bioinformatics | 81 / 200 (binary/multi-class/multi-label) | not evaluated |
| 5 | Nyamabo et al. SSI-DDI | 2021 | Brief. Bioinform. | 86 DrugBank | inductive baseline |
| 6 | Lin et al. MDF-SA-DDI | 2022 | Brief. Bioinform. | 65 / 100 | Tasks 1/2/3 |
| 7 | Lin et al. MDDI-SCL | 2022 | J. Cheminform. | 65 / 100 | Tasks 1/2/3 |
| 8 | Masumshah & Eslahchi DPSP | 2023 | Bioinform. Adv. | 65 / 100 / 185 | not evaluated |
| (9) | Yu et al. MSEDDI | 2023 | IJMS | 65 | Tasks 1/2/3 |

(9 papers in total since MSEDDI was added as a multi-scale variant. Effective count = 8 distinct method families.)

## Benchmarks crystallized by c5
- **Deng-65 (DS1)**: 572 drugs, 74,528 DDIs, 65 events. The de facto standard.
- **Deng-100 (DS2)**: 1,258 drugs, 323,539 DDIs, 100 events. Larger, used by MDF-SA-DDI and MDDI-SCL.
- **DPSP-185 (DS3)**: 645 drugs, 63,473 DDIs, 185 adverse effects. Bridge to polypharmacy.
- **DrugBank-86**: original DeepDDI sentence-template space, reused by SSI-DDI.
- **TWOSIDES-200 / -964**: polypharmacy side effects, multi-label. MUFFIN uses 200, Decagon uses 964.

## Cold-start protocol (Tasks 1/2/3) - canonical mapping to our S0/S1/S2
- Task 1 = S0 (both drugs seen, unseen pair).
- Task 2 = S1 (one drug new).
- Task 3 = S2 (both drugs new). All DDIMDL-family papers report dramatic degradation here (DDIMDL: 0.88 -> 0.41 acc).

## Class-imbalance handling: a maturity ladder
- Level 0 (none): DeepDDI, DDIMDL, MUFFIN, DPSP, Decagon, SSI-DDI.
- Level 1 (focal loss): MDF-SA-DDI (also mixup).
- Level 2 (focal + label smoothing + supervised contrastive): MDDI-SCL.
- None of the 9 papers reports per-mechanism (PK vs PD) performance breakdown - this is an open gap.

## Mapping classes to PK / PD (rough)
- Deng-65 / Deng-100: events come from DrugBank text. Crude PK markers in sentence templates: "metabolism", "serum concentration", "absorption", "excretion", "CYP", "transporter". PD markers: "therapeutic efficacy", "risk of", "adverse effects (any specific outcome)", "QT prolongation", "hypotension", "sedation". A back-of-envelope split of Deng-65 puts ~25 events on the PK side and ~40 on the PD side, but no c5 paper does this split explicitly.
- Decagon-964: almost entirely PD (specific clinical side effects).
- TWOSIDES-200: PD-dominant.
- DPSP-185: PD-dominant (polypharmacy effects).

This means **the Deng-65/100 benchmark is the only one where a PK-vs-PD split is meaningful** - which makes it the right target for testing our i1 hypothesis.

## Performance asymmetry across event types (direct support for i1)
- Decagon (2018): explicitly notes "models particularly well side effects with a strong molecular basis." Per-effect AUROC range 0.7 - 0.98.
- DDIMDL family: no per-event breakdown released, but the focal-loss / mixup / SCL additions in MDF-SA-DDI and MDDI-SCL are all motivated by per-event performance variance.
- SSI-DDI: order-of-drugs sensitivity is a hint that some event types (asymmetric mechanisms - "X inhibits Y's metabolism") are systematically harder than symmetric ones.

## Takeaways re: i1-i4

### i1 (PK vs PD = two paradigms)
- **Zero papers in c5 explicitly differentiate PK and PD events**. Every paper treats all 65/100/200/964 classes as flat softmax/sigmoid outputs.
- The closest is Decagon's molecular-vs-non-molecular observation and MUFFIN's structure-vs-KG dual branch, but neither is class-routed.
- **Strong open opportunity**: a PK/PD-aware grouping of Deng-65/100 plus mechanism-routed prediction is novel.

### i2 (meeting node + over-smoothing)
- Most c5 methods avoid drug-drug GNNs entirely (DDIMDL, MDF-SA-DDI, MDDI-SCL, DPSP, MUFFIN), so they sidestep over-smoothing.
- Decagon uses 2-hop R-GCN drug -> protein -> drug, which is effectively a "meeting-node" architecture - mild support for our i2 prescription, but it's transductive and pools uniformly.
- SSI-DDI uses GAT per-molecule with L=2 hops; no inter-drug propagation.
- **Gap**: nobody in c5 builds a learnable meeting-node graph that is mechanism-typed (PK meeting nodes = enzymes/transporters; PD meeting nodes = target proteins / pathways).

### i3 (uniform pooling = noisy, attention alone insufficient)
- DDIMDL averages 4 modality outputs -> baseline of "uniform pooling is noisy".
- MDF-SA-DDI and MSEDDI add self-attention across modalities -> shows attention helps but only over a small set of view tokens, not over neighborhoods.
- MDDI-SCL adds supervised contrastive loss as an external prior on the latent space -> directly supports i3's claim that pure attention is insufficient without an external structural prior.
- **Strong support for i3 from the c5 maturity ladder**: every newer method that beats DDIMDL adds an external prior (focal, mixup, contrastive, KG, multi-scale) beyond pure attention.

### i4 (node-name biomedical text)
- **Zero papers in c5 use drug or target NAMES as text features**. DeepDDI uses names only in output templates; MUFFIN/MSEDDI use KG entity IDs (vector lookups), not text.
- **Strongest open opportunity**: c5 cold-start failure on Task 3 is precisely where node-name biomedical text would help (a brand-new drug has a name even when targets/KG-edges are missing). No c5 method exploits this.

## Recommended c5 baselines for our paper
- **Primary**: DDIMDL (similarity-feature DNN), MDF-SA-DDI (transformer + focal), MDDI-SCL (contrastive), SSI-DDI (substructure GAT - the only one with native inductive ability).
- **Secondary**: MUFFIN (structure + KG dual branch), MSEDDI (multi-scale).
- **For polypharmacy extension**: Decagon, DPSP.

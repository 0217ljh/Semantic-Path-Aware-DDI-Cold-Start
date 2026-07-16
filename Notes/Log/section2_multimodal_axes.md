# Section 2 — Multi-Modal DDI Methods on Two Axes (verified 2026-06-21)

**Status**: Master comparison table for Section 2. Built by reading the actual PDFs of 8 representative papers (3 parallel agents, each quoting method sections). No fabrication — every cell traced to the paper's method section.

**PDFs**: `Paper/Reference/{MKG-FENN,TIGER,MUFFIN,MIRACLE,DDKG,KnowDDI,K-Path,MolecBioNet}.pdf`
**Extracts saved**: `Paper/Reference/{KnowDDI,K-Path,MolecBioNet}_extract.txt`

---

## ⚠ VERIFIED 2026-06-21 (6 agents read PDF + GitHub code per paper)

All 6 genuinely-multi-modal papers re-checked cell-by-cell against **both PDF and GitHub source code**. Result: **47/48 cells CONFIRMED, 1 corrected** (MolecBioNet fusion arity: paper says 4-stream concat, code does 6-stream). Three-tier classification emerged.

### TABLE 1 — Data-flow granularity (encode → pool → align), 7 papers, code-verified

| Paper | Venue | Tier | Mol encode | Mol pooled to | KG encode | KG pooled to | **Alignment granularity** |
|---|---|---|---|---|---|---|---|
| MKG-FENN | AAAI24 | A | substructure (no atom GNN; recast as KG triples) | drug | node (1-hop) | drug | **drug** |
| TIGER | AAAI24 | A | atom | drug (READOUT g_i) | subgraph | drug (s_i) | **drug** |
| MUFFIN | Bioinf21 | A | atom+bond (MPNN, offline pretrain) | drug (M1) | node (TransE, offline) | drug (M2) | **drug** |
| MolecBioNet | KDD25 | A | substructure (BRICS) + atom | **drug-pair (AGIPool)** | subgraph (k-hop) | drug-pair (CASPool) | **drug-pair** |
| MIRACLE¹ | WWW21 | B | atom+bond (bond-aware MPN) | drug | DDI-graph node/edge | drug | **drug** |
| DDKG | BIB22 | C | substructure (SMILES seq, Bi-LSTM) | drug (init only) | node+triple | drug | **drug** |
| EmerGNN | ICML23 | C | substructure (Morgan FP 1024-bit) | drug (init only) | path/subgraph (L=3 relation-aware) | drug | **drug** |

### TABLE 2 — Alignment mechanism (KG content + algorithm + fusion), code-verified

| Paper | Venue | Tier | KG content | Alignment algorithm | Fusion stage |
|---|---|---|---|---|---|
| MKG-FENN | AAAI24 | A | biomedical KG (only ch.1 real bio tails; 1-hop) | implicit (CE only) | late concat (4-channel) |
| TIGER | AAAI24 | A | biomedical KG | contrastive JS-MI (anchor = fused h↔each view) | late concat + MI loss |
| MUFFIN | Bioinf21 | A | biomedical KG (DRKG) | joint-embedding (outer + element-wise) | bilinear/intermediate + residual late-concat |
| MolecBioNet | KDD25 | A | biomedical KG (tsBKG) | **MI-MINIMIZATION (decorrelation; pushes apart)** | late concat (code 6-stream; paper Eq.20 4-stream) |
| MIRACLE¹ | WWW21 | B | **DDI graph, drug-drug only** | contrastive MI (pos=1-hop DDI neighbors) | intermediate (GCN injects DDI topology) |
| DDKG | BIB22 | C | biomedical KG | implicit (mol = node init only) | early init + hierarchical |
| EmerGNN | ICML23 | C | biomedical KG | implicit (Morgan FP = node feature init via Went) | early init + L=3 path reasoning |

**Footnote ¹ (MIRACLE)**: MIRACLE's second "modality" is the **drug-drug interaction graph itself**, not an external biomedical KG. Code (`main.py:139-154`) builds the graph purely from drug-pair interaction labels; nodes are drug SMILES; there are zero protein/pathway/biomedical entities. We include it because it does run a genuine parallel molecular encoder, but its KG view carries DDI topology (≈ task labels as a graph), not external biological knowledge.

### TABLE 3 — Cold-start scenario coverage (code/paper-verified 2026-06-21)

| Paper | Venue | Tier | S0 (warm) | S1 (known×new) | S2 (new×new) | S2 metric | cold-start = core? |
|---|---|---|---|---|---|---|---|
| MKG-FENN | AAAI24 | A | ✅ Task1 | ✅ Task2 | ✅ Task3 | ACC 0.455 / AUC 0.915 | secondary |
| TIGER | AAAI24 | A | ✅ 5-fold | ❌ | ❌ | — | **NO (warm-only)** |
| MUFFIN | Bioinf21 | A | ✅ 5-fold | ❌ | ❌ | — | **NO (warm-only)** |
| MolecBioNet | KDD25 | A | ✅ | ✅ Novel-Existing | ✅ Novel-Novel | ACC 0.388 | **YES (core)** |
| MIRACLE | WWW21 | B | ✅ 4:1 | ❌ | ❌ | — | **NO (transductive)** |
| DDKG | BIB22 | C | ✅ | ✅ DW-CV | ❌ (PW-CV ≠ S2) | DW-CV AUC 0.790 | secondary |
| EmerGNN | ICML23 | C | ✅ | ✅ S1 | ✅ S2 | F1 25.0±2.8 / ACC 46.3 | **YES (core)** |

**Verified judgment calls**:
- Only **3 papers test S2 (new×new)**: MKG-FENN (secondary), MolecBioNet (core), EmerGNN (core).
- **TIGER / MUFFIN / MIRACLE never hold out drugs** — pure warm-start, zero cold-start signal. (Explains the ColdDDI-table TIGER S2 = 0.56 we verified earlier — TIGER was not designed for cold-start.)
- **DDKG's PW-CV is NOT S2**: it holds out drug PAIRS, but each drug still appears in other training pairs → novel-combination, not both-unseen. Proof: PW-CV (0.739) is EASIER than DW-CV (0.717), the opposite of a true new×new setting. DDKG only reaches S1.
- **EmerGNN is the source of the S0/S1/S2 naming convention.**

**The white space our paper occupies**: No existing work is simultaneously (i) clean dual-modal (Tier A: parallel molecular branch + real biomedical KG), (ii) S2 cold-start as a core contribution, AND (iii) mechanism-level cross-modal alignment. MolecBioNet is Tier A + S2-core but uses MI-MINIMIZATION (the opposite of alignment, S2 ACC only 0.388); EmerGNN is S2-core but Tier C (molecular only inits nodes). The intersection is empty — that is our contribution.

### Four locked conclusions (verified against Tables 1-3)

1. **Late cross-modal mixing**: every genuinely dual-modal method encodes each drug INDEPENDENTLY to a drug-level representation, then mixes the two modalities afterward (concat / bilinear / MI). No cross-modal interaction happens DURING encoding. (Caveat: MUFFIN's bilinear is technically intermediate, but still operates on already-pooled drug-level vectors.)

2. **Pooling is universal** (most solid conclusion, zero exceptions): every method pools from many→few BEFORE alignment→fusion. All 7 pool molecular and KG to drug-level; MolecBioNet pools even coarser to drug-pair.

3. **Alignment mechanism is NOT homogeneous** (corrected from "mainly contrastive"): of the 7, only 2 use genuine contrastive (TIGER JS-MI, MIRACLE MI); 3 have NO explicit cross-modal alignment loss (MKG-FENN, DDKG, EmerGNN — task loss only); 1 uses joint-embedding (MUFFIN); and the strongest recent one (MolecBioNet) uses MI-MINIMIZATION, the OPPOSITE of contrastive. → The field has no consensus on how to align; designs range from no-alignment to contrastive to anti-alignment.

4. **Encoder backbones are GNN-centric**: KG side is always GNN/message-passing; molecular side is mostly GNN (MPNN/GCN/graph-transformer), with sequence-model exceptions (DDKG Bi-LSTM; EmerGNN uses fixed Morgan FP). The forward pass is fundamentally a GNN in 6/7.

### MolecBioNet MI-minimization — precise mechanism (closest-competitor distinction)

**Acts on**: h_uv (KG k-hop subgraph pair-embedding) and z_uv (molecular pair-embedding), projected to z_kg / z_mol (`loss.py:48-49`).

**Estimator** (paper Eq.15-18, code L52-62):
```
MI ≈ ce_kg_mol + ce_mol_kg − bi_di_kld
  ce_kg_mol = MSE(z_kg, z_mol.detach())     # conditional-entropy MSE surrogate
  ce_mol_kg = MSE(z_mol, z_kg.detach())
  bi_di_kld = KL(z_kg‖z_mol) + KL(z_mol‖z_kg)  # symmetric KL
```
Minimized (γ > 0 in `L = L_P + βL_C + γL_MI`).

**Stated purpose** (paper §3.3): "to ensure the embeddings are complementary while minimizing redundancy... high MI indicates redundancy which is undesirable... constrains h_uv and z_uv to capture unique and complementary aspects."

**Mechanistic effect**: DECORRELATES the two views — penalizes mutual predictability so KG and molecular embeddings encode disjoint/non-overlapping information, then CONCATENATES them. Pushes the two views toward statistical independence, NOT toward a shared space.

**Distinction from our alignment**: A contrastive/alignment loss MAXIMIZES agreement (pulls same-entity views together onto shared semantics). MolecBioNet does the OPPOSITE: it deliberately makes molecular and KG pair-embeddings encode DIFFERENT things, then concatenates for breadth — never establishing a cross-modal correspondence.

**Paper-ready distinction sentence**:
> "MolecBioNet minimizes a mutual-information proxy to force its molecular and KG pair-embeddings to be complementary (mutually non-redundant) and then concatenates them for prediction. We instead align molecular fragments with KG mechanism units, establishing a cross-modal correspondence onto shared mechanism semantics rather than a decorrelation."

**Nuances** (honest, for paper): (a) it is an upper-bound PROXY not exact MI (H(h,z) dropped, MSE Gaussian surrogate, `−bi_di_kld` can be negative); (b) `.detach()` makes the two MSE directions asymmetric; (c) it is a post-encoding REGULARIZER on already-fused dual-view embeddings — distinct from our built-into-encoding architectural alignment.

### Three-tier classification (verified)

| Tier | Criterion | Papers |
|---|---|---|
| **A. Clean dual-modal** | parallel molecular branch + real biomedical KG | MKG-FENN, TIGER, MUFFIN, MolecBioNet |
| **B. Narrow-KG** | parallel molecular branch but "KG" = DDI interaction graph | MIRACLE |
| **C. Init-only** | molecular info ONLY seeds KG node init (Morgan FP / SMILES embedding absorbed at layer 0) | DDKG, EmerGNN |

### Key caveats surfaced by code verification

- **EmerGNN** (Tier C, project code `baseline/emergnn/model.py`): `feat='M'` mode projects 1024-bit Morgan fingerprint via `Went = nn.Linear(1024, n_dim)` to seed drug-node features (`:71-72,114`), then L=3 relation-aware message passing over the KG (`:48,137`). Molecular info enters only as node-feature init — same init-only pattern as DDKG. Still factually multi-modal (it uses molecular fingerprints), just weakly.
- **DDKG** (Tier C): SMILES Bi-LSTM output used ONLY as drug node layer-0 init, absorbed into KG message passing. Ablation DDKG-E (random init) drops Acc only ~3% (KEGG) → molecular is an init prior, not co-equal modality.
- **MIRACLE** (Tier B): see footnote ¹ — KG is the DDI graph itself.
- **MolecBioNet** (Tier A, closest competitor): all 3 critical checks CONFIRMED via code — (a) AGIPool collapses fragments to ONE pair vector before fusion (`encoder.py:202`); (b) MI is MINIMIZED not maximized (`loss.py:69-71`, "minimize mutual information", positive γ) → pushes the two views APART; (c) NO fragment-to-KG-mechanism alignment anywhere. **Our fragment-level cross-modal alignment novelty is clear of MolecBioNet.**

**Repo note**: MolecBioNet working repo is `github.com/MengjieChan/MolecBioNet` (paper cites "Chan", README author "Chen" — working URL is "MengjieChan").

---

## Scope note

Of the 8 papers initially surveyed, **2 are actually KG-only** (KnowDDI, K-Paths — no molecular encoder; see Key Observation 1 below). The tables below keep only the **6 genuinely multi-modal methods** plus ours.

## Axis 1 — Granularity (encode granularity ≠ alignment granularity)

**Critical distinction**: the granularity at which each modality is *encoded* is NOT the granularity at which the two modalities are *aligned/fused*. Most methods encode finely (atom / substructure) but **pool to drug or drug-pair level before cross-modal interaction**. The "alignment granularity" column is the load-bearing one for cold-start.

| Paper | Venue | Mol encode granularity | KG encode granularity | **Alignment granularity** | Alignment type |
|---|---|---|---|---|---|
| **MKG-FENN** | AAAI 2024 | substructure (no atom GNN; mol recast as KG triples) | node (1-hop) | **drug** | drug-KG (single unified space) |
| **TIGER** | AAAI 2024 | atom (→ READOUT to drug vector g_i) | subgraph (per-drug) | **drug** (g_i ↔ s_i) | drug-KG (two views of same drug) |
| **MUFFIN** | Bioinformatics 2021 | atom+bond (MPNN → drug vector M1) | node (TransE entity M2) | **drug** (M1 ⊗ M2) | drug-KG |
| **MIRACLE** | WWW 2021 | atom+bond (bond-aware MPN → drug vector) | node/edge of DDI graph | **drug** | **drug-drug** (mol-view ↔ DDI-graph-view of same drug) |
| **DDKG** | Brief Bioinf 2022 | atom-SMILES seq (Bi-LSTM → drug init embedding) | node+triple | **drug** (mol seeds node init) | drug-KG |
| **MolecBioNet** | KDD 2025 | **substructure (BRICS) + atom** (hierarchical) | subgraph (k-hop) | **drug-pair** (AGIPool → z_uv) | drug-KG (pair level) |
| **OURS** | — | **substructure (BRICS fragment)** | **hyper-edge (mechanism class)** | **fragment ↔ mechanism** | fragment-mechanism |

→ **The killer column is "Alignment granularity": every existing method aligns at drug or drug-pair level. None aligns at the fragment / mechanism level — even MolecBioNet, which encodes with BRICS substructures, pools them to drug-pair before fusion.**

## Axis 2 — Cross-modal algorithm + fusion

| Paper | Venue | Alignment type | Alignment algorithm | Fusion stage |
|---|---|---|---|---|
| **MKG-FENN** | AAAI 2024 | drug-KG | **implicit** (task loss only; no cross-modal loss) | late (4-channel concat → MLP) |
| **TIGER** | AAAI 2024 | drug-KG | contrastive (JS mutual-information) | late (2-channel concat) + MI loss |
| **MUFFIN** | Bioinformatics 2021 | drug-KG | joint-embedding (cross-product + element-wise) | bilinear / intermediate |
| **MIRACLE** | WWW 2021 | drug-drug | contrastive (MI maximization, 1-hop DDI positives) | intermediate (GCN injects DDI topology) |
| **DDKG** | Brief Bioinf 2022 | drug-KG | implicit (mol = node init, no explicit aligner) | early init + hierarchical aggregation |
| **MolecBioNet** | KDD 2025 | drug-KG (pair) | **MI-MINIMIZATION** (decorrelation; pushes views APART) | late concat `[h_uv ∥ z_uv ∥ h_u ∥ h_v]` |
| **OURS** | — | fragment-mechanism | alignment (contrastive / cross-attn, TBD by ablation) | intermediate (fragment ↔ hyper-edge cross-modal) |

→ **Fusion is late or early-init in every case** — no method does cross-modal reasoning *during* encoding. Even the two methods with an explicit cross-modal loss (TIGER JS-MI, MolecBioNet MI-min) stitch the modalities by late concatenation; the loss only shapes the two independently-pooled vectors.

---

## Per-paper evidence (1-line each)

- **MKG-FENN**: "all four sources... represented in the form of triples ⟨drugs, relationships, entities⟩... Ê = E1⊕E2⊕E3⊕E4 then MLP" — molecular info is recast as KG triples, no separate molecular view, pure concat (no alignment loss).
- **TIGER**: `h_i = MLP(g_i || s_i)` where g_i = atom-READOUT, s_i = KG-subgraph readout; explicit `L_MI^hg, L_MI^hs` JS-MI between fused and each view.
- **MUFFIN**: MPNN mol feature M1 + TransE KG feature M2 → "cross-level unit (outer product + CNN) + scalar-level unit (element-wise product)" → concat.
- **MIRACLE**: bond-aware MPN (inter-view) + GCN on DDI graph (intra-view); contrastive MI "intra-view of anchor agrees with inter-view of positives (1-hop DDI neighbors)". KG = drug-drug interaction graph.
- **DDKG**: "initializes drug representations with embeddings derived from SMILES via encoder-decoder, then recursively propagates KG neighbor info along top-ranked paths". Mol enters only as node init.
- **KnowDDI**: "we do NOT use any molecular features of drugs... learning solely from external KG and DDI fact triplets" — explicitly KG-only.
- **K-Paths**: "training-free... K shortest loopless paths... transformed into natural language... appended to query" — KG path reasoning, drug as entity name, no molecular encoder.
- **MolecBioNet**: BRICS substructure graph + Morgan FP + atom GCN (molecular) and tsBKG subgraph + Graph Transformer (KG); "MI minimization regularization... constrains h_uv and z_uv to capture unique and complementary aspects" then `f_uv = [h_uv ∥ z_uv ∥ h_u ∥ h_v]`.

---

## Key observation 1 — Two of the 8 "multi-modal" papers are actually KG-only

**KnowDDI** and **K-Paths** have **no molecular encoder**:
- KnowDDI uses pure one-hot node features and explicitly states it learns "solely from the combination of external KG and DDI fact triplets"
- K-Paths carries drugs only as KG entity names / text for an LLM; the GNN variant uses one-hot relational features

→ They surface in multi-modal DDI searches because abstracts mention molecules, but they are **single-modality KG models**. For Section 2, cite them as KG-only baselines, NOT as multi-modal counterexamples.

The genuinely multi-modal set is therefore **6 papers**: MKG-FENN, TIGER, MUFFIN, MIRACLE, DDKG, MolecBioNet.

## Key observation 2 — The shared blind spot (= our contribution gap)

Across all 6 genuinely multi-modal methods, three properties are universal:

### (a) Molecular output granularity collapses to whole-drug or drug-pair
Even methods that encode at atom/bond/substructure level **pool to a single drug (or drug-pair) vector before fusion**:
- MUFFIN, TIGER, MIRACLE, DDKG: atom/bond → whole-drug readout
- MKG-FENN: substructure dissolved into drug-node KG triples
- **MolecBioNet: BRICS substructures pooled by AGIPool to drug/pair level before fusion**

→ **No method aligns at the FRAGMENT level.** The substructural signal — which is what distinguishes cerivastatin from atorvastatin (Section 1) — is averaged away before cross-modal interaction.

### (b) KG granularity is node / subgraph / path — never mechanism-class
- node: MKG-FENN, MUFFIN, DDKG
- subgraph: TIGER, KnowDDI, MolecBioNet
- path: K-Paths
- drug-drug edge: MIRACLE

→ **No method represents the KG at the level of a mechanism class (a typed set of co-perturbed anchors).** They reason over individual entities or unstructured subgraphs, not over mechanism units.

### (c) Alignment object is drug-identity or drug-pair, never mechanism
- drug-KG (per-drug): MKG-FENN, TIGER, MUFFIN, DDKG
- drug-drug (per-drug): MIRACLE
- drug-KG (per-pair): MolecBioNet

→ **Every method aligns or fuses at the drug or drug-pair level.** None aligns molecular substructures to KG mechanism units. This is exactly the alignment object that collapses under cold-start (drug/pair identity is OOD), per Section 1.

### (d) Fusion is predominantly late concat
Even methods with an explicit cross-modal loss do late concatenation at the architecture level:
- TIGER: JS-MI loss + late concat
- MolecBioNet: MI-min loss + late concat
- MKG-FENN, MUFFIN, DDKG: late / early-init concat

→ Late concat means **no cross-modal reasoning during encoding** — the two modalities are computed independently and stitched at the end.

---

## Key observation 3 — MolecBioNet is the closest competitor; precise distinction

**MolecBioNet (KDD 2025)** is the strongest recent multi-modal method and the closest to ours:
- ✅ Uses BRICS substructures (like us)
- ✅ Genuinely dual-modal (molecular + biomedical KG)
- ✅ Claims cold-start capability
- ✅ Top venue (KDD 2025)

**But its design differs from ours on every one of the three blind-spot axes**:

| Axis | MolecBioNet | OURS |
|---|---|---|
| Molecular granularity at fusion | BRICS substructures **pooled by AGIPool to drug-pair level** before fusion | BRICS fragments **kept at fragment level** for alignment |
| KG unit | k-hop enclosing **subgraph** | **hyper-edge (typed mechanism-class anchor set)** |
| Cross-modal objective | **MI-MINIMIZATION** — pushes the two pair-level views APART to be complementary | **alignment** — pulls fragment and mechanism-unit representations TOGETHER |
| Fusion | late concat `[h_uv ∥ z_uv ∥ h_u ∥ h_v]` | intermediate fragment↔hyper-edge cross-modal |

**The sharpest distinction**: MolecBioNet's MI-minimization is the **opposite** of mechanism alignment. It deliberately decorrelates the molecular pair-embedding and KG pair-embedding to capture "complementary" information, then concatenates them. It never establishes a correspondence between a specific fragment and a specific biological mechanism. Our framework does exactly that correspondence — and at the fragment-to-mechanism level rather than the pair level.

→ This is a clean, defensible novelty statement: **even the closest competitor that uses BRICS substructures and dual-modal fusion does not align substructures to mechanism units; it pools both modalities to the drug-pair level and concatenates them with a decorrelation regularizer.**

---

## Section 2 paper-ready summary paragraph

> "We survey representative multi-modal DDI methods along two axes: the granularity at which each modality is represented (Axis 1: molecular = atom / bond / substructure / whole-drug; KG = node / edge / path / subgraph) and the cross-modal design (Axis 2: alignment object, alignment algorithm, fusion stage). The survey reveals a homogeneous blind spot. First, although several methods encode molecules at the atom or substructure level, **all of them pool to a whole-drug or drug-pair representation before cross-modal interaction** (MUFFIN, TIGER, MIRACLE, DDKG via readout; MKG-FENN by dissolving substructures into KG triples; MolecBioNet by AGIPool), discarding the substructural signal that distinguishes drugs with similar scaffolds but divergent mechanisms. Second, the KG is represented at the node, subgraph, or path level — **never as a mechanism-class unit** — so the model reasons over individual entities rather than the typed co-perturbation sets through which DDIs arise. Third, and consequently, **every method aligns or fuses at the drug or drug-pair level**, the exact representation that goes out-of-distribution under cold-start. Even the strongest recent method, MolecBioNet (KDD 2025), which uses BRICS substructures and a dual-modal architecture, pools both modalities to the drug-pair level and combines them with a mutual-information *minimization* regularizer that pushes the two views apart — the opposite of establishing a fragment-to-mechanism correspondence. This motivates our design: align molecular fragments directly with KG hyper-edges representing mechanism classes, so that the cross-modal correspondence is established at the level where DDIs are causally generated and where it remains stable across the warm-to-cold distribution shift."

---

## Status

| Item | Status |
|---|---|
| 8 papers read + categorized on 2 axes | ✅ |
| KnowDDI / K-Paths flagged as KG-only (not multi-modal) | ✅ |
| MolecBioNet identified as closest competitor + precise distinction | ✅ |
| Three blind-spot axes identified | ✅ |
| Section 2 paper-ready summary paragraph | ✅ |
| Detailed per-method failure diagnosis (WHERE/WHAT-WAY/WHAT-PROCESS) | 🟡 next discussion (Task #6) |

## Cross-references
- PDFs + extracts: `Paper/Reference/`
- Section 1 (motivation): `Notes/Log/paper_locked_logic.md` Section 1
- TIGER code: https://github.com/Blair1213/TIGER
- MolecBioNet code: https://github.com/MengjieChen/MolecBioNet

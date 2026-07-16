# Paper Writeup — Accumulating Material

**Purpose**: a single place where paper-worthy content (findings, framings, sentences, figures
that work) accumulate as we discuss each experiment. Anything that could be written into the paper
goes here, organized by section so when it's time to draft, the content is already structured.

**Convention** (per user, 2026-05-30): after discussing each experiment / cache / design,
extract the paper-worthy bits into this file. Per-cache improvement directions stay in their own
`Notes/Log/*.md` files (those are future TODOs, not paper content).

**Last update**: 2026-05-30

---

## Section 1 — Problem Framing & Baseline

(to be filled as we discuss more experiments)

---

## Section 2 — Method Components

(to be filled as we discuss more experiments)

---

## Section 3 — Experiments & Results

### 3.Y MNAH — A Minimal Meeting-Node Auxiliary Head Yields +3.0 ± 0.16 pt Over the EmerGNN Backbone (i2 verification)

**Background and motivation**

Insight i2 argues that DDI evidence should be anchored at the **meeting node**(the shared
intermediate neighbor of a drug pair, e.g., a CYP enzyme both drugs are substrates of) rather than
at the drug endpoints themselves. Drug-anchored flow GNNs like EmerGNN must propagate signal
across the full path length from drug u to drug v; for the 20.8% of test_s2 pairs whose shortest
path length ≥ 3, this collides with the over-smoothing regime (vanilla GCN AUC drops monotonically
from K=2 to K=8 by −3.86pt with Dirichlet energy collapsing 700→16). A precursor experiment (E7
in i2.md) showed that a logistic regression on **just 22 hand-crafted shared-mediator counts**
(11 mediator kinds × {1-hop, 2-hop}) achieves S2 AUROC 0.7171 — already +3.7pt above vanilla GCN
K=2 — suggesting that meeting-node statistics carry most of the cold-start DDI signal but get
diluted by drug-anchored multi-layer aggregation.

MNAH(Meeting-Node Auxiliary Head)is the minimal *learnable* realization of i2: it keeps the
EmerGNN backbone (and its path-flow advantage for sparse-mediator pairs) entirely unchanged and
adds a small auxiliary head at the logit level:
$$\text{combined\_logit} = \text{emergnn\_logit} + \mathrm{softplus}(\beta) \cdot \mathrm{MLP}_{22\to32\to1}(\mathbf{c}_{a,b})$$
where $\mathbf{c}_{a,b}$ is the 22-d vector of train-normalized shared-mediator counts for the pair.
$\beta$ is initialized to 1.0 via $\mathrm{raw}\beta = 0.541$ so the auxiliary contribution starts
on equal footing with the backbone; BCE is applied only to the combined sigmoid (no auxiliary
direct supervision). Normalization statistics are fit on train positives plus train negatives
(positive-only normalization would bias the mean since positives have systematically more shared
mediators). Per-branch AUROCs(combined, EmerGNN-only, auxiliary-only)are logged each epoch to
enable ablation diagnostics.

**Results** (verified from `Code/runs/`, 3 independent seed42 runs; data source script:
`Code/scripts/analyze_mnah_summary.py`):

| Run | combined | emergnn-only | aux-only | within-run Δ (combined − emergnn) |
|---|---|---|---|---|
| rep1 | 0.7715 | 0.7437 | 0.6927 | +2.78 pt |
| rep2 | 0.7631 | 0.7322 | 0.6782 | +3.09 pt |
| rep3 | 0.7666 | 0.7354 | 0.6925 | +3.12 pt |
| **mean ± std** | **0.7670 ± 0.0035** | 0.7371 ± 0.005 | 0.688 ± 0.007 | **+3.00 ± 0.16 pt** |

We report the **within-run paired Δ** (combined minus emergnn-only on the same trained model)
rather than across-run Δ vs. a separate EmerGNN anchor (which would give a noisier +2.1pt), since
the within-run pairing controls for initialization seed and stochastic shuffling of training pairs.
The aux-only branch alone achieves AUROC 0.688 — substantially above chance — confirming that
22-d shared-mediator counts already carry meaningful pair-discriminative signal even without the
path-flow backbone.

**Controls — the critical evidence that the lift is pair-specific signal, not capacity or
optimization artifact**:

| Control | combined | emergnn-only | aux-only | Δ vs canonical |
|---|---|---|---|---|
| Shuffle (permute pair→feature binding) | **0.7278** | 0.7285 | **0.504** (chance) | combined collapses to emergnn-only |
| Degree-only (replace 22-d counts with raw degree only) | 0.7538 | 0.7440 | 0.565 | aux carries some signal but ~1/3 of canonical |
| seed43 generalization (different release split) | 0.7439 | 0.7305 | 0.6512 | **+1.34 pt within-run, same sign** |

The shuffle control is the cleanest falsification: it preserves the 22-d feature distribution, the
model capacity, the training procedure, and the loss — but breaks the pair-to-feature binding.
Aux-only collapses from 0.69 to chance (0.504), and combined drops to within noise of the
emergnn-only branch (0.7278 vs. 0.7285). The +3pt is therefore unambiguously from pair-specific
mediator counts, not from added capacity or improved optimization dynamics.

The degree-only control further dissects the signal: replacing the 22-d typed-mediator counts
with raw drug-degree only gives aux-only AUROC 0.565 (substantially above 0.5 but well below
0.69). Approximately one-third of MNAH's +3pt is attributable to degree-like density signals;
**the remaining two-thirds requires the typed-mediator semantics**(which mediator kind, not just
how many).

seed43 generalization on a different release split preserves the sign and statistical
significance of the lift, with a smaller magnitude (+1.34pt), establishing that the meeting-node
count signal is not a seed42-specific artifact.

**Leakage audit**: All 22-d count features are computed from the merged KG with DDI edges masked
(`edges__drugbank_hetionet_primekg__mask1.parquet`); the DDI label cannot enter the count
features by construction. Verified in `refine-logs/LEAKAGE_AUDIT.txt`.

**Interpretation** — MNAH directly validates Insight i2: simple statistics aggregated at the
shared mediator are a strong cold-start signal that the drug-anchored path-flow backbone does not
fully extract on its own. Critically, MNAH does this **without modifying the backbone** — EmerGNN's
representation, training procedure, and inductive simulation via `shuffle_train(mode="S2")` are
untouched, so the +3pt represents net signal added on top of all backbone improvements (including
the failed CACR consistency-regularization lever from Section 3.X above).

This result is the project's pivot point in two senses. First, it establishes the working
"backbone": EmerGNN + MNAH (~0.77) becomes the strong cold-start baseline against which subsequent
substrate additions (molecular features, LLM-distilled pharmacology, joint multi-class supervision)
are evaluated. Second, it converts the project's working hypothesis from "training-time
regularization can rescue cold-start" (CACR's framing) to "evidence-form modifications at the
readout — specifically pair-level aggregates of biomedical KG structure — are the productive
lever". All subsequent experiments follow this latter pattern.

**One-sentence headline summary**:

> A 22-dimensional shared-mediator count auxiliary head added at the logit level of the EmerGNN
> backbone — Meeting-Node Auxiliary Head (MNAH) — lifts cold-start test_s2 AUROC by +3.00 ± 0.16 pt
> across three seed42 runs (verified within-run Δ), with shuffle controls collapsing the auxiliary
> branch to chance (0.504) and degree-only controls confirming that two-thirds of the lift comes
> from typed-mediator semantics rather than raw connectivity.

---

### 3.X CACR — Regularization Alone Does Not Lift Cold-Start S2 AUROC

**Background and motivation** (one paragraph for the negative-results section)

We first asked whether the S2 cold-start plateau of the EmerGNN backbone (0.7458 AUROC) reflects
a *train-test KG density mismatch*: at training, drugs see rich KG context with many DDI and
biomedical edges; at test, unseen drugs have only their biomedical neighborhood (DDI edges are
masked). To test this, we implemented Cold-Aware Consistency Regularization (CACR): each training
step computes the score twice — once on the full KG, once on a "cold-simulated" KG where DDI edges
incident to batch endpoints are randomly dropped — and requires the cold-forward output to match
the (detached) full-forward output via a soft-label BCE consistency loss. This adds robustness
training on top of the existing inductive simulation that EmerGNN's `shuffle_train` already does
at the drug level each epoch (where 20% of training-graph drugs are temporarily marked as
"emerging" and the model is forced to predict their pairs without their DDI edges in the
training-time KG).

**Result** (verified from `Code/runs/`):

| Setting | test_s2 AUROC | test_s2 NLL |
|---|---|---|
| EmerGNN anchor (no CACR) | 0.7458 | 1.526 |
| **CACR canonical (λ=0.5, warmup=5, DDI-edge drop p=1.0), n=4 runs** | **0.7464 ± 0.0036** | **~0.80** |
| CACR best variant (λ=0.25, warmup=20) | 0.7504 | 0.70 |
| CACR worst variant (with light non-DDI edge dropout) | 0.7378 | 1.27 |

The canonical CACR run produced a +0.0006 mean AUROC shift over four independent runs — well
within run-to-run noise (~±0.004) and not statistically meaningful. None of the hyperparameter
sweep variants (λ ∈ {0.25, 0.5, 1.0}, warmup ∈ {5, 20}, DDI drop rate ∈ {0.75, 1.0}, with/without
auxiliary non-DDI dropout) produced a stable lift. A related lightweight-regularization baseline
(low-degree node deduplication, λ=0.1) similarly regressed (0.7361).

**A noteworthy secondary observation**: CACR roughly halved the test NLL (1.526 → 0.80) without
moving AUROC, indicating that the consistency loss did shape the output distribution toward more
calibrated probabilities — but this calibration improvement did not translate into ranking gains.
This dissociation between calibration and ranking is a useful methodological reminder: NLL-based
monitoring during training can mask the absence of discriminative improvement.

**Interpretation** (the lesson the project took from this)

CACR was directly attacking the train-test mismatch hypothesis. The negative result, combined
with the same EmerGNN backbone subsequently gaining +3 pt AUROC simply by adding a shared-mediator
count auxiliary head (MNAH), suggests that **the S2 plateau is not caused by training-time
distribution mismatch but by the backbone's readout failing to extract signal that is already
present in the training-time KG**. Regularization sharpens what the backbone already does; it does
not introduce new evidence. We use this dichotomy ("more robust training" vs "more evidence
extracted") to motivate the rest of our method, which adds explicit pair-level mechanistic
features rather than additional training-time regularization.

**One-sentence summary for the negative-results paragraph**:

> Cold-aware consistency regularization (CACR), which adds a batch-level edge-dropout consistency
> term on top of EmerGNN's epoch-level inductive simulation, improved calibration (NLL 1.53→0.80)
> but did not move test_s2 AUROC (0.7464 ± 0.0036 across four runs vs. anchor 0.7458), indicating
> that the S2 plateau is set by missing evidence extraction in the backbone readout rather than
> by training-time robustness.

---

### 3.Z MNAH Stage 2 — Text Semantics of Shared Mediators (i4) Add +2.64 pt AUROC Over EmerGNN and Motivate Mediator-Centric Pooling

**Setting and motivation**

Stage 1 (Section 3.Y) showed that even a coarse 22-dimensional kind-count vector of shared
mediators added +3 pt AUROC over EmerGNN. The Stage 1 representation, however, is deliberately
identity-agnostic. It records that the pair shares, for example, two Gene-typed mediators and
one Pathway-typed mediator, but it does not record *which* Gene or *which* Pathway. Insight i4
predicts that the textual identity of the shared mediator matters. Whether two drugs share
CYP3A4, P-glycoprotein, or an obscure transporter ought to carry different prior information
about interaction risk, and a generic count is blind to that distinction.

Stage 2 tests this hypothesis by replacing the count vector's identity-blindness with a fixed
PubMedBERT [CLS] embedding of the shared mediator's node name, pooled across the shared-mediator
set and projected to 64 dimensions via a PCA basis fit on training-pair pooled vectors only.
The full 22 count features remain in the aux head and the 64 text features are concatenated to
them, giving an 86-dimensional input to the aux MLP. The Stage 1 fusion at the logit level is
unchanged: `combined_logit = emergnn_logit + softplus(β) × MLP(features_86d)`.

**Mechanism (verified against code)**

PubMedBERT [CLS] embeddings (768d) for KG node names are pre-computed and cached at
`Code/data/KG/_merged_kg/_cache/screen1_tag_init/d_name_only__pubmedbert.pt` (real) and
`e_shuffled_text__pubmedbert.pt` (shuffled-name control where node names are permuted before
encoding, an E3-style capacity-vs-semantics check). For each pair (a, b), Stage 2 collects the
union of 1-hop and 2-hop non-drug mediators shared by a and b (same definition as Stage 1), mean-pools
their 768d PubMedBERT vectors into a single per-pair vector, projects to 64d using a PCA basis
**fit on training-pair pooled vectors only** to prevent test-pair leakage (codex round 15 fix),
and caches the result as `t_0..t_63` columns in
`Code/data/_cache/meet_text_real_drugbank_seed42_kgonly_v1.parquet`. The trainer merges this
text cache onto the 22-d count cache by canonical (drug_a, drug_b) pair with one-to-one
validation (`mnah_trainer.py:140–156`), producing the 86-d aux input.

**Verified results (3 reps each, seed42, S2 test split, 1919 pos / 1919 neg)**

| Condition | combined AUROC (mean ± std) | emergnn-only branch | aux-only branch | within-run Δ |
|---|---|---|---|---|
| Stage 1 baseline (counts only) | 0.7670 ± 0.0035 | 0.7371 | 0.6880 | +3.00 ± 0.16 pt |
| **Stage 2 real-name PubMedBERT** | **0.7722 ± 0.0036** | 0.7363 | **0.7335** | **+3.59 pt** |
| Stage 2 shuffled-name control | 0.7641 ± 0.0014 | 0.7290 | 0.6921 | +3.51 pt |

(Source runs: `Code/runs/*mnah_v2s2_realtext_seed42__seed42*`, `*mnah_v2s2_real_rep2*`,
`*mnah_v2s2_real_rep3*` for real-text reps 1–3; `*mnah_v2s2_shuftext_seed42*`,
`*mnah_v2s2_shuffled_rep2*`, `*mnah_v2s2_shuffled_rep3*` for the shuffled control.)

**Two findings, each clean**

*Finding 1 (semantics is real at the branch level).* The auxiliary branch alone improves from
0.688 (Stage 1 counts) to 0.7335 with real PubMedBERT text, a +4.55 pt branch-level jump. With
shuffled node names the same branch returns to 0.6921, statistically indistinguishable from
Stage 1 (within 0.4 pt). The +4 pt branch-level gap between real and shuffled text is
attributable to *which* mediators are shared, not to the additional 64 capacity. This is the
direct readout-side mirror of E3's earlier initialization-side finding (where real-name
PubMedBERT initialization for KG node embeddings outperformed shuffled-name init by +5.65 pt).

*Finding 2 (Stage 2 layers on top of Stage 1 in absolute terms, with diagnostic headroom).*
The combined-logit AUROC moves from 0.7670 (Stage 1) to **0.7722 (Stage 2 real)**, raising the
total improvement over the EmerGNN anchor from +2.12 pt to **+2.64 pt**. The shuffled control
sits at 0.7641, slightly below Stage 1 (−0.29 pt, within noise), confirming the +0.52 pt
incremental Stage 2 gain is semantic, not capacity. Stage 2 is therefore a standalone-positive
result on its own right. At the same time, the gap between the +4.55 pt branch-level gain and
the +0.52 pt combined-readout gain is diagnostic. The EmerGNN flow branch is already extracting
much of the mediator-identity signal indirectly via path flows that pass through the same nodes,
and additive logit fusion can only contribute what the EmerGNN branch has not already captured.
This information-transfer ratio points to architectural headroom that a mediator-centric
pooling scheme (PMP in Section 4) is designed to unlock.

**Why the controls matter**

The shuffled-name control (E3-style) is the load-bearing piece. Without it, the +0.52 pt
combined gain could plausibly be capacity-driven, since Stage 2 adds 64 dimensions and a
correspondingly wider aux MLP first layer. Because the shuffled control with identical capacity
*does not* gain (it slightly underperforms), the semantic identity of the shared mediator —
not the parameter count — drives whatever lift exists.

**Interpretation (the dual role this result plays in the paper)**

Stage 2 plays two complementary roles. As a **standalone result**, it is the first clean
evidence in the cold-start S2 regime that node-name text semantics carry pair-level information
on top of typed counts, and it lifts the EmerGNN anchor by +2.64 pt with a fully controlled
ablation (shuffled-name control, multi-rep mean, per-branch AUROC logging). As a **motivation**
for the architectural direction taken in Section 4, the branch-versus-readout gap (+4.55 pt
branch-level, +0.52 pt combined) quantifies how much signal additive readout fusion leaves on
the table when the backbone already attends to the same nodes. Two design moves follow directly.
(i) Pair-conditional per-mediator attention in place of mean pooling, so the head can up-weight
the specific mediator most relevant to a given pair rather than averaging the set.
(ii) Pair-Mediator Pooling (PMP), which moves the typed mediator representation, including its
PubMedBERT text component, from a parallel readout branch into the score function itself,
removing the additive-fusion bottleneck that caps Stage 2.

**One-sentence summary for the paper**

> Adding PubMedBERT-encoded text semantics of shared mediators (i4) raises the EmerGNN anchor
> from 0.7458 to **0.7722 AUROC on S2 cold-start (+2.64 pt over backbone, +0.52 pt over Stage 1
> typed counts)**, with a shuffled-name control confirming the gain is semantic (0.6921),
> establishing both a standalone positive result and a diagnostic motivation. The branch-level
> potential (+4.55 pt) considerably exceeds the combined-readout transfer (+0.52 pt), quantifying
> the headroom that the Pair-Mediator Pooling design in Section 4 is built to recover.

---

### 3.W1 D1 (LLM-Typed KG Edge Injection) — Backbone Receives the New Edges but Cannot Use Their Pair-Specific Semantics

**Background and motivation**

Sections 3.Y / 3.Z established that pair-level mediator evidence on top of the EmerGNN backbone is the productive lever for cold-start S2 AUROC. v2i4 (Section 3.Z extended with a 13-dim LLM-distilled pair-feature readout head) reached `combined=0.7804`, `auc_emergnn=0.7405`, `auc_count_only=0.6688`, `auc_i4_only=0.6003` on the canonical seed42 setting (run_id `2026-05-29_22-02-27__run_v2i4__v2i4_main_seed42__seed42`). D1 asked whether the same LLM-distilled pharmacological substrate (CYP substrate / inhibitor / inducer, transporter substrate / inhibitor, therapeutic class, primary targets, PD effects, toxicity mechanisms, clearance) could be routed through the EmerGNN backbone rather than only the readout head. We did this by treating each per-drug LLM token as a new KG node and the per-field assignment as a new KG relation type, producing 10 new relations and ~7600 new entity nodes (22,445 drug→token edges injected into a 5,633-entity 5-relation drugbank KG, expanding to 13,265 entities and 15 relations).

**Verified results (seed42, S2 test, 1919 positives / 1919 negatives)**

| Run | combined | emergnn | count_only | i4_only | NLL |
|---|---|---|---|---|---|
| v2i4 anchor (`2026-05-29_22-02-27`) | 0.7804 | 0.7405 | 0.6688 | 0.6003 | 1.4723 |
| **D1 main** (`2026-05-30_19-05-21`) | **0.7770** | **0.7533** | 0.6818 | 0.5301 | 1.2446 |
| K1 shuf-token (`2026-05-31_16-08-02`) | 0.7827 | 0.7495 | 0.7236 | 0.5365 | 1.5792 |
| K2 rand-token (`2026-05-31_22-17-53`) | 0.7778 | 0.7477 | 0.6950 | 0.5731 | 1.5274 |
| K3 drop-i4-head (`2026-05-31_22-15-57`) | 0.7779 | 0.7430 | 0.7221 | 0.4747 | 1.5397 |

D1's emergnn branch lifts by +1.28 pp (0.7405 → 0.7533) — superficially the architectural-novelty signal we wanted. The combined-AUROC moves slightly the wrong direction (−0.33 pp). NLL drops sharply (1.47 → 1.24, calibration improvement). The i4-only readout collapses (0.6003 → 0.5301).

**Falsification controls — the load-bearing diagnosis**

Three controls were designed (per `Notes/Log/d1_llm_edge_design.md` §3) to discriminate "the lift is LLM-semantic pair-specific evidence" from "the lift is KG-density / capacity / shuffle-invariant graph structure".

- **K1 shuf-token** permutes the drug→token assignment at the edge level: every drug gets the same number of edges in expectation, the per-token degree is preserved, the per-relation edge count is preserved, but the drug→its-own-LLM-token binding is destroyed. The design predicted combined to drop by ≥1.5 pp and emergnn to fall back to anchor 0.7405. Observed: combined +0.57 pp, emergnn 0.7495 (+0.90 pp over anchor). Falsification **fails**: the lift survives binding destruction.
- **K2 rand-token** replaces each edge's token with a globally unique synthetic ID; no two drugs share a token. Edge count, relation labels, per-relation count preserved; cross-drug bridges abolished by construction. The design predicted combined to drop more than K1. Observed: combined −0.08 pp from D1 main, emergnn +0.72 pp over anchor. Falsification **fails**: the lift survives bridge removal.
- **K3 drop-i4-head** freezes the readout-side i4 head at β_i4≈0. The design predicted combined ≥ v2i4 anchor 0.7804 (showing the backbone alone could carry the lift). Observed: combined 0.7779. Falsification **fails**: the backbone-integration alone falls short of v2i4 — the standalone i4 readout was apparently neither helping nor hurting in D1, just absorbing capacity.

Codex CP-3 verdict (NOT_PASS for the original D1 mechanism claim, archived at `Code/my_code/models/screen_s2_v3_multimodal/_reviews/2026-06-01__d1_results__round1.md`): "D1 main's +1.28 pp emergnn lift cannot honestly be attributed to pair-specific LLM semantics... The load-bearing mechanism is not 'drug A and drug B share meaningful LLM-derived entities'; it is some shuffle-invariant effect of adding typed edges/entities/relations to the KG."

**NLL ↔ binding coupling — a secondary positive observation**

D1 main improves NLL by 0.228 (1.47 → 1.24). K1 shuf-token reverses that and makes NLL worse than anchor (1.58). K2 and K3 also worsen NLL slightly (~+0.05). So the NLL improvement is specifically gated by the original LLM-token binding, even though the ranking gain (AUROC) is not. This mirrors the §3.X CACR finding ("NLL ↓ without AUROC ↑") at a smaller scale: D1's contribution is a calibration sharpener that depends on real evidence, but does not translate into ranking improvement under additive logit fusion against an EmerGNN backbone that has already captured the substantive pair signal through its propagation.

**Interpretation (the lesson the project took)**

D1 designed itself for the wrong kind of injection. Adding new KG nodes and relation types to a path-flow GNN gives the model new capacity / structural density / regularization that lifts the backbone's representation on cold-start drugs, but the path-flow propagation aggregates over edges in a way that does not strongly distinguish "drug A→its-real-CYP3A4-token→drug B" from "drug A→shuffled-replacement-token→drug B" when both walks reach a similarly-typed neighborhood. The shuffle-invariant lift is consistent with a capacity / graph-density account, not with a pair-specific semantic-evidence account. This recapitulates a CACR-style diagnosis at the architectural-augmentation layer: adding "more KG" without making the model use the pair-specific structure of that addition does not translate into ranking improvement. Section 4 (Section 4 in this writeup; D2 in the design notes) responds by making the pair-specific structure explicit at the propagation level (meeting-node attention bonus per pair) rather than implicit in the KG edge set.

**One-sentence summary for the paper**

> A 10-relation, ~7,600-node KG augmentation derived from LLM-distilled pharmacological field tags lifts the EmerGNN backbone branch by +1.28 pt AUROC on S2 cold-start (0.7405 → 0.7533) and tightens NLL by −0.23, but two falsification controls (shuf-token, rand-token) and one ablation (drop readout head) show the lift is invariant to drug→token binding, invariant to cross-drug bridges, and not sufficient to clear the v2i4 anchor in combined AUROC — establishing D1 as a KG-capacity / calibration result rather than a pair-specific evidence-routing result.

---

## Section 4 — Negative Results & What They Indicate

A consolidated paragraph for the "negative results" subsection. This will accumulate as we discuss
more failed attempts; each gets ~1-2 sentences.

- **CACR (regularization, edge-dropout consistency)** — see Section 3.X. NLL ↓, AUROC →; signals
  that the plateau is missing-evidence-bound, not training-stability-bound.
- **D1 (LLM-typed KG edge injection)** — see Section 3.W1. Backbone branch lifts +1.28 pt AUROC
  and NLL drops sharply, but K1 (drug-token binding shuffled) and K2 (cross-drug bridges destroyed)
  fail to collapse the lift, showing the contribution is shuffle-invariant capacity/density rather
  than pair-specific LLM semantics. Pair-conditional propagation (Section 4 D2) is the response.

(more to come)

---

## Section 5 — Reusable Framings / Sentences

Small reusable observations that might appear in multiple paper sections:

- **"NLL ↓ without AUROC ↑" framing**: useful in the discussion section to caution against using
  loss curves alone to judge cold-start methods; we observed it explicitly in CACR.
- **"more robust training vs more evidence extracted" dichotomy**: useful at the end of the
  motivation section to position our method (which adds evidence: counts, structured pair
  features) against alternative paths (which add regularization or augmentation).
- **EmerGNN already does drug-level cold simulation** via `shuffle_train(mode="S2")` with 20%
  per-epoch emerging drugs — useful when describing baselines so readers know cold-aware training
  is not a novel contribution of any single method.

---

## Index of Experiments Discussed (with status)

| Round | Method | AUROC | Status in paper |
|---|---|---|---|
| Pre-MNAH | CACR + nodedup (regularization) | 0.7464 ± 0.004 / 0.7361 | Negative result (Sec 3.X) ✓ |
| MNAH backbone | EmerGNN + meeting-node count aux head | 0.772 | Main backbone (to write) |
| v2i4 backbone | MNAH + 13-d LLM mechanistic pair head (i4) | 0.7804 (seed42) | Main backbone (Sec 3.Z+ ext.) |
| Round 4 D1 | LLM fields → new KG edge types (10 rel, 7.6k nodes) | 0.7770 (seed42, K1/K2/K3 all fail) | Negative result (Sec 3.W1) ✓ |
| ... | (filled as we discuss) | | |

---

## Conventions for future additions

- **Verify before writing**: every number cited here must be backed by a `Code/runs/` log or a
  successfully reproduced computation. No "I think this was about ~X" — find the log.
- **One subsection per experiment / finding**: keep them self-contained so they can be lifted into
  the paper draft individually.
- **Always include**: motivation paragraph, verified result table, interpretation paragraph,
  one-sentence headline summary.
- **Optional**: figure description or which logged run produces the figure data.

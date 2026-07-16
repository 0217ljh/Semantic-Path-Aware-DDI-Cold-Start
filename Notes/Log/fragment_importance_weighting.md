# Fragment Importance Weighting — EDT-Former Inspired

**Status**: not yet implemented; concrete plan inspired by EDT-Former (Jing et al., ICLR 2026)
**Logged**: 2026-05-30
**Cache**: `Code/data/_cache/molecular_fragments_brics.npz` (1994 drugs × 768-d uniform BRICS counts)
**Used by**: E-frag (`v2res_trainer.py::ResHead`); mol_head_alone failed to lift (0.577)
**Key paper**: Jing et al., "Entropy-Guided Dynamic Tokens for Graph–LLM Alignment in Molecular Understanding", ICLR 2026

---

## TL;DR

Our current BRICS fragment cache treats every fragment uniformly. EDT-Former (ICLR 2026) shows that
**entropy-guided segmentation** of SMILES via a lightweight Next-Atom Predictor (NAP) outperforms
BRICS-based patching by ~4-7pt on molecular tasks (BBBP, PAMPA). Two adaptation paths to our project:

- **Path A**: replace BRICS with entropy-driven segmentation (their direct approach)
- **Path B**: keep BRICS but add entropy-derived per-fragment importance weights (chemistry-preserving)

Both are cheap (NAP is 0.54M params, trains in minutes). Either path unifies cleanly with the
pair-conditioned attention design in `kg_typed_aggregation.md`: entropy gives a static per-drug
importance prior, pair-attention refines per-pair.

---

## Current state — what's actually been done

**Cache**: 1994 drugs × 768 uniform BRICS fragment counts, top-768 vocab by document frequency.

**Tested once in E-frag (2026-05-26)**:
- v2res_trainer fed it through proj_shared + proj_resid + InfoNCE alignment to typed-KG
- combined 0.7546, mol_head_alone **0.577** (close to chance)
- **NOT cleanly attributable to BRICS**: simultaneously changed source (Morgan→BRICS), target
  (k_u→k_typed), added residual branch, and switched to joint training

**Pre-session bag-of-motif test** (per handoff):
- BRICS-motif single logistic head → S2 AUC **0.5565**
- MNAH + motif oracle ensemble gain = **0.000**
- Interpretation at the time: "cheap molecular redundant with KG"

**What was NEVER tested**:
- BRICS as additive feature on top of MNAH (no shared/residual scaffolding) — i.e. is BRICS itself
  useful when fused additively to the strong KG+count baseline?
- Any non-uniform fragment weighting
- Any pair-conditioned aggregation of fragments
- Any entropy / saliency-based fragment importance

---

## EDT-Former: what it actually does

### Core mechanism (Sec 3.2 of paper)

For a SMILES string `(a_1, ..., a_T)`:

1. **NAP**: pre-train a lightweight Transformer (paper uses 2 layers, 2 heads, hidden 128, vocab 39,
   = 0.54M params) on canonical SMILES corpus (e.g. PubChem) to model `p(a_{t+1} | a_{1:t})`
2. **Surprisal**: `e_t = -log p(a_{t+1} | a_{1:t})` per position
3. **Peak detection**: local maxima of `e_t` + NMS (window Δ) + prominence threshold (γ) → split points
4. **Segmentation**: cut SMILES after retained peaks → variable-length, information-dense segments
5. **Graph mapping**: map SMILES positions to graph node indices via π; pool node embeddings within
   each segment via average → dynamic substructure token `z_k`

### Direct comparison to BRICS (Table 8 of paper)

The paper **explicitly compares** with BRICS (Jinsong et al., 2024 — the same paper our
`precompute_fragments.py` cites). Same MoleculeQA / PAMPA / BBBP benchmarks, matched token
budgets, prompts, query length:

| Method | BBBP Acc | PAMPA Acc | Avg drop vs Entropy |
|---|---|---|---|
| Entropy-Guided | **75.06** | **84.52** | 0% |
| BRICS | 73.59 | 71.90 | **-3.96%** |
| Random | 68.90 | 66.67 | -8.81% |
| None | 39.67 | 78.62 | -21.60% |

→ **Entropy beats BRICS by ~4-7 pt** on these benchmarks.

### Why entropy works (the theoretical claim, paper App A.1)

- Lemma 1: surprisal peaks coincide with **change-points** in the SMILES generative distribution
- Lemma 2: pooling loss within a segment is upper-bounded by within-segment entropy
- Prop 1: peak-cutting minimizes a budgeted upper bound on representation loss

Empirically (Table 37): entropy vs BRICS NMI ≈ 0.48 — meaningful overlap with chemistry-defined
fragmentation **but not identical**, suggesting entropy captures things BRICS misses.

### NAP is cheap (this matters for us)

- 0.54M parameters (`vocab_size=39`, 2 layers, hidden 128, 512 context)
- PubChem pre-training: 1 epoch, 4929 steps, ~minutes on a single RTX 3090, **0.07 GPU-hours**
- Trainable on **multi-core CPU within an hour** (no GPU needed)
- NAP size doesn't matter much (Table 38): 0.5M vs 50M vs 500M NAPs produce nearly identical
  segmentations (pairwise NMI > 0.85). So our 800-drug DrugBank subset is fine for a small NAP.

---

## Two adaptation paths for our DDI project

### Path A — Replace BRICS with entropy patching

Follow EDT-Former's direct recipe:

1. Train a NAP on DrugBank SMILES (1994 drugs, small dataset but NAP is also small — sufficient
   per Table 38). Use the 39-token vocab from the paper or extend with project-specific tokens.
2. Compute per-atom surprisal for each drug.
3. Cut SMILES at entropy peaks with NMS + prominence thresholds.
4. Map segments → graph nodes → per-segment representation (either avg-pool of atom-level
   ChemBERTa/Morgan, or just count vector over an entropy-derived vocab).
5. Output: new cache `molecular_fragments_entropy.npz` replacing `molecular_fragments_brics.npz`.

**Pros**:
- Direct application of validated paper (4-7pt above BRICS in their tasks)
- Theoretical backing (Lemma 1+2)
- Chemistry-aware (NMI ~0.48 with BRICS confirms it's not random)

**Cons**:
- Loses the chemical interpretability of BRICS fragments (entropy segments don't map to
  rule-based functional groups)
- Their tasks are per-molecule prediction; we have a pair-prediction (DDI) task — adaptation isn't
  guaranteed to transfer
- Doesn't solve our deeper pair-conditioning problem (see `kg_typed_aggregation.md`)

### Path B — Keep BRICS, add entropy importance weights (chemistry-preserving)

Hybrid: don't throw away the chemical regularity of BRICS, but add saliency:

1. Same NAP training.
2. For each BRICS fragment in a drug, **aggregate the surprisal** of its constituent atoms
   (mean / max / weighted by atomic-degree).
3. This gives a **per-fragment importance score** `imp(fragment, drug)`.
4. Augment the cache: store both `count[i]` (current) and `entropy_weight[i]` (new).
5. Downstream model can use:
   - `weighted_count[i] = count[i] × entropy_weight[i]` (simple)
   - or `concat([count, entropy_weight], dim=-1)` (let model decide)

**Pros**:
- Preserves BRICS's chemical interpretability → "the model focuses on **mechanistically meaningful
  fragments that are also information-dense**" is a stronger story than either alone
- Combines rule-based (BRICS) + data-driven (entropy) — orthogonal information sources
- Backward-compatible: cache structure extends, doesn't replace; old experiments still reproducible

**Cons**:
- Two-step pipeline (BRICS decomposition + NAP) — slightly heavier preprocessing
- The aggregation function (atom surprisal → fragment importance) is itself a design choice that
  needs ablation (mean vs max vs weighted)

---

## Concrete cache changes (per `kg_typed_aggregation.md` convention)

If we go Path A:
- New cache: `Code/data/_cache/molecular_fragments_entropy.npz` (1994 × variable-len-segments)
- Need to define segment representation (avg-pool of atom features? per-segment SMILES string with
  a learned encoder? — paper uses graph-node avg-pool, which we don't have a frozen graph encoder
  for in our pipeline)

If we go Path B:
- Extended cache: `Code/data/_cache/molecular_fragments_brics_v2.npz`
  - `drug_ids` (unchanged)
  - `frag_count` (current 1994 × 768)
  - `frag_entropy_weight` (new, 1994 × 768) — per-fragment mean surprisal of its atoms
  - `vocab` (unchanged)
  - `atom_surprisal` (optional debug, ragged per drug)

Path B is what I'd recommend for the **first attempt** since it's additive and reversible.

---

## Connection to other follow-up logs

### Relation to `kg_typed_aggregation.md` (pair-conditioned attention)

EDT-Former's entropy is **static per-drug**: drug A's fragment entropy weights don't change with
partner drug B. So entropy alone doesn't solve the pair-conditioning gap.

**But entropy weights compose cleanly with pair-attention**:
```
final_weight(i, drug_a, drug_b) =
    entropy_prior(i, drug_a)       # static, from EDT-Former-style NAP
  × pair_attention(i, drug_a, drug_b)  # learned, from architecture in kg_typed_aggregation.md
```

This is BLIP-2 / ALBEF style: entropy gives the *prior*, attention refines per-pair. Best of both.

### Relation to `kg_molecular_redundancy.md` (redundancy thesis)

The paper provides evidence that **substructure-level molecular signal IS useful** for per-molecule
prediction tasks (their MoleculeQA, PAMPA, BBBP, etc.). However:

- Their tasks are **per-molecule** prediction
- Our cold-start S2 DDI is **pair-prediction with KG biomedical neighborhoods retained**

So entropy-weighted fragments still need to escape the KG-saturation thesis: if MNAH path-flow
already encodes "what kind of drug this is" via its biomedical neighborhood, even better-weighted
molecular fragments may stay redundant.

**This becomes a clean test**: if entropy-weighted fragments **still** fail to lift on top of MNAH,
that's strong evidence for the redundancy thesis (since EDT-Former's mechanism is the strongest
substructure-saliency lever in the 2026 literature). If they **do** lift, it's evidence that
substructure-saliency information was missing all along.

### Relation to `molecular_alignment_design.md` (cross-attention architecture)

EDT-Former's Dynamic Query Transformer (their Sec 3.3) is essentially what `molecular_alignment_design.md`
prescribes:
- Anchors + dynamic tokens → self-attention + cross-attention over graph nodes → FFN → frozen-LLM
- Layered architecture (their L=8 layers)
- This **is** the ALBEF-style fuse-before-contrast pattern, just realized for molecular graphs

So if we ever build a cross-attention molecular-KG alignment (`molecular_alignment_design.md` Path D),
**EDT-Former's architecture is a direct template**:
- Their "anchors" ↔ our learnable modality anchors
- Their "dynamic tokens from entropy patches" ↔ our per-substructure tokens (BRICS or entropy)
- Their "cross-attn over graph node embeddings" ↔ our cross-attn between molecular substructures
  and KG-typed neighbor sequences

---

## Recommended experiment ordering (when this gets queued)

| Order | Experiment | Cost | Expected outcome |
|---|---|---|---|
| 1 | Train NAP on DrugBank SMILES (1994 drugs) | <1 GPU-hour | Baseline NAP, save model |
| 2 | Compute per-atom surprisal for all 1994 drugs | <30 min | Surprisal cache |
| 3 | **Path B**: compute per-fragment entropy weights, extend cache to `molecular_fragments_brics_v2.npz` | <1 hr | Augmented cache, backward-compatible |
| 4 | Re-run E-frag with entropy-weighted vs uniform BRICS, isolated A/B | ~70 min (2 runs) | Test if entropy weighting alone lifts on top of MNAH |
| 5 | If Path B is positive, try Path A (entropy patching instead of BRICS) | ~2 hr | Direct paper replication on our dataset |
| 6 | Combine with pair-attention from `kg_typed_aggregation.md` (entropy × pair-attention) | ~1 day | Unified mechanism, story-worthy |

---

## Critical caveat — why I'd test before trusting

EDT-Former's tasks are **per-molecule** (MoleculeQA, BBBP, PAMPA — predict a property of *one*
molecule). Our task is **pair-prediction with strong backbone** (MNAH 0.772 already extracts most
single-drug signal via path-flow). The 4-7pt advantage of entropy over BRICS in their benchmarks
**may not transfer** to our cold-start DDI setting.

The redundancy thesis (`kg_molecular_redundancy.md`) predicts that EVEN ENTROPY-WEIGHTED fragments
will fail to lift on top of MNAH, because MNAH already encodes the biomedical neighborhoods that
correlate with what entropy is picking up. The cleanest test is the experiment chain above.

**If entropy-weighted BRICS fails to lift, that's not a wasted experiment** — it's the most rigorous
test of the redundancy thesis to date, using the strongest substructure-saliency mechanism in
the 2026 literature.

---

## File ref

EDT-Former paper PDF: `C:\Users\27911\Downloads\15761_Entropy_Guided_Dynamic_T.pdf` (45 pages,
ICLR 2026)

Key sections to revisit when implementing:
- Sec 3.2 Entropy-Guided Sub-Graph Patching (algorithm)
- App C.1 (NAP config: 2-layer GPT-2, vocab 39, 0.54M params, PubChem pretrain settings)
- App D.9 (BRICS vs Entropy NMI analysis, Table 37, 38)
- App A.1 (theoretical analysis: surprisal peaks ↔ change points)
- App E.5 (anchor + dynamic token budget allocation — relevant for Path A's downstream usage)

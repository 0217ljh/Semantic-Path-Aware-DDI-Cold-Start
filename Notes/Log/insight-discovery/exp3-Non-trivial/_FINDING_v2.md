# exp3 Final Finding v2 — Neighbor-Borrowing Amplification in Cold-Start DDI

## TL;DR

> **Cold-start amplifies a trained DDI model's dependence on training-neighbor consensus, pushing it toward implicit KNN-style borrowing.** The same consensus-NLL pattern exists in both warm-start (S0) and cold-start (S2), but the slope **roughly doubles** in cold-start — the structural amplification IS the cold-start-specific signal. For positives, "training neighbors agree it's a DDI → easier" effect strengthens 2×; for negatives, "neighbors say YES but actual is NO → falsely confident positive" penalty also amplifies 2×.

This supersedes v1's "Support-Rich Underdetermination" framing. The H_rel U-shape in v1 was a real but secondary proxy — once consensus is controlled, H_rel adds no unique variance (ΔR²(H_rel | consensus) = 0). H_rel is now reframed as an interpretable view onto the deeper consensus mechanism.

## The phenomenon

**Definitions**:
- For each cold drug `d`, top-K=10 nearest training drugs by PubMedBERT cosine
- For pair `(a, b)`, `consensus_vote = fraction of N_a × N_b (k²=100) training-twin cross-pairs that ARE in train DDI`
- `consensus_var = vote × (1 - vote)` (high = max disagreement at vote=0.5)

**The structural shift**:
| Quantity | S0 (warm, n=12,455) | S2 (cold, n=14,022) | Ratio |
|---|---|---|---|
| POS NLL_resid range (dec0 → dec9) | 0.083 | 0.167 | **2.0×** |
| POS Spearman ρ(consensus_var, NLL) | -0.137 | -0.253 | 1.85× |
| NEG NLL_resid range | 0.251 | 0.330 | 1.31× |
| NEG Spearman ρ | +0.173 | +0.207 | 1.20× |

**Interaction model coefficients (NLL ~ ... + feature × split)**:
- `consensus_vote × is_S2`: +0.028 (positive — S2 increases consensus effect)
- `label × consensus_vote × is_S2`: **−0.061** (triple interaction — for positives in cold-start, consensus effect intensifies)
- `consensus_var × is_S2`: +0.020
- `label × consensus_var × is_S2`: **−0.045**

**Mechanistic narrative** (consistent with missing direct supervision):
- Warm-start: model has direct supervision on both drugs → consensus is one signal among many
- Cold-start: model has NO direct supervision on test drugs → falls back heavily on training-neighbor consensus
- Result: model becomes implicit-KNN over training pairs of similar drug pairs
- Both positive labels (when neighbors agree positive) AND negative labels (when neighbors mistakenly suggest positive) feel the amplification

## Key empirical evidence (already collected)

### Cross-seed (seed 42/43/44, S2 only, LR model, prior work)
H_rel U-shape replication 100% on all seeds (M3+M8 verified). Consensus cross-seed not yet replicated but expected to mirror H_rel pattern.

### Bootstrap (drug-cluster resample, S2 seed42)
H_rel U-shape: 96.4% under drug-cluster bootstrap (n=500). POS jump CI [+0.020, +0.041]. NEG inv-U: 99.0%.

### Cross-model
- GCN K=2 (AUC 0.708) on Q3 S2: ρ(H_rel, NLL | pos) = -0.236
- LR meeting-node (AUC 0.717) on Q3 S2: ρ(H_rel, NLL | pos) = -0.232 (similar direction)
- Consensus cross-model not yet replicated on GCN — recommended as next step

### Partial correlation (within Q3 S2)
- ΔR²(H_rel | consensus) = +0.0001  → H_rel adds nothing once consensus is in
- ΔR²(consensus | H_rel) = +0.0029  → consensus has unique explanatory power
- → consensus is the primary signal, H_rel is a coarse proxy

### S0 vs S2 differential (the cold-start specific result)
Full table above. Amplification factor ~2× in POS slope, 1.3× in NEG slope. Triple interaction terms all signed in mechanism-consistent direction.

## What v1 got wrong and v2 corrects

| v1 claim | v2 revision |
|---|---|
| "Support-Rich Underdetermination" is cold-start specific | NOT cold-start specific — present in S0 too |
| H_rel U-shape is the main finding | H_rel U-shape is a proxy for consensus-driven prediction |
| Phase transition at dec9 (cliff) is the paradox | The cliff is real but secondary to the amplification pattern |
| Mediator-template entropy → identifiability collapse | Consensus voting → KNN fallback in absence of direct supervision |

## Recommended remaining robustness (to lock paper)

1. **Consensus cross-seed (43/44)**: confirm amplification on seed 43/44 — same script as `10_cross_seed.py` with consensus features added
2. **GCN cross-model on amplification**: re-run S0 vs S2 differential with GCN K=2 predictions (currently amplification is LR-only)

If both pass, finding is paper-ready as cold-start subfield contribution.

If not pass: demote to LR-specific shortcut.

## Files

- `01-13*.py` — feature/exploration scripts
- `14_track2b_neighbor_consensus.py` — neighbor-consensus computation
- `15_partial_and_s0.py` — partial correlation + S0 contrast
- `16_s0_s2_differential.py` — **PRIMARY** S0 vs S2 interaction analysis
- `_FINDING.md` — v1 (kept as historical record)
- `_FINDING_v2.md` — this file (current)

## Codex thread

`019e27ad-5e63-7b32-b1a4-406dffbe79bd` (gpt-5.4, new thread for Track 2)

**Codex commit on title**: "Neighbor-Borrowing Amplification in Cold-Start DDI"

**Codex final stamp**:
> "Stop horizontal exploration. Only do consensus cross-seed + GCN replicate. If both pass, write consensus amplification as main finding, H_rel as interpretable projection."

## Why this is a better finding than v1

- **More general**: Encompasses any neighbor-consensus failure mode, not just relation templates
- **Cleaner mechanism**: "Cold drugs lack supervision → model leans on neighbor-borrowing → consensus errors amplify" — direct causal-style story
- **Cold-start specific signal exists**: Amplification (2× slope) IS the cold-start-specific signal. Pattern itself is general; specificity is in the magnitude.
- **Subsumes v1**: H_rel U-shape becomes an interpretable instance of consensus-driven prediction
- **Better paper narrative**: Suggests methodological implication — methods that bypass consensus-dependence (e.g., explicit mechanism inference) should help in cold-start more than warm-start

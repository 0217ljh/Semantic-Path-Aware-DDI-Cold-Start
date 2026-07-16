# Screen 5 — PK/PD Subgraph + Virtual Mediator Embedding (VME)

**Status**: Code skeleton DONE. Trainer raises `NotImplementedError` —
production version requires relation-vocab audit between PK/PD subgraphs.

**Backbone**: two parallel EmerGNN_TAG networks (one per subgraph) with a
late merge head Linear(8*n_dim, 1) over both branches' [head_emb,
tail_emb, head_hid, tail_hid].

## Motivation (E1b)

Statistically PK/PD pairs DO use different intermediates:
- PD pairs: PD-layer 55.6%, PK-layer 32.7% (margin 22.9pt)
- PK pairs: PK-layer 52.4%, PD-layer 44.0% (margin 8.4pt)
- χ² = 320.60, p = 2.41e-70

But effect-size asymmetric: PD signal strong, PK signal moderate.
**Screen 5 ⑤a (subgraph split) motivated for PD side; ⑤b (VME) is
exploratory.**

## Variants (per first_step_plan.md §4.7)

| ID | Subgraph | VME | Role |
|---|---|---|---|
| S0 | full KG | — | Anchor (= Screen 1 EmerGNN_TAG with full KG) |
| S1 | PK only | — | Layer ablation |
| S2 | PD only | — | Layer ablation |
| **S3** | **PK ∥ PD** | — | **⑤a primary** |
| **S4** | **PK ∥ PD** | **GPT-4o + PubMedBERT** | **⑤b primary** |
| S4b | PK ∥ PD | Qwen-72B embed-lookup | Generation-vs-lookup control |
| S5 | full KG | GPT-4o | VME effect indep. of split |
| S6 | random 2-split | — | Structural control |
| S7 | PK ∥ PD strict | — | Strict-split variant |
| **S8** | **PK ∥ PD** | **scrambled VME** | **Semantic VME control (critical)** |

## Files

- `__init__.py`
- `subgraph_builder.py` — PK / PD / random / strict-split edge extraction (fixed per Codex #3: classify by non-drug endpoint)
- `vme_gap_finder.py` — find structural gap candidates along shortest paths (Codex #3 flagged "gap definition" issue — kept as path-shortcut for now)
- `vme_generator.py` — GPT-4o generation ($50 hard cap) + Qwen-72B embed-lookup fallback
- `_per_mode_pkpd.py` — dual-flow trainer SKELETON (NotImplementedError until rel-vocab audit done)
- `smoke_vme.py` — small VME generation smoke (~$0.005, sandbox-blocked in autonomous run)

## How to use (when ready)

### Step 1: Generate VMEs (one-time, ~$50 budget, ~30 min)

```bash
python -c "
from my_code.models.screen5_pkpd_subgraph.vme_gap_finder import enumerate_all_gaps
from my_code.models.screen5_pkpd_subgraph.subgraph_builder import split_edges_pk_pd
# ... load drug pairs, split KG, find gaps, generate via GPT-4o
"
```

### Step 2: Launch dual-flow training (when trainer is implemented)

```bash
python -u Code/my_code/models/screen5_pkpd_subgraph/run_screen5.py \
  --variant S3 --init-variant D --projection P1 --epochs 100 --seed 42
```

## Outstanding work

1. **Relation-vocab audit**: when restricting merged KG to PK / PD
   subgraphs, the relation IDs in `EmerGNN.rel_kg` Embedding must map
   correctly. Need to verify the kg_builder/`build_edge_lists_from_triplets`
   logic produces consistent relation indices across PK/PD instances or
   build per-subgraph relation vocabs.

2. **VME gap criterion**: current `vme_gap_finder.find_gaps_for_pairs`
   takes `(path[i-1], path[i+1])` as a gap, which is more "path shortcut"
   than "missing mediator". Reframe before using; or redefine as
   distance-≥-2 pairs along BFS frontier where no observed intermediate
   exists.

3. **Qwen contract**: `vme_generator.generate_vme_via_qwen_embed` returns
   8192d tensor; downstream expects 64d. Add projection step (random or
   PCA, sharing Screen 1's projection module) before consumption.

4. **Drug-drug edge handling**: Hetionet `CrC` (Compound-resembles-Compound)
   survives the `mask1` filter. With `overlap_mode="include_both"`, these
   leak into both PK and PD subgraphs as drug-similarity shortcuts. Consider
   adding `--exclude-CrC` flag or always treating CrC separately.

5. **Codex review #5**: review post-fix subgraph + VME code (after
   relation-vocab audit + VME gap re-definition).

## Codex review history

- **Round 3 (initial)**: NEEDS_FIX
  - CRITICAL #1: PK/PD classify by non-drug endpoint (FIXED)
  - CRITICAL #3: VME gap definition wrong (OUTSTANDING — see #2 above)
- **Round 4 (post-fix Screen 3 / verify Screen 5)**: PASS for Screen 3,
  Screen 5 trainer intentionally skeleton — no verdict; WARNs about CrC
  leakage and Disease-Gene edge classification (OUTSTANDING — see #4 above).

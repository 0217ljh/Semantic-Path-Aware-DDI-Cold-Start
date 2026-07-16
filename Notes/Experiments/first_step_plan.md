# First Step Plan — LLM-Enhanced Flow GNN Screening

**Status**: v1, locked specs
**Date**: 2026-05-20
**Project**: Semantic-Path-Aware-DDI-Cold-Start
**Purpose**: Lock environment + data + code-layout conventions BEFORE running any screening experiment, so we do not drift mid-execution.

---

## 0. Hard rules (read every session)

1. **Env**: WSL2 + conda env `project_1` only. No other env.
2. **GPU**: single RTX 5090, 32 GB VRAM. Plan batch sizes accordingly.
3. **Backbone**: EmerGNN multimode (the official binary baseline as of 2026-05-20) is the fixed comparison anchor. Every screening change is a single-point variation against it.
4. **Code lives ONLY in** `Code/my_code/`. Do not modify `Code/baseline/*` or `Code/my_code/models/GCN/*` (existing baselines).
5. **One screen = one folder**. `Code/my_code/models/screen{N}_{slug}/`. Never mix screens.
6. **Three-seed reporting** (42, 43, 44) for any AUC claim. Single-seed is exploratory only and must be labeled as such.
   - **Current execution scope (2026-05-20)**: ALL screening experiments run **seed 42 + 800-drug only** until further notice. Multi-seed extension and 1900-drug promotion deferred. Adjust compute budgets in §4.5/4.6/4.7 by skipping seeds 43/44 (cut total by ~50-60%).
7. **Comparison anchor numbers** (EmerGNN multimode, binary, seed 42, 800-drug legacy):
   - test_s0 AUC = 0.9895
   - test_s1 AUC = 0.8328
   - test_s2 AUC = 0.7462
   These are seed-42 only. Re-run seeds 43, 44 before relying on them.

---

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows 11 host + WSL2 (Ubuntu) |
| Conda env | `project_1` |
| Activation | `wsl bash -ic "conda activate project_1 && <cmd>"` (Windows shell) |
| GPU | NVIDIA RTX 5090, 32 GB |
| CUDA | whatever `project_1` ships (do not change) |
| Project root (WSL view) | `/mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start` |
| Project root (Windows view) | `D:\My-Research\03-Projects\Semantic-Path-Aware-DDI-Cold-Start` |

---

## 2. Data inventory (USE THESE PATHS — do not invent new ones)

### 2.1 DDI splits

Use **`private/outputs_full/splits_legacy/`** — these are the canonical 3-seed splits.

| Dataset | Path | Seeds available |
|---|---|---|
| 800-drug (legacy P0) | `Code/data/private/outputs_full/splits_legacy/800drug/latest_drugbank_ddi-Binary_cls-{42..46}+cold_start_split_fair_step-and-fair_negatives_step.pkl` | 42, 43, 44, 45, 46 |
| 1900-drug P0 | `Code/data/private/outputs_full/splits_legacy/1900drug/P0/latest_drugbank_ddi-Binary_cls-{42,43,44}+...pkl` | 42, 43, 44 |
| 1900-drug P4 (with 1-hop subgraph sequences pre-extracted) | `Code/data/private/outputs_full/splits_legacy/1900drug/P4/6-latest_drugbank_ddi-Binary_cls-{42,43,44}+...pkl` | 42, 43, 44 |

**Default for screening**: 800-drug seeds {42, 43, 44}. Comparable to the EmerGNN multimode baseline.
**Promotion target after screen passes**: 1900-drug seeds {42, 43, 44}.

### 2.2 Annotations (per-pair labels)

`Code/data/private/outputs_full/annotations/`
- `pkpd.parquet` — PK / PD class per DDI pair (used in screens ② ⑤)
- `ab.parquet` — A/B asymmetry labels
- `action_pairs.parquet` — action-type labels
- `mediating_entities.parquet` — mediator labels

### 2.3 Knowledge graph

`Code/data/KG/_merged_kg/`
- `nodes__drugbank_hetionet_primekg.parquet` — 178,029 nodes, columns `[id, kind, name, source_kg]`
- `edges__drugbank_hetionet_primekg__mask1.parquet` — edge list

Per-source KGs in `Code/data/KG/{drugbank, hetionet, primekg, ddinter, twosides}/` if needed for ablation.

### 2.4 Drug text (key for screen 1)

`Code/data/KG/drug_text/`
- `drug_profiles.json` — structured per-drug profile
- `drug_profiles_flat.csv` — flat one-row-per-drug, has `text` column (already concat of name + type + groups + targets + enzymes). Use this as the TAG text source for drug nodes.

### 2.5 Mol graphs (for Step 2, not now)

`Code/data/KG/mol_graphs/` — reserved.

### 2.6 API credentials

| Service | Path | Notes |
|---|---|---|
| OpenAI (GPT-4o for Screen 5 S4 VME generation, Screen 2 T2 path labeling) | `D:\My-Research\03-Projects\Semantic-Path-Aware-DDI-Cold-Start\API-KEY\API-KEY.txt` | Plain text. Protected by `.gitignore` rule `/API-KEY/**`. Never read into transcripts. Code should load via `Path(...).read_text().strip()` at runtime only. |

**Hard rule**: never print, log, or echo the key. Use it only as `os.environ["OPENAI_API_KEY"]` set inside the script that needs it.

---

## 3. Code layout (under `Code/my_code/`)

```
Code/my_code/
├── data/             # existing — do not modify
├── datasets/         # existing — extend with new loader if needed (new file, not edit)
├── eval/             # existing — extend collapse_metrics.py etc. via new files
├── io/               # existing
├── kg_lib/           # existing — KG loader
├── models/
│   ├── GCN/          # existing baseline — DO NOT MODIFY
│   ├── screen1_tag_init/        # NEW — Screen 1: PubMedBERT TAG init
│   │   ├── __init__.py
│   │   ├── encoder.py           # PubMedBERT wrapper, caches to .pt
│   │   ├── node_text_builder.py # build (node_id → text) from KG + drug_profiles
│   │   ├── init_features.py     # produce {random, type_onehot, node2vec, pubmedbert, pubmedbert_shuffled, typename_only} init tensors
│   │   ├── emergnn_with_init.py # thin subclass of EmerGNN backbone that consumes external init tensor
│   │   └── run_screen1.py       # entry: 6 init variants × 3 seeds × EmerGNN backbone
│   ├── screen2_pkpd_prior/      # NEW — Screen 2 (PK/PD attention prior)  [run later]
│   ├── screen3_meet_in_middle/  # NEW — Screen 3 (junction-node readout)  [run later]
│   ├── screen4_llm_verifier/    # NEW — Screen 4 (LLM decoder verifier)   [run later, conditional]
│   └── screen5_pkpd_subgraph/   # NEW — Screen 5 (PK/PD subgraph + LLM imputation) [run later]
├── paths/
├── pipeline/
├── predict/
├── train/            # existing trainer; reuse, do not modify
└── utils/
```

**Hard rules on code layout**:
- Each `screenN_xxx/` is self-contained. No cross-screen imports except shared utilities in `utils/`.
- Never edit existing modules. If a shared piece needs change, copy + extend in the screen folder.
- Each screen folder MUST have a `run_screen{N}.py` entry script + a `README.md` (one-paragraph: what it tests, what gate, what output).

---

## 4. Result conventions

| Item | Convention |
|---|---|
| Run output dir | `Code/runs/{timestamp}__screen{N}__{tag}__seed{S}/` |
| Required files in each run dir | `config.json`, `results.json`, `train.log` |
| Aggregated report | `Notes/Experiments/_results/screen{N}/{date}__{slug}.md` |
| Comparison table | report file MUST include paired bootstrap CI vs EmerGNN multimode anchor on s0/s1/s2 |
| Tracking | one row per (screen, init/variant, seed) — do not aggregate into mean before logging raw |

---

## 4.5 Screen 1 (TAG) — full variant matrix

Backbone fixed: EmerGNN multimode. Only `external_init` tensor varies.

### Text source per node (shared by all PubMedBERT/Qwen variants)

| Node kind | Text source | Fallback if missing |
|---|---|---|
| Drug | `drug_profiles_flat.csv` `text` column (name + type + groups + targets + enzymes) | KG `name` |
| Drug — name-only sub-variant | `drug_profiles_flat.csv` `name` column (e.g. "Aspirin") | KG `name` |
| Non-drug (Gene/SE/Disease/Pathway/etc.) | merged KG `nodes.parquet` `name` column | zero vector |
| ID-only / unnamed | — | zero vector |

### 8 init variants

| ID | Init | Encoder | Text fed (drug) | Text fed (non-drug) | Role |
|---|---|---|---|---|---|
| A | Random Gaussian 64d | — | — | — | lower bound |
| B | Node-type one-hot → 64d | — | — | — | topology lower bound |
| C | Node2Vec 64d (pretrained on merged KG) | — | — | — | topology strong baseline |
| D | PubMedBERT [CLS] → 64d | PubMedBERT-base | **full profile text** | `name` | primary claim |
| **D-name** | **PubMedBERT [CLS] → 64d** | **PubMedBERT-base** | **drug `name` ONLY (e.g. "Aspirin")** | **`name`** | **isolates profile-richness vs name-only contribution** |
| E | PubMedBERT [CLS] → 64d | PubMedBERT-base | full profile text with **name shuffled within kind** | `name` shuffled within kind | semantic-vs-capacity control |
| F | PubMedBERT [CLS] → 64d | PubMedBERT-base | **kind string only** ("Drug") | kind string only ("Gene") | type-name-only control |
| **H** | **Qwen-72B input-embedding mean-pool → 64d** | **Qwen2.5-72B (or Qwen3-72B) — tokenizer + `embed_tokens` only, no transformer forward** | full profile text | `name` | **non-contextual-but-frontier-LM control** |

Projection (768d / 8192d → 64d) runs TWO methods per variant:
- (P1) frozen random Gaussian projection (seed 0)
- (P2) PCA-to-64 fit on all node vectors

So total runs per seed = 8 inits × 2 projections = **16 configs per seed**. (A/B/C have no projection, fixed at 8 + 2×5 = 18 — but A/B/C are projection-invariant so still count as 8 effective configs.) Practical compute: skip P2 for A/B/C → 13 configs/seed.

### Gates

1. **Primary**: D > E by ≥ 2pt on test_s2 AUC, paired bootstrap CI excluding 0, holds under both P1 and P2.
2. **Secondary**: D > D-name by ≥ 1pt → profile richness matters (else name alone is enough — paper narrative changes).
3. **Tertiary**: H vs D — checks whether contextual encoding (PubMedBERT) beats non-contextual frontier-LM token-embed (Qwen).
4. **Sanity**: D > F by ≥ 1pt → text content beats type label alone.

### Qwen-72B embedding-layer extraction (variant H)

- Load **only** `model.embed_tokens.weight` from the safetensors shard via `safetensors.safe_open`. Do not instantiate the full model.
- Approximate size: vocab 152K × hidden 8192 × FP16 ≈ 2.5 GB tensor — fits in 5090 32GB easily, also fits in CPU RAM.
- Pipeline: tokenize text → look up embeddings → mean-pool over tokens → 8192d vector → project to 64d.
- **Download location**: full Qwen-72B weights are ~145 GB. Do NOT download to `C:` or `D:`. Set `HF_HOME=G:/hf_cache` (or another drive with ≥200 GB free) before any HuggingFace download. If only the embedding shard is needed (~5 GB), `D:` is acceptable but `G:` is still preferred for consistency.
  - WSL view of G drive: `/mnt/g/hf_cache`
  - Command pattern: `HF_HOME=/mnt/g/hf_cache huggingface-cli download Qwen/Qwen2.5-72B --include "*.safetensors" --include "tokenizer*"`
  - Once downloaded, point loader at `/mnt/g/hf_cache/hub/models--Qwen--Qwen2.5-72B/snapshots/<hash>/`
- **If 72B download/extraction fails (HF auth issue, disk space on G:, etc.)**: drop H entirely from the matrix; report D/D-name/E/F only. Do NOT substitute a smaller Qwen model — variant H's whole point is "frontier scale". A smaller Qwen would muddle the comparison.

### Compute budget (5090, EmerGNN multimode ~11.6 h/run)

- One-time encoding: PubMedBERT 178K nodes ≈ 10 min; Qwen embed lookup 178K nodes ≈ 2 min; cached to `Code/data/KG/_merged_kg/_cache/`
- Per-seed training: 13 configs × 11.6 h ≈ 150 GPU-h
- Three seeds: ~450 GPU-h ≈ 19 days continuous

**Practical strategy**: seed 42 first runs all 13 configs (~6 days). If gates met → seeds 43, 44 only run the 4 critical configs (D, D-name, E, H — under P1 only) ≈ 4 × 11.6 × 2 = 93 GPU-h ≈ 4 days. Total ≈ 10 days for screen 1.

## 4.6 Screen 3 (Meet-in-Middle Pooling) — full variant matrix

Backbone fixed: EmerGNN multimode. Only readout changes. Flow propagation (L hops bidirectional) unchanged.

### Core change

EmerGNN original readout:
```
h_s, h_t = flow(L hops from source), flow(L hops from target)
score = MLP([h_s, h_t])    # terminal-only
```

Meet-in-middle readout:
```
H_s = {layer-l node embeddings for s, l=1..L}    # keep all intermediate layers
H_t = {layer-l node embeddings for t, l=1..L}
junctions = define_junctions(s, t, kg)            # J1 (structural) or J3 (kind-restricted)
h_junc = aggregate(H_s[junctions] ⊕ H_t[junctions])
score = MLP([h_s, h_t, h_junc])
```

`define_junctions` and `aggregate` are new modules; EmerGNN flow internals untouched.

### Junction definitions

| Type | Definition | Notes |
|---|---|---|
| J1 | shared 1-hop neighbors of (s, t) | structural, no hyperparameter |
| J3-PK | shared neighbors with `kind ∈ {Gene, Protein, enzyme, transporter, pathway}` (E1b "molecular" layer) | mechanism-aware |
| J3-PD | shared neighbors with `kind ∈ {SideEffect, Disease, Anatomy, Phenotype}` (E1b "effect_system" layer) | mechanism-aware |
| J3-both | J3-PK ∪ J3-PD | combined |

J2 (top-K attention-weighted) intentionally **skipped** — would confound with screen ② attention prior.

### 8 readout variants

| ID | Readout content | Junction | Role |
|---|---|---|---|
| R0 | EmerGNN terminal only | — | anchor (= EmerGNN multimode baseline) |
| R1 | terminal + junction | J1 | primary claim (hyperparameter-free) |
| R2 | terminal + junction | J3-PK | mechanism-aware (molecular layer) |
| R3 | terminal + junction | J3-PD | mechanism-aware (effect-system layer) |
| R4 | terminal + junction | J3-both | combined mechanism layers |
| R5 | junction only (no terminal) | J1 | ablation: is terminal still needed |
| R6 | terminal + random-set with same cardinality as J1 | random | **capacity-vs-structure control** |
| R7 | terminal + junction at 2-hop intersection | shared 2-hop neighbors | does deeper junction help |

### Prerequisites (gating)

1. **plan E7 (LR-over-meeting-features)** must pass first. Gate: LR-count AUC within 5 pt of GCN K=2 baseline. If gap > 5 pt, meeting-node signal is too weak and screen 3 is dropped.
2. **plan E1b (path-endpoint PK/PD layer asymmetry)** must pass for R2 / R3 / R4. If E1b chi-square omnibus is not significant, drop R2/R3/R4; run only R0/R1/R5/R6/R7.

### Gates (vs EmerGNN multimode on test_s2 AUC, paired bootstrap CI)

| Comparison | Threshold | Interpretation |
|---|---|---|
| R1 > R0 | ≥ +1 pt, CI excludes 0 | junction adds real signal |
| R1 > R6 | ≥ +1.5 pt | gain is from structural selection, not from added capacity |
| R4 > R1 | ≥ +0.5 pt | mechanism awareness adds incremental value (nice-to-have) |
| R5 vs R0 | descriptive | informs paper narrative: "augment terminal" vs "replace terminal" |
| R7 vs R1 | descriptive | informs whether deeper junction is worth its cost |

### Compute budget (5090, EmerGNN multimode ~11.6 h/run)

- Seed 42: 8 variants × 11.6 h ≈ 93 GPU-h ≈ 4 days
- If gates pass: seeds 43, 44 run only {R0, R1, R6, R4} = 4 × 2 × 11.6 ≈ 93 GPU-h ≈ 4 days
- Total ≈ 8 days

### Code layout

```
Code/my_code/models/screen3_meet_in_middle/
├── __init__.py
├── junction_finder.py        # J1 / J3-PK / J3-PD / J3-both / random / 2-hop  — no learnable params
├── junction_aggregator.py    # sym aggregation at junction nodes
├── emergnn_mim_readout.py    # EmerGNN backbone subclass, swaps readout
├── run_screen3.py            # entry: 8 variants × seeds
└── README.md
```

## 4.7 Screen 5 (PK/PD Subgraph + Virtual Mediator Embedding) — full variant matrix

Backbone fixed: EmerGNN multimode with screen 1 winner init. Two changes:
1. **⑤a**: KG split into PK and PD subgraphs (per E1b layer mapping); flow runs on each subgraph in parallel, late-merge readout.
2. **⑤b**: At structural gaps in subgraphs (distance ≥ 2 pairs with no intermediate node), inject a **Virtual Mediator Embedding (VME)** — an LLM-derived vector that acts as a virtual neighbor in message passing. **No actual nodes are added to the KG.**

### Subgraph definitions (per E1b layer mapping)

| Subgraph | Edge inclusion rule |
|---|---|
| PK | at least one endpoint has `kind ∈ {Gene, Protein, enzyme, transporter, pathway}` OR is `Drug` |
| PD | at least one endpoint has `kind ∈ {SideEffect, Disease, Anatomy, Phenotype}` OR is `Drug` |
| Drug-only edges (both endpoints Drug) | included in BOTH subgraphs |

Drug nodes preserved in both subgraphs (otherwise drug embeddings cannot update).

### VME generation pipeline

**Concept**: a PD path conceptually requires intermediate physiological-system nodes that the KG may lack. We do NOT add these nodes structurally. Instead, for each structural gap (a, c) we precompute an LLM-derived embedding that the flow uses as a virtual neighbor's contribution.

**Step 1 — gap pruning (`vme_gap_finder.py`)**:
- For each train/test drug pair (s, t), find top-3 shortest paths in each of PK / PD subgraphs
- A "gap" = position on a path where distance(a, c) ≥ 2 and no intermediate node exists in the subgraph
- Deduplicate by (node_a.id, node_c.id, layer)
- Hard cap: **20K gaps total**; if exceeded, keep highest-frequency gaps

**Step 2 — VME query (`vme_generator.py`)** — primary variant S4:
- LLM: **GPT-4o** via `openai` SDK
- Prompt template:
  ```
  In a {layer} drug-interaction pathway, name and briefly describe the
  most likely intermediate biological entity between "{a.name}" and "{c.name}".
  Respond in ≤2 sentences.
  ```
- Concurrency 20, retry 3, ~30-60 min total for 20K queries
- Encode LLM response with PubMedBERT [CLS] → 768d → project to 64d (reuse screen 1's projection)

**Step 3 — fallback variant S4b** (`vme_generator_qwen.py`):
- No API call. Construct prompt locally, tokenize with Qwen-72B tokenizer, look up `embed_tokens`, mean-pool over tokens → 8192d → project to 64d
- Free, ~30 min total, fully reproducible

**Caching**:
- Key: `sha256(f"{node_a.id}|{node_c.id}|{layer}")`
- Location: `Code/data/KG/_merged_kg/_cache/vme_gpt4o/{key}.json` (and `vme_qwen/{key}.json` for S4b)
- One-time generation; archive cache with paper release for reproducibility

**Budget guardrail**:
- Maintain `Code/data/KG/_merged_kg/_cache/vme_gpt4o/_cost_log.csv`
- Hard cap: **$50 USD**. Script aborts if exceeded.
- Expected actual cost at 20K queries × ~400 token avg: ~$50.

### Flow injection (S3/S4/S4b/S5 implementation)

EmerGNN message aggregation at layer l (per subgraph):
```
m_v^(l+1) = AGG_{u in N_subgraph(v)} flow_msg(h_u^(l), h_v^(l), edge_uv)    # real neighbors
          + w_l · sum_{(v,c) in VME_gaps_v} LLM_VME(v, c)                    # virtual neighbors (VME-variant only)
```

- `w_l` is a learnable scalar gate per layer, initialized small (1e-3) so model decides how much to trust VME
- VMEs are frozen during training (LLM not in compute graph)
- Two flow networks (PK + PD) do NOT share parameters (different mechanisms); if VRAM tight, fall back to shared backbone + subgraph-id embedding

### 9 variants (final)

| ID | Subgraph | VME | Role |
|---|---|---|---|
| S0 | full KG (no split) | — | anchor (= EmerGNN multimode + screen 1 winner init) |
| S1 | PK only | — | layer ablation |
| S2 | PD only | — | layer ablation |
| **S3** | **PK ∥ PD** | — | **⑤a primary** |
| **S4** | **PK ∥ PD** | **GPT-4o generated text → PubMedBERT embedding** | **⑤b primary** |
| **S4b** | **PK ∥ PD** | **Qwen-72B embedding-lookup (zero-cost fallback)** | **generation-vs-lookup control** |
| S5 | full KG | GPT-4o (S4 pipeline) | controls for VME effect independent of split |
| S6 | random 2-split (size-matched to PK/PD) | — | **structural control: is PK/PD split meaningful or any 2-split works** |
| S7 | PK ∥ PD strict (drug-only edges assigned to one side only) | — | strict-split variant |
| S8 | PK ∥ PD | **scrambled VME (same VME vectors but randomly assigned to wrong gaps)** | **semantic control: VME gain is from semantics not capacity** |

### Prerequisites

1. **plan E1b must pass** (PK/PD path endpoint asymmetry). If chi-square omnibus not significant, drop screen 5 entirely.
2. **Screen 1 winner init must be locked**. All S0-S8 use this same init so VME / split effects are not confounded with init.
3. LLM choice (GPT-4o vs Qwen lookup) decided **before** any training run; cache regenerated only if LLM changes.

### Gates (vs S0 on test_s2 AUC, paired bootstrap CI)

| Comparison | Threshold | Interpretation |
|---|---|---|
| S3 > S0 | ≥ +1.5 pt, CI excludes 0 | PK/PD split works |
| S3 > S6 | ≥ +1 pt | split is mechanism-based, not just any 2-partition |
| S4 > S3 | ≥ +1 pt | VME adds value |
| **S4 > S8** | **≥ +1 pt** | **VME gain comes from semantics, not from added capacity (critical gate for ⑤b)** |
| S4 > S5 | ≥ +0.5 pt | VME and split are complementary, not redundant |
| S4 vs S4b | descriptive | decides paper narrative: GPT-4o generation vs Qwen embed-lookup |
| S1, S2 | each > S0 - 2 pt | both layers contribute non-trivially |

### Compute budget (5090)

- VME generation (one-time, cached): GPT-4o ~$50 + 30-60 min; Qwen lookup ~30 min, $0
- Dual-flow training per run: ~1.5-1.8× EmerGNN baseline ≈ **18 h**
- Seed 42: 9 variants × 18 h ≈ 162 GPU-h ≈ **7 days**
- If gates pass, seeds 43, 44 run only {S0, S3, S6, S4, S8} = 5 × 2 × 18 = **180 GPU-h ≈ 7.5 days**
- **Total ≈ 14-15 days** (heaviest screen; dual-flow contributes the 1.5× per-run overhead, and S8 control cannot be dropped)

### Code layout

```
Code/my_code/models/screen5_pkpd_subgraph/
├── __init__.py
├── subgraph_builder.py        # PK / PD / random-split / strict-split extraction
├── vme_gap_finder.py          # enumerate candidate gaps, prune to 20K
├── vme_generator.py           # GPT-4o API + PubMedBERT encoding + cache
├── vme_generator_qwen.py      # Qwen-72B embed-lookup fallback (S4b)
├── vme_scrambler.py           # control variant S8: random-assign VMEs to wrong gaps
├── emergnn_dual_flow_vme.py   # EmerGNN dual-channel backbone with VME virtual-neighbor injection
├── run_screen5.py             # entry: 9 variants × seeds
└── README.md
```

## 4.8 Screen 2 (PK/PD Path Prior) — full variant matrix

Backbone fixed: EmerGNN multimode. Only path-level attention gets a soft bias term; KG topology and readout unchanged.

### Core change

EmerGNN flow original path attention:
```
α_path = softmax(MLP(h_s, h_t, h_path))
```

With prior:
```
α_path = softmax(MLP(h_s, h_t, h_path) + λ · prior(pair_mech, path_mech))
```

`λ` is a learnable scalar (init 0.1) so the model decides how much to trust the prior. `prior(·)` is a small lookup table: pair_mech == path_mech → +1.0; partial match (pair_mech is "both" or path is "both") → +0.5; mismatch → −1.0; unknown → 0.

### Relationship to Screen 5

| Dimension | Screen 5 | Screen 2 |
|---|---|---|
| Change location | input side (KG split) | inside flow (attention bias) |
| Topology | dual subgraph | full KG, unchanged |
| Per-run cost | 1.5-1.8× EmerGNN | ~1.05× EmerGNN |
| Not mutually exclusive — can stack | — | — |

**Execution dependency**: if Screen 5 ⑤a (S3) already secures ≥ +1.5 pt via PK/PD structural split, Screen 2's soft prior is likely dominated. Therefore Screen 2 runs **only if Screen 5 fails or yields marginal gain**.

### Label sources

- **Pair PK/PD label**: `Code/data/private/outputs_full/annotations/pkpd.parquet` (PK / PD / both / neither per pair) — must reach ≥ 80% coverage on seed-42 800-drug train+test, else screen 2 is dropped.
- **Path mechanism label** — three strategies:

| Strategy | Implementation | One-time cost |
|---|---|---|
| Rule-based (primary, T1) | majority vote of path intermediate kinds, using plan E1b's kind→layer mapping | $0 |
| LLM-labeled (T2) | GPT-4o reads path text, outputs PK/PD/both/neither; cache by path hash | ~$30 for ~50K paths |
| Hybrid (optional) | rule-based default; LLM only for ambiguous "mixed"/"neither" paths | ~$10 |

### 6 variants

| ID | Prior | Role |
|---|---|---|
| T0 | none (EmerGNN multimode raw) | anchor |
| **T1** | **rule-based prior (E1b mapping)** | **primary (zero-cost)** |
| T2 | LLM-labeled prior (GPT-4o) | high-quality label control |
| T3 | **random prior (same distribution, scrambled assignment)** | **critical control: prior gain is from semantics, not from added bias term** |
| T4 | hard prior (mismatched paths masked out entirely) | extreme variant — over-constrains? |
| T5 | T1 + screen 1 winner init | composition with best init |

### Prerequisites

1. **plan E1b passes** (PK/PD path-endpoint asymmetry + label audit ≥ 85% agreement)
2. **pkpd.parquet coverage check**: ≥ 80% of seed-42 800-drug train+test pairs have PK / PD / both label, else drop screen 2
3. **Screen 5 result is known**: only run Screen 2 if Screen 5 ⑤a fails to meet +1.5 pt gate (otherwise Screen 2 is likely dominated)

### Gates (vs T0 on test_s2 AUC, paired bootstrap CI)

| Comparison | Threshold | Interpretation |
|---|---|---|
| T1 > T0 | ≥ +1 pt | rule-based prior adds signal |
| T1 > T3 | ≥ +1 pt | gain comes from mechanism semantics, not from the extra bias term capacity |
| T2 > T1 | ≥ +0.5 pt | LLM labels worth the ~$30 |
| T4 vs T1 | descriptive | informs paper narrative: soft vs hard prior |

### Compute budget (5090, seed 42 + 800-drug only)

- Per-run training ≈ 1.05× EmerGNN baseline ≈ **12 h**
- 6 variants × 12 h ≈ **72 GPU-h ≈ 3 days**
- One-time LLM labeling (T2 only) ~$30
- **Lightest of the four screens**

### Code layout

```
Code/my_code/models/screen2_pkpd_prior/
├── __init__.py
├── pair_label_loader.py      # reads annotations/pkpd.parquet
├── path_labeler_rule.py      # rule-based path mechanism labeling via E1b kind→layer (T1)
├── path_labeler_llm.py       # GPT-4o path labeling + cache (T2)
├── prior_scrambler.py        # T3: same-distribution random reassignment
├── emergnn_with_prior.py     # EmerGNN subclass, adds prior to attention logits
├── run_screen2.py            # entry: 6 variants × seed 42
└── README.md
```

## 4.9 Screen 4 (LLM Decoder Verifier) — DEFERRED

Originally proposed: LLM as closed-vocab discriminator scoring path-effect consistency, output fed back to path attention.

**Status (2026-05-20): DROPPED from current execution scope.** Will reconsider only if Screen 1, 3, 5 collectively fail to reach a publishable gain on test_s2 AUC.

Reasons for deferral:
- Highest implementation cost (2-3 weeks) among the screens
- Hard to disambiguate from "LLM logits as feature" baseline in ablation
- Inference-time LLM cost (recurring, not one-time like Screen 5 VME)

If revived later, design will be specified in a future §4.9 expansion. No code skeleton, no folder created.

## 5. Screening plan (anchored to project plan)

| # | Screen | Folder | Order | Gate (vs EmerGNN multimode on test_s2) |
|---|---|---|---|---|
| ① | TAG node init (PubMedBERT) | `screen1_tag_init/` | **FIRST** | real PubMedBERT > shuffled-name by ≥ 2pt AUC, paired CI excluding 0 |
| ③ | Meet-in-middle pooling | `screen3_meet_in_middle/` | after E7 passes | ≥ 1pt over EmerGNN multimode |
| ② | PK/PD path prior | `screen2_pkpd_prior/` | after E1b passes | ≥ 1pt over EmerGNN multimode |
| ⑤ | PK/PD subgraph + LLM imputation | `screen5_pkpd_subgraph/` | after E1b passes, after ① | (a) +1.5pt subgraph alone (b) +1pt imputation increment |
| ④ | LLM decoder verifier | `screen4_llm_verifier/` | conditional (only if ①③⑤ underwhelm) | ≥ 2pt over EmerGNN multimode |

Prereqs (motivation experiments from main plan): E1b (PK/PD endpoint), E1c (readability), E7 (LR meeting node) — run concurrently with screen 1.

---

## 6. EmerGNN multimode anchor: re-run seeds 43, 44

Seed 42 numbers are in `Code/baseline/emergnn/_results/2026-05-20__binary_cls__seed42__final.md`. Seeds 43 and 44 are NOT yet run. Must run them before any screen reports paired CI.

Command template (run for seed in {43, 44}):
```bash
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_baseline.py --baseline emergnn --kg-source drugbank --seed 43 --epochs 100 --tag emergnn_seed43"
```
Expected wall time per seed: ~12 h on 5090.

---

## 7. What to do if uncertain

1. Re-read this file.
2. Check existing baseline code for prior art (`Code/baseline/emergnn/`).
3. Never invent a new path or filename — use only paths listed in section 2.
4. Never modify code outside `Code/my_code/screen{N}_*` (and even then, only inside your own screen folder).

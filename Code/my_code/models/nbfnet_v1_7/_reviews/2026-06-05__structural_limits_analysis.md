> # ⛔ INVALIDATED / 作废 (2026-06-06)
>
> **This entire analysis is VOID. It was computed on the WRONG knowledge graph.**
>
> The NBFNet run AND every structural number below were on the legacy **5-bucket
> DrugBank KG** (`build_kg_from_kb`: 5633 entities, 5 relations, 18792 edges,
> bipartite). That is NOT the project's canonical KG. The reference methods
> (EmerGNN / MNAH / PMP v1.1–v1.6 / v2i4) and the intended NBFNet setting use the
> **merged DrugBank+Hetionet+PrimeKG KG** (`build_kg_from_merged_parquet`:
> **22049 entities, 18 drug-incident relations, 319303 edges**, with drug–drug
> resemblance edges). `precompute_pmp_cache.py:10` confirms PMP uses the merged KG.
>
> Therefore the core premises here — "5 relations", "bipartite", "path-algebra
> collapse", "2-hop overlap AUC 0.704", "47% zero-overlap" — **do not hold** for
> the real setting. The 3 "structural gaps" and the codex discussion built on them
> are not valid evidence about NBFNet on our data.
>
> Corrected merged-KG structural stats (verified, `analyze_nbfnet_kg_structure.py
> --kg merged`): drug KG-degree median 75 (vs 6), trivial shared-neighbor AUC
> test_s2 = 0.6926, 79% of S2 positives share ≥1 neighbor. A corrected analysis
> will be written once the NBFNet@merged number is available.
>
> Kept (not deleted) for history per CLAUDE.md. **Do not cite anything below.**
>
> ---

# Why NBFNet v1.7 is structurally limited on our cold-start S2 DDI — analysis

## Meta

- **Date**. 2026-06-05
- **Primary reviewer**. Claude (claude-opus-4-8) — gathered structural evidence, drove the discussion, synthesized.
- **Independent reviewer**. codex (gpt-5.4, `model_reasoning_effort=xhigh`, MCP) — thread `019e99e5-1746-7350-9dd2-654076e989d6`, 3 rounds.
- **Trigger**. User. "use auto-research skill (→ research-review) + codex, ≤5 rounds, give the structural gaps + feasible directions for why NBFNet is limited on our KG/DDI."
- **Scope**. Structural analysis only (NOT a code review, NOT a fix loop). The trainer/model already passed Round 7 GO (see `2026-06-05__paper_faithful_implementation.md`).

## Evidence base (all verified, not estimates)

Run that produced the structural stats. `Code/scripts/analyze_nbfnet_kg_structure.py` (seed42 800-drug pkl).
Run that produced the NBFNet numbers. `Code/runs/2026-06-05_15-16-09__run_nbfnet__nbfnet_v1_7_seed42__seed42/`.

**KG (bipartite drug→entity, 5 drug-centric relations)**
- 5633 entities, 18792 edges, 2106 drugs in vocab.
- relation edges. target 8481, enzyme 4914, transporter 2889, carrier 733, pathway 1775.
- No entity-entity edges; no drug-drug edges except the DDI relation. → drug↔drug paths are even-length alternating (min length 2 = a shared entity).
- drug KG-degree. mean 9.0, median 6, min 1, max 304 (heavy skew). G1=640, G2=160; **8/160 G2 drugs have ZERO KG edges**.

**Trivial 2-hop-overlap floor (count of shared KG entities as the only score)**
- test_s2 AUC = **0.7043** (POS mean shared 1.27, 53.1% ≥1; NEG mean 0.23, 14.4% ≥1)
- val_s2 AUC = 0.6861; test_s0 AUC = 0.6734
- **~47% of S2 positive pairs share ZERO KG entities** (no 2-hop path exists).

**NBFNet v1.7 result (seed42, 100ep cap, early-stopped)**
- test_s2 AUC **0.7597**, AUPRC 0.7756, NLL 0.585. best val_auc 0.7335 @ epoch 3, then flat 10-epoch plateau (train loss kept dropping). → NBFNet adds only **+5.5pt over the trivial 2-hop overlap**.
- Reference band. EmerGNN 0.7458, NBFNet 0.7597, MNAH 0.7670, v1.5A 0.7764, v2i4 0.7804.

## The core finding

Four very different inductive biases land within ~3.5pt (EmerGNN→PMP), and a *dumb* shared-entity count already explains AUC 0.704. NBFNet's full 6-layer learned Bellman-Ford buys only +5.5pt over that floor. When architectures this different converge, the usual cause is **the available signal is nearly exhausted under the current KG + static side-information regime** — not insufficient model capacity. ~0.78 reads as a practical *information* ceiling for this benchmark, not an absolute ceiling of DDI prediction.

## Structural gaps (codex Round 1, 6 gaps; Round 2 narrowed to the 3 NBFNet-specific ones)

The 3 that are the real, NBFNet-specific story (paper-worthy):

1. **Path-algebra collapse on a drug-centric bipartite incidence graph.** NBFNet's selling point is query-conditioned *relational path composition*. Here the "path language" is impoverished: 5 drug-centric relations, no entity-entity edges, one binary `interact` query. The dominant informative motif is just a 2-hop shared entity; deeper paths recycle that signal. *Evidence*. trivial overlap 0.704; +5.5pt only. *Direction*. enrich the graph so composition means something (entity-entity / ontology / pathway-hierarchy / entity-similarity edges), or drop path reasoning for a bipartite set-matching model.

2. **Source-rooted reachability is the wrong geometry for symmetric cold-cold DDI.** NBFNet asks "from source drug, what reaches the target?"; S2 is a *pairwise neighborhood-compatibility* problem between two cold drugs. Post-hoc `h(a,b)+h(b,a)` is two one-sided passes, not native joint reasoning. *Evidence*. the strong overlap baseline; meet-node/PMP (dual-source) already capture most of the gain. *Direction*. dual-source / meet-node / cross-attention over the two entity sets — which is exactly the project's MNAH/PMP line.

3. **Depth >2 hops degenerates into seen-DDI bridge chasing, not richer mechanism.** For the 47% zero-overlap positives the only route is cold→entity→(G1 drug)→DDI→(G1 drug)→entity→cold. That is bridge-mediated *transfer through training topology*, not direct cold-cold evidence — exactly where hub/popularity shortcuts and weak generalization enter. *Evidence*. 47% zero-overlap; eval KG uses G1-G1 train DDI; val peaks @ep3 while train loss keeps falling. *Direction*. separate short-path from bridge evidence; regularize/downweight hub-mediated long paths.

Generic cold-start gaps (real but NOT the NBFNet story): zero-/low-degree cold drugs are unsalvageable for a feature-free topology-only model (8/160 G2 isolated; median degree 6); exact-overlap-only (no entity similarity) misses "different but related" biology; the single binary query underuses NBFNet's query-conditioning. These argue for side information (molecular/LLM/ATC), which the project already adds via MNAH/PMP — and which still only reach ~0.78.

## Feasible improvement directions (judged against existing project assets)

- Molecular alone is a known-weak lever here (prior internal. molecular ~0.733 < KG ~0.823), but may *specifically* rescue the 47% zero-overlap positives → only worth it if a subset analysis shows side channels help exactly that bin.
- Dual-source / meet-node and entity-similarity edges are *already* the MNAH/PMP line and still cap ~0.78 → diminishing returns from "better aggregator."
- Graph enrichment (entity-entity / ontology / similarity edges) is the one direction that could raise the *information* ceiling rather than re-extract the same signal. Highest upside, highest cost.
- For the NBFNet branch specifically. an architectural investment is only justified if the diagnostics below show NBFNet extracts *specific bridge structure* (not hub popularity).

## Decision (codex Round 3) — recommended: (b) run 2 cheap diagnostics, then freeze

Freeze NBFNet v1.7 as a baseline data point after running two ~1h diagnostics. Flip to "invest in an NBFNet-branch change" ONLY if BOTH diagnostics say: removing G1-G1 DDI at eval causes a material drop AND degree-preserving rewiring also causes a material drop (esp. on the zero-overlap subset) — i.e. NBFNet uses specific bridge topology, not hubs.

### Two diagnostics (highest insight per GPU-hour, no retraining)

1. **Eval-time removal of G1-G1 DDI edges**. run the trained model with eval KG = base KG only. Tests whether the +5.5pt over overlap comes from seen-DDI bridges.
2. **Degree-preserving rewiring of G1-G1 DDI edges**. swap-rewire the seen-DDI subgraph (5-10 graphs, average), no retrain. Tests whether bridge signal is specific structure vs hub/popularity exposure.

(Secondary. subset decomposition by shared-entity bin {0,1,2+} across overlap-baseline / NBFNet / KG-only / molecular-only / LLM-only / PMP; degree/exposure-matched negatives on the zero-overlap subset.)

### Results-to-claims matrix (codex Round 3, verbatim framing)

Rows = G1-G1 DDI removal; Cols = degree-preserving rewiring.

| | Hub-shortcut (rewiring ≈ original) | Structure-dependent (rewiring << original) |
|---|---|---|
| **Bridge-dependent** (remove DDI << original) | NBFNet's S2 gain depends on seen-seen DDI bridges, but is largely explained by exposure to high-degree bridge drugs, not specific mechanistic path structure. | NBFNet's S2 gain comes from specific seen-seen DDI bridge topology, not just hub exposure; but it is still bridge-mediated transfer, not direct cold-cold mechanism. |
| **Bridge-independent** (remove DDI ≈ original) | NBFNet does not meaningfully use seen-seen DDI bridges; on this bipartite drug KG it behaves mainly as a learned local-overlap / degree-biased scorer over the base drug-entity graph. | Internally inconsistent — rerun and resolve before claiming anything. |

## Bottom line

On this bipartite, drug-centric, 5-relation KG with cold-cold S2 pairs, NBFNet's core inductive bias (query-conditioned multi-relational *path* reasoning, source-rooted reachability) is aimed at the wrong structure. It works (beats EmerGNN, +5.5pt over 2-hop overlap) but cannot escape the ~0.78 information ceiling that all KG+static-side-info methods share. Recommended next action is the two cheap diagnostics, then freeze NBFNet v1.7 as a baseline and keep the narrative on inductive-bias mismatch.

## Codex thread

`019e99e5-1746-7350-9dd2-654076e989d6` (gpt-5.4, xhigh, 3 rounds). Round-1/2/3 content summarized above; key matrix and decision quoted verbatim.

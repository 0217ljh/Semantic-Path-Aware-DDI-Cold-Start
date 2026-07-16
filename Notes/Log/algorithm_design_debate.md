# Algorithm Design — Multi-Round Codex Debate Log

**Phase**: DESIGN ONLY (implementation is a SEPARATE later phase, per user).
**Goal**: 10-20 rounds of codex debate to harden the algorithm to a theoretically-grounded, reliable design.
**Started**: 2026-06-22

## Locked design intent (from user)

```
BASE: GNN. Fuse v1.6 hyper-edge + NBF-NET (NBF-NET SUBSUMED inside the hyper-edge
      framework, with appropriate simplification).

CORE: mechanism-level multi-modal alignment (molecular fragment ↔ KG mechanism),
      NOT drug-identity alignment.

3-LEVEL GRANULARITY (large→small scale, justified by over-smoothing theory):
   Mol. Fragment
      ↓
   coarse type A(τ)   [大类 common neighbor, 1.6-style]
      ↓
   fine entity within type   [小类 common neighbor, NBF-NET does path reasoning]
      ↓
   cross-type: entities of different types co-occur on the SAME drug-A→B path
               (not necessarily directly linked)

TRAINING:
   pretrain: mechanism-prior alignment (no DDI label)
   finetune: partial DDI label (transferable to binary/multi-cls/multi-label)
   contrastive: construct "should-NOT-interact" molecule-KG structures as negatives,
                to fix the positive-DDI distribution bias (model never sees
                non-responding structures otherwise)

KEY ARGUMENT: hyper-edge is LESS noisy than EmerGNN because large→small-scale
              layered processing is NECESSARY (over-smoothing theory), not by-design.
```

## Debate plan (themes, ~10-20 rounds)

| Round | Theme |
|---|---|
| 1-2 | Architecture skeleton: 3-level hierarchy + how NBF-NET is subsumed |
| 3-4 | Coarse-to-fine NECESSITY: over-smoothing theory (why large→small is forced) |
| 5-6 | Molecular fragment ↔ mechanism alignment mechanism |
| 7-8 | NBF-NET simplification within hyper-edge; cross-type path encoding |
| 9-10 | Training: mechanism pretrain objective (label-free) |
| 11-12 | Contrastive negatives: "should-not-interact" structure construction |
| 13-14 | DDI-label finetune + cross-task transfer |
| 15-16 | Theoretical consolidation (over-smoothing + common-neighbor preservation) |
| 17-18 | Falsification pass: where could the design fail / be attacked |
| 19-20 | Full integration + final design freeze |

(Plan is a guide; will adapt as debate surfaces issues.)

## Round log

### Round 1 — Architecture skeleton + NBF-NET subsumption (codex thread `019eee84`)

Codex produced a usable skeleton. Key decisions established:

**3-level hierarchy (formal):**
- KG `G=(V,E,φ,ψ)`, type `φ(v)∈T`, relation `ψ(e)`. Coarse types `T={τ_1..τ_K}`, K=12 (locked v1.6 set, REVISIT — may go to mechanism-vocab K=80).
- Fragment level: BRICS sets `F(a)={f_i^a}`, embeddings `z_i^a = g_frag(f_i^a)`.
- Coarse type level: typed neighbor `N_τ(a)`, typed common neighbor `A_τ(a,b)=N_τ(a)∩N_τ(b)` (v1.6).
- Fine entity level: anchor set `U_τ(a,b)=A_τ(a,b)`, plus local context for path reasoning.
- Cross-type: path `p=(a=v_0,...,v_L=b)`, type-trace `type(p)=(φ(v_1)..φ(v_{L-1}))`. Cross-type = ∃ s<t with φ(v_s)=τ, φ(v_t)=τ' on the SAME path. Handled implicitly by shared path membership.

**NBF-NET subsumed as fine layer (key move):**
- Per type τ, build a **type-anchored relevant subgraph** `G_τ^(a,b)` = subgraph of a,b + anchors `U_τ` + intermediate nodes within H hops on candidate a→b paths.
- Anchor mask `α_τ(v)=1[v∈U_τ(a,b)]`.
- Restricted Bellman-Ford: `h_τ^(0)(v)=x_a if v=a else 0`; for t=1..H:
  `m_τ^(t)(v)=⊕_{(u,v)∈E_τ} M_t(h_τ^(t-1)(u), e_uv, x_v, α_τ(u), α_τ(v))`,
  `h_τ^(t)(v)=U_t(h_τ^(t-1)(v), m_τ^(t)(v))`. Fine summary `p_τ(a,b)=R_τ({h_τ^(t)(b)})`.
- **Simplification of NBF-NET**: share M across t; small hop budget H∈{2,3,4}; relation-agnostic or light relation-aware msg `M=γ_τ(α_u,α_v)⊙σ(W_h h + W_e e + W_x x)`. Crucial restriction = **anchor conditioning** (not full-KG reasoning → this is the anti-EmerGNN-noise lever).

**Coarse routing + coarse-to-fine gating (v1.6):**
- Coarse pool `c_τ(a,b)=AttnPool_τ({x_u: u∈A_τ(a,b)})`.
- Learnable K×K transition `Q`: `c̃_τ=Σ_τ' softmax(Q_{τ,τ'}) c_τ'`.
- Gate `g_τ=σ(w_g^T[c_τ ‖ c̃_τ ‖ m_τ^frag])`; fused `r_τ=[c̃_τ ‖ g_τ·p_τ(a,b)]`; final `r(a,b)=Fuse({r_τ}, Q)`.
- Logic: **coarse type evidence decides which fine path computations matter** (= coarse-to-fine necessity hook).

**Where fragments enter:** at MECHANISM level via `m_τ^frag(a,b)=FragAlign_τ(F(a),F(b))` (FragAlign left for R5-6). Can modulate (1) coarse activation `c_τ' = c_τ + W_cf m_τ^frag`, (2) fine messages `M_τ(·; m_τ^frag)`, (3) gate g_τ. Never via drug identity.

**Forward pass:** SMILES→BRICS frags → typed common-nbr hyper-edges A_τ → coarse pool c_τ → frag signal m_τ^frag injected → Q-route → build G_τ subgraph + restricted BF → p_τ → gate → fuse → MLP score.

**Provenance:** v1.6 = A_τ, attn pool, Q matrix. NBF-NET = BF recurrence, path-sum. NEW = 3-level hierarchy, type-anchored subgraph G_τ, coarse→fine gating, frag-to-mechanism entry, cross-type as same-path co-occurrence.

**OPEN DECISIONS flagged (for later rounds):**
1. FragAlign_τ definition: frags→type prototypes vs →entity sets vs →path states. [R5-6]
2. Fine recurrence relation-agnostic vs relation-aware vs relation-buckets. [R7-8]
3. How to construct G_τ^(a,b): hop-limited / top-P paths / differentiable pruning. [R7-8]
4. Cross-type: implicit same-path only, vs explicit pairwise type-path score s_{τ,τ'}. [R7-8]
5. Q global-learned vs structured/fixed by biomedical type-transition prior. [R3-4]

**My concerns to push in R2 (attack the skeleton):**
- (a) Cost: building a separate H-hop subgraph + BF per type τ per pair × K types = K parallel NBF passes. Is this tractable at S2 scale? v1.6 was cheap precisely because it AVOIDED per-pair path reasoning.
- (b) Does `A_τ(a,b)=N_τ(a)∩N_τ(b)` (DIRECT common neighbor) even survive cold-start? In S2 both drugs unseen → do they HAVE typed KG neighbors at all? Need to confirm cold drugs are still in KG with edges (they are — DrugBank entities), but the common-neighbor set may be sparse/empty for many pairs → fine layer has nothing to anchor.
- (c) The cross-type "same path" notion needs the path to actually exist a→...→b; but a,b are drugs and most a→b paths route through the very mechanism entities — is the subgraph G_τ then just "all short a-b paths"? Then τ-restriction is post-hoc labeling, not a real architectural separation.

### Round 2 — Attack the skeleton (codex thread `019eee84`). MAJOR REVISION.

All 3 objections forced revision. Codex agreed with the suspected direction. **New skeleton:**

> **multi-hop typed reachability + ONE shared propagation + typed multi-head readout + coarse gating**

**Obj A (tractability) — REVISE.** K separate BF passes per pair is wrong at S2 scale (≈ v1.6 + K-fold overhead). Fix: ONE shared pair-conditioned propagation on a single compact subgraph `G^(a,b)`, then K typed readouts.
- `h^(0)(v)=x_a if v=a else 0`; `m^(t)(v)=⊕_{(u,v)∈E^(a,b)} M_t(h^(t-1)(u),e_uv,x_v)`; `h^(t)(v)=U_t(...)`.
- Typed readout: `p_τ(a,b)=Readout_τ({h^(t)(v): v∈V^(a,b), φ(v)=τ})`.
- `G^(a,b)` = CHEAP NON-differentiable retrieval (`V^(a,b)={v: dist(a,v)≤h_a ∧ dist(b,v)≤h_b}`, optional degree/score cap). Learning is in propagation+readout, not retrieval.
- Sparse gating: only top-Q types by coarse score contribute (soft-top-Q / straight-through / train-dense-infer-sparse). Doesn't break training.

**Obj B (cold-start common-neighbor survival) — REVISE STRONGLY.** Direct `A_τ=N_τ(a)∩N_τ(b)` (1-hop) is brittle in S2. Generalize to **multi-hop typed common reachability**:
- `R_τ^(ℓ)(a)={u: φ(u)=τ, dist(a,u)≤ℓ}`; `A_τ^(ℓ)(a,b)=R_τ^(ℓ)(a)∩R_τ^(ℓ)(b)`. Default ℓ∈{2,3}.
- ℓ=1 recovers v1.6 as special case. Architecture must NOT depend on 1-hop overlap density.
- `c_τ=AttnPool_τ({x_u: u∈A_τ^(ℓ)(a,b)})`.
- (NOTE: must verify in code whether v1.6 is 1-hop or multi-hop — its 0.776 says 1-hop is useful but not the right core abstraction.)

**Obj C (τ-restriction real or post-hoc) — REVISE.** My diagnosis confirmed: τ-subgraphs are overlapping views of the same path neighborhood. So type is NOT a partition of propagation; it is a **partition of mechanism summarization** (multi-head typed readout over shared path states). Cross-type effects already encoded in `h^(t)(v)` via mixed-type paths. Cleaner + resolves A.
- `r_τ=[c̃_τ ‖ p_τ ‖ m_τ^frag]`; `α_τ=Gate_τ(r_τ)`; `r(a,b)=Σ_τ α_τ W_τ r_τ`.

**Coarse-to-fine necessity MOVES (critical):** no longer "need separate per-type propagation." Now:
`coarse reachable-mechanism SELECTION (A_τ^(ℓ), explicit set feature) → fine typed path SUMMARIZATION (shared propagation + typed heads)`.

**Provenance after R2:** v1.6 = mechanism-type channels + routing + common-structure bias (now as explicit reachability feature). NBF-NET = path-based inductive propagation. NEW = multi-hop typed reachability coarse anchors + one shared propagation + typed multi-head readout + coarse gating.

**🚨 NEW DANGER introduced by R2 (must resolve in R3):** "ONE shared propagation over the a-b neighborhood G^(a,b)" is now structurally close to **EmerGNN** (which propagates over the pair's KG neighborhood with soft attention). Our KEY differentiator was "less noisy than EmerGNN." Must establish: (1) what makes G^(a,b)+typed-readout provably less noisy than EmerGNN's full-KG attention-pruned propagation; (2) the coarse explicit common-reachability feature A_τ^(ℓ) sits OUTSIDE propagation — it is the structural variable MPNN/EmerGNN CANNOT represent (BUDDY ICLR23 Thm 4.2 / MPLP NeurIPS24 / SEAL). That preservation argument is now the load-bearing novelty. R3 must nail it or the whole "less noisy" claim collapses.

### Round 3 — The "are we just EmerGNN?" make-or-break round (codex thread `019eee84`). THESIS LOCKED.

**VERDICT: thesis survives, but in narrower + honest form. The fine propagation is NOT the differentiator from EmerGNN.** Novelty load is correctly placed on: (1) explicit coarse pairwise structural extraction, (2) typed mechanism-conditioned readout/gating. The fine layer refines within preserved mechanism support; it does not carry novelty alone.

**Pillar 1 — structural-variable preservation (load-bearing).**
- Coarse signature `s^(ℓ)(a,b)=(s_τ^(ℓ))_τ`, `s_τ^(ℓ)=|A_τ^(ℓ)(a,b)|`, or pooled `c_τ^(ℓ)=Pool({x_u: u∈A_τ^(ℓ)(a,b)})`. Computed JOINTLY on (a,b) by set intersection BEFORE any propagation.
- ✅ DEFENSIBLE: EmerGNN-style propagation does not EXPLICITLY compute `A_τ^(ℓ)`/`s_τ^(ℓ)`; it computes nodewise states and combines at the scorer (SEAL/BUDDY/MPLP line: pairwise structural vars not natively represented by nodewise MP). Ours computes it exactly by construction, not learned approximation.
- ❌ OVERREACH TO AVOID: do NOT claim "EmerGNN provably cannot represent s_τ at all" in absolute generality. Safe version: standard MP does not PRESERVE this pairwise variable as an explicit statistic; any recovery is implicit through node embeddings + downstream pair function — exactly the regime where common-neighbor signals are known to be problematic.
- "Exact set intersection = anti-noise" is SOUND as a COMPUTATIONAL claim (removes estimation noise for that feature), NOT a claim the feature is biologically causally sufficient.

**Pillar 2 — over-smoothing → coarse-to-fine NECESSITY (state as expressivity-stability TRADEOFF, not hard depth bound).**
- ✅ DEFENSIBLE: recovering `A_τ^(ℓ)` from pure propagation requires transporting both ℓ-hop neighborhoods into embeddings then combining; larger ℓ → broader receptive field → repeated MP contracts representations (over-smoothing, Li AAAI18 / Oono-Suzuki ICLR20) → less structural selectivity. So "explicit coarse extraction THEN shallow fine propagation" is necessary if you want BOTH pairwise mechanism access AND bounded propagation noise.
- ❌ OVERREACH: do NOT say "MPNN must use depth 2ℓ" or "over-smoothing destroys signal at L=3" — too brittle/architecture-dependent.
- WEAKEST POINT (reviewer attack): "your fine layer ALSO propagates — why doesn't it over-smooth?" ANSWER: we do not ask the fine layer to DISCOVER mechanism support; coarse extracts support first + gates fine; fine stays shallow (small H) and only summarizes mechanism-conditioned local paths. Necessity is COMPARATIVE, not absolute.

**Pillar 3 — noise advantage, ranked by defensibility.**
1. (STRONGEST) exact coarse structural statistic — not soft-attention-estimated.
2. smaller support — fine runs on compact pair-conditioned G^(a,b), not full augmented KG (depends on how G^(a,b) is built → R7-8).
3. (WEAKEST, design-bias not theorem) typed multi-head readout decomposes evidence into mechanism channels vs single global attention.
- ❌ OVERREACH: do NOT claim "universally less noisy." Right claim is CONDITIONAL: relative to propagation-only attention pruning, ours reduces AVOIDABLE noise in extracting pairwise mechanism structure.

**🔒 LOCKED THESIS (cleanest sentence):**
> "Our model is less noisy than propagation-only DDI GNNs because it preserves the pairwise mechanism support A_τ^(ℓ)(a,b) explicitly by exact typed intersection before any message passing, making coarse-to-fine processing necessary: coarse extraction isolates the structural signal that pure propagation tends to blur, and shallow fine propagation only refines paths within that preserved support."

**🔒 FRAMING RULE:** STOP comparing the fine layer to EmerGNN as a novel propagator. Paper says: novelty is NOT a better neighborhood propagator; it is a **mechanism-preserving pairwise structural front-end + mechanism-typed refinement**.

### Round 4 — Earn the fine layer or cut it (codex thread `019eee84`). DECISION: (C) Hybrid.

Given R3 (fine prop not the differentiator), must justify keeping NBF-style fine layer. Three options: (A) keep BF as in R2/R3; (B) drop propagation entirely, pure explicit features; (C) hybrid — explicit features for ALL structure discovery + very shallow prop for semantic refinement only.

**VERDICT: (C). The fine layer survives ONLY in a demoted role — it cannot be the mechanism-structure discoverer.**

**Q1 — what fine layer can do that coarse cannot:**
- (a) cross-type same-path co-occurrence: `s_τ^(ℓ)` only counts per-type reachability INDEPENDENTLY; it cannot say a τ-entity and τ'-entity lie on the SAME bounded a→b path. BUT this does NOT uniquely require BF — it requires path-AWARE features, which can be made EXPLICIT: `s_{τ,τ'}^(ℓ)(a,b) = #{(u,v): φ(u)=τ,φ(v)=τ', ∃ path p:a→b len≤ℓ with u,v∈p}`.
- (b) ordering/direction: also explicit-able: `s_{τ≺τ'}^(ℓ)`.
- (c) relation-semantic weighting: this is where learned propagation helps MOST naturally (context-dependent composition of relation types/directions/attrs). Not strictly necessary but most natural.
- RANKING — genuinely require propagation: (1) context-dependent semantic path weighting/composition, (2) soft scoring of alternative path motifs, (3) generalization beyond fixed templates. Do NOT require propagation: per-type counts, cross-type same-path existence/count, explicit ordering.

**Q2 — pure explicit alternative (B):** `r(a,b)=MLP([s^(ℓ) ‖ s_pair^(ℓ) ‖ m^frag])`. Strictly weaker than BF in ONE sense: restricted to a fixed library of hand-specified structural statistics; cannot learn flexible compositions over path content. But it is the PUREST version of the "structure is explicit, propagation blurs it" thesis. Too austere given lead wants NBF subsumed + multimodal mechanism refinement.

**Q3 — DECISION (C) Hybrid:**
- Explicit structural features carry ALL structure discovery: `s_coarse=[{s_τ^(ℓ)}_τ, {s_{τ,τ'}^(ℓ)}_{τ,τ'}]`.
- These define support subgraph G^(a,b) + active type/type-pair channels.
- Shallow prop (H=1 or 2) on compact G^(a,b) ONLY refines node states: `h^(t)(v)=U(h^(t-1)(v), ⊕_{(u,v)∈E} M(h^(t-1)(u),e_uv))`, t≤2.
- Typed support-conditioned readout: `p_{τ,τ'}(a,b)=Readout_{τ,τ'}({h(v): v∈G^(a,b), φ(v)∈{τ,τ'}})`.
- Final: fuse explicit structural features + refined typed readouts + fragment alignment.
- NBF "subsumed with appropriate simplification" honestly satisfied: H≤2, compact subgraph, no structure-discovery claim, only refines embeddings of already-selected supports.

**🔒 LINE FOR THE LEAD (flag at design review):** "NBF can stay, but only as a shallow semantic refiner on top of explicitly extracted mechanism structure; if it is asked to discover the structure itself, it weakens the paper's core claim." (B-vs-C is a genuine lead decision: B = sharper/simpler thesis but drops learned path semantics; C = keeps NBF + multimodal refinement. Recommending C.)

**🔒 ARCHITECTURE AFTER R4:**
> explicit per-type reachability `s_τ^(ℓ)` + explicit cross-type co-path features `s_{τ,τ'}^(ℓ)` + typed coarse routing + very shallow propagation (H≤2) for refinement only + fragment↔mechanism alignment.

### Round 5 — CORE: mechanism-level fragment↔KG alignment (codex thread `019eee84`). FragAlign DESIGNED.

**Decision: FragAlign is HIERARCHICAL, matching the 3-level structure: `z_i^d → τ → v → (τ,τ')`. Not "pick one target."**

**Q1 — alignment targets (3-stage):**
1. Fragment ↔ mechanism TYPE τ (primary, most stable inductive unit): `α_{i,τ}^d = softmax_τ sim(W_f z_i^d, μ_τ)`, μ_τ = learnable type prototype SHARED across drugs.
2. Fragment ↔ KG ENTITY v (fine, ONLY within activated types, conditional on type to avoid noisy global retrieval): `β_{i,v}^d = softmax_{v∈V_τ^(a,b)} sim(U_f z_i^d, U_h h(v))`.
3. Fragment-pair ↔ cross-type CHANNEL (τ,τ') (pair-level, corridor activation): `γ_{ij,τ,τ'}^{a,b} = α_{i,τ}^a·α_{j,τ'}^b · σ((W_c z_i^a)^T M_{τ,τ'} (W_c z_j^b))`.
- Biological rationale: type = broad biochemical functionality (stable before single target known); entity = sharpened to specific protein/pathway; cross-type = fragments from a AND b JOINTLY activate a corridor (single frag doesn't align to a path alone).

**Q2 — FragAlign injects in THREE places, all through shared mechanism objects (no drug-id):**
1. Type gating: `m_τ^d=Σ_i α_{i,τ}^d z_i^d`; `m_τ^{a,b}=Fuse(m_τ^a,m_τ^b)`; modulates `c̃_τ=Fuse(c_τ, s_τ^(ℓ), m_τ^{a,b})`.
2. Entity readout modulation: `e_τ^d=Σ_i α_{i,τ}^d Σ_{v∈V_τ, φ(v)=τ} β_{i,v}^d h(v)` (mechanism-typed frag-to-entity retrieval). `r_τ=Fuse(c̃_τ, e_τ^a, e_τ^b, p_τ)`.
3. Standalone alignment score (label-free corridor estimate): `q_{τ,τ'}^{a,b}=Σ_{i,j} γ_{ij,τ,τ'}^{a,b}`.
- **NO-LEAK GUARANTEE:** all alignment passes through g_frag (shared), μ_τ (shared), h(v) (shared), M_{τ,τ'} (shared). NO drug-id embedding, NO neighbor-drug substitution. Only drug-specific input = SMILES fragments + KG neighborhood.

**Q3 — cold-start transfer argument (KEY, beats MKG-FENN kNN):**
- Transferable inductive unit = `fragment pattern → mechanism type/entity/channel association`.
- Unseen drug d: `d ↦ F(d)={f_i^d} ↦ {α_{i,τ}^d, β_{i,v}^d}` via SHARED params. No learned vector for d needed.
- kNN substitution (MKG-FENN) assumes cold drug ≈ nearby TRAINING drug identities → FAILS when new drug is a recombination of mechanism-relevant fragments not jointly seen in any one training drug.
- FragAlign composes attentively over fragments (`m_τ^d=Σ_i α_{i,τ}^d z_i^d`) → novel combination activates novel mechanism profile even if no training drug had that exact combination.
- 🔒 ONE-LINE: **"composition over shared fragment-to-mechanism mappings generalizes; identity imputation does not."**

**Q4 — label-free pretraining signal (mechanism-prior, no DDI label) — placed, detailed in R9-10:**
- Pseudo-targets from KG+SMILES connectivity alone:
  - type multi-label: `y_τ^(d)=1[R_τ^(ℓ)(d)≠∅]` (or count).
  - entity positives: `(d,v)` positive if `v∈R_τ^(ℓ)(d)`.
  - cross-type corridor: `y_{τ,τ'}^(d)=1[∃ bounded path from d containing both τ,τ']`.
- Objectives: `L_type=Σ BCE(ŷ_τ^(d),y_τ^(d))`; `L_ent` = entity InfoNCE (frag-emb vs h(v+) vs h(v-)); `L_chan=Σ BCE(ŷ_{τ,τ'}^(d),y_{τ,τ'}^(d))`. Uses ONLY drug–KG connectivity + SMILES fragments, zero DDI labels.

**OPEN for R6:** (1) μ_τ free param vs pooled from KG entities vs both; (2) entity alignment over all V_τ vs explicit anchor subset; (3) γ multiplicative vs bilinear vs OT-style; (4) sparsity/interpretability regularization on alignment; (5) symmetric vs role-aware (directed mechanism order) frag alignment across the two drugs.

### Round 6 — Resolve 5 FragAlign open choices (codex thread `019eee84`). One real revision: DIRECTIONALITY.

**(1) μ_τ prototypes — GROUNDED RESIDUAL + anti-collapse.** `μ_τ = λ·h̄_τ + (1-λ)·μ̃_τ`, h̄_τ=Pool{h(v):φ(v)=τ} (KG type centroid), μ̃_τ learnable residual, λ→1 early. Collapse risk REAL (free μ_τ → α ignores fragment, predicts marginal type freq). Three constraints TOGETHER: (a) grounding `L_ground=Σ_τ‖μ_τ−h̄_τ‖²`; (b) **difference-based supervision** — predict relative type-profile variation across drugs `Δy_τ^{d,d'}=y_τ^(d)−y_τ^(d')`, blocks "always predict common types"; (c) entropy-BAND reg (target moderate entropy, not min). Do NOT rely on frag-diversity contrastive alone (forces artificial separation of similar frags).
- **🔒 DIAGNOSTIC (prove fragment-specificity, anti-collapse):** **fragment permutation test** — shuffle fragments across drugs preserving drug-level type marginals; if perf/α barely change → FragAlign only learned priors; if drops → genuinely fragment-specific. Plus MI/retrieval test (predict held-out KG type profile from m^d vs global-freq baseline).

**(2) DIRECTIONALITY — REAL REVISION. Role-aware internals, label-compatible head.**
- DDI mechanisms often DIRECTED (A inhibits enzyme metabolizing B). Make internal mechanism channels ROLE-AWARE; symmetry ONLY at output if dataset labels undirected.
- Directed features: `s_{τ≺τ'}^(ℓ)(a→b)=#{(u,v): φ(u)=τ,φ(v)=τ', ∃ p:a→...→u→...→v→...→b}` (ordered type occurrence on directed bounded path). `γ_{ij,τ,τ'}^{a→b}=α_{i,τ}^{a,src}·α_{j,τ'}^{b,tgt}·σ(bilinear)` with src/tgt role projections.
- Undirected-label head: two directed passes `r^{a→b}=Φ(a,b)`, `r^{b→a}=Φ(b,a)`, then `ŷ=σ(MLP([r^{a→b} ‖ r^{b→a} ‖ r^{a→b}⊙r^{b→a}]))` OR symmetric `σ(MLP(r^{a→b}+r^{b→a}))`.
- Keep symmetric `s_τ^(ℓ)(a,b)` only as AUXILIARY; main mechanism features DIRECTIONAL.
- ⚠️ MUST VERIFY IN CODE/DATA: is our DDI benchmark task symmetric (interaction y/n) or directed (A→B effect)? Determines whether output head needs symmetrization. [verify before implementation]

**(3) Entity alignment scope — EXPLICIT ANCHOR SUBSET** `A_τ^(ℓ)(a,b)` (not all V_τ). Consistent with "explicit structure first"; avoids drifting into noisy soft search.

**(4) γ scoring — BILINEAR** `γ=(α_{i,τ}^a α_{j,τ'}^b)·σ((W_a z_i^a)^T M_{τ,τ'} (W_b z_j^b))`. OT = overkill/unstable; multiplicative-only too weak. Bilinear = minimal upgrade capturing frag-pair compatibility within channels.

**(5) Sparsity/interpretability — entropy-band + group sparsity.** `L_{α-sparse}=Σ(H(α_i^d)−h_0)²` (stable sparse, not brittle L1); `L_{γ-group}=Σ_{τ,τ'}‖q_{τ,τ'}‖₂` (concise channel explanations). Inference: report top-k fragments/types/channels → interpretable "fragment X → mechanism τ → corridor (τ,τ')" (selling point vs EmerGNN post-hoc beam search).

**🔒 ARCHITECTURE AFTER R6:** R4 architecture + hierarchical FragAlign (R5) + **DIRECTED/role-aware internal mechanism features** (src/tgt), symmetric output head iff labels undirected.

### Round 7 — Make explicit features COMPUTABLE + cheap (codex thread `019eee84`). Tractability SOLVED.

**Q1 — `s_{τ,τ'}^(ℓ)(a→b)` without path enumeration (KEY: bounded-WALK semantics, not simple paths).**
- (u,v) with φ(u)=τ,φ(v)=τ' active for a→b at budget ℓ iff directed walk `a⇝u⇝v⇝b` of length ≤ℓ exists ⟺ **`d(a,u)+d(u,v)+d(v,b) ≤ ℓ`** (shortest directed distances). EXACT for walk semantics (concatenate shortest paths ⟹ walk; walk exists ⟹ inequality holds).
- Count: `s_{τ,τ'}^(ℓ)(a→b)=Σ_{u:φ=τ}Σ_{v:φ=τ'} 1[d(a,u)+d(u,v)+d(v,b)≤ℓ]`. Pooled/weighted: `Σ w(u,v)[x_u‖x_v‖e_{τ,τ'}]`, `w=exp{-η·(dist sum)}·1[≤ℓ]`.
- Walk (allows repeats) NOT simple path: simple-path count is #P-hard; walks = cheap (BFS/matrix). NOT a cheat — structural claim is bounded mechanism connectivity/order on short supports (ℓ∈3..5), shortest-distance composition captures whether τ,τ' can jointly lie on an a→b corridor.
- **Complexity per pair** (n=|V^(a,b)|, m=|E_sub|): BFS from a O(m) + reverse-BFS from b O(m) + bounded all-pairs O(n·m_ℓ) + type-pair scan O(n²). For n in low hundreds → feasible, polynomial, exact.

**Q2 — compact support G^(a,b) + hub control (the EmerGNN failure mode):**
- Base: `V_0^(a,b)={v: d(a,v)≤h_a ∧ d(v,b)≤h_b}` (between-a-and-b corridor), h_a+h_b≤ℓ_max. Plus anchors `A_τ^(ℓ)={u:φ=τ, d(a,u)+d(u,b)≤ℓ}`. `V^(a,b)=V_0 ∪ ⋃_τ A_τ`.
- **3-stage cap (in order):** (1) distance budget `d(a,v)+d(v,b)≤ℓ_max` (strongest filter); (2) type-wise top-K_τ by `r(v)=-[d(a,v)+d(v,b)] - λ·log(1+deg(v))` (degree-PENALIZED so true hubs like CYP3A4 survive when close, not blanket-removed); (3) hard global cap n_max≈200-500.
- Degree-penalized ranking >> raw degree thresholding (latter discards true mechanism hubs). PPR rejected as MAIN filter (makes support approximate/opaque, contradicts explicit-structure thesis); optional fallback only.

**Q3 — shallow refinement H≤2 (semantic only):**
- Init `h^(0)(v)=[e_v ‖ e_{φ(v)} ‖ p(v;a,b)]` for non-drug v; drugs a,b use ONLY structural/fragment features, NO id embedding.
- `p(v;a,b)=MLP([d(a,v), d(v,b), d(a,v)+d(v,b)])` (directed-distance positional encoding).
- **🔒 INDUCTIVE-SAFETY RULING:** learnable embeddings for NON-DRUG KG entities (proteins/enzymes/pathways) are INDUCTIVE-SAFE — they are shared/seen, NOT the cold-start objects. Only DRUG-id embeddings are a leak. (Regularize entity embs toward type/feature structure to avoid memorizing artifacts.)
- Message (relation-BUCKETED, 58 rel → ~5 buckets enzyme-inhib/target-bind/pathway-membership/side-effect/unspecified): `m^(t)(v)=Σ_{(u,r,v)} α_{u,r,v}^(t) W_r h^(t-1)(u)`, attn over in-neighbors. Update `h^(t)(v)=LayerNorm(h^(t-1)(v)+MLP([h^(t-1)(v)‖m^(t)(v)]))`. H∈{1,2}. Cost O(Hmd)/pair.

**🟡 THE ONE HONEST APPROXIMATION (state plainly in paper):** support TRUNCATION under hub control. Features are EXACT on the compact support defined by the retrieval rule; APPROXIMATE relative to the full uncapped corridor only when pruning activates.

### Round 8 — Directed-distance conflation FIXED (codex thread `019eee84`). Correction to R6/R7.

**Problem:** R6/R7 built directionality on DIRECTED KG distances, but biomedical KG edges are mostly curation triples (subj,pred,obj) whose direction is an artifact (`part_of`, `associated_with` are biologically symmetric). Directed a→b reachability would be artificially sparse/meaningless → many pairs disconnected.

**Q1 — DECISION: distances run on the SYMMETRIZED KG `G_sym`** (each triple bidirectional for reachability). `d_sym(x,y)=shortest hop in G_sym`. Direction is NOT from path orientation. Mechanism direction recovered via: (1) ordered drug ROLES (a=src, b=tgt), (2) relation semantics + traversal direction along corridor.

**Q2 — corridor existence reality check + MANDATORY fallback:**
- Realistic motifs: 2-hop `a–protein–b`, 3-hop `a–protein–pathway–b`, 4-hop `a–P1–pathway/P2–b`. So ℓ∈{3,4} on G_sym is the regime. Captures many mechanism-sharing pairs but NOT all; many cold pairs have empty/tiny corridor within budget (KG incompleteness real).
- **🔒 EXPLICIT REQUIREMENT: graceful degradation to FRAGMENT-ONLY prediction.** When `V^(a,b)=∅` or all `s_τ=s_{τ,τ'}=0`, model still scores from fragment-side alignment `ŷ=f_frag(F(a),F(b))`. Architecture MUST include fragment-only fallback branch + **learned KG-availability gate**. Not optional — S2 must not collapse when support sparse.

**Q3 — relation-direction as a FEATURE (direction survives without path orientation):**
- Each traversed edge → (relation bucket r, traversal-direction flag δ∈{+1,-1} = stored-triple-direction or reversed). Undirected support edge = two labeled options `(u,v,r,+1)`,`(v,u,r,-1)`. Refinement/readout use `e_{r,δ}` not just `e_r`.
- Co-path feature gains relation-semantic weight: `s_{τ,τ'}^(ℓ)(a,b)=Σ 1[d_sym sum ≤ℓ]·ρ(u,v;a,b)`, where `ρ=Pool(hist(a⇝u),hist(u⇝v),hist(v⇝b))` over relation-bucket/direction (r,δ) histograms on shortest-path segments.
- Role-aware head learns e.g. "a's fragment INHIBITS → enzyme (reverse semantics) → b's fragment METABOLIZED-BY that enzyme class."

**Q4 — restated under symmetrized reachability:**
- `s_{τ,τ'}^(ℓ)(a,b)=Σ_{u:φ=τ,v:φ=τ'} 1[d_sym(a,u)+d_sym(u,v)+d_sym(v,b)≤ℓ]` — now SYMMETRIC as support-existence `s_{τ,τ'}(a,b)=s_{τ',τ}(b,a)`.
- Asymmetry reintroduced in HEAD: `r^{a⇒b}_{τ,τ'}=Fuse(s_{τ,τ'}, ρ-weighted s, m_τ^{a,src}, m_{τ'}^{b,tgt}, γ^{a⇒b})`. Undirected labels: `ŷ=σ(MLP([r^{a⇒b} ‖ r^{b⇒a}]))`.
- **NET: SIMPLIFIES computation** (one undirected BFS, one shared support, reused both role orders) while keeping mechanism direction in features.

**Effect on core arguments:** inductive-safety UNCHANGED (no id leak). Noise argument IMPROVES — symmetrized support avoids orientation-artifact sparsity. New phrasing: "explicit support from robust undirected biomedical connectivity + mechanism direction modeled separately via relation semantics & drug roles → avoid BOTH propagation noise AND orientation artifacts."

### Round 9 — Label-free mechanism-prior PRETRAINING (codex thread `019eee84`). Honest verdict.

**Q1 — VERDICT: pretraining is CONDITIONALLY essential, NOT automatically valuable just for being label-free.**
- Case A (test drugs have RICH KG neighborhoods): predicting a drug's own KG neighborhood from fragments is PARTLY REDUNDANT (can read KG edges directly at inference). Value modest: regularize alignment, denoise sparse channels, help fragment-only fallback.
- Case B (test drugs have PARTIAL/incomplete KG): pretraining is SUBSTANTIVELY important — fragment branch learns to COMPLETE missing mechanism associations from structure. Real downstream transfer.
- Naive "predict your own observed KG neighbors" ≈ adjacency reconstruction = weak shortcut. Meaningful version: predict HELD-OUT associations from fragments + partial context under masking.
- **⚠️ DATA FACT TO VERIFY before implementation:** are our S2 cold-test drugs' KG neighborhoods rich or sparse? Determines essential-vs-marginal. (Merged KG: 178k nodes / 7.1M edges incl 8048 drugs — need per-cold-drug KG-degree distribution.)

**Q2 — RIGHT PRETEXT = masked KG-completion-from-structure (subsumes R5 targets):**
- For drug d: mask fraction m of mechanism edges/entities/channels; train FragAlign to recover from fragments + remaining context + shared structure. Directly matches cold-drug-with-incomplete-KG use case + serves R8 fragment-only fallback.
- 3-level targets (computed on `R_τ^(ℓ)(d)={u:φ(u)=τ, d_sym(d,u)≤ℓ}`): type completion `y_τ^(d)=1[R_τ≠∅]`/count; entity completion (masked v∈R_τ as positives); channel completion `y_{τ,τ'}^(d)`.
- Masking CURRICULUM: m=0.2 early → 0.4-0.5 later → occasional FULL-KG mask (forces fragment-only recovery).
- Loss: `L_pre = λ1·L_type(BCE) + λ2·L_ent(InfoNCE frag vs h(v+)/h(v-)) + λ3·L_chan(BCE) + λ4·L_ground`.

**Q3 — leakage/protocol (state EXPLICITLY, don't overclaim):**
- Using test-drug KG neighborhoods in pretraining (no DDI label) = **"transductive on KG, inductive on DDI labels"** — defensible, common for KG cold-start, SAME as EmerGNN. Claim is NOT "never saw these drugs"; it IS "never saw DDI supervision involving them, but may use their non-DDI KG + molecular info."
- Stricter purely-inductive claim → must exclude test drugs from pretraining entirely (cleaner but weakens mechanism-completion case).
- **🔒 RECOMMENDATION: report BOTH if feasible** — (1) main = transductive-KG/inductive-DDI (fairest vs prior); (2) strict = full inductive drug holdout (isolates transferable frag→mechanism gain). If only one, use (1) + state plainly.

**Q4 — pretrain→finetune handoff:**
- Pretrain: g_frag, μ_τ, entity embeddings/encoder, FragAlign (α,β,γ), shallow refinement, type/channel readout. NOT the final DDI head.
- Finetune: ALL with differential LR (lower for pretrained backbone, higher for DDI head + task routing). Do NOT hard-freeze alignment (too rigid).
- Anti-forgetting: keep auxiliary alignment loss `L_total=L_DDI + η·L_align-aux` (lighter type/channel completion on minibatch drugs); anneal η down but not to zero unless it hurts val.

**🔒 CLEAN TRAINING STORY:** pretrain frag→mechanism alignment to recover MASKED type/entity/channel associations → finetune on DDI with alignment prior retained as auxiliary constraint.

### Round 10 — Contrastive "should-NOT-interact" negatives (codex thread `019eee84`). Honest: WEAKEST component.

**Q1 — the REAL bias = EASY-NEGATIVE bias (not "never sees negatives").** Random negatives exist but are mostly mechanistically UNRELATED (no shared reachability, no corridor, weak channel overlap) → trivially separable. Model undertrained on the dangerous region: **hard non-interactors with strong mechanism-like structure** (high s_τ, high s_{τ,τ'}, plausible channels, but NO DDI). With our architecture amplifying mechanism signals, model would learn "corridor presence ⟹ interaction" — too coarse. 🔒 FRAMING: standard negatives = coarse class separation; contrastive = teach **mechanism support is NECESSARY but NOT SUFFICIENT**.

**Q2 — principled construction (avoid false-negative trap):**
- ❌ DO NOT use option (a) "high-corridor but unlabeled pairs" as supervised negatives — that's exactly where positive-unlabeled contamination is worst (high mechanism overlap → maybe they DO interact, unrecorded).
- ✅ (b) **counterfactual mechanism-ablation negatives**: from a positive pair (a,b) with high channel (τ,τ'), build broken support `G̃^(a,b)` by removing top anchors in A_τ / deleting key cross-type corridor edges / zeroing relation-semantic features on decisive segment / masking dominant γ. KEY: this is NOT a fake drug pair claiming non-interaction — it is a fake pair-STRUCTURE VIEW; requires ablated structure to score LOWER than intact. Safe (no false-negative pair label).
- ✅ (c) **confident disjoint-mechanism negatives**: small pool of pairs with no bounded corridor + no shared types + near-disjoint fragment profiles. High-confidence anchors.

**Q3 — where contrastive attaches:** TWO levels — (1) MAIN: pair representation/logit `L_cf-logit=max(0, m − ŷ(a,b;G) + ŷ(a,b;G̃))`; (2) channel-level (mechanism-local, interpretable): dominant channel `q_{τ*,τ'*}(a,b;G) > q_{τ*,τ'*}(a,b;G̃)` margin. NOT on raw α (too indirect). Mostly FINETUNING, not core pretraining (R9 KG-completion is cleaner/more fundamental); can be light auxiliary pretraining loss.

**Q4 — NECESSITY VERDICT: SUPPORTING ablation, NOT headline.** Less fundamental than explicit-structure+FragAlign core. Well-motivated (our architecture amplifies mechanism signals → real failure mode "strong corridor ⇏ DDI") but: theoretically less clean, negative construction delicate/attackable, paper doesn't NEED it. 🔒 EARNS inclusion ONLY if ablations show it specifically improves: high-corridor non-interacting pairs, precision@high-recall, FP reduction in mechanism-rich regions. Small avg-AUC bump ⟹ not worth narrative weight.
- 🔒 PAPER SENTENCE: "We further reduce mechanism-driven false positives with a counterfactual contrastive objective that distinguishes intact interaction corridors from structurally ablated corridors, without requiring potentially noisy unlabeled-pair negatives."
- **🔒 FLAG FOR LEAD:** contrastive recommended as ABLATION/optional module, not a headline contribution. (Lead originally listed it as a core training pillar — recommending demotion.)

### Round 11 — Cross-task transfer binary/multi-cls/multi-label (codex thread `019eee84`). Real, if framed right.

**Q1 — what transfers: the MECHANISM-STRUCTURED DECOMPOSITION, not a generic embedding.** Shared backbone = s_τ, s_{τ,τ'}, FragAlign (α,β,γ), q_{τ,τ'}, r_{τ,τ'}. All 3 tasks = different projections of ONE latent question "which mechanism channels are active for this pair?" (binary=any harmful channel active; multi-label=which labels active; multi-class=which dominates). Architecture produces a FACTORIZED MECHANISM BASIS.
- ⚠️ BUT channel↔DDI-type is NOT identity: our (τ,τ') = KG entity types (protein/enzyme/pathway); DDI K=80 = effect labels ("increased risk of X"). Different ontologies. Mapping channels→labels is LEARNED. Claim = "mechanism-grounded latent basis", NOT one-to-one semantic alignment.

**Q2 — defensible claim = (b)+(c):** (b) mechanism channels PRE-STRUCTURE the output space (mechanism-grounded basis, more stable than any single task's label space); (c) mechanism-prior pretraining improves DATA EFFICIENCY across tasks. ❌ AVOID (a) "task-agnostic backbone, only head changes" (weak, any model claims it). ❌ AVOID claiming channels = DDI classes.
- **🔒 DIAGNOSTIC (make (b) real not wishful):** freeze backbone, train LINEAR PROBE from {q_{τ,τ'}} / {r_{τ,τ'}} to multi-cls/multi-label labels; compare vs probe on generic pair-embedding baseline. If channels align → probe predicts labels with minimal supervision. Second: MI/CKA/cluster-purity between DDI classes and channel activation patterns.

**Q3 — joint vs sequential: SEQUENTIAL (not equal-status multitask).** Mainline: pretrain (mechanism completion) → finetune PER TASK from same backbone init → report same backbone transfers. Multitask muddies (label spaces differ in semantics/noise; reviewers attribute gains to more supervision). Task hierarchy: multi-label richest, `y_binary=1[Σ_k y_k^ML>0]`, multi-class = single-dominant special case. Optional supporting expt: pretrain+finetune multi-label → adapt to binary with limited labels vs binary-only.

**Q4 — cold-start × transfer: MULTI-LABEL is the most NATURAL fit + best showcase.** Binary supervision is WEAK for our architecture (doesn't say which channel responsible); multi-label REWARDS identifying which mechanism/effect active → aligns with FragAlign+typed channels → stronger cold-start signal. Positioning: broadly applicable to all 3, ESPECIALLY well-matched to multi-label mechanism-aware DDI. Caveat: multi-label labels can be sparse/noisy → advantage conditional on label quality.

**🔒 STRONGEST PAPER CLAIM:** "Our mechanism-channel backbone transfers across DDI task formulations because it factorizes pair evidence at the mechanism level, which is more stable than any single task's label space." Support: (1) linear probes channels→labels, (2) few-label cross-task adaptation, (3) multi-label supervision improves cold-start binary more than binary-only.

### Round 12 — Theory consolidation (codex thread `019eee84`). "Prove less, claim it cleanly."

**PRINCIPLE: only C1 is theorem-grade. C2 (over-smoothing) and C3 (noise) → downgrade to analysis + empirical.**

**Q1 — C1 (PROVABLE): a SEPARATION PROPOSITION, not a universal impossibility theorem.**
- 🔒 PROPOSITION: For a standard L-layer MPNN computing endpoint embeddings `h_a^(L), h_b^(L)` (query-independent) + any pair decoder `g(h_a,h_b)`, ∃ pairs (a,b),(a',b') in typed multi-relational KGs s.t. (1) `h_a^(L)=h_{a'}^(L)` and `h_b^(L)=h_{b'}^(L)` for EVERY such GNN, but (2) `s_τ^(ℓ)(a,b)≠s_τ^(ℓ)(a',b')` (sim. s_{τ,τ'}). ⟹ these features not recoverable from nodewise MP + pair decoder in general.
- SEPARATION SKETCH (proof-by-construction): make endpoint-rooted L-hop neighborhoods of a,a' isomorphic (and b,b'), by duplicating local typed structure — so any MPNN gives identical endpoint embeddings — while changing ONLY the cross-linking BETWEEN the two neighborhoods (shared vs disjoint common neighbor for s_τ; on-same-corridor vs not for s_{τ,τ'}). Then pairwise feature differs but MPNN can't distinguish.
- HONEST PHRASING: "standard nodewise MP does not IN GENERAL preserve these pairwise structural variables; we give a separation construction" — NOT "MPNNs cannot represent these features."

**Q2 — C2 over-smoothing necessity: DOWNGRADE to empirical motivation.** No clean finite-depth theorem ("must go deep enough to over-smooth"); over-smoothing lit is asymptotic/architecture-conditional → hard necessity theorem = overclaim. Honest: recovering broader pairwise structure via propagation needs larger receptive field; deeper prop known to increase smoothing/reduce discriminability; explicit extraction avoids the tension. **Paper treatment = EMPIRICAL CURVE** (sweep L in EmerGNN-style/shared-prop baseline; show hard-structural-case perf doesn't improve monotonically / degrades; ours retains perf without increasing L) + supporting citation.

**Q3 — C3 noise: DOWNGRADE to restricted analytical remark + empirical.** Narrow formalization OK: explicit set-intersection/distance features are DETERMINISTIC given KG+retrieval rule (ZERO estimation variance for that statistic); learned attention estimates it indirectly (nonzero estimator variance/approx error). BUT only valid if explicit about WHICH statistic — does NOT prove whole-prediction less noisy globally. Restricted claim: "less noisy in EXTRACTING the explicit pairwise mechanism-support statistic", not universal. Back with empirical (explicit vs learned-only on hard negatives/sparse-corridor/hub-heavy; stability+interpretability of channels).

**Q4 — 🔒 CONSOLIDATED THEORY SECTION STRUCTURE (every claim labeled):**
1. Preliminaries/definitions (s_τ^(ℓ), s_{τ,τ'}^(ℓ), MPNN setting, our extractor) — formal.
2. Prior-theory connection (SEAL/BUDDY/MPLP) — CITED theory.
3. Our separation proposition + proof — OUR THEOREM (core theory contribution).
4. Corollary: explicit extraction preserves these exactly by construction — direct implication.
5. Over-smoothing as DESIGN MOTIVATION (analysis + citations, NOT theorem).
6. Noise as RESTRICTED ANALYTICAL REMARK (exact-conditional-on-support) → fast to experiments.
7. Empirical validation (explicit-vs-learned ablation, depth sweep, hard-negative/hub/sparse subsets, channel stability).

**🔒 FINAL THEORY POSITION:** (1) PROVEN: explicit coarse features preserve pairwise structural variables nodewise MP can fail to preserve. (2) ARGUED+EMPIRICAL: deeper propagation to recover such structure → stability/selectivity tradeoff. (3) EMPIRICAL: exact extraction improves robustness on mechanism-rich cases. Review-safe.

### Round 13 — Falsification pass / hostile reviewer (codex thread `019eee84`). 🚨 KEY RISK SURFACED.

| Attack | Lethality | Defense / required experiment |
|---|---|---|
| **A1** "just hand-crafted features, GBDT matches you" | SERIOUS | Ablation ladder; MUST run GBDT/XGBoost on {s_τ, s_{τ,τ'}, pooled stats} vs full. If GBDT close → novelty shifts to "feature discovery" (survivable if perf+theory strong, less ambitious). |
| **A2** "v1.6 already 0.776; complexity unearned" | POTENTIALLY FATAL | Strict ablation ladder: 1-hop → +multi-hop → +co-path → +role-head → +FragAlign → +refine → +pretrain. Total story coherent (need not be monotone every step). Subset analysis (empty-1-hop / sparse-corridor / hub-heavy / unseen-fragment). If full only matches v1.6 on avg but wins on sparse-corridor subset → still saves paper. |
| **A3** "molecular modality is DECORATION; ablating fragments barely drops" | **MOST LIKELY FATAL** | THE headline risk. Project's OWN prior: PK 0.733 < KG 0.823 (molecular weaker). Defense = CONDITIONAL utility NOT universal: restrict multimodal claim to regime where fragments needed (sparse/missing KG, empty corridor, rare/unseen fragment combos, fallback). |
| **A4** "transductive-KG weakens cold-start claim" | SERIOUS-manageable | State explicitly inductive-on-DDI/transductive-on-KG (= EmerGNN); provide stricter fully-inductive variant; compare fragment gains under KG masking. |
| **A5** "counterfactual negatives are circular (teaching model its own bias)" | MANAGEABLE | Keep secondary = regularization vs mechanism-overconfidence, NOT biological proof. On/off ablation on hard-negative precision/calibration/PR-AUC. If minor → appendix. |
| **A6** "gains within seed variance (cold-start notoriously high-var)" | FATAL IF IGNORED | 5-10 S2 splits, 5 seeds, paired bootstrap/t-test vs EmerGNN+v1.6, per-split win counts, subgroup consistency. NOT single split / 3 seeds. |
| **A7** "co-path 'exact' is misleading — pruning removes true corridors" | SERIOUS | Word precisely: exact GIVEN retrieval support, approximate vs full KG if pruning active. Sensitivity-to-cap experiment; show perf saturates + selected corridors stable. |

**🚨🚨 SINGLE MOST LIKELY REJECTION REASON: A3 — fragments don't add convincing value beyond the cold drug's KG neighborhood.** Threatens the paper's IDENTITY. If true: "multimodal alignment" overstated, FragAlign decorative, real contribution = explicit KG structural features (= a DIFFERENT, still-good paper).

**🔒 THE ONE EXPERIMENT THAT MOST DE-RISKS THE PROGRAM (run EARLY in implementation, before locking paper identity):**
> **Stratified KG-availability ablation with controlled masking.** (1) train/eval full KG; (2) progressively mask cold-drug KG neighborhoods at test/val; (3) compare KG-only-explicit vs KG+FragAlign vs fragment-only; (4) stratify by corridor density. If FragAlign truly matters it shows up exactly when KG sparse/incomplete. Addresses A3+A4+partly A2 in one shot.

**🔒 PAPER-IDENTITY HEDGE (design to survive either outcome — do NOT lock multimodal headline before the de-risk experiment):**
- BEST CASE: fragments materially help under sparse/incomplete KG → keep full multimodal headline.
- FALLBACK CASE: explicit mechanism structure is the main result → molecular alignment = robustness/fallback module, paper becomes "explicit pairwise mechanism-structure for cold-start DDI" with multimodal as secondary.

### Round 14 — FINAL INTEGRATION + DESIGN FREEZE (codex thread `019eee84`). ✅ READY FOR IMPLEMENTATION.

**🔒 FROZEN FORWARD PASS (13 steps), for query pair (a,b):**
1. Build symmetrized retrieval graph `G_sym` (each edge bidirectional for reachability; keep relation label + traversal-direction flag). [supersedes R6/R7 directed distance]
2. Retrieve compact support: `V_0^(a,b)={v: d_sym(a,v)+d_sym(v,b)≤ℓ_max}` ∪ typed anchors `A_τ^(ℓ)`; rank `r(v)=-[d_sym(a,v)+d_sym(v,b)]-λlog(1+deg(v))`; top-K_τ per type + global cap n_max → G^(a,b).
3. Explicit features: `s_τ^(ℓ)=|A_τ^(ℓ)|`; `s_{τ,τ'}^(ℓ)=Σ 1[d_sym(a,u)+d_sym(u,v)+d_sym(v,b)≤ℓ]`; relation-semantic `s_{τ,τ'}=Σ 1[·]·ρ(u,v;a,b)` (ρ=relation-bucket+direction histograms on segments).
4. Encode fragments: BRICS `F(a),F(b)` → `z_i^a=g_frag(f_i^a)` (shared inductive encoder).
5. Init support-node states (non-drug v): `h^(0)(v)=[e_v ‖ e_{φ(v)} ‖ p(v;a,b) ‖ x_v^stat]`. Drugs a,b: NO id embedding.
6. Shallow refinement H≤2, relation-bucketed `(r,δ)`: `m^(t)(v)=Σ α W_r h^(t-1)(u)`, `h^(t)(v)=LN(h^(t-1)+MLP([h‖m]))`. [refinement only, structure already fixed in steps 2-3]
7. Grounded prototypes: `μ_τ=λ·h̄_τ+(1-λ)·μ̃_τ`, h̄_τ=Pool{h(v):φ(v)=τ}.
8. FragAlign frag→type: `α_{i,τ}^d=softmax_τ sim(W_f z_i^d, μ_τ)` → role-aware profiles `m_τ^{a,src}, m_τ^{a,tgt}, m_τ^{b,src}, m_τ^{b,tgt}`.
9. FragAlign frag→entity (over anchors A_τ): `β_{i,v}^d=softmax sim(U_f z_i^d, U_h h(v))` → `e_τ^a, e_τ^b`. [refine THEN align — coherent]
10. FragAlign frag-pair→channel (role a⇒b): `γ_{ij,τ,τ'}^{a⇒b}=(α_{i,τ}^{a,src} α_{j,τ'}^{b,tgt})·σ((W_a z_i^a)^T M_{τ,τ'} (W_b z_j^b))`; `q_{τ,τ'}^{a⇒b}=Σ γ`. Repeat b⇒a.
11. KG-availability gate `κ(a,b)∈[0,1]` from support size/nonzero-structure → fragment-only fallback when corridor empty/sparse.
12. Role-aware channel fusion: `r_{τ,τ'}^{a⇒b}=Fuse(s_τ, s_{τ,τ'}, ρ-s_{τ,τ'}, m_τ^{a,src}, m_{τ'}^{b,tgt}, e_τ^a, e_{τ'}^b, q_{τ,τ'}^{a⇒b})` modulated by κ; aggregate → `r^{a⇒b}`, `r^{b⇒a}`.
13. Task head: undirected labels → `r^sym=SymFuse(r^{a⇒b}, r^{b⇒a})`; binary=BCE, multi-cls=softmax CE, multi-label=per-class BCE.

**Q2 — consistency: NO architecture-level contradictions. Loose ends:**
1. ✅ directionality resolved (symmetrized dist + relation-direction features).
2. ✅ propagation = refinement BEFORE entity alignment — coherent.
3. **🔒 τ (KG mechanism ENTITY types) ≠ DDI output classes — keep SEPARATE NAMES everywhere in impl** (recurring conflation).
4. exact-on-retained-support vs full-KG wording must be consistent.
5. undirected-vs-directed head depends on DATA (verify).
6. β anchor scope = anchors by default; local expansion = ablation not uncertainty.
7. pretraining = transductive-KG/inductive-DDI unless stricter split imposed.

**Q3 — 🔍 VERIFY IN CODE/DATA before implementation (ordered by design impact):**
1. **Do fragments help when KG sparse/incomplete?** (A3 — paper identity).
2. Do S2 cold drugs have rich/sparse KG neighborhoods? (FragAlign+pretrain value).
3. Is DDI label directed or undirected in our data? (head symmetry).
4. Typical a-b corridor length + empty-corridor rate at ℓ=3,4? (features usable at all).
5. What exactly is v1.6 — 1-hop/multi-hop? unimodal/multimodal? true S2 number? (ablation ladder + novelty).
6. KG entity type taxonomy for τ (count, semantics, imbalance).
7. Support-size distribution under cap (tractability + pruning distortion).

**Q4 — 🔒 PHASED BUILD PLAN (falsify riskiest first, each phase has go/no-go):**
- **Phase 1 (minimal falsification):** symmetrized retrieval + explicit s_τ, s_{τ,τ'} + MLP/XGBoost head. GO/NO-GO: does explicit structure ALONE beat/match v1.6 & challenge EmerGNN? Fail badly → STOP (thesis wrong or retrieval broken).
- **Phase 2 (KG structural backbone):** + role-aware head + hub-control tuning + shallow refinement + typed channel fusion. GO/NO-GO: materially > Phase 1 & v1.6? Negligible → complex multimodal path likely not worth it.
- **Phase 3 (FragAlign core):** + fragment encoder + grounded prototypes + α,β,γ + KG-gate/fallback. **RUN STRATIFIED KG-MASKING DE-RISK EXPERIMENT HERE.** GO/NO-GO: fragments help on sparse-corridor/masked-KG? No → downgrade to KG-core + molecular fallback framing.
- **Phase 4 (pretraining):** + masked mechanism-completion + finetune w/ aux alignment. GO/NO-GO: improves cold-start under KG incompleteness/low-label? Marginal w/ full KG → keep but frame conditionally.
- **Phase 5 (optional contrastive):** + counterfactual corridor-ablation loss. GO/NO-GO: hard-neg precision/calibration? Small/unstable → CUT from main.
- **Phase 6 (cross-task transfer):** reuse backbone across binary/multi-cls/multi-label; linear-probe diagnostic.

**Q5 — 🔒 FREEZE VERDICT: YES, ready for implementation. No major architecture contradictions; remaining uncertainties are EMPIRICAL not conceptual.**

**🔒 LEAD-LEVEL DECISIONS — ALL LOCKED 2026-06-22:**
1. ✅ **Paper identity:** design to survive BOTH outcomes, do NOT lock multimodal headline before the KG-masking de-risk experiment (Phase 3).
2. ✅ **Protocol:** transductive-KG/inductive-DDI as MAIN (same as EmerGNN), explicitly stated; stricter fully-inductive as auxiliary if feasible.
3. ✅ **Task emphasis:** multi-label = flagship showcase; binary = benchmark comparison.
4. ✅ **Fine layer = C** — keep shallow (H≤2) refinement propagation (satisfies "subsume NBF"). [B noted as sharper but not chosen]
5. ✅ **Contrastive = ablation** (demoted from lead's original core-pillar framing).

---

## DESIGN PHASE COMPLETE — 14 rounds (2026-06-22)

**Major pivots during debate (not in original sketch):**
- R2: K-separate-NBF-reasoners → ONE shared propagation + typed multi-head readout (tractability).
- R3: novelty is NOT a better propagator; it's the mechanism-preserving explicit pairwise structural front-end + typed readout. Stop comparing fine layer to EmerGNN as a novel propagator.
- R4: fine propagation DEMOTED to shallow semantic refinement (H≤2); cross-type made an EXPLICIT feature s_{τ,τ'}, not propagation-discovered.
- R8: distances on SYMMETRIZED KG (not directed); direction = role-asymmetry + relation-direction features. Fragment-only fallback now mandatory.
- R9: pretraining = masked mechanism-COMPLETION (conditionally essential, not automatic).
- R10: contrastive demoted to ablation.
- R12: theory = prove C1 (separation prop) only; C2/C3 empirical.
- R13: A3 (fragments may not beat KG) = headline rejection risk; stratified-KG-masking = the de-risk experiment.

**Next:** implementation per phased plan, gated by go/no-go. Design and implementation kept SEPARATE per lead instruction.

---

## HYPER-EDGE BIOLOGICAL MOTIVATION — LOCKED (2026-06-23, codex 2 rounds, thread 019ef2c2/019ef2c3)

Reviewer-defensible (codex confirmed). Three claims, all NARROWED to minimal forms:

**M1 — COARSE (group same-subtype + position-agnostic):**
- Coarse unit = per mechanism **SUBTYPE**, operationalized by **relation bucket** (enzyme / transporter / target / pathway / effect; the primary carrier — node-type is incidental) plus coarse endpoint type where available. NOT broad PK/PD, NOT raw node-type alone.
- Pool same-subtype shared mediators as **category-level convergence evidence**; entity identity preserved within subtype by the learned pooling (not erased).
- Hop distance treated as a **CONFOUNDED proxy** (mixes biology with annotation depth / ontology), so NOT used as a monotone strength signal; pooled position-agnostically within a small hop limit for robustness to incomplete cold-drug annotation.
- CAVEAT to state: this is an **operational coarse-graining induced by available KG relations**, NOT a claim that direct=distal mechanisms are equivalent NOR a fully-resolved biological role taxonomy (e.g. protein+target still mixes receptor/ion-channel/enzyme-target).
- ❌ DELETE "distance is a curation artifact" → ✅ "distance is a confounded proxy".

**M2 — FINE (cross-type relations between hyper-edges, NOT molecular):**
- Cross-type coupling is an important **SUBSET** of DDIs (esp. PK perturbation modulating PD risk; distinct upstream layers converging on a shared phenotype). Modeling relations between typed mediator sets MAY recover mechanisms missed by isolated same-type sharing.
- ❌ do NOT claim "dominant" / "many" without data → quantify empirically (Phase B).
- Implemented by cross-type co-path s_{τ,τ'} + routing W_T.

**M3 (NEW) — molecular fragment ↔ hyper-edge alignment, coarse + fine:**
- Fragments provide **imperfect but transferable PRIORS** over mechanism categories + candidate mediator compatibility for cold drugs; SOFT, AUXILIARY alignment to **denoise** typed mediator sets — NOT literal fragment-to-target binding.
- fragment→mechanism-**TYPE** (coarse, transferable) = primary claim; fragment→specific-**ENTITY** (fine, selective) = secondary, must be empirically supported.
- ❌ "structural cause" overstated (mechanism depends on 3D conf/ionization/stereochem/exposure/metabolites/context); BRICS crude (not pharmacophores).

Weakest→strongest: M3 > M2 > M1. All pass with the above narrowing.

### PHASE-B EMPIRICAL VALIDATION (2026-06-23, 1848 both-unseen positive pairs, AND support, codex thread 019ef2c8)

Ran `analyze_spmn_v1_path_distribution.py`. Data forces corrections:

**M1(i) — FORCE-NARROW.** 95.8% of shared mediators are 2-hop → NO relation bucket. So relation-bucket is NOT the primary subtype carrier (kills the Round-2 reconciliation). **NODE-TYPE + multi-hop context is the bulk carrier; relation buckets refine only the ~4% 1-hop minority.** "mean 4.8 subtypes/pair" overstated (counts the 2hop category). Corrected: positive pairs are supported by **heterogeneous mediator NODE-TYPE categories + multi-hop context**, not direct relation buckets.

**M1(ii) — SUPPORT (strength + liability → strengthens molecular necessity).** Only 1.6% of shared mediators are direct (1,1) common neighbors; 93.2% at (2,2). → multi-hop reachability captures ~60× more than classic 1-hop CN (strong result). BUT the (2,2) bulk is low-specificity downstream convergence → **this is EXACTLY why molecular pruning is necessary, not just helpful**: reachability gives an abundant but low-specificity candidate set; the molecular module selects the few mediators that align with drug structure. "Reachability necessary but insufficient without pruning."

**M2 — SUPPORT (as semantic heterogeneity, NOT mechanistic purity).** 80.9% of co-paths are cross-type (off-diagonal) over 8015 instances → can now claim cross-type co-occurrence DOMINATES corridor structure. BUT top pairs (disease→protein_gene, side_effect→protein_gene, side_effect→disease) are semantically MIXED, not clean protein→effect chains. State as "cold-start DDI evidence is typically semantically mixed/cross-type," NOT "clean mechanistic cross-type chains."

**What the data KILLS:** (a) relation-bucket as primary subtype carrier in the bulk; (b) direct common neighbors as the main shared-support pattern.

**Net:** locked motivation SURVIVES in narrower form — **multi-hop, semantically mixed, mostly node-type-level support that NEEDS molecular pruning to recover specificity.** This data-grounded "reachability abundant-but-noisy → molecular pruning necessary" is the strongest version of the multi-modal-necessity argument.

### 🔒 FINAL LOCKED MOTIVATION (codex round 2, thread 019ef2cc) — data-grounded, reviewer-defensible

Node-type bulk: disease 17.4% / protein_gene 16.6% / compound 13.4% / side_effect 12.2% / bio_process 12.0% / anatomy 11.4% / mol_function 6.6% / pathway 4.5% / ... → heterogeneous, **32.6% generic biology** (anatomy+bio_process+mol_func+cell_comp). Per-pair specific relation subtypes: mean **3.51**, only 3.8% zero-specific.

- **M1 COARSE (SUPPORTED):** Shared support is (a) overwhelmingly multi-hop (1.6% direct CN) and (b) heterogeneous across mediator NODE-TYPE categories (~1/3 generic biology); relation-level subtypes (enzyme/transporter/target/effect) present ~3.5/pair but **primarily through the 1-hop minority** (not "only on" — association not exclusivity). Pool same-node-type as coarse category, position-agnostic (hop = confounded proxy), relation subtype as refinement where available.

- **🔑 NECESSITY (SUPPORTED, corrected wording — non-circular):** The data establishes that **KG reachability is abundant but low-specificity** (93% at 2-hop, 1/3 generic biology), so reachability ALONE cannot isolate the mechanistically relevant mediators. ⟹ "**An AUXILIARY SPECIFICITY SIGNAL is necessary; molecular fragment alignment is our chosen instantiation of it.**" (NOT "molecular pruning is necessary" — data shows insufficiency of reachability, not uniqueness of molecular as the denoiser.)

- **M2 FINE (SUPPORTED):** a-b corridors dominated by **cross-type co-occurrence (80.9% off-diagonal)** → cold-start DDI evidence is typically **semantically mixed** across mediator types, motivating modeling relations BETWEEN typed hyper-edges. NOT ordered mechanistic chains (co-occurrence/corridor composition, not validated pathway).

**Honest limitations (state in paper):** (M1) heterogeneity measured at corpus level, not per-pair guaranteed; (necessity) shows reachability insufficient, not molecular uniquely valid; (M2) off-diagonal = mixed-type co-occurrence, not causal ordering.

**Optional PROOF analysis (not required to lock motivation, → experiments):** compare mediator ranking/enrichment by molecular alignment vs a NON-molecular denoiser within the reachable set — would upgrade "auxiliary signal needed (motivated)" to "molecular specificity proven."

---

## FACTS VERIFIED against code/data (2026-06-22)

1. **v1.6 is 2-hop, not 1-hop.** `M = N_≤2(a) ∩ N_≤2(b)` typed common neighbors on merged KG; 12 types (11 kind groups + "other"); EmerGNN backbone (Morgan-FP seeded → mol-as-INIT, NOT fragment multimodal). **S2 AUC = 0.779** (seed42 ndim32, `Code/runs/2026-06-04_21-22-40__run_pmp_v1_6...`). [precompute_pmp_cache.py:1-32]. ⟹ CORRECTION to R2: new model's increment over v1.6 = cross-type co-path `s_{τ,τ'}` + FragAlign, NOT "1-hop→multi-hop".
2. **KG taxonomy:** 178,029 nodes / 7,099,528 edges; `kind` → 12 groups (gene/protein, side_effect, disease, pathway...); **59 base relations (117 w/ inverses)**. Schema: nodes `id/kind`; edges `src/dst/relation/directed/src_kind/dst_kind`. [analyze_ddi_merged_kg_structure summary.json]
3. **Cold-drug KG degree:** mean 166, median 85, max 1476, **3.4% (65 drugs) degree=0**.
4. **Corridor existence (POSITIVE pairs):** dist2 76.5%, dist3 14.9%, ≥4/disconnected 8.0%, ≤3 connected 92%. Common-nbr: 77% pairs ≥1 (median 6). ⚠️ pathway-type shared = 0%; signal in gene/protein (57.8% ≥1). ⟹ **ℓ=2~3 sweet spot; pathway type effectively dead.** CAVEAT: these are positive-pair stats; random/negative pairs likely sparser.
5. **Support size / hub:** 2-hop common-nbr mean 27/p90 83/max 1280; 3-hop walks mean 431/p90 1231/**max 19342** → hub control mandatory (validates R7 degree-penalized top-K).
6. **Labels:** binary = SYMMETRIC; **K=80 multi-cls/label = DIRECTIONAL** (templates "metabolism of [subj] decreased when combined with [obj]"). [ddi_type_map_v3_0.json] ⟹ VALIDATES R6/R8 role-aware design; directionality is a real need since multi-label is flagship.
7. **Fact "do fragments help" (A3) = NOT verifiable now** — it is the Phase-3 experiment outcome. Structural facts confirm a real sparse-corridor population exists (3.4% isolated + 8% disconnected) where fragment fallback could matter.

**Two-KG note:** `data_utils/kg.py` KnowledgeGraph = simple drug→5-entity-type bipartite (enzyme/target/transporter/carrier/pathway), used by baseline eval harness. `Code/data/KG/_merged_kg/` = full 178k multi-hop KG (12 types, 59 rel), needed for corridor/co-path features. New model needs the MERGED KG; fair-comparison harness uses data_utils — split comparability is a Phase-1 item.

---

## IMPLEMENTATION — Phase 1 (started 2026-06-22)

Decisions locked this session: only merged KG; EmerGNN backbone DROPPED (clean less-noisy thesis, v1.6 floor approximate); SMILES → BRICS slices → per-slice atom-graph GNN (Phase 3); v1.6-as-floor wiring (additive residual + zero-init gates) for the neural model; XGBoost = side-diagnostic + A1 baseline, NOT the model head.

**Modules built + validated (`Code/my_code/models/spmn_v1/`):**
- `retrieval.py` — symmetrized merged-KG (dedup nnz=9,675,024, max degree 17,369), vectorized bounded BFS + per-drug neighborhood cache, corridor support `d_sym(a,v)+d_sym(v,b)<=l_max` with degree-penalized per-type top-K + global cap. l_max=3 = 0.001s/pair (cached). codex-reviewed, all findings fixed.
- `struct_features.py` — exact `s_τ` (uncapped) + directional co-path `s_{τ,τ'}` (bounded-walk semantics, R7). **Co-path verified against independent brute-force (exact match).** 2.8ms/pair l_max=3.
- `Code/scripts/run_spmn_v1_phase1.py` — go/no-go probe (GBDT on explicit features, same 800-drug seed42 pkl as v1.6).

**🔑 Data fact corrected:** v1.6's 0.779 is on the **800-drug** `coldddi_legacy/800drug/seed42.pkl` (NOT 1900). Phase-1 uses the same pkl. All 799 drugs present in merged KG.

**PHASE-1 RESULT (l_max=3, s_τ + sym co-path, sklearn HGB):**
- test_s2 AUC = **0.7053** (val_s2 0.6945), AUPRC 0.7117. feat_dim=104. vs v1.6 **0.779**, EmerGNN ~0.710.
- **Stratification (the key finding):** empty-or-OOV support = 14.4% of test, and these are EASY (pos_rate 0.236 ⟹ "no corridor → likely no DDI", model exploits correctly). NONEMPTY-support pairs (85.6%) AUC = only **0.6810** — that's the hard regime.
- **Interpretation:** explicit structural COUNTS carry real cold-start signal (0.705 >> 0.5, retrieval/features validated end-to-end), but do NOT alone match v1.6. The gap is on nonempty-support pairs where counts are too coarse — exactly what v1.6's LEARNED mediator-embedding pooling captures. ⟹ motivates Phase 2 (neural coarse head = learned pooling over mediator embeddings, v1.6 is its special case).
- **A1 reviewer baseline established:** GBDT-on-explicit-features = 0.705. Neural model must beat this AND v1.6's 0.779.
- **A3/fragment target refined:** empty-support POSITIVES (~130 test pairs that interact but have no KG corridor) are the precise population fragments must recover (Phase 3), not "all empty support" (most of which are correctly-negative).

**CO-PATH ABLATION (the cross-type novelty over v1.6):**
- s_τ only (feat_dim=26): test_s2 AUC = **0.6878**.
- s_τ + sym co-path (feat_dim=104): **0.7053** (+1.75 pts). ⟹ cross-type co-path `s_{τ,τ'}` adds real signal over per-type common neighbors (v1.6 only has the latter). Small but positive and consistent (val also +0.8).

**🔒 PHASE-1 GO/NO-GO VERDICT = GO.**
- Explicit zero-param structure carries clear cold-start signal (0.705 >> 0.5); retrieval + features validated end-to-end on real data (co-path brute-force-exact).
- Nearly matches EmerGNN (0.705 vs ~0.710) with ZERO learned KG parameters — supports the "explicit structure competitive with noisy propagation" thesis.
- Does NOT match v1.6 (0.779) — expected, since v1.6 adds LEARNED mediator-embedding attention pooling that raw counts lack. This is precisely Phase 2's job.
- Comparison caveat: EmerGNN 0.710 was 1900-drug; should re-run EmerGNN on this 800-drug seed42 split for a strict same-split number.
- **Next: Phase 2** = neural coarse head (learned attention pooling over mediator embeddings + routing), wired so v1.6 is a recoverable special case; target = beat 0.705 (A1 baseline) and close on / pass v1.6 0.779.

### Position decision (2026-06-22, lead): NO absolute position; keep relative order only.
- **Rationale (lead):** absolute hop-distance is scale-inconsistent across pairs / KG-density regions (hub-dense areas → smaller hops), so as a feature it is more likely noise than stable signal.
- **KEEP:** relative order = directional co-path `s_{τ,τ'} ≠ s_{τ',τ}` (a-side vs b-side); this is mechanism-meaningful (directional DDI) and cross-pair comparable.
- **DROP:** absolute-position features — single ℓ only (NO multi-ℓ length banding, NO (d_a,d_b)-bucketed co-path counts).
- **Distances still used as the reachability GATE** (`d(a,u)+d(u,v)+d(v,b) ≤ ℓ`) — that is reachability, not a position feature; unaffected.
- **CONSEQUENCE for Phase 2 (frozen step 5 revision):** the per-node positional encoding `p(v;a,b)=MLP([d_a,d_b,d_a+d_b])` injected absolute distances into the learned refinement → DROP it for consistency. Replace with at most a RELATIVE side indicator (e.g. sign/argmin of d_a vs d_b, "closer-to-a vs closer-to-b") if a positional cue is needed; node_init otherwise = entity_embed ‖ type_embed (‖ optional relative-side flag). Revisit only if Phase 2 ablation shows the relative cue helps.

## IMPLEMENTATION — Phase 2 (neural coarse head, KG-only binary)

`Code/my_code/models/spmn_v1/coarse_head.py` — `SPMNCoarseHead`:
- entity_embed (over all 178029 KG nodes, only non-drug ids gathered → inductive-safe) + type_embed; `phi(m)=MLP([entity_embed, type_embed])` (NO absolute position, per decision).
- per-(pair,type) attention pool (pyg_softmax over group = pair*K+type, scatter_add) → c[B,K,d] — learned pooling over each hyper-edge's members.
- routing `c̃[b,k]=Σ_j softmax(W_T)[k,j] c[b,j]` (K×K, v1.6-style).
- head: `MLP([c̃.reshape(B,K*d) ‖ struct_feats])` → logit. Fuses LEARNED pooling + EXACT structural features. v1.6 ≈ representable subset (pooling+routing path); Phase-1 ≈ subset (struct path).
- `Code/scripts/run_spmn_v1_phase2.py` — one-pass precompute+cache of ragged supports + symmetric struct vectors (104-d, shared `symmetric_binary_vector` in struct_features.py); GPU training (Adam, BCEWithLogits, TrainProgress, best-by-val_s2-AUC), eval test_s2.

**codex review (thread 019ef09c):** pooling + routing math confirmed correct. Handled in code: best_state deep-clone ✓, float32/long dtypes ✓, eval sigmoid vs BCEWithLogits ✓, empty-support pairs safe (no NaN). **Leakage check: edges are `__mask1` (DDI removed from KG) → features use only protein/enzyme/pathway connectivity, no label leakage; transductive-KG/inductive-DDI as accepted.** ONE actionable flagged: raw struct counts (s_τ/co-path reach 10s–100s) may swamp the ~O(1) routed channels → standardize struct on train-stats if Phase-2 underwhelms.

### 🔑 DESIGN RESOLUTION — v1.6 is FACETS of one hyper-edge incidence (not a bolt-on)

Lead intent clarified: SUBSUME v1.6 into the hyper-edge framework, then extend with molecular. Unified concept: **a hyper-edge = an ATTRIBUTED higher-order incidence between drug pair (a,b) and a set of typed mediators, with a MARGINAL (per-drug) / JOINT (per-pair) structure, read DIRECTIONALLY.** v1.6's three pieces map onto facets:
- relation features rel(a→m),rel(m→b) = member ATTRIBUTES of the incidence (intrinsic). Multi-hop → relation-sequence/bucket histogram (more natural than v1.6's crude REL_2HOP token).
- per-drug affinity `affinity[d,k]=log1p|N_τ(d)|` = the MARGINAL incidence; gates the JOINT pooling (marginal/joint pairing, like CN normalized by degree). NOTE: tested affinity as a CONCAT FEATURE = refuted; v1.6 uses it as ATTENTION ROUTING — different, the right form.
- a/b-separated evidence = DIRECTIONAL readout of the (symmetric) joint hyper-edge; composes with directional cross-type co-path.
- v1.6's free W_T (K×K) is REPLACED/grounded by the structural cross-type co-path `s_{τ,τ'}` (W_T optional learned smoothing).
- Molecular = ANOTHER member attribute (fragment↔mediator alignment), gating the same incidence. Natural, not a new structure.

**Pretraining arc is natural (verified):** pretraining target = per-drug MARGINAL incidence — the SAME object the supervised model gates on (affinity). FragAlign is the shared module (pretrain: fragment→marginal incidence completion; finetune: fragment→joint hyper-edge alignment). Masking = set-member dropout on explicit incidence sets. Marginal→joint composition (`A_τ=N_τ(a)∩N_τ(b)`) IS the cold-start fragment fallback. Order: unified head (subsume v1.6) → Phase 3 molecular attribute → Phase 4 marginal-completion pretrain → finetune. Monotone layering, no rebuild.

**PHASE-2 RESULT v1 (simple pool, d=32, Adam, raw struct, 60 ep):** test_s2 AUC = **0.7307** (best val 0.7397), AUPRC 0.7386. Beats Phase-1 GBDT 0.705 (+2.6pts → learned pooling helps), below v1.6 0.779.
- **🚩 SEVERE OVERFIT:** val AUC peaks at EPOCH 1 (0.7397) then monotonically degrades (ep30 0.699, ep60 0.677) while train loss falls 0.64→0.46. The 5.7M-param entity-embed table memorizes train mediator patterns that don't transfer to unseen S2 pairs. best-ckpt-by-val saved the reported number but model barely trains before overfitting → real headroom if regularized.
- **Fixes applied:** (1) standardize struct on train-stats (codex), (2) Adam→AdamW (decoupled wd). Running reg sweep: wd∈{1e-2,1e-1}, dropout 0.5, d∈{16,32}.
- Phase-2 verdict so far: GO direction (learned pooling > counts), gap to v1.6 is a regularization problem not architecture.

### 🚨 GAP ROOT CAUSE FOUND — support-definition BUG (2026-06-22)

Tried to close the v1.6 gap; REFUTED in sequence: regularization (sweep plateaus 0.734–0.737), Adamic-Adar weighting (GBDT 0.700 / NN 0.733 flat), affinity-as-feature (0.696), affinity-as-routing+a/b-sep incidence head (0.728, WORSE — K-channel→2d info bottleneck without relation features). All cheap fixes failed.

**Then found the real cause: my support was a MISIMPLEMENTATION of my own design.**
- R2 defines `A_τ^(ℓ)(a,b) = R_τ^(ℓ)(a) ∩ R_τ^(ℓ)(b) = {u: dist(a,u)≤ℓ ∧ dist(b,u)≤ℓ}` (AND / intersection).
- I implemented `build_pair_support` as `{u: d_a+d_b ≤ l_max}` (SUM budget) — a DIFFERENT, much smaller set that DROPS the (2,2) mediators (2 hops from both).
- Decisive test (GBDT, 30k train sample, test_s2): SUM → mean support **87**, AUC **0.712**, protein_gene **0.628**; AND `{d_a≤2 ∧ d_b≤2}` (= v1.6 N_≤2∩N_≤2, = correct A_τ^(2)) → mean support **1430**, AUC **0.741**, **protein_gene 0.743 (+11.5pt)**.
- ⟹ I was pooling ~6% of v1.6's mediators. The (2,2) layer (shared deeper entities: pathways/processes/2-hop proteins) carries the protein_gene discriminative signal; 1-hop common neighbors are hub-dominated noise. This fixes the worst failure stratum and is very likely THE main gap (NOT relation features, NOT regularization).
- ALL prior SPMN numbers (Phase-1 0.705, Phase-2 0.734) were on the buggy 6%-size support.
- FIX: `build_pair_support(support_and=True)` added (retrieval.py); default going forward = AND (the correct A_τ^(ℓ)). `--support-sum` kept to reproduce the bug for ablation. co-path s_{τ,τ'} unchanged (it's a bounded-WALK feature, a different object from the AND intersection — consistent with R2/R7).
- Rebuilding full pipeline (cache `v3and`) + rerunning GBDT/NN; v1.6 reproduction also running (~40min) for an env-comparable 0.779 target.

### RESULTS on corrected AND support (2026-06-22)

- **Simple pool + AND support + co-path** (coarse_head): test_s2 AUC **0.7720**, AUPRC **0.7949**. vs buggy-support 0.734 (+3.8pt). ≈ v1.6 (0.779 / 0.7943) — AUPRC slightly ABOVE. **Support fix was the whole gap.**
- **Incidence head (affinity routing + a/b sep) + AND support**: test AUC **0.7728**, AUPRC 0.7928 — **IDENTICAL to simple pool**. ⟹ on the correct support, v1.6's affinity routing + a/b separation add NOTHING. The hyper-edge framework matches v1.6 with LESS machinery.
- **Co-path KG-only ablation** (AND support, GBDT): s_τ only 0.7345, +AA (no co-path) **0.7417**, +co-path 0.7373 (slightly WORSE). ⟹ **cross-type co-path has NO KG-only net gain** — confirms lead's hypothesis: co-path's value is realized only when fragments align to the cross-type channel (γ: fragment-pair→(τ,τ')), per the lead's diagram. Co-path's role = scaffold defining cross-type channels for MOLECULAR alignment (Phase 3), NOT a KG-only feature. AA gives a small gain (+0.7pt).

### 🔒 PHASE-2 CHARACTERIZATION (clarified to lead)
Phase-2 (KG-only) = **a clean equivalent of v1.6**, NOT yet an upgrade: same hyper-edge core (AND-intersection typed common neighbors + attention pool + K×K routing), v1.6's affinity/a-b-sep routing DROPPED (shown redundant on correct support), plus explicit structural variables (s_τ, AA) whose net KG-only gain is small, plus co-path (no KG-only gain — reserved for molecular). It MATCHES v1.6 (~0.772 ≈ 0.779) with less machinery. **It is the geba (foundation), not the novelty.** The novelty = molecular fragment↔mechanism alignment (Phase 3), which must beat this ~0.772 floor. Components shown REDUNDANT on correct support: affinity routing, a/b separation, co-path-as-KG-feature.

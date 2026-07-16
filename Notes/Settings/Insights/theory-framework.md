# Theoretical Framework: Cold-Start σ-Field Sufficiency

> **Master concept**: cold-start relational prediction is generalization through a *stable, sufficient mediator σ-field* under anchor-support shift. Three architectural obstructions are special cases of "predictor measurable on a too-coarse σ-field".

Co-developed with codex (gpt-5.4) over 4 rounds. This file is the consolidated theory output; subsequent paper drafting will hang off this structure.

---

## 0. Lead paragraph (theory section opener)

> In cold-start relational prediction, generalization is possible only through a query-conditioned σ-field of evidence that is both **stable** across the train/test entity-support shift and **sufficient**, up to slack, for the target. Meeting-node retrieval with external semantic priors is designed to operate on such a σ-field; anchor-dependent representations, source-conditioned flow endpoint readouts, and attention-only selectors are not, and therefore admit target-risk lower bounds independent of sample size.

---

## 1. Master formal object

**Stable-Mediator Cold-Start Problem**: `Π = (Q, M, Z, A, P_tr, P_te, Y)`.

- `Q`: query space (entity pairs `(u, v)` or k-tuples)
- `M(q)`: candidate mediator/evidence set
- `Z(q, c)`: stable features of candidate `c ∈ M(q)` — measurable at test time without training-pair support (e.g., PubMedBERT name embeddings, typed KG local features)
- `A(q, c)`: anchor features — informative in training but unsupported / shifted at test time (e.g., learned drug-ID embeddings, pair-trained drug embeddings)
- `P_tr, P_te`: train/test distributions over queries and evidence
- `Y(q)`: target label

**Cold-start regime**: `supp(P_te^A) ⊄ supp(P_tr^A)` (or weaker, badly shifted) while `Z` remains measurable and approximately stable.

**Stable sufficiency**: `Y` is approximately measurable w.r.t. `σ(Z_{W*})` for some witness subset `W* ⊆ M(q)`, up to slack `(η, κ)`.

**Domain applicability**:
- DDI: drugs as entities; mediators = shared enzymes/transporters/effects/phenotypes
- Recsys cold-start: user-item; mediators = content attributes, stereotype clusters
- Inductive KGC: head-tail; mediators = shared neighbors, typed local evidence
- DTI w/ unseen targets: drug-target; mediators = pathways, functions, phenotypes
- Few-shot relational learning more broadly

---

## 2. Lead theorem — Cold-Start Sufficiency Principle

**Theorem (positive)**. For `Π` with witness selector `W_hat(q)` measurable on stable features only, stable scorer, approximate sufficiency of `σ(Z_{W*})` for `Y`, and bounded witness-identification error:

```
R_te(h) ≤ transfer(Z) + identification(W_hat) + estimation(selected Z)
```

**Theorem (impossibility)**. If a method's prediction is measurable only w.r.t. a σ-field `G` strictly coarser than the stable sufficient σ-field needed for `Y`, there exist cold-start target environments with identical `G`-observations but different labels/witnesses, yielding a constant target-risk lower bound independent of training sample size.

Specializations:
- `G = σ(S)` vs unsupported anchor fields → C1
- `G = σ(path signatures up to L, endpoints)` → C2
- `G = σ(training-supported internal selector features)` → C3

---

## 3. Specialization I (C1) — Transfer under approximate stable sufficiency

### Setup
Decompose observed features `X = (S, B, U)`:
- `S`: stable mediator features
- `B`: identifiable anchor features (test support covered, or adaptively estimable)
- `U`: unsupported anchor features

For the pure meeting-node theorem, set `B = ∅`, `Z = S`.

### Assumptions
- **A1 (stable-label representation)**: `E_{Z ∼ P_te^Z} |μ_te(Z) − μ_tr(Z)| ≤ κ`
- **A2 (approximate mediator sufficiency)**: `E_{(Z,U) ∼ P_e} |η_e(Z, U) − μ_e(Z)| ≤ η_e` for `e ∈ {tr, te}`
- **A3 (adaptable feature shift)**: `d_{HΔH}(P_tr^Z, P_te^Z)` finite/small

### Theorem (C1)
For every `h ∈ H_Z`:
```
R_te(h) ≤ R_tr(h) + ½ d_{HΔH}(P_tr^Z, P_te^Z) + λ_Z + η_tr + η_te + κ
```
where `λ_Z := inf_{h ∈ H_Z} (R_tr^μ(h) + R_te^μ(h))` is the joint surrogate-Bayes error.

### Proof skeleton
1. Surrogate problem: `Ȳ_e | Z ∼ Bernoulli(μ_e(Z))`; surrogate risk `R_e^μ(h)`.
2. True-vs-surrogate gap: `|R_e(h) − R_e^μ(h)| ≤ E|η_e(Z, U) − μ_e(Z)| ≤ η_e` (Bernoulli TV).
3. Apply Ben-David `HΔH` adaptation on `Z`.
4. Account for `μ_tr ≠ μ_te` via `κ = E|μ_te(Z) − μ_tr(Z)|`.
5. Combine.

### Corollary (Bayes-gap)
`R_te(h_Z*) − R_te(h_{Z,U}*) ≤ η_te`. (Bayes error `min(p, 1−p)` is 1-Lipschitz in `p`.)

### Necessity / two-point impossibility
Set `U = 0` a.s. in source; `U = 1` w.p. `α` in target. Target environments `Q_+, Q_-`:
- `Q_+`: `Y = Z` when `U = 1`
- `Q_-`: `Y = 1 − Z` when `U = 1`

Both agree on source-observable events and on `P(S, Y)`. Any source-trained learner has identical output law under both; one of `{Q_+, Q_-}` incurs error ≥ `α/2`.

**Interpretation**: anchor-dependent predictors are not uniformly identifiable under cold-start support shift. The bound `≤ η_te` is the price paid for the slack; without it, error is `Ω(α)`.

### Failure modes / fallbacks
- κ small not provable: carry κ explicitly; argue (not prove) PubMedBERT + KG-typed features make it small.
- `η_e` unobservable: structural parameter, not estimand; sensitivity analysis acceptable.
- "Just domain adaptation": novelty is the decomposition `S/B/U` + impossibility under unrestricted `U`.

---

## 4. Specialization II (C2) — Flow blind spot as coarse path-signature σ-field

### Setup
For `L ≥ 1` and pair `(u, v)`, define **relation-labeled path signature** `Sig_L(G; u, v)` = multiset of all relation sequences `(r_1, ..., r_t)`, `t ≤ L`, that connect `u → v` in `G`.

Shared mediator set for relation pattern `(r, s)`: `I_{r,s}(u, v) = { m : (u, r, m) ∈ E, (m, s, v) ∈ E }`.

**EmerGNN-style class** `H_E^L`:
- Pair-conditioned relation gating: `α_r(u, v)` depends only on `(f_u, f_v, r)`
- Source-rooted init: `h_x^(0) = 0` for `x ≠ u`; `h_u^(0) = T_0(f_u)`
- Relation-only propagation: `h_t^(ℓ+1) = Σ_{(s,r,t) ∈ E} F_r^(ℓ)(h_s^(ℓ); α_r(u, v))` — no node-instance feature of `t` or `s` beyond what's in `h_s^(ℓ)`
- Endpoint readout: `score(u, v) = D(h_v^(L), f_u, f_v)`

**Meeting-node class** `H_M = { h(u, v) = ρ(Σ_{m ∈ I_{r,s}(u,v)} φ(x_m)) }` where `x_m` = mediator node-instance features (e.g., PubMedBERT).

### Assumptions
- **E1 (load-bearing)**: intermediate node-instance features are not injected during propagation except through the source-rooted hidden state. Verified for EmerGNN: message `φ(h, h_r) = α_r · (h ⊙ h_r)` uses only `h` (source-recursive) and relation embedding `h_r`, never `x_m`.
- **E2**: ∃ mediators `a, b` with same structural role under `(r, s)` but `x_a ≠ x_b`
- **E3**: target depends on mediator semantics, `f*(u, v, G) = Σ_{m ∈ I_{r,s}} g(x_m)`, `g` non-constant

### Theorem (C2 — path-signature invariance)
For every `h ∈ H_E^L`, ∃ `Ψ_h` such that
```
score_h(u, v, G) = Ψ_h(f_u, f_v, Sig_L(G; u, v))
```

### Corollary (strict separation)
Fix `L = 2`. Construct `G, G'` with same endpoints, same `Sig_2`, but `f*(G) ≠ f*(G')`. Concrete witness: `G` has mediator `a` with `u -r→ a -s→ v`; `G'` has mediator `b` with `u -r→ b -s→ v`; both have signature `{(r, s)}`, but `g(x_a) ≠ g(x_b)`. Every `h ∈ H_E^2` outputs the same score; meeting-node class realizes `f*` exactly with `φ = g`, `ρ = id`.

### Proof skeleton
Induction on layer depth:
- Layer 0: only `u` has nonzero state → all states are functions of `f_u`.
- Inductive step: `h_t^(ℓ+1)` is sum of relation-specific transforms over incoming neighbors; relation-level gating + no new node-instance features → `h_t^(ℓ+1)` depends only on `f_u` and relation-labeled paths `u → t` of length ≤ `ℓ+1`.
- At `t = v` after `L` steps: `h_v^(L)` is a function of `(f_u, Sig_L(G; u, v))`. Decoder adds `f_v`. □

### Why this bites EmerGNN specifically
The obstruction names EmerGNN's choices:
- pair-conditioned gating at **relation-type** level (not edge-instance)
- propagation rooted at `f_u` only
- readout only at `v`
- mediator-node semantics not injected as first-class evidence

### Generalization
Safe statement:
> Any source-conditioned path-flow architecture with endpoint readout whose representation is generated only by endpoint initialization plus relation/edge-type propagation, and which does not inject intermediate node-instance semantics as first-class inputs along the path, is invariant to changes in intermediate node-instance semantics conditional on the same path-signature statistics.

Includes EmerGNN directly. NBFNet's core formulation is source-conditioned + relation-compositional; the C2 obstruction applies to subclasses that omit intermediate node-instance injection. **Do not overclaim "all NBFNet variants"** without inspecting the specific inductive variant compared against.

### Fallback (moment-collision)
If a successor architecture injects per-instance features, weaken to: there exist `G, G'` matching on all path-signature statistics AND mediator-feature sums up to degree `d`, but differing on a higher-order moment. Becomes a finite-dimensional moment-collision theorem.

---

## 5. Specialization III (C3) — Witness identifiability gap

### Setup
Each query `q` has candidate mediator set `M(q)` and true witness set `W*(q)`, `|W*(q)| ≤ k`.

- `S(q, c)`: stable features (PubMedBERT name, typed KG local)
- `A_tr(q, c)`: training-supported internal features (learned drug embeddings, ID-derived pair features)

**Attention-only selector**: `T_att(q) = T((A_tr(q, c))_{c ∈ M(q)})`
**Externally-primed selector**: `T_ext(q) = T((r_ext(S(q, c)), S(q, c))_{c ∈ M(q)})`

### Lower bound (C3-LB): internal attention is non-identifying

**Assumption CI1 (coarse feature experiment)**: ∃ target subfamily and environments `P_+, P_-` such that:
- `Law((A_tr(q, c))_{c ∈ M(q)} | P_+) = Law((A_tr(q, c))_{c ∈ M(q)} | P_-)`
- but `W*_+(q) ≠ W*_-(q)` with positive probability

**Theorem CI-LB**. For any attention-only `T_att`, ∃ prior over `{P_+, P_-}` such that
```
E[Recall(T_att(q), W*(q))] ≤ ½    (single-witness)
E[|T_att ∩ W*|/|W*|] ≤ ρ_0       (general, symmetry baseline)
P[W*(q) ⊆ T_att(q)] ≤ ½          (binary)
```

**Proof skeleton**: total variation between observed experiments is 0 (Le Cam degenerate); any measurable selector has same output law under both; mean recall ≤ ½ by symmetry. □

### Upper bound (C3-UB): margin-based external prior recovery

**Assumption CI2 (margin + calibration)**: ∃ population witness score `r*(q, c)`. External prior satisfies `|r_ext − r*| ≤ δ` uniformly. Witness margin:
```
min_{c ∈ W*} r*(q, c) − max_{c ∉ W*} r*(q, c) ≥ γ
```

**Theorem CI-UB**. If `δ < γ/2`, top-`k` selection by `r_ext` recovers `W*(q)` exactly.

Noisy version: if `P(sup_c |r_ext − r*| > t) ≤ ψ(t)`, then `P[W_hat_ext ≠ W*] ≤ ψ(γ/2)`.

**Proof skeleton**: failure ⇒ some non-witness outranks some witness under `r_ext` ⇒ perturbation > γ/2 somewhere. □

### Margin vs AUROC
- **Theorem**: margin (exact subset recovery is the natural object)
- **Empirical bridge**: AUROC (measurable, ties to E3 PubMedBERT AUC 0.708, but does not imply margin alone)

### End-to-end composition with C1

Stable downstream scorer `h(q) = G({z(q, c) : c ∈ W_hat(q)})`. Sparse-witness margin: `Y(q) = sign(Σ_{c ∈ W*} s_q(c) − τ_q)` with margin `γ_s`, nuisance budget `β`.

```
R_te(h) ≤ P[W*(q) ⊄ W_hat(q)]
       + P[estimation error on selected witnesses > γ_s − β]
       + transfer_terms_C1
```

Where `transfer_terms_C1 = ½ d_{HΔH}(P_tr^Z, P_te^Z) + λ_Z + η_tr + η_te + κ`.

Composes cleanly when both selector and scorer are `Z`-measurable. **Subtlety**: downstream scorer must be restricted to stable selected-witness features, else training-set unstable anchor info sneaks in.

### What's new
- Problem formalization (cold-start witness selection on mediator sets) is new
- Selection impossibility / prior-enabled recovery pair is new in this setting
- Composition with cold-start transfer is new

**Not claimed as new**: sufficiency/Blackwell/Le Cam machinery itself; ranking/AUC calibration theory.

---

## 6. Cross-domain implications (theory remark)

The framework predicts:
> Cold-start is structurally easier in domains where the Bayes-optimal decision is well-approximated by stable mediator evidence and harder where residual anchor dependence is large.

Quantified via `η_tr + η_te + κ` and witness margins.

- **DDI**: plausibly small slack — node-name semantics + KG mediators carry main signal
- **Recsys**: larger slack — collaborative anchor signal dominant
- **Inductive KGC**: intermediate — depends on relation-local semantics informativeness

Testable cross-domain prediction; include as theory remark, not main theorem.

---

## 7. Scope / non-claims

The theory does NOT claim:
- All flow GNNs fail (only those satisfying E1)
- All external priors help (only those satisfying CI2)
- AUC ⇒ witness recovery
- "Just domain adaptation" (the decomposition `S/B/U` + impossibility on unrestricted `U` is the novelty)

Success depends on stable-sufficiency slack (η, κ) and witness margin (γ).

---

## 8. Prior art positioning

- **Ben-David et al. 2010** — `HΔH` domain adaptation backbone for C1
- **Blackwell sufficiency / Le Cam two-point** — classical machinery for C1 necessity + C3 lower bound
- **Deep Sets (Zaheer et al. 2017)** — canonical permutation-invariant set functions; can't claim "set aggregation is new"
- **Maron et al. 2019** — universality of invariant networks
- **Wagstaff et al. 2019** — latent dimension caveats for set universality
- **NBFNet (Zhu et al. 2021)** — source-conditioned Bellman-Ford path flow; C2 generalizes to this class with caveat
- **GraIL (Teru et al. 2020)** — local subgraph inductive bias on unseen entities; "local structure helps induction" is not new; our contribution is sharper (cold-start support-shift + sparse-witness + depth-locality)
- **EmerGNN (Zhang et al. 2023)** — direct comparator; C2's E1 verified against published equations
- **Oono & Suzuki 2020** — deep GNN expressivity collapse; useful background for C2 fallback intuition
- **Topping et al. 2022** — over-squashing / information contraction; W3 fallback citation
- **Xu et al. (GIN)** — MPNN/WL expressivity background

What's genuinely ours:
- Unifying concept: cold-start σ-field sufficiency
- Master object: stable-mediator cold-start problem `Π`
- Three architecture/representation obstructions as coarse-σ-field failures
- Composition into end-to-end target-risk decomposition

---

## 9. Theory section TOC (paper-ready)

1. **Setup** — define `Π`, stable sufficiency, witness sets, anchor-support shift, architecture measurability
2. **Lead Theorem** — Cold-Start Sufficiency Principle (upper + impossibility)
3. **Specialization I (C1)** — transfer under approximate stable sufficiency; Bayes-gap corollary; necessity
4. **Specialization II (C2)** — flow blind spot; EmerGNN-specific corollary; generalized flow class remark
5. **Specialization III (C3)** — witness identifiability gap; attention-only LB; external-prior UB; combined risk
6. **Implications Across Cold-Start Domains** — DDI / recsys / inductive KGC / DTI; comparative prediction via slack
7. **Discussion of Scope** — what is and isn't claimed

**Length estimate** (standard ML theory prose):
- Main paper: 2.5–4 pages compressed
- Supplement: 3–5 pages proofs/constructions

Split: definitions + lead theorem + 1 clean statement per specialization + short proof ideas in main; formal proofs in appendix.

---

## 10. What goes back into the DDI paper

The DDI paper uses this theory section as **motivation + theoretical guarantee** for our meeting-node + PubMedBERT-init + pair-conditional-selection method:

- C1 motivates **why our representation has to be stable-feature-based** (PubMedBERT not learned-ID); this is i4 theorem-level
- C2 motivates **why we don't traverse paths** (we read meeting nodes locally); this is i2 + EmerGNN-boundary theorem-level
- C3 motivates **why selection needs external prior** (pair-conditional with PubMedBERT prior, not vanilla attention); this is i3 theorem-level
- Lead theorem ties i1-i4 architecture choices to a single theoretical principle

E2/E3/E6/E7/E8.6/E8.7-hetero empirical results instantiate the slack terms and the witness identification rate on ColdDDI seed=42 S2 split.

---

## 11. Next steps

- [ ] Draft `Setup` subsection in paper-prose form
- [ ] Draft lead theorem statement + 1-paragraph proof idea
- [ ] Draft each specialization as theorem block + proof sketch
- [ ] Cross-link to i1.md / i2.md / i3.md / i4.md / i2-vs-emergnn.md
- [ ] Verify NBFNet message form against the specific inductive variant we plan to baseline
- [ ] Decide whether cross-domain implications subsection goes in main or moves to discussion

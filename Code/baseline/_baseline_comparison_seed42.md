# Baseline Comparison on Project Data (seed 42)

Consolidates 3 baselines (EmerGNN, HDN-DDI, TIGER) on the **800-drug
DrugBank cold-start PKL** (`Code/data/coldddi_legacy/800drug/seed42.pkl`),
single seed 42. To be used as **comparison baselines for our proposed
method**.

All numbers verified from each baseline's `_results/` doc — no
fabrication; line citations in §6.

---

## 1. Setup (shared across all 3 baselines)

| param | value |
|---|---|
| dataset | 800-drug DrugBank cold-start PKL (`Code/data/coldddi_legacy/800drug/seed42.pkl`) |
| n_drug | 800 |
| n_train (DDI pos) | 53,743 |
| n_test_s0 | 2,986 pos × 2,986 neg = 5,972 (binary) / 2,985 (MC, 1 OOV) |
| n_test_s1 | 15,258 × 15,258 = 30,516 (binary) / 15,139 (MC, 119 OOV) |
| n_test_s2 | 1,919 × 1,919 = 3,838 (binary) / 1,887 (MC, 32 OOV) |
| seed | 42 (single) |

**S0 / S1 / S2** = ColdDDI cold-start ladder
- S0 warm: both drugs in train pool
- S1 half-cold: exactly one drug unseen
- S2 full-cold: both drugs unseen

---

## 2. Binary classification (per-pair, pos vs neg, AUC + F1)

| baseline | epochs | wall time | **S0 AUC** | **S1 AUC** | **S2 AUC** | S0 F1 | S1 F1 | S2 F1 |
|---|---|---|---|---|---|---|---|---|
| HDN-DDI | 100 | ~64 min | 0.6595 | 0.6412 | 0.6201 | 0.5988 | 0.6009 | 0.5886 |
| TIGER (DW, no cold-patch) | 50 | ~7.1 h | 0.8642 | 0.7122 | 0.5842 | 0.7884 | 0.6130 | 0.4220 |
| **EmerGNN (multimode)** | 100 | ~11.6 h | **0.9895** | **0.8328** | **0.7462** | **0.9432** | **0.7727** | **0.6667** |

### Cold-start degradation (S0 → S2)

| baseline | ΔAUC (S0−S2) | ΔF1 (S0−S2) | interpretation |
|---|---|---|---|
| HDN-DDI | -3.9 pt | -1.0 pt | flat; no warm-start advantage at all (mol-graph only) |
| TIGER | -28.0 pt | -36.6 pt | steep crash; BKG channel + mol channel both fail on unseen drugs without cold-start patch |
| EmerGNN multimode | -24.3 pt | -27.7 pt | steep but **starts much higher**, S2 still leads field by 12-14 pt AUC |

### Binary winner by split

| split | winner | margin vs 2nd |
|---|---|---|
| S0 | EmerGNN multimode | +12.5 pt vs TIGER |
| S1 | EmerGNN multimode | +12.1 pt vs TIGER |
| S2 | EmerGNN multimode | +12.6 pt vs HDN |

EmerGNN sweeps. KG-aware inductive propagation is the dominant advantage.

---

## 3. Multi-class classification (K-way DDI type, K=152 observed)

All 3 baselines automatically adjusted from default `n_classes=86` →
observed 152 unique `ddi_type` in train (project taxonomy finer than
paper's 86).

| baseline | epochs | wall time | **S0 top1** | **S1 top1** | **S2 top1** | S0 macro_AUC | S1 macro_AUC | S2 macro_AUC | S0 macro_F1 | S1 macro_F1 | S2 macro_F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HDN-DDI** | 100 | ~29 min | **0.8044** | **0.4330** | **0.2109** | 0.9717 | **0.8616** | **0.6775** | **0.6722** | **0.2648** | **0.0619** |
| EmerGNN | 100 | ~31 min | 0.5591 | 0.3673 | 0.2284 | 0.9488 | 0.8637 | 0.6675 | 0.1478 | 0.0604 | 0.0410 |
| TIGER (DW, no cold-patch) | 50 | ~4.6 h | 0.5310 | 0.3314 | 0.1404 | **0.9501** | 0.8495 | 0.5703 | 0.1044 | 0.0419 | 0.0136 |

### Multi-class winner by metric × split

| metric × split | winner | margin vs 2nd |
|---|---|---|
| top1 S0 | HDN-DDI | +24.5 pt vs EmerGNN |
| top1 S1 | HDN-DDI | +6.6 pt vs EmerGNN |
| top1 S2 | EmerGNN | +1.8 pt vs HDN |
| macro_AUC S0 | HDN-DDI | +2.2 pt vs TIGER |
| macro_AUC S1 | EmerGNN | +0.2 pt vs HDN (~tie) |
| macro_AUC S2 | HDN-DDI | +1.0 pt vs EmerGNN |
| macro_F1 S0 | HDN-DDI | +52.4 pt vs EmerGNN (HDN's predictions much more peaked) |
| macro_F1 S1 | HDN-DDI | +20.4 pt vs EmerGNN |
| macro_F1 S2 | HDN-DDI | +2.1 pt vs EmerGNN |

HDN-DDI dominates on warm + half-cold; **on full cold-start S2,
EmerGNN narrowly leads top1 but HDN leads macro_F1 / macro_AUC**.

---

## 4. Cross-task reading (binary vs MC on the same baseline)

| baseline | binary S0 AUC | MC S0 macro_AUC | binary S2 AUC | MC S2 macro_AUC |
|---|---|---|---|---|
| HDN-DDI | 0.6595 | 0.9717 (+31.2 pt) | 0.6201 | 0.6775 (+5.7 pt) |
| TIGER | 0.8642 | 0.9501 (+8.6 pt) | 0.5842 | 0.5703 (-1.4 pt) |
| EmerGNN | 0.9895 | 0.9488 (-4.1 pt) | 0.7462 | 0.6675 (-7.9 pt) |

- HDN-DDI's binary task is severely under-specified (rel_total=1
  collapses 152 relation semantics into "exists?"); MC fixes this on
  warm-start. On cold-start S2 the MC advantage drops to ~6 pt.
- TIGER + EmerGNN: binary is the cleaner signal because their
  encoders incorporate KG context; MC just spreads the same signal
  over 152 classes.
- EmerGNN binary > EmerGNN MC across all splits — its KG-flow path
  representation is strongest when the head is binary (no class
  bottleneck).

---

## 5. Headline observations for paper baseline framing

1. **EmerGNN multimode is the strongest binary baseline** —
   0.99 / 0.83 / 0.75 across S0/S1/S2 AUC. Sets a high bar.

2. **HDN-DDI is the strongest MC warm-start baseline** — 0.80 top1 on
   S0 dominates. Mol-graph encoder memorizes warm pairs very well.

3. **All 3 baselines crash on cold-start S2**:
   - Binary S2 AUC range: 0.58 (TIGER) — 0.75 (EmerGNN). Even the
     best is 25 pt below its S0 number.
   - MC S2 top1 range: 0.14 (TIGER) — 0.23 (EmerGNN). All below
     1/4 of S0 top1.
   - **This is the headroom for the proposed method to claim
     contribution.** Cold-start S2 is where existing baselines
     genuinely struggle.

4. **TIGER cold-start patch was OFF** in the reported run
   (paper-faithful default). Codex review noted enabling
   `--tiger-cold-start-patch` for S1/S2 would likely lift S2 numbers
   — to be quantified in a future run if needed for fair comparison.

5. **Single seed, no std deviation.** Multi-seed mean ± std is
   required before any paper claim against these baselines (see
   §7 caveats).

---

## 6. Source files (all numbers verified)

| baseline × task | result doc | run_dir | numbers verified |
|---|---|---|---|
| EmerGNN binary | `baseline/emergnn/_results/2026-05-20__binary_cls__seed42__final.md` | `Code/runs/2026-05-20_00-52-15__run_baseline__emergnn_multimode_seed42__seed42/` | §4-5 of doc |
| EmerGNN MC | `baseline/emergnn/_results/2026-05-18__multi_cls__seed42.md` | (in doc) | §4 |
| HDN-DDI binary | `baseline/hdn_ddi/_results/2026-05-18__binary_cls__seed42.md` | (in doc) | §4 |
| HDN-DDI MC | `baseline/hdn_ddi/_results/2026-05-18__multi_cls__seed42.md` | (in doc) | §4 |
| TIGER binary | `baseline/tiger/_results/2026-05-19__binary_seed42_e50.md` | `Code/runs/2026-05-19_01-33-31__run_baseline__tiger_binary_e50_seed42__seed42/` | results.json eval |
| TIGER MC | `baseline/tiger/_results/2026-05-19__mcc_seed42_e50.md` | `Code/runs/2026-05-19_01-33-37__run_baseline__tiger_mcc_e50_seed42__seed42/` | results.json eval |

Code review docs (all PASS):
- `baseline/emergnn/_reviews/2026-05-18__baseline_round5_PASS.md`
- `baseline/hdn_ddi/_reviews/2026-05-17__baseline.md`
- `baseline/tiger/_reviews/2026-05-18__baseline_round5_PASS.md`

---

## 7. Caveats (must address before paper claims)

1. **Single seed (42)** for all 6 runs. Need multi-seed (e.g. 0, 42,
   100) mean ± std before claiming statistical significance.
2. **Different epoch counts**: HDN/EmerGNN ran 100 ep; TIGER ran 50 ep.
   TIGER might still be under-trained; a 100-ep TIGER run would
   tighten the comparison.
3. **Different wall times** (29 min — 11.6 h). Not directly relevant
   for accuracy comparison but matters for "training cost vs
   accuracy" framing.
4. **TIGER cold-start patch = OFF** (paper-faithful default). If
   the user wants a "TIGER with all extensions enabled" comparison,
   add `--tiger-cold-start-patch` run. Current numbers are the
   paper-faithful default.
5. **HDN-DDI binary task is mismatch** (rel_total=1 collapses 152
   relations to "exists?"). Its low binary numbers (0.66 / 0.64 / 0.62)
   reflect this — not a bug. Apples-to-apples comparison is
   `HDN-DDI multi_cls` vs `EmerGNN multi_cls` vs `TIGER multi_cls`.
6. **EmerGNN binary multimode is post-fix (2026-05-20)**. The earlier
   single-S2-mode binary run on 5/18 had degenerate S0/S1 (collapsed
   to AUC=0.5). Make sure to cite the 5/20 numbers.
7. **K=152 multi-class** observed (project taxonomy finer than paper's
   86). When comparing to literature numbers reported on paper's 86-class
   setting, scale comparison accordingly.

---

## 8. How to use these baselines

When reporting your proposed method's numbers, compare against:

| your method's task | compare against |
|---|---|
| binary (S0/S1/S2 AUC + F1) | EmerGNN multimode (this table §2) — **set the bar**. Also HDN binary (as a "weak binary baseline" reference for the task-mismatch story). |
| multi-class (S0/S1/S2 top-k acc + macro F1 + macro AUC) | HDN-DDI MC (this table §3) — sets the warm-start bar. EmerGNN MC ties on S2 cold-start. |
| ablation (KG vs no-KG) | HDN-DDI binary AUC 0.62-0.66 = no-KG floor; EmerGNN multimode 0.75-0.99 = KG ceiling. Gap quantifies KG contribution. |
| cold-start positioning | All baselines crash on S2. Headroom for proposed method is largest there. |

---

## 9. To reproduce any single number

All numbers above produced via:

```bash
# EmerGNN binary (multimode, official)
python Code/scripts/run_baseline.py --baseline emergnn --kg-source drugbank --seed 42 --epochs 100 --tag emergnn_seed42

# EmerGNN MC
python Code/scripts/run_baseline.py --baseline emergnn-mcc --kg-source drugbank --seed 42 --epochs 100 --tag emergnn_mc_seed42

# HDN-DDI binary
python Code/scripts/run_baseline.py --baseline hdn_ddi --seed 42 --epochs 100 --tag hdn_ddi_binary_seed42

# HDN-DDI MC
python Code/scripts/run_baseline.py --baseline hdn_ddi-mcc --seed 42 --epochs 100 --tag hdn_ddi_mc_seed42

# TIGER binary (paper-faithful default; randomWalk extractor, no cold-start patch)
python Code/scripts/run_baseline.py --baseline tiger --kg-source merged --epochs 50 --tag tiger_binary_e50_seed42

# TIGER MC
python Code/scripts/run_baseline.py --baseline tiger-mcc --kg-source merged --epochs 50 --tag tiger_mcc_e50_seed42
```

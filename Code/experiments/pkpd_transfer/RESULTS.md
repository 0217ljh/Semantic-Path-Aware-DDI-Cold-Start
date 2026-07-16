# PK/PD Transfer Experiments — Results & Analysis

NBFNet v1.7, 800-drug seed42, merged KG, L=3 dim=16. Tests whether the model
trained on one mechanism regime (PK/PD) transfers to the other, and whether
joint training hurts PD ("fighting"). All numbers verified from run artifacts.

## Transfer matrix (AUROC, test_s2)

| train ＼ test | PD | PK |
|---|---|---|
| ALL (PK+PD) | 0.727 | 0.855 |
| PK only | **0.580** | ~0.840 (PK-val proxy) |
| PD only | **0.724** | **0.569** |

Extra metrics (base-rate ≈ 0.33):
- PD→PD: AUC 0.7242, AUPRC 0.5935, NLL 0.6496 (n_pos 944 / n_neg 1919)
- PD→PK: AUC 0.5692, AUPRC 0.4001, NLL 0.7002 (n_pos 964 / n_neg 1919)
- PK→PD: AUC 0.5798, AUPRC 0.4301, NLL 0.7301 (n_pos 944 / n_neg 1919)
- ALL→PD / ALL→PK: 0.727 / 0.855 (AUROC; from post-hoc bucket split of the
  reference run nbfnet_merged_L3_dim16 test_s2_scores.npz)

Sources: runs/exp2_PKtrain_PDtest/results.json, runs/exp3_PDtrain_PKPDtest/results.json,
runs/2026-06-06_00-14-08__run_nbfnet__nbfnet_merged_L3_dim16__seed42 (reference).

## Findings

1. **Cross-regime transfer collapses both ways to near-chance.**
   PK→PD = 0.580, PD→PK = 0.569 (chance = 0.5). PK-learned propagation has no
   PD ability and vice versa → PK and PD are two largely disjoint regimes
   (consistent with the structural finding: PK loads on enzyme/transporter
   channels, PD on side-effect/contraindication; the channels barely overlap).

2. **No negative transfer ("they do NOT fight").**
   PD→PD = 0.724 ≈ ALL→PD = 0.727 (Δ 0.003). Adding PK data to training neither
   helps nor hurts PD. PK data is inert for PD. The earlier "two regimes fight
   in cold-start" hypothesis is NOT supported — refined to: the regimes are
   ORTHOGONAL (non-transferable but non-interfering); the shared model has
   capacity to learn both relation-channel sets independently.

3. **PD is intrinsically hard, not dragged down by PK.**
   PD→PD itself tops out at 0.724, vs PK→PK ~0.84–0.855. With all the PD data
   available, PD still caps ~0.72. PD's deficiency is intrinsic to the PD regime
   in this KG (diffuse neighborhood, no discriminative channel), NOT a
   multi-task interference effect.

## Implication for method direction

- Do NOT build PK/PD routing/separation to "prevent fighting" — there is no
  fighting to prevent.
- The binding constraint is the intrinsic PD→PD ceiling (~0.72): PD's
  discriminative signal is missing from the KG structure. The only lever that
  can lift it is INJECTING new PD-discriminative signal (node semantics /
  molecular / phase-direction). This turns strategy a (phase) / b (node
  semantics) from optional to necessary.

## Caveats

- Single seed (42).
- exp3 used batch_size 8 (OOM mitigation); exp1/exp2 used 16. So PD→PD (0.724,
  bs8) vs ALL→PD (0.727, bs16) spans a minor config difference — still strong
  enough for "no meaningful negative transfer", but a same-config PD→PD re-run
  would make it airtight.
- PK→PK is a val proxy (0.840), not an independent test; ALL→PK = 0.855 is test.
- exp3 early-stopped at ep7 (PD-val); PD→PK is "best-PD model evaluated on PK".

## Reproduce

```
# exp2 PK->PD
python Code/scripts/run_nbfnet_pkpd_transfer.py --train-pairs Code/experiments/pkpd_transfer/data/train_PK.csv --val-pairs Code/experiments/pkpd_transfer/data/val_s2_PK.csv --test-pairs Code/experiments/pkpd_transfer/data/test_s2_PD.csv --tag exp2_PKtrain_PDtest
# exp3 PD->{PK,PD}  (batch 8 + expandable_segments for OOM)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python Code/scripts/run_nbfnet_pkpd_transfer.py --train-pairs Code/experiments/pkpd_transfer/data/train_PD.csv --val-pairs Code/experiments/pkpd_transfer/data/val_s2_PD.csv --test-pairs Code/experiments/pkpd_transfer/data/test_s2_PK.csv --extra-test-pairs Code/experiments/pkpd_transfer/data/test_s2_PD.csv --tag exp3_PDtrain_PKPDtest --batch-size 8
```

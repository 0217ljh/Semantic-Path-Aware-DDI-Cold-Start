# PK/PD Transfer Experiments (NBFNet v1.7)

Quick test: does a conditional-message-passing model trained on one mechanism
regime (PK or PD) transfer to the other? Tied to the PD-structural-deficiency
diagnosis (PD = diffuse-neighborhood, hard to predict).

All on the **800-drug legacy seed42** dataset, **merged KG**, NBFNet v1.7,
config matched to the reference run `nbfnet_merged_L3_dim16`
(kg_source=merged, L=3, dim=16, 100 epochs, batch=16) so numbers are comparable.
Only the injected/supervised positives (train) and the evaluated positives
(test) change by PK/PD bucket. Eval negatives = standard split negatives.

## Layout (self-contained; scripts live in Code/scripts per project rule)

```
Code/experiments/pkpd_transfer/
├── data/                       # bucketed pair lists (this folder)
│   ├── train_ALL/PK/PD.csv     # 53743 / 28457 / 25257
│   ├── val_s2_ALL/PK/PD.csv    #  1918 /   929 /   980
│   └── test_s2_ALL/PK/PD.csv   #  1919 /   964 /   944
├── runs/                       # experiment outputs (results.json, test_scores.npz)
└── README.md

Scripts (Code/scripts/):
  prepare_pkpd_transfer_data.py      # build data/ (already run)
  run_nbfnet_pkpd_transfer.py        # train+eval one transfer config
  analyze_pkpd_transfer_negatives.py # post-hoc: negatives PK-like vs PD-like
```

## Experiments

| exp | train | test | status |
|---|---|---|---|
| exp1 | PK+PD (ALL) | PD | **done** via reference run (ALL→PD test_s2 AUC = 0.727; ALL→PK = 0.855) |
| exp2 | PK | PD | run command below |
| exp3 | PD | PK | run command below |

Reference (already measured, same config, from `runs/2026-06-06_00-14-08__run_nbfnet__nbfnet_merged_L3_dim16__seed42`):
ALL→PD = 0.727, ALL→PK = 0.855 (split of test_s2_scores.npz by bucket).

## Commands (run from project root, WSL conda env project_1)

exp2  PK → PD:
```
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_nbfnet_pkpd_transfer.py --train-pairs Code/experiments/pkpd_transfer/data/train_PK.csv --val-pairs Code/experiments/pkpd_transfer/data/val_s2_PK.csv --test-pairs Code/experiments/pkpd_transfer/data/test_s2_PD.csv --tag exp2_PKtrain_PDtest"
```

exp3  PD → PK:
```
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/run_nbfnet_pkpd_transfer.py --train-pairs Code/experiments/pkpd_transfer/data/train_PD.csv --val-pairs Code/experiments/pkpd_transfer/data/val_s2_PD.csv --test-pairs Code/experiments/pkpd_transfer/data/test_s2_PK.csv --tag exp3_PDtrain_PKtest"
```

(optional) fresh exp1 ALL → PD (redundant with reference, skip unless wanted):
```
... run_nbfnet_pkpd_transfer.py --train-pairs .../train_ALL.csv --val-pairs .../val_s2_ALL.csv --test-pairs .../test_s2_PD.csv --tag exp1_ALLtrain_PDtest
```

Note: each run ≈ 8h on the merged KG (matches reference fit_sec 30043s). For a
quick-and-dirty signal add `--epochs 30` (loses exact comparability to 0.727).

## Post-hoc negative analysis (no training needed, run anytime)
```
wsl bash -ic "conda activate project_1 && cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start && python Code/scripts/analyze_pkpd_transfer_negatives.py --seed 42"
```
Answers: are the test_s2 negatives' confluent channel signature closer to
PK-positives or PD-positives.

## Outputs per run
`runs/<tag>/results.json` (config + test AUC/AUPRC/NLL) and
`runs/<tag>/test_scores.npz` (pair_a, pair_b, y_true, y_score) for further
PK/PD slicing.
```

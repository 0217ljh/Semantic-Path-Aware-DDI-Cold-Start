#!/bin/bash
# Idempotent driver: gate-full, budget REMOVED (weight 0), V-REx 0.5 kept (codex R-anchor2).
# Resumable: skips any seed whose json exists. Run: wsl bash Code/scripts/_run_gate_nobudget.sh
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate project_1 2>/dev/null || true
cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNDIR=Code/runs/spmn_v2_standalone
for s in 42 43 44; do
  f="$RUNDIR/aware1_and_seed${s}_mech_sdeg_dpool_topk_cw0.1_det_gatefull_vrex0.5_bf_lmax3_d32.json"
  if [ -f "$f" ]; then echo "SKIP seed=$s (exists)"; continue; fi
  echo "== RUN gate=full nobudget seed=$s =="
  python Code/scripts/run_spmn_v2_aware.py --seed "$s" --aware-step 1 --use-mech-feats \
    --support-degrade --degrade-pool --degrade-topk --consistency-weight 0.1 --deterministic \
    --mediator-gate --gate-input full --gate-vrex-weight 0.5 --gate-budget-weight 0.0 \
    --branch-finetune 2>&1 | tail -2
done
echo ALL_DONE_NOBUDGET

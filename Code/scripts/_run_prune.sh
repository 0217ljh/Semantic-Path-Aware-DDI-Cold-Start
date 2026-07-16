#!/bin/bash
# Idempotent driver: upstream redundancy prune (frac 0.15) on the locked pipeline, 3 seeds.
# Resumable: skips seeds whose json exists. Run: wsl bash Code/scripts/_run_prune.sh
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate project_1 2>/dev/null || true
cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNDIR=Code/runs/spmn_v2_standalone
for s in 42 43 44; do
  f="$RUNDIR/aware1_and_seed${s}_mech_sdeg_dpool_topk_cw0.1_det_prune0.15_bf_lmax3_d32.json"
  if [ -f "$f" ]; then echo "SKIP seed=$s (exists)"; continue; fi
  echo "== RUN prune0.15 seed=$s =="
  python Code/scripts/run_spmn_v2_aware.py --seed "$s" --aware-step 1 --use-mech-feats \
    --support-degrade --degrade-pool --degrade-topk --consistency-weight 0.1 --deterministic \
    --prune-pool-frac 0.15 --branch-finetune 2>&1 | tail -2
done
echo ALL_DONE_PRUNE

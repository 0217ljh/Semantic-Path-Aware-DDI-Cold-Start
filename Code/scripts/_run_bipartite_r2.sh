#!/bin/bash
# R2: bipartite with GRL-null REMOVED (bp-null-weight 0), keep inv=0.1 sep=0.1. 3 seeds.
# Idempotent/resumable. Usage: wsl bash Code/scripts/_run_bipartite_r2.sh
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate project_1 2>/dev/null || true
cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNDIR=Code/runs/spmn_v2_standalone
for s in 42 43 44; do
  f="$RUNDIR/aware1_and_seed${s}_mech_sdeg_dpool_topk_cw0.1_det_bipartite_n0i0.1s0.1_bf_lmax3_d32.json"
  if [ -f "$f" ]; then echo "SKIP seed=$s (exists)"; continue; fi
  echo "== RUN bipartite null=0 seed=$s =="
  python Code/scripts/run_spmn_v2_aware.py --seed "$s" --aware-step 1 --use-mech-feats \
    --support-degrade --degrade-pool --degrade-topk --consistency-weight 0.1 --deterministic \
    --bipartite --bp-null-weight 0.0 --bp-inv-weight 0.1 --bp-sep-weight 0.1 \
    --branch-finetune 2>&1 | tail -2
done
echo ALL_DONE_BIP_R2

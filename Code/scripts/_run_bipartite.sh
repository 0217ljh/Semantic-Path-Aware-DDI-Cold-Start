#!/bin/bash
# Idempotent driver: BIPARTITE invariance method (z_C/z_D + GRL null + cross-view inv + sep)
# on the locked pipeline, 3 seeds. Resumable: skips seeds whose json exists.
# Usage: wsl bash Code/scripts/_run_bipartite.sh [null] [inv] [sep]  (weights optional)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate project_1 2>/dev/null || true
cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNDIR=Code/runs/spmn_v2_standalone
NW=${1:-0.1}; IW=${2:-0.1}; SW=${3:-0.1}
for s in 42 43 44; do
  f="$RUNDIR/aware1_and_seed${s}_mech_sdeg_dpool_topk_cw0.1_det_bipartite_bf_lmax3_d32.json"
  # only the default-weight run uses the canonical tag; weighted sweeps get a suffix tag
  if [ "$NW" = "0.1" ] && [ "$IW" = "0.1" ] && [ "$SW" = "0.1" ]; then
    tgt="$f"
  else
    tgt="$RUNDIR/aware1_seed${s}_bipartite_n${NW}_i${IW}_s${SW}.SENTINEL"
  fi
  if [ -f "$f" ]; then echo "SKIP seed=$s (exists)"; continue; fi
  echo "== RUN bipartite n=$NW i=$IW s=$SW seed=$s =="
  python Code/scripts/run_spmn_v2_aware.py --seed "$s" --aware-step 1 --use-mech-feats \
    --support-degrade --degrade-pool --degrade-topk --consistency-weight 0.1 --deterministic \
    --bipartite --bp-null-weight "$NW" --bp-inv-weight "$IW" --bp-sep-weight "$SW" \
    --branch-finetune 2>&1 | tail -2
done
echo ALL_DONE_BIPARTITE

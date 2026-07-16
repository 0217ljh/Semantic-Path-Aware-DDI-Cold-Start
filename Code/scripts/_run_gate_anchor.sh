#!/bin/bash
# Idempotent driver for the mediator-gate anchor ablation (degree vs full, 3 seeds).
# Resumable: skips any (gate,seed) whose result json already exists, so relaunching
# after a session-teardown kill continues where it left off without wasted reruns.
# Run: wsl bash Code/scripts/_run_gate_anchor.sh   (from project root, env project_1)
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate project_1 2>/dev/null || true
cd /mnt/d/My-Research/03-Projects/Semantic-Path-Aware-DDI-Cold-Start
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUNDIR=Code/runs/spmn_v2_standalone

run_one () {
  g=$1; s=$2
  f="$RUNDIR/aware1_and_seed${s}_mech_sdeg_dpool_topk_cw0.1_det_gate${g}_vrex0.5_gbud0.1_bf_lmax3_d32.json"
  if [ -f "$f" ]; then echo "SKIP gate=$g seed=$s (exists)"; return 0; fi
  echo "== RUN gate=$g seed=$s =="
  python Code/scripts/run_spmn_v2_aware.py --seed "$s" --aware-step 1 --use-mech-feats \
    --support-degrade --degrade-pool --degrade-topk --consistency-weight 0.1 --deterministic \
    --mediator-gate --gate-input "$g" --gate-vrex-weight 0.5 --gate-budget-weight 0.1 \
    --branch-finetune 2>&1 | tail -2
}

for g in degree full; do
  for s in 42 43 44; do
    run_one "$g" "$s"
  done
done
echo ALL_DONE_GATE_ANCHOR

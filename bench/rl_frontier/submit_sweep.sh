#!/bin/bash
# Submit the full frontier sweep: 2 tasks x 9 conditions x 5 seeds = 90 one-core LSF jobs.
# Budgets come from protocol_{reach,push}.json (total_timesteps); no --steps override.
# Run on gc211:  bash bench/rl_frontier/submit_sweep.sh [TASKS]   default TASKS="push reach"
# Supplementary: bash bench/rl_frontier/submit_sweep.sh push_staged  (no sparse arm: reuses push/sparse)
# Already-completed runs are skipped.
set -euo pipefail
B=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench
CODE=$B/code_rl
RUNS=$B/rl_runs/sweep
mkdir -p $RUNS/logs
CONDS="sparse dense auroc0.55 auroc0.60 auroc0.65 auroc0.70 auroc0.80 auroc0.90 auroc1.00"
for TASK in ${1:-push reach}; do   # longer task first
  for SEED in 0 1 2 3 4; do
    for COND in $CONDS; do
      [ "$TASK" = push_staged ] && [ "$COND" = sparse ] && continue
      OUT=$RUNS/$TASK/$COND/seed$SEED
      if [ -e "$OUT/status.json" ] && grep -q '"completed": true' "$OUT/status.json"; then continue; fi
      # never duplicate a queued/running job (run_one.sh would move its live dir aside)
      if bjobs -w -J eb_rl_${TASK}_${COND}_s${SEED} 2>/dev/null | grep -qE ' (RUN|PEND|SSUSP|USUSP|PSUSP) '; then continue; fi
      bsub -q cpu_ubuntu_wrn -n 1 -R 'rusage[mem=4000]' -W 8:00 \
        -J eb_rl_${TASK}_${COND}_s${SEED} -o $RUNS/logs/rl_%J.log \
        $CODE/bench/rl_frontier/run_one.sh $TASK $COND $SEED $OUT
    done
  done
done | tee -a $RUNS/logs/submitted_$(date -u +%Y%m%dT%H%M%SZ).txt

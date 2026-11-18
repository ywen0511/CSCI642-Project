#!/bin/bash
# One formal RL run (one LSF job, one core): run_job.sh <checkout> <config (repo-relative)> <phase> <suite_dir> <job_name>
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
CO=$1 CFG=$2 PHASE=$3 OUT=$4 NAME=$5
cd $CO || exit 2
DIRTY=$(git status --porcelain --untracked-files=no)
echo "host=$(hostname) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) head=$(git rev-parse HEAD) dirty=[$DIRTY] start=$(date -u +%FT%TZ)"
[ -z "$DIRTY" ] || { echo "CHECKOUT_DIRTY"; exit 3; }
/usr/bin/time -v -o $OUT/$NAME.resources $PY -m rl.suite --config $CFG --phase $PHASE \
  --feedback-dir $C/repo/artifacts/rl_feedback --out $OUT --run-job $NAME
rc=$?
echo "RUNJOB_END rc=$rc end=$(date -u +%FT%TZ)"
exit $rc

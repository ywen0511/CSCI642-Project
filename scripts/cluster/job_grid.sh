#!/bin/bash
# usage: job_grid.sh <config> [<config> ...]   trains all grid candidates of each config
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg/scripts_ops/job_env.sh
for c in "$@"; do
  $PY scripts/run_grid.py --config $c --prep-dir $PREP --out-root $RUNS || echo "GRID_FAILED $c"
done
echo JOB_DONE

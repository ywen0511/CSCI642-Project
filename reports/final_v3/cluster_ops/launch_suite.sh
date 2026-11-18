#!/bin/bash
# launch_suite.sh <commit> <config (repo-relative)> <phase> <suite_name> [walltime]
# pinned clean checkout of <commit>, rl.suite --prepare, then one LSF job per planned run.
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1 CFG=$2 PHASE=$3 NAME=$4 W=${5:-4:00}
CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments
[ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA || exit 2
[ "$(git -C $CO rev-parse HEAD)" = "$(git -C $C/repo rev-parse $SHA)" ] || { echo "checkout mismatch"; exit 2; }
OUT=$C/runs/$NAME
mkdir -p $C/runs $C/logs/$NAME
cd $CO && $PY -m rl.suite --config $CFG --phase $PHASE --feedback-dir $C/repo/artifacts/rl_feedback --out $OUT --prepare || exit 4
for job in $($PY -c "import json;print(' '.join(j['name'] for j in json.load(open('$OUT/plan.json'))['jobs']))"); do
  bsub -q cpu_ubuntu_wrn -n 1 -R "span[hosts=1]" -W $W -J "v3_${NAME}_$job" -o $C/logs/$NAME/$job.%J.out \
    $C/ops/run_job.sh $CO $CFG $PHASE $OUT $job > /dev/null
done
echo "LAUNCHED $NAME $(ls $C/logs/$NAME | wc -l) jobs submitted from $CO"
bjobs -noheader -J "v3_${NAME}_*" 2>/dev/null | wc -l

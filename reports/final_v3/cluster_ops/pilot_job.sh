#!/bin/bash
# one preference pilot run: pilot_job.sh <checkout> <config> <condition> <subject|-> <seed> <out_root>
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
CO=$1 CFG=$2 COND=$3 SUBJ=$4 SEED=$5 ROOT=$6
cd $CO; DIRTY=$(git status --porcelain --untracked-files=no)
echo "host=$(hostname) head=$(git rev-parse HEAD) dirty=[$DIRTY] start=$(date -u +%FT%TZ)"; [ -z "$DIRTY" ] || exit 3
NAME=${COND}_$([ "$SUBJ" = - ] && echo shared || echo $SUBJ)_s$SEED
EXTRA=""; [ "$SUBJ" = - ] || EXTRA="--subject $SUBJ"
/usr/bin/time -v -o $ROOT/$NAME.resources $PY -m rl.train --config $CO/$CFG --phase pilot --condition $COND $EXTRA --seed $SEED \
  --feedback-dir $C/repo/artifacts/rl_feedback --out $ROOT/$NAME > $ROOT/$NAME.log 2>&1
echo "PILOT_END rc=$? end=$(date -u +%FT%TZ)"

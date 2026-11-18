#!/bin/bash
# post.sh <analysis_commit> <suite_dir_name> [...]: rl.summarize + rl.analysis into $C/analysis/<suite>
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1; shift; CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments
[ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
cd $CO; echo "analysis HEAD $(git rev-parse HEAD)"
mkdir -p $C/analysis
for S in "$@"; do
  $PY -m rl.summarize $C/runs/$S > $C/analysis/$S.summarize.log 2>&1; echo "SUMMARIZE $S rc=$?"
  rm -rf $C/analysis/$S
  $PY -m rl.analysis $C/runs/$S --feedback-dir $C/repo/artifacts/rl_feedback --out $C/analysis/$S > $C/analysis/$S.analysis.log 2>&1; echo "ANALYSIS $S rc=$?"
done
echo POST_DONE

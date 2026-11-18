#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1; CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments; [ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
cd $CO; $PY -m unittest rl.test_stage2 rl.test_final_v3 > $C/logs/tests_$SHA.log 2>&1 || { echo TESTS_FAILED; tail -20 $C/logs/tests_$SHA.log; exit 1; }
grep -E "^Ran|^OK" $C/logs/tests_$SHA.log
for w in w100 w200 w400; do $C/ops/launch_suite.sh $SHA configs/final_v3/pref_candidates/pref_$w.json validation pref_validation_$w 6:00 | grep LAUNCHED; done
echo LAUNCH_PREF_VAL_DONE

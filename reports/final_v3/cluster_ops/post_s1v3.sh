#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
$C/ops/post.sh 3222d27 sensitivity_w010_final sensitivity_w050_final v3_validation_w050 v3_validation_w100 v3_validation_w200
cd $C/checkouts/3222d27
for w in w010 w050; do $PY scripts/final_v3_determinism_check.py --new $C/runs/sensitivity_${w}_final --reference $C/restore/suites/rl_v2_final --out $C/analysis/determinism_$w.json; done
$PY scripts/rl_select_candidate.py $C/runs/v3_validation_w050 $C/runs/v3_validation_w100 $C/runs/v3_validation_w200 --metric auc --out $C/analysis/v3_selection.json --markdown $C/analysis/v3_selection.md
echo POSTS1_DONE

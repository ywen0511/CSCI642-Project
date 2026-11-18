#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
cd $C/dev
[ -d .git ] || { git init -q && git add -A && git -c user.name=dev -c user.email=dev@x commit -qm dev; }
git add -A && git -c user.name=dev -c user.email=dev@x commit -qm dev >/dev/null 2>&1
ln -sfn $C/repo/artifacts/rl_feedback artifacts/rl_feedback 2>/dev/null || { mkdir -p artifacts; ln -sfn $C/repo/artifacts/rl_feedback artifacts/rl_feedback; }
$PY -m unittest rl.test_stage2 rl.test_final_v3 2>&1 | tail -30
rm -rf /tmp/kz_geo; $PY scripts/final_v3_pref_geometry_check.py --config configs/final_v3/pref_candidates/pref_w100.json --out /tmp/kz_geo 2>&1 | tail -40
echo DEVTEST_DONE

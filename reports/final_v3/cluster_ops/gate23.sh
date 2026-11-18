#!/bin/bash
# Stage 2/3 gate at a pinned commit: tests, v3 smoke, preference geometry check, preference smoke.
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1; CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments
[ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
cd $CO; echo "HEAD $(git rev-parse HEAD) dirty=[$(git status --porcelain --untracked-files=no)]"
G=$C/gate23_${SHA:0:7}; mkdir -p $G
$PY -m unittest rl.test_stage2 rl.test_final_v3 -v > $G/tests.log 2>&1; echo "TESTS rc=$?"; grep -E "^Ran|^OK|FAILED" $G/tests.log
$PY -m rl.smoke --config configs/final_v3/v3_candidates/v3_w100.json --feedback-dir $C/repo/artifacts/rl_feedback --out $G/v3_smoke > $G/v3_smoke.log 2>&1; echo "V3_SMOKE rc=$?"
$PY scripts/final_v3_pref_geometry_check.py --config configs/final_v3/pref_candidates/pref_w100.json --out $G/pref_geometry > $G/pref_geometry.log 2>&1; echo "PREF_GEOMETRY rc=$?"
$PY -m rl.smoke --config configs/final_v3/pref_candidates/pref_w100.json --feedback-dir $C/repo/artifacts/rl_feedback --out $G/pref_smoke > $G/pref_smoke.log 2>&1; echo "PREF_SMOKE rc=$?"
for s in v3_smoke pref_smoke; do $PY -c "import json;r=json.load(open(\"$G/$s/smoke_report.json\"));print(\"$s ok\",r[\"ok\"],r.get(\"error\"))"; done
echo GATE23_DONE

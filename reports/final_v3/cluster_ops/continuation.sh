#!/bin/bash
# continuation.sh <tool_commit> <training_commit> <snapshot_dir> <label> [steps]
# check script from <tool_commit>; rl.train runs in a clean checkout of <training_commit> (--checkout)
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
TOOL=$1 SHA=$2 SNAP=$3 LABEL=$4 STEPS=${5:-2000}
for s in $TOOL $SHA; do CO=$C/checkouts/${s:0:7}; git -C $C/repo fetch -q origin final-v3-experiments; [ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $s; done
TCO=$C/checkouts/${SHA:0:7}; echo "training checkout $(git -C $TCO rev-parse HEAD) dirty=[$(git -C $TCO status --porcelain --untracked-files=no)]"
W=$C/continuation/$LABEL.$(date -u +%Y%m%dT%H%M%S); mkdir -p $C/continuation
cd $C/checkouts/${TOOL:0:7}
$PY scripts/rl_continuation_check.py --snapshot $SNAP --feedback-dir $C/repo/artifacts/rl_feedback --steps $STEPS --work $W --out $W.report.json --checkout $TCO > $W.log 2>&1
echo "CONTINUATION $LABEL rc=$? $W.report.json"; python3 -c "import json;d=json.load(open(\"$W.report.json\"));print({k:d.get(k) for k in (\"ok\",\"restored_transitions_unchanged\",\"restored_slots_checked\",\"replay_check\",\"steps_increased\",\"updates_increased\",\"replay_grew\")})"

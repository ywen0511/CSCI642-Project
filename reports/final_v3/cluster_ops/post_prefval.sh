#!/bin/bash
# post_prefval.sh <tool_commit>: summarize + analysis of the 3 preference validation suites, declared selection
# (--metric preferred_pass) and rl.freeze of the selected candidate into $C/analysis/pref_frozen (committed from there).
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1
$C/ops/post.sh $SHA pref_validation_w100 pref_validation_w200 pref_validation_w400
cd $C/checkouts/${SHA:0:7}
$PY scripts/rl_select_candidate.py $C/runs/pref_validation_w100 $C/runs/pref_validation_w200 $C/runs/pref_validation_w400 \
  --metric preferred_pass --out $C/analysis/pref_selection.json --markdown $C/analysis/pref_selection.md || exit 5
SEL=$(python3 -c "import json;print(json.load(open('$C/analysis/pref_selection.json'))['selected'])")
RAT=$(python3 -c "import json;print(json.load(open('$C/analysis/pref_selection.json'))['rationale'])")
rm -rf $C/analysis/pref_frozen
$PY -m rl.freeze --validation-suite $SEL --rationale "$RAT" --out $C/analysis/pref_frozen && echo "FROZEN $SEL"
cp $C/analysis/pref_selection.json $C/analysis/pref_frozen/selection.json; cp $C/analysis/pref_selection.md $C/analysis/pref_frozen/selection.md
echo POST_PREFVAL_DONE

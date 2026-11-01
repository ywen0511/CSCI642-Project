#!/bin/bash
# Post-training: selection + evaluation (main), extra decoder seeds, supplementary evaluation, export.
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg/scripts_ops/job_env.sh
ART=$P/artifacts; SEL=$ART/selection
set -e
for m in eegnet cbramod; do
  $PY scripts/select_and_evaluate.py --config configs/models/$m.yaml --runs-root $RUNS --prep-dir $PREP --selection-dir $SEL
done
# extra decoder seeds for the selected configuration of each main model
EXTRA=()
for m in eegnet cbramod; do
  OVR=$($PY -c "import json;s=json.load(open('$SEL/$m.json'));c=[c for c in s['candidates'] if c['run_id']==s['selected_run_id']][0];print(' '.join(c['overrides']))")
  for seed in 43 44 45 46; do
    $PY scripts/run_grid.py --config configs/models/$m.yaml --prep-dir $PREP --out-root $RUNS --seed $seed --overrides $OVR
    RID=$($PY -c "import sys;sys.path.insert(0,'src');from errp.tuning import candidates;print([c['run_id'] for c in candidates('configs/models/$m.yaml',$seed) if sorted(c['overrides'])==sorted('$OVR'.split())][0])")
    $PY scripts/select_and_evaluate.py --config configs/models/$m.yaml --runs-root $RUNS --prep-dir $PREP --run-id $RID
    EXTRA+=($RID)
  done
done
SUPP=()
for m in eeginception_erp eegnex ctnet eegconformer labram eegpt; do
  if [ -f $RUNS/bd_${m}_s42/train_summary.json ]; then
    $PY scripts/select_and_evaluate.py --config configs/models/supplementary/$m.yaml --runs-root $RUNS --prep-dir $PREP --run-id bd_${m}_s42 && SUPP+=(bd_${m}_s42)
  else echo "SUPP_MISSING $m"; fi
done
[ -f $RUNS/bd_xdawn_riemann_s42/metrics.json ] && SUPP+=(bd_xdawn_riemann_s42)
$PY scripts/export_artifacts.py --artifacts $ART --prep-dir $PREP --extra-seed-runs ${EXTRA[@]} --supplementary-runs ${SUPP[@]}
echo JOB_DONE

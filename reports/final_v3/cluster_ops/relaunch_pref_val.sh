#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=ae633ea
for w in w100 w200 w400; do
  N=pref_validation_$w; A=$C/aborted/${N}_killed_20261118T0805Z; mkdir -p $A/runs $A/logs
  for r in $C/runs/$N/*/; do b=$(basename $r); grep -qs '"completed": true' $r/status.json && continue; mv $r $C/runs/$N/$b.log $C/runs/$N/$b.resources $A/runs/ 2>/dev/null; done
  mv $C/logs/$N/* $A/logs/ 2>/dev/null
  $C/ops/submit_suite.sh $SHA configs/final_v3/pref_candidates/pref_$w.json validation $N 6:00
  sleep 20
done
echo RELAUNCH_DONE

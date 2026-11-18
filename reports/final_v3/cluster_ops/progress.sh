#!/bin/bash
# per suite: planned / completed (status.json completed) / failed (run dir with status not completed and no LSF job alive)
C=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3
alive=$(bash -lc "bjobs -noheader -o job_name 2>/dev/null" | grep ^v3_)
for s in $C/runs/*/; do n=$(basename $s)
  if [ -f $s/plan.json ]; then planned=$(grep -c "\"name\"" $s/plan.json); else planned=$(ls -d $s/*/ 2>/dev/null | wc -l); fi
  done_=$(grep -l "\"completed\": true" $s/*/status.json 2>/dev/null | wc -l)
  fail=0; for st in $(grep -L "\"completed\": true" $s/*/status.json 2>/dev/null); do r=$(basename $(dirname $st)); grep -q "_${r}\$" <<< "$alive" || fail=$((fail+1)); done
  echo "$n planned=$planned completed=$done_ failed=$fail alive=$(grep -c "^v3_${n}_" <<< "$alive")"
done

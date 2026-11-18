#!/bin/bash
# submit_suite.sh <commit> <config> <phase> <suite_name> [walltime]: submit plan jobs that have no run dir and no LSF job
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1 CFG=$2 PHASE=$3 NAME=$4 W=${5:-4:00}; CO=$C/checkouts/${SHA:0:7}; OUT=$C/runs/$NAME
mkdir -p $C/logs/$NAME
active=$(bjobs -noheader -o job_name 2>/dev/null | grep "^v3_${NAME}_")
n=0
for job in $($PY -c "import json;print(\" \".join(j[\"name\"] for j in json.load(open(\"$OUT/plan.json\"))[\"jobs\"]))"); do
  [ -e $OUT/$job ] && continue; grep -qx "v3_${NAME}_$job" <<< "$active" && continue
  bsub -q cpu_ubuntu_wrn -n 1 -W $W -J "v3_${NAME}_$job" -o $C/logs/$NAME/$job.%J.out $C/ops/run_job.sh $CO $CFG $PHASE $OUT $job >/dev/null && n=$((n+1))
done
echo "SUBMITTED $NAME $n"

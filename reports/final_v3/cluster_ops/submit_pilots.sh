#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
CO=$C/checkouts/a09817f; P=$C/runs/pref_pilots; mkdir -p $P $C/logs/pref_pilots
for seed in 900 901; do for spec in sparse:- dense:- ideal:- eegnet:s11 cbramod:s13; do
  IFS=: read cond subj <<< "$spec"
  bsub -q cpu_ubuntu_wrn -n 1 -W 4:00 -J v3_pref_pilot_${cond}_$seed -o $C/logs/pref_pilots/${cond}_$seed.%J.out $C/ops/pilot_job.sh $CO configs/final_v3/pref_candidates/pref_w100.json $cond $subj $seed $P >/dev/null
done; done; echo SUBMITTED_PILOTS

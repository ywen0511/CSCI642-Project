#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=afa0531; CO=$C/checkouts/$SHA
git -C $C/repo fetch -q origin final-v3-experiments; [ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
for g in g1_original_300k g2_corridor10_90_150k g3_corridor10_90_300k; do
  P=$C/runs/pref_pilots_$g; mkdir -p $P $C/logs/pref_pilots_$g
  for seed in 900 901; do for cond in ideal sparse; do
    bsub -q cpu_ubuntu_wrn -n 1 -W 6:00 -J v3_gate2_${g}_${cond}_$seed -o $C/logs/pref_pilots_$g/${cond}_$seed.%J.out \
      $C/ops/pilot_job.sh $CO configs/final_v3/pref_pilot_gate/$g/pref_w100.json $cond - $seed $P >/dev/null
  done; done
done
bsub -q cpu_ubuntu_wrn -n 1 -W 1:00 -J v3_geo_g2 -o $C/logs/geo_g2.%J.out "cd $CO && $PY scripts/final_v3_pref_geometry_check.py --config configs/final_v3/pref_pilot_gate/g2_corridor10_90_150k/pref_w100.json --out $C/stage3_geometry_corridor10_90 | tail -20" >/dev/null
echo SUBMITTED_GATE2

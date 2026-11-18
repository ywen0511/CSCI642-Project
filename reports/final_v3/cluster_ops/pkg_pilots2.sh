#!/bin/bash
C=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3
for g in g1_original_300k g2_corridor10_90_150k g3_corridor10_90_300k; do
  $C/ops/package.sh ae633ea pilot $C/runs/pref_pilots_$g pref_pilots_$g "preference pilot-gate round 2 variant $g (ideal/sparse only)"
done
echo PKG_PILOTS2_DONE

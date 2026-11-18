#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
P=$C/ops/package.sh; T=afa0531
$P $T suite $C/runs/sensitivity_w010_final sensitivity_w010_final
$P $T suite $C/runs/sensitivity_w050_final sensitivity_w050_final
for w in w050 w100 w200; do $P $T suite $C/runs/v3_validation_$w v3_validation_$w; done
$P $T pilot $C/runs/pref_pilots pref_pilots_round1 "preference pilots round 1 (gate failed); w=1.0, seeds 900/901"
$P $T smoke $C/gate23_a09817f/v3_smoke v3_smoke
$P $T smoke $C/gate23_a09817f/pref_smoke pref_smoke
echo PACKAGE_BATCH1_DONE

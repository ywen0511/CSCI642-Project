#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
$C/ops/post.sh 30ef93c v3_final
$C/ops/package.sh 30ef93c suite $C/runs/v3_final v3_final
$C/ops/media.sh 30ef93c v3_final v3_final
echo POSTV3_DONE

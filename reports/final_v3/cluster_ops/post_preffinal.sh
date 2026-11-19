#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
$C/ops/post.sh 43479e3 pref_final
$C/ops/package.sh 43479e3 suite $C/runs/pref_final pref_final
$C/ops/media.sh 43479e3 pref_final pref_final
$C/ops/media_package.sh pref_final pref_final pref_final
echo POSTPREF_DONE

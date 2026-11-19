#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1 TAG=$2; CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments; [ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
cd $CO; V=$C/verify/$TAG.$(date -u +%Y%m%dT%H%M%S); mkdir -p $V
$PY scripts/rl_verify_release.py --tag $TAG --reports-only --download-dir $V/download --extract-dir $V/extract --out $V/report.json > $V/verify.log 2>&1
echo "VERIFY $TAG rc=$? $V"

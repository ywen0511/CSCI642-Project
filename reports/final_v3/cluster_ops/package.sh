#!/bin/bash
# package.sh <tool_commit> <scope: suite|pilot|smoke> <runs_dir> <package_name> [note]
# final snapshots only in the archives (periodic snapshots stay on the filer, listed with SHA-256), 100 MB parts.
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1 SCOPE=$2 RUNS=$3 NAME=$4 NOTE=${5:-}
CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments; [ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
cd $CO; mkdir -p $C/packages
FLAG=""; [ $SCOPE = pilot ] && FLAG="--pilot"; [ $SCOPE = smoke ] && FLAG="--smoke"
SNAP="--final-snapshots-only"; [ $SCOPE = smoke ] && SNAP=""
rm -rf $C/packages/$NAME
$PY -m rl.package --runs $RUNS --out $C/packages/$NAME $FLAG $SNAP --max-part-bytes 100000000 ${NOTE:+--scope-note "$NOTE"} > $C/packages/$NAME.log 2>&1
echo "PACKAGE $NAME rc=$? $(du -sh $C/packages/$NAME | cut -f1) $(ls $C/packages/$NAME | wc -l) files"

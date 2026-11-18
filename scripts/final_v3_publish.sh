#!/bin/bash
# Laptop-side publication of one final-v3 Release (the cluster has no gh):
#   final_v3_publish.sh <tag> <commit> <title> <status> <config|-> <scope>=<cluster package name> [...]
# 1. copy packages from the project filer, 2. stage with "<scope>__" names + SHA256SUMS (rl_stage_release.sh),
# 3. write notes, 4. create the Release pinned to <commit> and upload until complete (rl_upload_missing.sh, which
#    also removes incomplete "starter" assets). Fresh-download verification runs separately on the cluster.
set -euo pipefail
TAG=$1 COMMIT=$2 TITLE=$3 STATUS=$4 CONFIG=$5; shift 5
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=${STAGE_ROOT:-$HOME/Documents/gm_ucb_project/csci642-final-v3-staging}
C=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3
mkdir -p $W/pkgs
specs=()
for spec in "$@"; do
  scope=${spec%%=*} name=${spec#*=}
  [ -f $W/pkgs/$name/SHA256SUMS ] || rsync -a --partial gc211:$C/packages/$name/ $W/pkgs/$name/
  specs+=("$scope=$W/pkgs/$name")
done
STAGE=$W/stage/$TAG
[ -d $STAGE ] || bash $ROOT/scripts/rl_stage_release.sh $STAGE $TAG $COMMIT "$TITLE" /dev/null "${specs[@]}"
CFG=(); [ "$CONFIG" != - ] && CFG=(--config "$CONFIG")
python3 $ROOT/scripts/final_v3_release_notes.py --tag $TAG --commit $COMMIT --title "$TITLE" --status "$STATUS" \
  --stage-dir $STAGE "${CFG[@]}" --extra "${EXTRA_NOTES:-}" --out $W/notes_$TAG.md
REPO=ywen0511/CSCI642-Project
FULL=$(git -C $ROOT rev-parse $COMMIT)
gh release view $TAG --repo $REPO >/dev/null 2>&1 || gh release create $TAG --repo $REPO --target $FULL --title "$TITLE" --notes-file $W/notes_$TAG.md
bash $ROOT/scripts/rl_upload_missing.sh $STAGE $TAG

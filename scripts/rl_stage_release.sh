#!/bin/bash
# Combine package directories into one Release asset folder with "<scope>__<file>" names, write SHA256SUMS
# over every asset, then (optionally) create the Release and upload with retries.
# usage: rl_stage_release.sh <stage_dir> <tag> <target_sha> <title> <notes_file> <scope>=<package_dir> [...]
#        set UPLOAD=1 to create/upload; otherwise only stage.
set -euo pipefail
STAGE=$1 TAG=$2 TARGET=$3 TITLE=$4 NOTES=$5; shift 5
[ -e "$STAGE" ] && { echo "stage dir exists: $STAGE"; exit 1; }
mkdir -p "$STAGE"
for spec in "$@"; do
  scope=${spec%%=*} dir=${spec#*=}
  (cd "$dir" && sha256sum -c SHA256SUMS --quiet)  # package-internal check before staging
  for f in "$dir"/*; do cp -p "$f" "$STAGE/${scope}__$(basename "$f")"; done
done
(cd "$STAGE" && sha256sum $(ls | grep -v '^SHA256SUMS$') > SHA256SUMS)
echo "staged $(ls "$STAGE" | wc -l) assets, $(du -sh "$STAGE" | cut -f1)"
[ "${UPLOAD:-0}" = 1 ] || exit 0
REPO=ywen0511/CSCI642-Project
gh release view "$TAG" --repo $REPO >/dev/null 2>&1 || \
  gh release create "$TAG" --repo $REPO --target "$TARGET" --title "$TITLE" --notes-file "$NOTES"
for f in $(cd "$STAGE" && ls -Sr); do
  for try in 1 2 3 4; do
    timeout 3600 gh release upload "$TAG" "$STAGE/$f" --repo $REPO --clobber >/dev/null 2>&1 && { echo "OK $f"; break; }
    echo "retry $try $f"; sleep 20
  done
done
echo UPLOAD_DONE

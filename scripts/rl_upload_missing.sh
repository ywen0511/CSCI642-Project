#!/bin/bash
# Upload assets of <stage_dir> that are missing (or differ in size) on Release <tag>; repeat until complete.
# usage: rl_upload_missing.sh <stage_dir> <tag>
set -uo pipefail
STAGE=$1 TAG=$2 REPO=ywen0511/CSCI642-Project
for round in 1 2 3 4 5 6; do
  have=$(gh api "repos/$REPO/releases/tags/$TAG" --paginate -q '.assets[] | select(.state=="uploaded") | "\(.name) \(.size)"' 2>/dev/null)
  missing=()
  for f in $(cd "$STAGE" && ls -Sr); do
    sz=$(stat -c %s "$STAGE/$f")
    grep -qx "$f $sz" <<<"$have" || missing+=("$f")
  done
  echo "round $round: ${#missing[@]} missing"
  if [ ${#missing[@]} -eq 0 ]; then
    # failed partial uploads ("starter" assets) of this new Release are leftovers of our own retries: remove them
    rid=$(gh api "repos/$REPO/releases/tags/$TAG" -q .id)
    for id in $(gh api "repos/$REPO/releases/$rid/assets?per_page=100" --paginate -q '.[] | select(.state!="uploaded") | .id'); do
      gh api -X DELETE "repos/$REPO/releases/assets/$id" && echo "removed incomplete asset $id"
    done
    echo ALL_UPLOADED; exit 0
  fi
  for f in "${missing[@]}"; do
    out=$(timeout 3600 gh release upload "$TAG" "$STAGE/$f" --repo $REPO --clobber 2>&1) && echo "OK $f" || { echo "FAIL $f: $(echo "$out" | tail -1)"; sleep $((30*round)); }
  done
done
echo INCOMPLETE; exit 1

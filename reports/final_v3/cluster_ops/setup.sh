#!/bin/bash
# final-v3 setup (cpu_ubuntu_wrn): repo checkout, RL env from the historical lock, OSMesa prefix, frozen feedback,
# restore of the v2 validation (w010, w050) and final suites from their Releases (hash-verified).
set -x
C=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3
CONDA=/data/rtxw01/nbak/robot_intel/kztrgg/miniconda3/bin/conda
export CONDA_PKGS_DIRS=/data/rtxw01/nbak/robot_intel/kztrgg/conda_pkgs PIP_CACHE_DIR=/data/rtxw01/nbak/robot_intel/kztrgg/pip_cache
mkdir -p $C/logs $C/restore
cd $C
if [ ! -d repo/.git ]; then
  mkdir -p repo && git -C repo init -q && git -C repo remote add origin https://github.com/ywen0511/CSCI642-Project.git
fi
git -C repo fetch -q origin final-v3 && git -C repo checkout -q -B final-v3 FETCH_HEAD && git -C repo log --oneline -1
[ -x venv/rl/bin/python ] || $CONDA create -y -q -p $C/venv/rl python=3.12.3 pip
PY=$C/venv/rl/bin/python
$PY -m pip install -q -r repo/rl/requirements-lock.cluster.txt
$PY -m pip freeze > logs/pip_freeze_rl.txt
diff <(sort -f repo/rl/requirements-lock.cluster.txt) <(sort -f logs/pip_freeze_rl.txt) && echo LOCK_IDENTICAL || echo LOCK_DIFF
$PY -V
[ -d gl ] || $CONDA create -y -q -p $C/gl -c conda-forge mesalib=24.3
ls gl/lib/libOSMesa* 
cd repo && $PY scripts/download_feedback.py --out-dir artifacts/rl_feedback; cd $C
# restore v2 suites
for TAG in rl-v2-validation-w010-20261110-cfe0305 rl-v2-validation-w050-20261110-cfe0305 rl-v2-final-20261110-25bb0f4; do
  D=$C/restore/$TAG/download; mkdir -p $D
  curl -sfL "https://api.github.com/repos/ywen0511/CSCI642-Project/releases/tags/$TAG" -o $D/../release.json
  $PY - $D/../release.json > $D/../urls.txt <<'PYEOF'
import json,sys
for a in json.load(open(sys.argv[1]))['assets']: print(a['browser_download_url'], a['name'], a['size'])
PYEOF
  while read url name size; do
    [ -f $D/$name ] && [ $(stat -c %s $D/$name) = $size ] && continue
    curl -sfL --retry 5 -o $D/$name "$url" || echo "DLFAIL $name"
  done < $D/../urls.txt
  (cd $D && sha256sum -c SHA256SUMS --quiet && echo "SUMS_OK $TAG") || echo "SUMS_BAD $TAG"
done
echo SETUP_DONE

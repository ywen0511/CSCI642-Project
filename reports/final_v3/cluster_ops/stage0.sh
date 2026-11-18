#!/bin/bash
# Stage 0 gate: tests on the merge (RL + decoder suites), decoder env from the decoder study lock,
# restore of the v2 suites from the Releases, centers recomputation check.
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
CONDA=/data/rtxw01/nbak/robot_intel/kztrgg/miniconda3/bin/conda
export CONDA_PKGS_DIRS=/data/rtxw01/nbak/robot_intel/kztrgg/conda_pkgs PIP_CACHE_DIR=/data/rtxw01/nbak/robot_intel/kztrgg/pip_cache
cd $C/repo
git fetch -q origin final-v3-experiments && git checkout -q -B final-v3-experiments FETCH_HEAD
echo "HEAD $(git rev-parse HEAD) dirty=[$(git status --porcelain --untracked-files=no)]"
mkdir -p $C/stage0
$PY -m unittest rl.test_stage2 -v > $C/stage0/test_stage2.log 2>&1; echo "RL_TESTS rc=$?"; tail -3 $C/stage0/test_stage2.log
# decoder test env (Python 3.11, decoder study lock)
[ -x $C/venv/dec/bin/python ] || $CONDA create -y -q -p $C/venv/dec python=3.11 pip
$C/venv/dec/bin/python -m pip install -q -r requirements-lock.txt > $C/stage0/dec_pip.log 2>&1; echo "DEC_PIP rc=$?"
$C/venv/dec/bin/python -m pip freeze > $C/stage0/dec_pip_freeze.txt
$PY scripts/download_feedback.py --profile configs/decoder_improvement/rl_feedback_vnext.json --out-dir $C/fb_vnext > $C/stage0/fb_vnext.log 2>&1; echo "VNEXT_DL rc=$?"
VNEXT_DIR=$C/fb_vnext $C/venv/dec/bin/python -m unittest tests.test_dimp tests.test_feedback_vnext tests.test_closeout -v > $C/stage0/test_decoder.log 2>&1; echo "DEC_TESTS rc=$?"; tail -3 $C/stage0/test_decoder.log
# restore v2 suites
R=$C/restore
for spec in validation_w010:rl-v2-validation-w010-20261110-cfe0305:rl_v2_validation/candidate_w010 \
            validation_w050:rl-v2-validation-w050-20261110-cfe0305:rl_v2_validation/candidate_w050 \
            final:rl-v2-final-20261110-25bb0f4:rl_v2_final; do
  IFS=: read scope tag dest <<< "$spec"
  [ -d $R/suites/$dest ] || { mkdir -p $(dirname $R/suites/$dest); $PY scripts/final_v3_restore_suite.py --download $R/$tag/download --scope $scope --out $R/suites/$dest --report $C/stage0/restore_$scope.json > $C/stage0/restore_$scope.log 2>&1; echo "RESTORE $scope rc=$?"; }
  $PY -m rl.summarize $R/suites/$dest > $C/stage0/summarize_$scope.log 2>&1; echo "SUMMARIZE $scope rc=$?"
done
$PY scripts/final_v3_compute_centers.py --feedback-dir artifacts/rl_feedback --out $C/stage0/centers.json 2>&1 | tail -3 || true
echo STAGE0_DONE

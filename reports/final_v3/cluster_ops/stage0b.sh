#!/bin/bash
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
cd $C/repo; echo "HEAD $(git rev-parse HEAD)"
$C/venv/dec/bin/python -m pip install -q --force-reinstall --no-deps nvidia-nccl-cu13==2.30.7; echo "NCCL_FIX rc=$?"
$C/venv/dec/bin/python -m unittest tests.test_dimp tests.test_closeout -v > $C/stage0/test_decoder_dec_env.log 2>&1; echo "DEC_TESTS rc=$?"; tail -3 $C/stage0/test_decoder_dec_env.log
VNEXT_DIR=$C/fb_vnext $PY -m unittest tests.test_feedback_vnext rl.test_stage2 -v > $C/stage0/test_vnext_rl_env.log 2>&1; echo "VNEXT_TESTS rc=$?"; tail -3 $C/stage0/test_vnext_rl_env.log
echo STAGE0B_DONE

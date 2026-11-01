#!/bin/bash
# usage: job_verify.sh <download_dir> <tag>  -- verify a set of (downloaded) release assets
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg/scripts_ops/job_env.sh
DL=$1; TAG=$2; V=$P/verify/$TAG
rm -rf $V; mkdir -p $V
EXP=""
[ -f $DL/_expected_sizes.json ] && EXP="--expected-sizes $DL/_expected_sizes.json"
$PY scripts/verify_release.py --download-dir $DL --extract-dir $V/extract --prep-dir $PREP --out $V/verify.json $EXP
for r in $($PY -c "import json;m=json.load(open('$DL/manifest.json'));print(' '.join(e['run_id'] for e in m['checkpoints'].values() if e['group']=='main'))"); do
  $PY scripts/check_resume.py --mode continue --ckpt $V/extract/runs/$r/checkpoints/last.ckpt --prep-dir $PREP \
      --work-dir $V/resume_work_$r --epochs 2 --out $V/resume_$r.json
done
echo JOB_DONE

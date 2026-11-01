#!/bin/bash
P=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg
export CUDA_CACHE_PATH=/tmp/kztrgg_cuda_cache MPLCONFIGDIR=/tmp/kztrgg_mpl TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HOME=/data/rtxw01/nbak/robot_intel/hf_cache ERRP_WEIGHTS_DIR=$P/weights CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONWARNINGS=ignore::FutureWarning
cd $P/repo; PY=$P/env/bin/python; PREP=$P/data/derived/prep_v1; OUT=$P/runs_smoke
for m in eegnet cbramod; do
  $PY scripts/check_resume.py --mode equivalence --config configs/models/$m.yaml --prep-dir $PREP --work-dir $OUT/resume_eq_$m --epochs 3 --out $OUT/resume_eq_$m.json
done
for m in eeginception_erp eegnex ctnet eegconformer labram eegpt; do
  $PY scripts/train.py --config configs/models/supplementary/$m.yaml --run-id bench_$m --prep-dir $PREP --out-root $OUT --max-epochs 2 --no-resume 2>&1 | grep -v Warning | tail -6
done
echo SMOKE2_DONE

#!/bin/bash
# GPU smoke test + benchmark + resume check (one self-terminating job).
P=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg
export CUDA_CACHE_PATH=/tmp/kztrgg_cuda_cache MPLCONFIGDIR=/tmp/kztrgg_mpl TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HOME=/data/rtxw01/nbak/robot_intel/hf_cache ERRP_WEIGHTS_DIR=$P/weights CUBLAS_WORKSPACE_CONFIG=:4096:8
cd $P/repo
PY=$P/env/bin/python
PREP=$P/data/derived/prep_v1
OUT=$P/runs_smoke
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print((torch.randn(1000,1000,device='cuda')@torch.randn(1000,1000,device='cuda')).sum().item())" || exit 3
set -x
# 1. overfit sanity on a small balanced training subset
$PY scripts/train.py --config configs/models/eegnet.yaml --run-id smoke_overfit_eegnet --prep-dir $PREP --out-root $OUT --smoke-train-subset 64 --max-epochs 60 --override train.patience=1000 --override train.periodic_ckpt_every=0 --no-resume
$PY scripts/train.py --config configs/models/cbramod.yaml --run-id smoke_overfit_cbramod --prep-dir $PREP --out-root $OUT --smoke-train-subset 64 --max-epochs 25 --override train.patience=1000 --override train.periodic_ckpt_every=0 --no-resume
# 2. short benchmark on the full training split (3 epochs)
$PY scripts/train.py --config configs/models/eegnet.yaml --run-id bench_eegnet --prep-dir $PREP --out-root $OUT --max-epochs 3 --no-resume
$PY scripts/train.py --config configs/models/cbramod.yaml --run-id bench_cbramod --prep-dir $PREP --out-root $OUT --max-epochs 3 --no-resume
# 3. resume: continue the benchmark runs from last.ckpt for one more epoch
$PY scripts/train.py --config configs/models/eegnet.yaml --run-id bench_eegnet --prep-dir $PREP --out-root $OUT --max-epochs 4
$PY scripts/train.py --config configs/models/cbramod.yaml --run-id bench_cbramod --prep-dir $PREP --out-root $OUT --max-epochs 4
echo SMOKE_DONE

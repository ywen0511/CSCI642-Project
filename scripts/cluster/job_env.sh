# sourced by every GPU job script
P=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-eeg
export CUDA_CACHE_PATH=/tmp/kztrgg_cuda_cache MPLCONFIGDIR=/tmp/kztrgg_mpl TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HOME=/data/rtxw01/nbak/robot_intel/hf_cache ERRP_WEIGHTS_DIR=$P/weights CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONWARNINGS=ignore::FutureWarning
cd $P/repo; PY=$P/env/bin/python; PREP=$P/data/derived/prep_v1; RUNS=$P/artifacts/runs
mkdir -p $RUNS
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
echo "SOURCE_SHA=$(cat SOURCE_SHA) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host=$(hostname)"

#!/bin/bash
# bash bench/worker.sh TASKFILE PARALLEL : runs the task lines with PARALLEL concurrent processes, then exits.
B=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench
export CUDA_CACHE_PATH=/tmp/kztrgg_cuda MPLCONFIGDIR=/tmp/kztrgg_mpl XDG_CACHE_HOME=/tmp/kztrgg_xdg HF_HOME=$B/hf_cache \
  ERRP_WEIGHTS_DIR=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-decoder/weights CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONWARNINGS=ignore EB_THREADS=${EB_THREADS:-2} OMP_NUM_THREADS=${EB_THREADS:-2} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} tasks=$(wc -l < $1) start=$(date -u +%FT%TZ)"
while read -r line; do o=$(echo "$line" | grep -o -- '--out [^ ]*' | cut -d' ' -f2); mkdir -p "$(dirname "$o")"; done < "$1"
xargs -P "$2" -d '\n' -I{} bash -c '{}' < "$1"
echo "end=$(date -u +%FT%TZ) done=$(grep -c . "$1")"

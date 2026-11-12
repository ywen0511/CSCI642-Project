#!/bin/bash
# Usage: run_one.sh TASK CONDITION SEED OUT_DIR [SMOKE_STEPS]
# One CPU core, torch threads = 1 (protocol), caches on /tmp (avoid NFS lock hangs).
set -euo pipefail
TASK=$1; COND=$2; SEED=$3; OUT=$4; STEPS=${5:-}
CODE=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench/code_rl
PY=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-rl/repo/venv/rl/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
T=/tmp/kz_errpbench_${LSB_JOBID:-$$}
mkdir -p $T
export MPLCONFIGDIR=$T/mpl XDG_CACHE_HOME=$T/xdg CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
trap 'rm -rf $T' EXIT
cd $CODE
if [ -e "$OUT/status.json" ] && grep -q '"completed": true' "$OUT/status.json"; then
  echo "already complete: $OUT"; exit 0
fi
if [ -e "$OUT" ]; then  # incomplete earlier attempt: move aside, never overwrite
  mv "$OUT" "$OUT.failed_$(date +%s)"
fi
mkdir -p "$(dirname "$OUT")"
ARGS=(--task "$TASK" --condition "$COND" --seed "$SEED" --out "$OUT")
[ -n "$STEPS" ] && ARGS+=(--steps "$STEPS")
echo "host=$(hostname) start=$(date -u +%FT%TZ) ${ARGS[*]}"
$PY -u bench/rl_frontier/train_frontier.py "${ARGS[@]}"
echo "end=$(date -u +%FT%TZ)"

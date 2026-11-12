#!/bin/bash
# bash bench/qworker.sh QUEUE_DIR SLOTS : SLOTS parallel loops; each claims one task file from QUEUE_DIR/todo by an
# atomic mv into QUEUE_DIR/running, runs it, moves it to done/ if the fit wrote metrics.json, else back to todo/
# (at most 3 attempts, then failed/). Exits when todo/ is empty and its own tasks are finished.
B=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/errp-bench
Q=$1; SLOTS=$2
export CUDA_CACHE_PATH=/tmp/kztrgg_cuda MPLCONFIGDIR=/tmp/kztrgg_mpl XDG_CACHE_HOME=/tmp/kztrgg_xdg HF_HOME=$B/hf_cache \
  ERRP_WEIGHTS_DIR=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-decoder/weights CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONWARNINGS=ignore EB_THREADS=${EB_THREADS:-2} OMP_NUM_THREADS=${EB_THREADS:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $Q/todo $Q/running $Q/done $Q/failed
echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} slots=$SLOTS start=$(date -u +%FT%TZ) job=$LSB_JOBID"
slot() {
  while true; do
    t=""
    for f in $(ls $Q/todo 2>/dev/null | shuf -n 20); do
      if mv $Q/todo/$f $Q/running/$f.$LSB_JOBID.$1 2>/dev/null; then t=$f; break; fi
    done
    [ -z "$t" ] && { [ -z "$(ls $Q/todo)" ] && return; sleep 5; continue; }
    r=$Q/running/$t.$LSB_JOBID.$1
    o=$(grep -o -- '--out [^ ]*' $r | cut -d' ' -f2); mkdir -p "$(dirname "$o")"
    bash $r
    if [ -f "$o/metrics.json" ]; then mv $r $Q/done/$t
    else n=$(( $(cat $Q/failed/$t.n 2>/dev/null || echo 0) + 1 )); echo $n > $Q/failed/$t.n
      if [ $n -ge 3 ]; then mv $r $Q/failed/$t; else mv $r $Q/todo/$t; sleep 30; fi
    fi
  done
}
for i in $(seq 1 $SLOTS); do slot $i & pids="$pids $!"; sleep 2; done
wait $pids
echo "end=$(date -u +%FT%TZ)"

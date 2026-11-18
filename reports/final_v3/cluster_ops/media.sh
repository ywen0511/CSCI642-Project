#!/bin/bash
# media.sh <tool_commit> <suite> <out_name>: representative trajectories (RL seed 0, EEG participant s02, goals 30000-30002),
# all-final-policy rollouts + table, and two-stage GIFs (torch record -> torch-free OSMesa render, replay checked).
source /data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3/ops/env.sh
SHA=$1 S=$2 NAME=$3; CO=$C/checkouts/${SHA:0:7}
git -C $C/repo fetch -q origin final-v3-experiments; [ -d $CO ] || git -C $C/repo worktree add -q --detach $CO $SHA
cd $CO; M=$C/media/$NAME; rm -rf $M; mkdir -p $M/gif
fin() { ls -d $C/runs/$S/$1/checkpoints/final_* | tail -1; }
REP=(); for r in sparse_shared_s0 dense_shared_s0 ideal_shared_s0 eegnet_s02_s0 cbramod_s02_s0 independent_eegnet_s02_s0 independent_cbramod_s02_s0; do REP+=(--run $r=$(fin $r)); done
$PY -m rl.trajectories "${REP[@]}" --seeds 30000 30001 30002 --out $M/trajectories_representative > $M/trajectories_representative.log 2>&1; echo "TRAJ_REP rc=$?"
ALL=(); for d in $C/runs/$S/*/; do r=$(basename $d); ALL+=(--run $r=$(fin $r)); done
$PY -m rl.trajectories "${ALL[@]}" --seeds 30000 30001 30002 --out $M/trajectories_all_final_runs --no-plot > $M/trajectories_all_stdout.json 2>&1; echo "TRAJ_ALL rc=$?"
$PY scripts/rl_trajectory_table.py $M/trajectories_all_final_runs/trajectories.json --out $M/trajectory_table.md > /dev/null 2>&1; echo "TABLE rc=$?"
for r in sparse_shared_s0 dense_shared_s0 ideal_shared_s0 eegnet_s02_s0 cbramod_s02_s0 independent_eegnet_s02_s0 independent_cbramod_s02_s0; do
  $PY -m rl.render_rollout record $(fin $r) --seeds 30000 30001 30002 --out $M/gif/$r.actions.json > /dev/null 2>&1 || echo "RECORD_FAIL $r"
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH=$C/gl/lib:$LD_LIBRARY_PATH $PY -m rl.render_rollout render $M/gif/$r.actions.json --out $M/gif/$r.gif > $M/gif/$r.render.log 2>&1 || echo "RENDER_FAIL $r"
done
ls $M/gif/*.gif | wc -l; cat $M/gif/*.gif.json | grep -h -E "replay_matches|pillar_drawn" | sort | uniq -c
echo MEDIA_DONE

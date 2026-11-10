# RL protocol v2: execution record

Assignment: [`GPU_AGENT_HANDOFF.md`](../GPU_AGENT_HANDOFF.md) (source branch `rl-proposal-v2`,
`450205ed7d97f2ee6c7fbe37ba5542bfe3bf4a6b`). Execution branch: `rl-v2-experiments`, created from that commit.
Plan declared before validation: [`rl_v2_experiment_plan.md`](rl_v2_experiment_plan.md).

## 1. Hardware and environment

| item | value |
|---|---|
| compute used for every RL run | GM Warren LSF queue `cpu_ubuntu_wrn`: Intel Xeon Gold 6242 @ 2.80 GHz, 32 cores and 187 GB RAM per node, Ubuntu 24.04.5; one core per run, one torch thread |
| GPU | one 1-GPU throughput benchmark was submitted to gc175 (A100-SXM4-40GB); see Section 5 |
| durable storage | project filer `/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-rl` (`repo/` git checkout of the execution branch, `repo/venv/rl`, `repo/artifacts/`) |
| Python | 3.12.3 |
| key packages | gymnasium 1.4.0, gymnasium-robotics 1.4.2, mujoco 3.11.0, stable-baselines3 2.9.2, torch 2.15.0 (+cu132 wheel, used on CPU), numpy 2.4.6; matplotlib 3.10.8 for figures only |
| full lock | [`rl/requirements-lock.cluster.txt`](../rl/requirements-lock.cluster.txt) (`pip freeze` of the run environment) |
| rendering | headless OSMesa from a separate conda prefix (`mesalib` 24.3, conda-forge), `MUJOCO_GL=osmesa`; used only for GIFs |

Deviation from the handoff's setup commands: the compute nodes' `/usr/bin/python3.12` has no `ensurepip`, so
`python3.12 -m venv` fails. The same `venv/rl` prefix was created with conda (`python=3.12`) and
`pip install -r rl/requirements.txt` was run inside it. The EEG environment was not touched.

Workspace provenance: the filer checkout was created with `git init`, `git fetch` of the execution branch from
`ywen0511/CSCI642-Project` and a checkout of the fetched commit (no `git clone`, no disconnected history). Every
run records `git rev-parse HEAD`, a clean-tree check is printed in every job log, and `run.json` records SHA-256
hashes of the four training sources (`train.py`, `environment.py`, `feedback.py`, `checkpoint.py`).

## 2. Feedback inputs

`python scripts/download_feedback.py --out-dir artifacts/rl_feedback` downloaded `validation_predictions.csv`,
`test_predictions.csv` and `manifest.json` from Release `eeg-stage1-20261101-bccbb8a`; sizes, SHA-256, schema,
decoder seed 42, checkpoint/preprocessing/calibration IDs and participant coverage all matched
`configs/rl_feedback.json` (`ok: true`). No EEG model was retrained, recalibrated or re-run.

## 3. Tests and smoke runs

`python -m unittest rl.test_stage2 -v`: **30 tests pass** on the execution code (21 inherited + 9 added for seed
lists, suite sizes 33/45/75, disjoint episode seeds, evaluation-seed separation, AUC/duplicate handling, AUROC,
macro aggregation, the SB3 `info` key and observation content).

`python -m rl.smoke --out artifacts/rl_v2_smoke_r4` on training commit `8131761` (wall time 57 s, CPU), report
[`reports/rl_v2_smoke/smoke_report.json`](rl_v2_smoke/smoke_report.json):

- FetchReach contract: 50 random steps, sparse reward and success match the 5 cm threshold, truncation at 50.
- Scripted real-simulator feedback check (ideal feedback, w = 0.5): holding still moved the gripper 0.71 mm
  (inside the 1 mm deadband) -> no label, no sample, EEG reward 0; moving toward the goal for 5 steps ->
  label 0, EEG reward +0.25; moving away -> label 1, EEG reward -0.25; exactly 2 events counted.
- All seven conditions: 250 simulator steps (5 episodes), 230 SAC gradient updates, final checkpoint reload with
  replay restoration and fixed inference probes reproduced exactly (max |action difference| 0.0, tolerance 1e-6).
- Transition logs: every labeled segment has the label implied by the sign of the distance change; ideal
  rewards are exactly +-w/2; decoded rewards equal `w (0.5 - p_error)` with source label = simulated label
  (independent controls excepted); sparse/dense have no EEG events; neutral rows have blank label, probability,
  trial and event fields.
- The policy observation space is exactly `{observation, achieved_goal, desired_goal}` (shapes 10/3/3);
  no probability or label enters the policy.
- Periodic snapshot (step 80) verified and continued to step 120 in a new directory: restored 80 replay
  transitions, then more updates and replay entries; final snapshot verified.

Earlier smoke runs (`rl_v2_smoke`, `_r2`, `_r3`) passed on earlier commits and are kept as history only.

## 4. Problems found and fixed during execution

1. **SB3 `info["episode"]` collision (inherited).** The first pilot batch (16 runs) failed after 4 episodes:
   the wrapper copied its integer episode counter into `info["episode"]`, which SB3's logger reads as the
   Monitor's episode-statistics dict. Fixed by exposing it as `episode_index` (CSV column unchanged); test added;
   smoke lengthened to 250 steps so the logging path is exercised. Failed runs are preserved as
   `rl_v2_pilots_attempt1_failed_sb3_info_episode` and packaged as a failure record.
2. **Two different policies evaluated at the final step (inherited).** SB3 runs the rollout callback (periodic
   evaluation), then one more gradient update; the runner then evaluated the final policy at the same step
   count. In the 150k pilots the two records differed (e.g. 0.85 vs 0.95 success). The periodic evaluation at
   the final boundary is now skipped; evaluations record `n_updates`; duplicates must be identical.
3. **Correlated training goals across RL seeds (inherited).** Episode *k* of seed *s* used goal seed `s + k`,
   making seeds shifted copies of one goal sequence. Now `s * 1,000,000 + k`, with range checks.
4. Pilot logging: the first job wrapper filtered stderr and hid the traceback; fixed (resource usage now goes
   to a separate file).

## 5. Pilots and throughput

See plan Section 7 and [`reports/rl_v2_pilots/pilot_report.json`](rl_v2_pilots/pilot_report.json). Pilots ran on
commit `3113b6b`-`5454ec0` code (before fix 2 above); for the five 150k pilots the report keeps the final
policy's evaluation and records the dropped pre-update value. Throughput 50-56 environment steps/s per core.

GPU benchmark: GPU_BENCH_PLACEHOLDER

## 6. Validation and final execution

RUNS_PLACEHOLDER

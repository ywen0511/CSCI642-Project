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

`python -m rl.smoke --out artifacts/rl_v2_smoke_r4` on commit `cfe0305` (training sources identical to `8131761`; wall time 57 s, CPU), report
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
commit `50888a9` (after fix 1, before fix 2 above); for the five 150k pilots the report keeps the final
policy's evaluation and records the dropped pre-update value. Throughput 50-56 environment steps/s per core.

GPU benchmark: an identical 20k-step dense run with `--device cuda` was submitted to gc175 (1 GPU) at the start
of the pilots. It stayed pending for the whole execution because all gc175 GPUs were occupied by another user's
jobs (LSF: "Host's available GPU resources cannot meet the job's requirements"); it never held a GPU and was cancelled at the end of the final suite. No GPU
speedup is therefore claimed or measured, and every reported run used the CPU. All checkpoints are CPU-trained,
so there is no GPU-to-CPU inference roundoff to measure: CPU reload reproduced the stored probe actions exactly.

## 6. Validation and final execution

All formal runs were launched with `rl.suite --prepare` (writes `plan.json` from the protocol's seed lists) and
one LSF job per planned run, `rl.suite --run-job NAME` (same `rl.train` command and output layout as
`--execute`), one CPU core each, wall limit 4 h (actual 55-60 min). Clean checkout verified before each launch.

| suite | config | commit | runs | env. steps | completed | core-hours | wall time |
|---|---|---|---|---|---|---|---|
| pilots (50k/150k/20k) | `rl/protocol.json` + `--steps` | `50888a9` | 15 | 1.22 M | 15 | 6.3 | ~50 min |
| validation `candidate_w010` | `configs/rl_v2/candidate_w010.json` | `cfe0305` | 33 | 4.95 M | 33 | | |
| validation `candidate_w050` | `configs/rl_v2/candidate_w050.json` | `cfe0305` | 33 | 4.95 M | 33 | 91.7 (all three) | ~1 h 5 min (all three in parallel) |
| validation `candidate_w100` | `configs/rl_v2/candidate_w100.json` | `cfe0305` | 33 | 4.95 M | 33 | | |
| final | `configs/rl_v2/frozen/protocol.json` | `25bb0f4` | 75 | 11.25 M | 75 | 68.8 | ~1 h |

No formal run failed, was interrupted or resumed. `rl.summarize` passed for every suite: complete plan, protocol
identity, run identity, no resumption, full budget, shared evaluation goals, identical training-source hashes
(the training sources of `cfe0305` and `25bb0f4` are identical; later commits changed analysis, reports and
release tooling only).

Selection and freeze: `scripts/rl_select_candidate.py` (committed before the validation results existed) applied
the declared rule and wrote the rationale; `rl.freeze` wrote `protocol.json` with the evidence hash; both were
committed (`25bb0f4`) before the final suite was launched, and the final suite ran from that committed copy
(byte-identical to the freeze output). Sparse and dense validation runs, which do not use the EEG weight, were
bit-for-bit identical across the three candidate suites.

Post-processing (CPU): `rl.analysis` for every suite; `rl.trajectories` for the representative runs and for all
75 final policies; `rl.render_rollout` for GIFs. A first media job crashed (segmentation fault) when torch and the
OSMesa library were loaded in the same process; GIF rendering was split into a torch-only rollout recording and a
torch-free replay/render step that verifies the replayed positions (maximum deviation 0.0 m).

Storage: raw run directories 4.2 GB (pilots), 49 GB (validation), 37 GB (final) on the project filer.

## 7. Delivery and fresh-download verification

Six new Releases were created in `ywen0511/CSCI642-Project`, each tag pinned to the training commit of its runs;
existing EEG tags and Releases were not changed. Assets were packaged on the cluster (`rl.package`; archives above
100 MB split losslessly into `.part-NN`), copied to the laptop and uploaded with `gh` (`scripts/rl_stage_release.sh`,
`scripts/rl_upload_missing.sh`). Upload problems and how they were handled:

- Intermittent TLS errors (`bad record MAC`) on the laptop's connection made large single files fail repeatedly;
  later packages use 100 MB parts and the upload loop re-uploads only missing or size-mismatched assets.
- A failed upload left an incomplete ("starter") asset in the pilot Release, which made the first pilot
  verification fail; it was removed (its content is published as verified parts), and the upload script now
  removes such leftovers. The first verification report is kept as
  `rl-v2-pilots-20261110-50888a9.attempt_with_starter_asset.json`.
- The laptop-cluster link dropped to about 1.4 MB/s. At the user's instruction ("upload only important ckpt,
  upload through this computer"), the final-suite and w = 0.1 Releases were re-packaged with final snapshots only
  (`rl.package --final-snapshots-only`); their partial full-scope uploads were deleted first so that no Release
  mixes scopes. The w = 0.5 and w = 1.0 validation Releases had already completed with every snapshot.

Verification results are in the final project report, Section 8, and in `reports/rl_v2_release_verification/`.
All six Releases pass; two continuation checks from downloaded snapshots pass.

Durable copies: every run directory (including all periodic snapshots and replay buffers) remains on the project
filer under `repo/artifacts/` (pilots 4.2 GB, validation 49 GB, final 37 GB), together with the full-scope
packages in `repo/artifacts/rl_v2_release/`.

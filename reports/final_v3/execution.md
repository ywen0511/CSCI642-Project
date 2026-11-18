# Final round: execution record

Assignment: [FINAL_V3_HANDOFF.md](../../FINAL_V3_HANDOFF.md). Branch `final-v3-experiments` (from `final-v3` at
`81813de`). Status: [`status.json`](status.json); every declared rule, freeze and deviation, with UTC timestamp and
commit: [`decisions.jsonl`](decisions.jsonl) (append-only).

## 1. Environment and hardware

- **Scheduler and hardware.** GM Warren LSF, queue `cpu_ubuntu_wrn`, Intel Xeon Gold 6242 (2.80 GHz; 32 cores and
  187 GB per node). Every RL run was one LSF job with one core and one torch thread (`OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `PYTHONHASHSEED=0`). No GPU was used for RL; every checkpoint was trained on CPU, as in v2.
  The host, CPU model, checkout commit and `git status` of every job are printed at the start of its LSF log, and
  each job refuses to run in a dirty checkout.
- **RL environment.** `venv/rl`: Python 3.12.3, with `pip freeze` identical to
  [`rl/requirements-lock.cluster.txt`](../../rl/requirements-lock.cluster.txt) (the v2 lock: gymnasium 1.4.0,
  gymnasium-robotics 1.4.2, mujoco 3.11.0, stable-baselines3 2.9.2, torch 2.15.0, numpy 2.4.6). Freezes:
  [`stage0/`](stage0/).
- **Decoder test environment** (Stage 0 tests only): Python 3.11 from the decoder branch lock, with the documented
  NCCL reinstall ([`../decoder_improvement/REPRODUCE.md`](../decoder_improvement/REPRODUCE.md)). No decoder was
  retrained.
- **GIF rendering.** OSMesa (Mesa 24.3, built into the workspace), torch-free render stage, as in v2.
- **Workspace.** The v2 filer directories had been purged on 2026-11-13, so a new filer workspace was created
  (`git init` + fetch of this repository, so commits stay attached to the real history). The v2 validation suites
  needed for the sensitivity freeze were restored from their Releases, with every hash checked
  ([`scripts/final_v3_restore_suite.py`](../../scripts/final_v3_restore_suite.py), reports in [`stage0/`](stage0/)).
  Every suite ran from a pinned, clean `git worktree` of its training commit.
- **Storage.** About 700 TB were free on the filer. All periodic snapshots of every run are kept there. The Releases
  carry final snapshots plus all logs, and list every periodic snapshot with its SHA-256 in the package manifest.
- **GitHub.** Write access was confirmed with the `gh` API (`push: true`). The cluster has no `gh`, so Releases were
  staged on the filer and uploaded from a workstation with `gh` ([`scripts/final_v3_publish.sh`](../../scripts/final_v3_publish.sh)).
  Fresh-download verification ran on the cluster with anonymous API downloads.

## 2. Commands

Cluster wrappers (LSF submission, post-processing, packaging, verification) are in [`cluster_ops/`](cluster_ops/).
Each wrapper creates or reuses a pinned worktree of the commit it is given and then calls the repository tools
below (paths relative to the repository root, environment `venv/rl`).

```bash
# Stage 0
python -m unittest rl.test_stage2 rl.test_final_v3 -v
python scripts/final_v3_restore_suite.py --download <Release download> --scope validation_w010 --out <suite> --report <json>  # hash-checked
# Stage 1: freeze from existing v2 validation evidence, then one final suite per weight
python -m rl.freeze --validation-suite <v2 validation candidate_w010> --rationale "..." --out configs/final_v3/sensitivity_w010
python -m rl.suite --config configs/final_v3/sensitivity_w010/protocol.json --phase final --out runs/sensitivity_w010_final --prepare
python -m rl.suite ... --run-job <job>                  # one LSF job per planned run (cluster_ops/launch_suite.sh)
python scripts/final_v3_determinism_check.py --new runs/sensitivity_w010_final --reference <v2 final runs> --out sensitivity_determinism.json
# Stage 2
python scripts/final_v3_compute_centers.py --feedback-dir artifacts/rl_feedback --out configs/final_v3/centers.json
python scripts/final_v3_make_configs.py ...             # v3 candidates w 0.5/1/2 and preference candidates
python -m rl.smoke --config configs/final_v3/v3_candidates/v3_w050.json --out reports/final_v3/v3_smoke
python scripts/rl_select_candidate.py runs/v3_validation_w{050,100,200} --metric auc --out selection.json --markdown selection.md
python -m rl.freeze --validation-suite runs/v3_validation_w050 --rationale "..." --out configs/final_v3/v3_frozen
# Stage 3
python scripts/final_v3_pref_geometry_check.py --config configs/final_v3/pref_candidates/pref_w100.json --out reports/final_v3/preference/geometry_corridor10_90
python scripts/final_v3_pilot_report.py --variant round1=... --variant g1=... --out reports/final_v3/preference/pilot_report.json
python scripts/rl_select_candidate.py runs/pref_validation_w{100,200,400} --metric preferred_pass --out selection.json --markdown selection.md
python -m rl.freeze --validation-suite runs/<selected> --rationale "..." --out configs/final_v3/pref_frozen
# every suite
python -m rl.summarize runs/<suite>
python -m rl.analysis runs/<suite> --feedback-dir artifacts/rl_feedback --out reports/final_v3/<suite>
python scripts/final_v3_write_reports.py sensitivity|v3|preference ...   # generated tables + number audit
# media (cluster_ops/media.sh): trajectories, all-final-policy table, two-stage GIFs
python -m rl.trajectories --run <name>=<final snapshot> ... --seeds 30000 30001 30002 --out <dir>
python scripts/rl_trajectory_table.py <dir>/trajectories.json --out trajectory_table.md
python -m rl.render_rollout record <snapshot> --seeds 30000 30001 30002 --out a.json
MUJOCO_GL=osmesa python -m rl.render_rollout render a.json --out a.gif
# packaging, Releases and verification
python -m rl.package --runs runs/<suite> --out packages/<name> --final-snapshots-only --max-part-bytes 100000000
scripts/final_v3_publish.sh <tag> ...                   # workstation: stage, notes, gh release create, upload missing
python scripts/rl_verify_release.py --tag <tag> --download-dir <new> --extract-dir <new> --out report.json
python scripts/rl_continuation_check.py --snapshot <downloaded snapshot> --steps 2000 --checkout <clean worktree of training commit> ...
python scripts/final_v3_usage.py --runs-root runs --suites ... --out usage.json --markdown usage.md
```

## 3. Training commits

| suites | training commit | note |
|---|---|---|
| sensitivity w = 0.1, 0.5 (final) | `f63dc24` | RL training sources byte-identical to the v2 final commit `25bb0f4` |
| v3 smoke, v3 validation (3 candidates), preference smoke, round-1 pilots | `a09817f` | v3 and preference code + tests |
| v3 final | `41109cb` | frozen v3 protocol (w = 0.5) committed |
| preference pilot round 2 (G1-G3) | `afa0531` | geometry variants declared before round 2 |
| preference validation (3 candidates) | `ae633ea` | declared plan (G3) committed |
<!--PREF_FINAL_COMMIT-->

`rl.summarize` enforces identical training-source hashes within a suite.

## 4. Runs, wall time and core-hours

Generated by `scripts/final_v3_usage.py` from the run records: core-hours are the sum of each run's training
wall time (`status.json` `elapsed_seconds`, one core per run); suite wall clock runs from the first `run.json` to
the last `status.json`; killed attempts are counted from their LSF logs. Data: [`usage.json`](usage.json).

<!--USAGE-->

## 5. Failures, reruns and operational deviations

Every one is logged in [`decisions.jsonl`](decisions.jsonl) before any dependent run.

- **Duplicate submissions.** Neither duplicate produced a result.
  - A submission call timed out on the submit host but kept running, so 33 duplicate `v3_validation_w200` jobs and
    10 duplicate preference pilots were queued. The duplicates were killed while pending, or exited at once on the
    existing run directory. They did not start training. The pilot duplicates did truncate the original pilots'
    stdout logs; the run records were not affected.
  - Lag in the `bjobs` listing caused 12 duplicate `pref_validation_w400` jobs. They were killed while pending.
- **External kill of the preference validation.** All 99 runs of the first attempt (started 06:20Z) were killed
  by a signal from outside LSF at 08:04-08:05Z, on five hosts at once, at 260k-290k of 300k steps. They had not
  written a `status.json` or final evaluation, and none of their results were inspected. They were moved to
  `aborted/` on the filer and all 99 runs were resubmitted unchanged from the same commit. Their core-hours are
  counted in Section 4.
- **Continuation check.** The first attempt assumed a replay buffer that was not full, and reported `ok=false` on a
  full 100k buffer. The check was generalized to circular buffers (`c927499`) and rerun: ok. The first report is
  kept (`release_verification/*attempt1*`).
- **Pre-existing finding, not caused by this round.** Downloading the `rl_feedback_vnext` profile reports
  "Unexpected CSV schema" (3 extra columns), but its asset hashes are OK. The profile is not used for RL.
- **Pilot gate.** The declared preference gate failed in round 1 and in all three declared round-2 variants. The
  declared fallback G3 was used ([`preference_plan.md`](preference_plan.md)). This is a scientific outcome, not an
  operational failure.
- **No training run failed.** Apart from the killed preference validation attempt, every formal run completed on
  its first attempt, with no resumption, and `rl.summarize` passed for every suite.

## 6. Releases and verification

<!--RELEASES-->

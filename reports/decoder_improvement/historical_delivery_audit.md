# Historical delivery audit (EEG stage and RL v2)

Generated from a fresh paginated inventory of every Release and asset (all upload states) taken
2026-11-11T04:34:21.717675+00:00 ([`remote_inventory.json`](remote_inventory.json)) and a file-by-file comparison of every
expected historical run on the project filer with the published package manifests
([`historical_delivery_audit.json`](historical_delivery_audit.json), `scripts/decoder_improvement/audit_historical.py`).

## Remote state

| Release | assets | states | GB |
|---|---|---|---|
| `rl-v2-validation-w100-20261110-cfe0305` | 103 | {'uploaded': 103} | 8.24 |
| `rl-v2-validation-w050-20261110-cfe0305` | 103 | {'uploaded': 103} | 8.24 |
| `rl-v2-validation-w010-20261110-cfe0305` | 37 | {'uploaded': 37} | 0.82 |
| `rl-v2-smoke-20261110-cfe0305` | 45 | {'uploaded': 45} | 0.01 |
| `rl-v2-pilots-20261110-50888a9` | 40 | {'uploaded': 40} | 1.74 |
| `rl-v2-final-20261110-25bb0f4` | 89 | {'uploaded': 89} | 1.90 |
| `eeg-stage1-closeout-20261102` | 3 | {'uploaded': 3} | 0.00 |
| `eeg-stage1-20261101-bccbb8a` | 25 | {'uploaded': 25} | 5.52 |

All historical Releases were fully uploaded at audit time: no `starter`, zero-size or duplicate entries. (The
incomplete entries noted when the handoff was prepared were in-progress uploads that later completed or were
replaced; the published assets were not changed by this audit.)

## Per-run classification

| Release | categories |
|---|---|
| `rl-v2-smoke-20261110-cfe0305` | independently_published: 32 |
| `rl-v2-pilots-20261110-50888a9` | independently_published: 15, incomplete_execution: 15 |
| `rl-v2-validation-w010-20261110-cfe0305` | recoverable_upload: 33 |
| `rl-v2-validation-w050-20261110-cfe0305` | independently_published: 33 |
| `rl-v2-validation-w100-20261110-cfe0305` | independently_published: 33 |
| `rl-v2-final-20261110-25bb0f4` | recoverable_upload: 75 |
| `eeg-stage1-20261101-bccbb8a` | independently_published: 19 |

- `independently_published`: every file of the run is inside an uploaded archive whose manifest hash matches the
  local file.
- `recoverable_upload` (108 runs: RL v2 final 75, validation w010 33): every run is complete and its final snapshot,
  logs and evaluation records are published; the **15 periodic snapshots per run** (every 10,000 steps) are
  present on the filer with the manifest hashes but were not uploaded, because the final and w010 Releases were
  re-scoped to final snapshots at the user's instruction of 2026-11-10 (slow upload link).
- `incomplete_execution` (15 runs): the preserved failed first pilot attempt
  (`rl_v2_pilots_attempt1_failed_sb3_info_episode`, SB3 `info['episode']` collision), published as a failure
  record in `rl-v2-pilots-20261110-50888a9`. Not a gap.
- Missing local artifacts: **0**. Suite completeness: every suite has its expected number of completed runs
  (pilots 15, validation 3 x 33, final 75).

## Recovery

The complete original packages (built before the re-scoping and kept on the filer) were reassembled and each
archive was checked against the full-archive SHA-256 in its original package manifest (final 76 archives,
w010 34 archives, all matching). They are published, without re-running any training and without touching the
original Releases, as self-contained recovery Releases with 50 MiB parts and member-level manifests:

RECOVERY_STATUS

Original EEG stage: all 19 run archives and the predictions/manifests are published and were independently
verified after a fresh download on 2026-11-01; this audit re-confirmed the inventory.

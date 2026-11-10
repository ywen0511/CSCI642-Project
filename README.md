# EEG-Assisted Robot Learning: proposal-aligned RL follow-up

Repository: **ywen0511/CSCI642-Project**. This is branch **rl-v2-experiments**, the executed version of the
`rl-proposal-v2` handoff.

## Status on this branch (executed 2026-11-09/10)

The handoff in [GPU_AGENT_HANDOFF.md](GPU_AGENT_HANDOFF.md) was carried out end to end: v2 smoke, pilots, three
complete validation candidates, a validation-only freeze, the 75-run final comparison, analysis, trajectories,
reports and Release delivery with fresh-download verification.

- **Result:** on FetchReach with SAC (150k steps), every condition reaches high final success (0.94-1.00).
  Learning speed (normalized AUC): dense 0.915 > ideal 0.764 > sparse 0.659 ≈ EEGNet 0.641 > CBraMod 0.571.
  Replayed EEG error probabilities did not speed up learning over sparse reward; EEGNet feedback was slightly
  better than CBraMod and than its label-independent control. See [reports/rl_v2_results.md](reports/rl_v2_results.md).
- **Reports:** [final project report](reports/final_project_report.md), [results](reports/rl_v2_results.md),
  [experiment plan (declared before validation)](reports/rl_v2_experiment_plan.md),
  [execution record](reports/rl_v2_execution.md), [generated tables](reports/rl_v2_final/results_tables.md).
- **Frozen protocol and evidence:** [configs/rl_v2/frozen/](configs/rl_v2/frozen/).
- **Releases:** final comparison [rl-v2-final-20261110-25bb0f4](https://github.com/ywen0511/CSCI642-Project/releases/tag/rl-v2-final-20261110-25bb0f4); validation, pilot and smoke Releases and their fresh-download verification are listed in the [final project report, Section 8](reports/final_project_report.md#8-artifacts).

The sections below are the original handoff text from `rl-proposal-v2`, kept for reference.

## Start here

**Executing agent: read [GPU_AGENT_HANDOFF.md](GPU_AGENT_HANDOFF.md) in full, then complete its required implementation, validation, experiments, analysis, report and verified Release delivery.** That document is the complete current assignment; it takes precedence over historical stage instructions. Use this repository for source and new deliverables. No access to the private Savaw repository is needed. Do not retrain the frozen EEG decoders.

The user only needs to provide the handoff MD link. No user-side cloning or setup is required, and the executing agent must not run `git clone`. Reuse an existing workspace or retrieve the exact source archive/API content yourself. The agent must execute the work and directly upload code, results, all checkpoints and Release assets, then verify fresh downloads; do not stop at an upload-ready folder.

## What is ready

- EEGNet and CBraMod training, calibration, evaluation, feedback probabilities and EEG checkpoints were completed previously. Original assets remain in this repository's Releases.
- FetchReach/SAC training, evaluation, checkpoint/replay/RNG saving, continuation, suite planning and packaging are implemented under `rl/`.
- Protocol **v2** uses `r_task + w_eeg * (0.5 - p_error)` at eligible feedback events.
- Distance changes inside the inclusive deadband receive **no EEG feedback**, no sampled EEG trial and no binary label. Task reward is retained.
- Neutral segments still end on schedule. Logs distinguish segment boundaries, eligibility and actual feedback.

The v2 preparation performed unit/static checks only. It did **not** run SAC training, simulator smoke runs, parameter selection or final experiments. Historical v1 smoke results do not validate v2. The default 50,000 steps and five final seeds are still the old planner defaults; the next agent must implement the explicit seed/budget plan in the handoff before formal runs.

## Proposal stages and current status

| Proposal stage | Status |
|---|---|
| 1: reaching/SAC and sparse/dense baselines | **Done** on this branch: formal sparse/dense baselines in every validation and final suite (150k steps) |
| 2: EEG audit, training, calibration and evaluation | Completed earlier; weights, splits and predictions kept frozen |
| 3: EEG integration and main comparison | **Done**: smoke, pilots, 3 validation candidates, freeze, 75-run final comparison; optional route preference not performed |
| 4: analysis, curves, trajectories and report | **Done**: curves, AUC, diagnostics, trajectories/GIFs, reports, verified Releases |

Historical names `eeg-stage1` and `rl-stage2` are delivery names, not the proposal's numbered stages.

## Useful files

- [Complete next-agent assignment](GPU_AGENT_HANDOFF.md)
- [Current RL mechanics and commands](rl/README.md)
- [v2 change and verification record](reports/rl_protocol_v2.md)
- [Frozen EEG inputs](docs/rl_handoff.md) and [machine-readable profile](configs/rl_feedback.json)
- [Historical EEG specification](EEG_README.md), [measured EEG results](reports/eeg_results.md)
- [Historical v1 execution report](rl/execution_report.md)

Small feedback files can be downloaded and verified using only Python's standard library:

```bash
python3 scripts/download_feedback.py --out-dir artifacts/rl_feedback
```

This downloads from Releases in **this same repository**. Reading probabilities requires neither raw EEG nor GPU inference. For training setup use `rl/requirements.txt`, not the EEG dependencies at the repository root.

All new documentation, reports, code comments, commit messages and Release notes must be in English. Preserve every periodic/final checkpoint and replay buffer until the new Release has been freshly downloaded and verified. Do not overwrite old tags or upload raw third-party EEG.

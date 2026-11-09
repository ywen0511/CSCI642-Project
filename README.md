# EEG-Assisted Robot Learning: proposal-aligned RL follow-up

Repository: **ywen0511/CSCI642-Project**. Active branch: **rl-proposal-v2**.

## Start here

**Executing agent: read [GPU_AGENT_HANDOFF.md](GPU_AGENT_HANDOFF.md) in full, then complete its required implementation, validation, experiments, analysis, report and verified Release delivery.** That document is the complete current assignment; it takes precedence over historical stage instructions. Use this repository for source and new deliverables. No access to the private Savaw repository is needed. Do not retrain the frozen EEG decoders.

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
| 1: reaching/SAC and sparse/dense baselines | Code and historical short checks exist; formal v2 baselines must be run with the common protocol |
| 2: EEG audit, training, calibration and evaluation | Completed; keep weights, splits and predictions frozen |
| 3: EEG integration and main comparison | v2 interface implemented; new smoke, pilot, validation and final comparisons remain |
| 4: analysis, curves, trajectories and report | EEG analysis exists; integrated RL analysis/report/delivery remain |

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
